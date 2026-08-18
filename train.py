"""
We4Restore - KLA Hackathon 2026 Training Pipeline
Model: Native NAFNet Super-Resolution (Width=32, Upscale=2x)
Features: AMP FP16, MixUp Augmentation, 8x Validation TTA, Cosine Decay
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_skimage
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
import lpips


# ==========================================
# 1. DATASET PIPELINE
# ==========================================
class PairedRestorationDataset(Dataset):
    """
    Paired dataset loader for degraded (.npy) inputs and ground truth (.npy) targets.
    Applies random spatial crops and D4 dihedral group geometric augmentations.
    """
    def __init__(self, root_dir: str, patch_size: int = 256, train: bool = True):
        self.root_dir = Path(root_dir)
        self.lr_dir = self.root_dir / "NoisyLR"
        self.hr_dir = self.root_dir / "GT"
        self.patch_size = patch_size
        self.train = train

        if not self.lr_dir.exists() or not self.hr_dir.exists():
            raise FileNotFoundError(f"Missing NoisyLR or GT directories in {self.root_dir}")

        self.image_names = sorted([f for f in os.listdir(self.lr_dir) if f.lower().endswith(".npy")])
        if len(self.image_names) == 0:
            raise ValueError(f"No .npy files found in {self.lr_dir}")

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        filename = self.image_names[idx]
        lr_path = self.lr_dir / filename
        hr_path = self.hr_dir / filename

        lr_img = np.load(lr_path).astype(np.float32)
        hr_img = np.load(hr_path).astype(np.float32)

        # Standardize arrays to (C, H, W) format
        if lr_img.ndim == 2:
            lr_img = np.expand_dims(lr_img, axis=0)
        elif lr_img.ndim == 3 and lr_img.shape[-1] <= 3:
            lr_img = np.transpose(lr_img, (2, 0, 1))

        if hr_img.ndim == 2:
            hr_img = np.expand_dims(hr_img, axis=0)
        elif hr_img.ndim == 3 and hr_img.shape[-1] <= 3:
            hr_img = np.transpose(hr_img, (2, 0, 1))

        lr_tensor = torch.from_numpy(lr_img)
        hr_tensor = torch.from_numpy(hr_img)

        if self.train:
            _, h, w = lr_tensor.shape
            if h >= self.patch_size and w >= self.patch_size:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                lr_tensor = lr_tensor[:, top : top + self.patch_size, left : left + self.patch_size]
                hr_tensor = hr_tensor[:, top : top + self.patch_size, left : left + self.patch_size]

            # D4 Dihedral Symmetry Augmentations
            if random.random() < 0.5:
                lr_tensor = torch.flip(lr_tensor, [1])
                hr_tensor = torch.flip(hr_tensor, [1])
            if random.random() < 0.5:
                lr_tensor = torch.flip(lr_tensor, [2])
                hr_tensor = torch.flip(hr_tensor, [2])
            if random.random() < 0.5:
                rot = random.choice([1, 2, 3])
                lr_tensor = torch.rot90(lr_tensor, rot, [1, 2])
                hr_tensor = torch.rot90(hr_tensor, rot, [1, 2])

        return lr_tensor, hr_tensor


# ==========================================
# 2. ARCHITECTURE DEFINITION
# ==========================================
class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int, DW_Expand: int = 2, FFN_Expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.sg = SimpleGate()
        self.norm1 = nn.LayerNorm(c, eps=1e-6)
        self.norm2 = nn.LayerNorm(c, eps=1e-6)

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp.permute(0, 2, 3, 1)
        x = self.norm1(x).permute(0, 3, 1, 2)
        x = self.conv3(self.sg(self.conv2(self.conv1(x))))
        y = inp + x * self.beta

        x = y.permute(0, 2, 3, 1)
        x = self.norm2(x).permute(0, 3, 1, 2)
        x = self.conv5(self.sg(self.conv4(x)))
        return y + x * self.gamma


class NAFNetFeatureTrunk(nn.Module):
    def __init__(self, img_channel: int = 3, width: int = 32, middle_blk_num: int = 12,
                 enc_blk_nums: list = [2, 2, 4, 8], dec_blk_nums: list = [2, 2, 2, 2]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, h, w = inp.shape
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x) + enc_skip
            x = decoder(x)

        return x[:, :, :h, :w]


def icnr_init(tensor: torch.Tensor, upscale_factor: int = 2) -> torch.Tensor:
    out_channels, in_channels, kh, kw = tensor.shape
    scale_sq = upscale_factor ** 2
    sub_kernel = torch.nn.init.kaiming_normal_(torch.empty(out_channels // scale_sq, in_channels, kh, kw))
    return sub_kernel.repeat_interleave(scale_sq, dim=0)


class NativeNAFNetSR(nn.Module):
    def __init__(self, width: int = 32, upscale: int = 2):
        super().__init__()
        self.trunk = NAFNetFeatureTrunk(img_channel=3, width=width, middle_blk_num=12,
                                        enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
        self.upsample = nn.Sequential(
            nn.Conv2d(width, width * (upscale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(width, 1, kernel_size=3, padding=1)
        )
        self.upsample[0].weight.data.copy_(icnr_init(self.upsample[0].weight, upscale_factor=upscale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        features = self.trunk(x.repeat(1, 3, 1, 1))
        return base + self.upsample(features)


# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[We4Restore] Initializing Training Engine on Device: {device}")

    # Dataset Preparation
    full_dataset = PairedRestorationDataset(root_dir=args.data_dir, patch_size=256, train=True)
    indices = list(range(len(full_dataset)))
    random.seed(args.seed)
    random.shuffle(indices)

    split = int(0.95 * len(full_dataset))
    train_ds = torch.utils.data.Subset(full_dataset, indices[:split])
    val_ds = torch.utils.data.Subset(full_dataset, indices[split:])

    print(f"[INFO] Dataset Partitioning -> Training: {len(train_ds)} | Validation: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model & Loss Initialization
    model = NativeNAFNetSR(width=args.width, upscale=2).to(device)
    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device).eval()

    if args.weights and os.path.exists(args.weights):
        print(f"[INFO] Loading Pretrained Weights: {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint.get("params", checkpoint))

        new_state_dict = {}
        model_keys = set(model.state_dict().keys())
        for k, v in state_dict.items():
            if k in model_keys:
                new_state_dict[k] = v
            elif f"trunk.{k}" in model_keys:
                new_state_dict[f"trunk.{k}"] = v
            elif k.startswith("trunk.") and k[6:] in model_keys:
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v

        missing, _ = model.load_state_dict(new_state_dict, strict=False)
        print(f"[SUCCESS] Weights Applied Successfully (Missing Keys: {len(missing)})")

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "val_previews"
    img_dir.mkdir(parents=True, exist_ok=True)

    best_psnr = 0.0

    print(f"\n[INFO] Starting Optimization Loop ({args.epochs} Epochs)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()

        for lr_t, gt_t in train_loader:
            lr_t, gt_t = lr_t.to(device), gt_t.to(device)

            # MixUp Augmentation
            if random.random() < 0.5:
                lam = np.random.beta(1.2, 1.2)
                idx = torch.randperm(lr_t.size(0)).to(device)
                lr_t = lam * lr_t + (1.0 - lam) * lr_t[idx]
                gt_t = lam * gt_t + (1.0 - lam) * gt_t[idx]

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(lr_t)
                loss = F.l1_loss(pred, gt_t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # Validation Step with 8x Test-Time Augmentation
        model.eval()
        psnr_vals, ssim_vals, lpips_vals = [], [], []

        with torch.inference_mode():
            for b_idx, (lr_t, gt_t) in enumerate(val_loader):
                lr_t, gt_t = lr_t.to(device), gt_t.to(device)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    preds = []
                    for rot in [0, 1, 2, 3]:
                        lr_r = torch.rot90(lr_t, rot, [2, 3])
                        preds.append(torch.rot90(model(lr_r), -rot, [2, 3]))
                        preds.append(torch.rot90(torch.flip(model(torch.flip(lr_r, [3])), [3]), -rot, [2, 3]))

                    pred = torch.mean(torch.stack(preds), dim=0)
                    pred = torch.clamp(pred, 0.0, 1.0)

                    pred_rgb = (pred.repeat(1, 3, 1, 1) * 2.0) - 1.0
                    gt_rgb = (gt_t.repeat(1, 3, 1, 1) * 2.0) - 1.0
                    lpips_batch = loss_fn_vgg(pred_rgb, gt_rgb).mean().item()
                    lpips_vals.append(lpips_batch)

                if b_idx == 0:
                    lr_up = torch.clamp(F.interpolate(lr_t, scale_factor=2, mode="bilinear", align_corners=False), 0.0, 1.0)
                    grid = vutils.make_grid(torch.cat([lr_up[:3], pred[:3], gt_t[:3]], dim=0), nrow=3)
                    vutils.save_image(grid, img_dir / f"epoch_{epoch:03d}.png")

                p_np, g_np = pred.cpu().numpy()[:, 0], gt_t.cpu().numpy()[:, 0]
                for i in range(p_np.shape[0]):
                    mse = np.mean((p_np[i] - g_np[i]) ** 2)
                    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else 50.0
                    ssim = ssim_skimage(g_np[i], p_np[i], data_range=1.0)
                    psnr_vals.append(psnr)
                    ssim_vals.append(ssim)

        avg_psnr = np.mean(psnr_vals)
        avg_ssim = np.mean(ssim_vals)
        avg_lpips = np.mean(lpips_vals)

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Elapsed: {time.time()-epoch_start:.1f}s | "
              f"Val PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), out_dir / "nafnet_sr_best.pth")
            print(f"  --> Saved New Best Model Checkpoint ({best_psnr:.2f} dB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="We4Restore Training Pipeline")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)