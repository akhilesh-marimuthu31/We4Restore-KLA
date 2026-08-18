"""
We4Restore - KLA Hackathon 2026 Inference Engine

Architecture: Native NAFNet Super-Resolution (SR)
Features: 8x Test-Time Augmentation (TTA), Offline Execution
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. STANDALONE NAFNET ARCHITECTURE
# ==========================================
class SimpleGate(nn.Module):
    """
    Activation-free gating mechanism.
    Splits the channel dimension and multiplies the two halves.
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free Block for image restoration.
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
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

    def forward(self, inp):
        x = inp
        
        # Spatial Feature Extraction
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        
        y = inp + x * self.beta
        
        # Channel Mixing
        x = y.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        
        return y + x * self.gamma


class NAFNetFeatureTrunk(nn.Module):
    """
    Encoder-Decoder backbone utilizing NAFBlocks.
    """
    def __init__(self, img_channel=3, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
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

    def forward(self, inp):
        B, C, H, W = inp.shape
        x = self.intro(inp)
        encs = []
        
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
            
        x = self.middle_blks(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            
        return x[:, :, :H, :W]


def icnr_init(tensor, upscale_factor=2):
    """
    Initialization strategy to prevent checkerboard artifacts in PixelShuffle.
    """
    out_channels, in_channels, kh, kw = tensor.shape
    scale_sq = upscale_factor ** 2
    sub_kernel = torch.nn.init.kaiming_normal_(torch.empty(out_channels // scale_sq, in_channels, kh, kw))
    return sub_kernel.repeat_interleave(scale_sq, dim=0)


class NativeNAFNetSR(nn.Module):
    """
    Top-level module containing the NAFNet trunk and the SR upsampling head.
    """
    def __init__(self, width=32, upscale=2):
        super().__init__()
        self.trunk = NAFNetFeatureTrunk(img_channel=3, width=width, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
        self.upsample = nn.Sequential(
            nn.Conv2d(width, width * (upscale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(width, 1, kernel_size=3, padding=1)
        )
        self.upsample[0].weight.data.copy_(icnr_init(self.upsample[0].weight, upscale))

    def forward(self, x):
        base = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        features = self.trunk(x.repeat(1, 3, 1, 1))
        return base + self.upsample(features)


# ==========================================
# 2. HACKATHON INFERENCE ENGINE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="We4Restore - KLA Hackathon Inference Script")
    parser.add_argument("input_dir", type=str, help="Directory containing input .npy files")
    parser.add_argument("output_dir", type=str, help="Directory to save restored .npy files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[We4Restore] Initializing Inference on: {device}")

    model = NativeNAFNetSR(width=32, upscale=2).to(device)
    
    script_dir = Path(__file__).resolve().parent
    weight_path = script_dir / "models" / "nafnet_sr_best.pth"
    
    if not weight_path.exists():
        print(f"[ERROR] Model weights not found at {weight_path}")
        print("Ensure 'nafnet_sr_best.pth' is placed inside the 'models/' directory.")
        sys.exit(1)
        
    model.load_state_dict(torch.load(str(weight_path), map_location=device), strict=True)
    model.eval()

    input_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    if not input_files:
        print(f"[WARNING] No .npy files found in {input_dir}.")
        sys.exit(0)

    print(f"[We4Restore] Found {len(input_files)} arrays. Executing 8x TTA Restoration...")
    
    with torch.inference_mode():
        for filename in input_files:
            file_path = input_dir / filename
            lr_img = np.load(file_path).astype(np.float32)

            if lr_img.ndim == 2:
                lr_tensor = torch.from_numpy(lr_img).unsqueeze(0).unsqueeze(0)
            elif lr_img.ndim == 3:
                lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).unsqueeze(0)
            else:
                continue 
            
            lr_tensor = lr_tensor.to(device)
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = []
                for rot in [0, 1, 2, 3]:
                    lr_r = torch.rot90(lr_t=lr_tensor, k=rot, dims=[2, 3])
                    
                    pred_r = model(lr_r)
                    preds.append(torch.rot90(pred_r, k=-rot, dims=[2, 3]))
                    
                    lr_r_flip = torch.flip(lr_r, [3])
                    pred_r_flip = model(lr_r_flip)
                    preds.append(torch.rot90(torch.flip(pred_r_flip, [3]), k=-rot, dims=[2, 3]))
                
                pred = torch.mean(torch.stack(preds), dim=0)
                pred = torch.clamp(pred, 0.0, 1.0)
            
            out_np = pred.squeeze().cpu().numpy()
            out_np = np.nan_to_num(out_np, nan=0.0, posinf=1.0, neginf=0.0)

            np.save(output_dir / filename, out_np)
            print(f" -> Successfully Restored: {filename}")

    print(f"\n[SUCCESS] We4Restore Inference Complete. {len(input_files)} files saved to {output_dir}")


if __name__ == "__main__":
    main()