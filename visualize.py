import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize.py <input.npy> <output.npy>")
        sys.exit(1)

    inp = np.load(sys.argv[1]).astype(np.float32).squeeze()
    out = np.load(sys.argv[2]).astype(np.float32).squeeze()

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor("#0a0e17")

    axes[0].imshow(inp, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Degraded Input ({inp.shape[1]}x{inp.shape[0]})", color="#00E5FF", fontsize=11, pad=10)
    axes[0].axis("off")

    axes[1].imshow(out, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"We4Restore Output ({out.shape[1]}x{out.shape[0]})", color="#39FF14", fontsize=11, pad=10)
    axes[1].axis("off")

    plt.suptitle("We4Restore: Offline SEM Wafer Restoration", color="white", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()