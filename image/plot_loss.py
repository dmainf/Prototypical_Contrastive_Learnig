"""
Loss curve from checkpoints.

Usage:
  python3 plot_loss.py
  python3 plot_loss.py --checkpoints-dir ./checkpoints --output ./loss_curve.png
"""

import argparse
import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints-dir", default="./checkpoints")
    p.add_argument("--output", default="./loss_curve.png")
    return p.parse_args()


def main():
    args = parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.checkpoints_dir, "pcl_epoch*.pth")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {args.checkpoints_dir}")

    epochs, losses = [], []
    for path in ckpts:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "avg_loss" not in ckpt:
            print(f"  skip {os.path.basename(path)} (avg_loss not saved)")
            continue
        epoch_num = int(os.path.basename(path).replace("pcl_epoch", "").replace(".pth", ""))
        epochs.append(epoch_num)
        losses.append(ckpt["avg_loss"])
        print(f"  epoch {epoch_num:4d}  loss={ckpt['avg_loss']:.4f}")

    if not epochs:
        print("avg_loss が .pth に保存されていません。次回学習分から有効になります。")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, losses, marker="o", markersize=4, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("PCL Training Loss", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(left=0)

    min_idx = int(np.argmin(losses))
    ax.annotate(
        f"min={losses[min_idx]:.4f} (ep {epochs[min_idx]})",
        xy=(epochs[min_idx], losses[min_idx]),
        xytext=(10, 10), textcoords="offset points",
        fontsize=9, color="crimson",
        arrowprops=dict(arrowstyle="->", color="crimson", lw=1),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
