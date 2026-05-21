"""
t-SNE visualization of learned time-series representations (cf. Figure 4 in the PCL paper).
色は時系列上の時刻位置を表す（青=始め → 赤=終わり）。

Usage:
  python3 visualize_tsne.py \
      --checkpoint ./checkpoints/pcl_epoch0200.pth \
      --output ./tsne_epoch200.png
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE

from pcl.model import PCL
from pcl.dataset import TimeSeriesDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-path", default="./datasets/ETT-small/ETTh1.csv")
    p.add_argument("--variables", default="univariate",
                   choices=["univariate", "multivariate"])
    p.add_argument("--target-col", default="OT")
    p.add_argument("--seq-len", default=96, type=int)
    p.add_argument("--stride", default=1, type=int)
    p.add_argument("--samples", default=500, type=int, help="t-SNEで使うサンプル数")
    p.add_argument("--perplexity", default=30.0, type=float)
    p.add_argument("--output", default="./tsne.png")
    p.add_argument("--workers", default=2, type=int)
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> PCL:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    model = PCL(
        encoder_type=saved_args.get("encoder", "transformer"),
        in_channels=1 if saved_args.get("variables", "univariate") == "univariate"
                   else saved_args.get("in_channels", 1),
        seq_len=saved_args.get("seq_len", 96),
        d_model=saved_args.get("d_model", 64),
        nhead=saved_args.get("nhead", 4),
        num_layers=saved_args.get("num_layers", 2),
        dim=saved_args.get("dim", 128),
        queue_size=saved_args.get("queue_size", 4096),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def extract_features(model: PCL, loader, device: torch.device) -> np.ndarray:
    feats = []
    for x, _ in loader:
        f = nn.functional.normalize(model.encoder_k(x.to(device)), dim=1)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats)


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    print(f"Loaded: {args.checkpoint}")

    ds = TimeSeriesDataset(
        path=args.data_path, variables=args.variables,
        target_col=args.target_col, seq_len=args.seq_len,
        stride=args.stride, split="train", transform=None,
    )
    n = len(ds)
    samples = min(args.samples, n)
    indices = np.linspace(0, n - 1, samples, dtype=int).tolist()
    loader = DataLoader(Subset(ds, indices), batch_size=256, shuffle=False,
                        num_workers=args.workers)

    print(f"Extracting features from {samples} windows...")
    feats = extract_features(model, loader, device)

    print(f"Running t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity,
                max_iter=1000, random_state=42, verbose=1)
    emb = tsne.fit_transform(feats)

    time_pos = np.linspace(0, 1, samples)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=time_pos, cmap="coolwarm",
                    s=8, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="time position (0=start, 1=end)")

    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    ax.set_title(f"t-SNE  [{ckpt_name}]  {samples} windows", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
