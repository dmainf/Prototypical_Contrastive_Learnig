"""
Figure 3 style visualization (Wang & Isola, ICML 2020)

左列: 正例ペアの特徴距離ヒストグラム (Alignment)
右列: PCA 2D投影 → KDE (Uniformity)

Usage:
  python3 visualize_alignment_uniformity.py --checkpoints-dir ./checkpoints_s128
  python3 visualize_alignment_uniformity.py --checkpoints-dir ./checkpoints_s128 --epochs 0 50 100
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from pcl.model import PCL
from pcl.dataset import (
    TimeSeriesDataset, TwoViewTransform, IndexedDataset,
    TimeSeriesAugmentation,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints-dir", default="./checkpoints_s128")
    p.add_argument("--data-path", default="./datasets/electricity/electricity.csv")
    p.add_argument("--variables", default="individuals")
    p.add_argument("--target-col", default="OT")
    p.add_argument("--output-dir", default="./align_uniform")
    p.add_argument("--epochs", nargs="+", type=int, default=None,
                   help="可視化するエポック番号（省略時はすべて）")
    p.add_argument("--n-pairs", default=3000, type=int,
                   help="Alignment計算に使う正例ペア数")
    p.add_argument("--n-feats", default=5000, type=int,
                   help="Uniformity可視化に使うサンプル数")
    p.add_argument("--workers", default=0, type=int)
    return p.parse_args()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    model = PCL(
        encoder_type=a.get("encoder", "transformer"),
        in_channels=a.get("in_channels", 1),
        seq_len=a.get("seq_len", 512),
        d_model=a.get("d_model", 64),
        nhead=a.get("nhead", 4),
        num_layers=a.get("num_layers", 2),
        dim=a.get("dim", 128),
        chronos_model_name=a.get("chronos_model", "amazon/chronos-bolt-small"),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, a


@torch.no_grad()
def get_pos_pair_dists(model, loader, device, n_pairs):
    """正例ペアの L2 距離を収集する。loaderは TwoViewTransform 付き。"""
    dists = []
    for (views, _) in loader:
        x1, x2 = views[0].to(device), views[1].to(device)
        f1 = F.normalize(model.encoder_k(x1), dim=1)
        f2 = F.normalize(model.encoder_k(x2), dim=1)
        dists.append((f1 - f2).norm(dim=1).cpu())
        if sum(len(d) for d in dists) >= n_pairs:
            break
    return torch.cat(dists)[:n_pairs].numpy()


@torch.no_grad()
def get_features(model, loader, device, n_feats):
    """全サンプルの特徴量を収集する。"""
    feats = []
    for x, _ in loader:
        f = F.normalize(model.encoder_k(x.to(device)), dim=1)
        feats.append(f.cpu())
        if sum(len(f) for f in feats) >= n_feats:
            break
    return torch.cat(feats)[:n_feats].numpy()


def compute_uniformity(feats, t=2):
    """Wang & Isola (2020) Eq.2: log E[exp(-t * ||z_i - z_j||^2)]"""
    f = torch.tensor(feats)
    sq_dists = torch.pdist(f, p=2).pow(2)
    return torch.log(torch.exp(-t * sq_dists).mean()).item()


def project_pca2d(feats):
    """L2正規化済み特徴量をPCAで2次元に投影する。"""
    pca = PCA(n_components=2)
    return pca.fit_transform(feats)


def plot_kde(ax, xy, title):
    """2D PCA投影の KDE を描画する。"""
    pad = (xy.max() - xy.min()) * 0.1 or 0.1
    lo, hi = xy.min() - pad, xy.max() + pad
    try:
        kde = gaussian_kde(xy.T, bw_method=0.15)
        grid = np.linspace(lo, hi, 200)
        xx, yy = np.meshgrid(grid, grid)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=20, cmap="Greens", alpha=0.85)
    except Exception:
        ax.scatter(xy[:, 0], xy[:, 1], s=1, alpha=0.3, color="seagreen")

    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("PC1", fontsize=9)
    ax.set_ylabel("PC2", fontsize=9)
    ax.set_title(title, fontsize=9)


def make_figure(epoch, dists, feats_pca2d, uniformity, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(f"epoch = {epoch}", fontsize=11)

    # --- 左: Alignment（正例ペア距離ヒストグラム）---
    ax = axes[0]
    ax.hist(dists, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(dists.mean(), color="red", linestyle="--", linewidth=1.2,
               label=f"mean = {dists.mean():.3f}")
    ax.set_xlabel(r"$\ell_2$ distance", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Alignment\n(Positive Pair Feature Distances)", fontsize=9)
    ax.legend(fontsize=9)

    # --- 右: Uniformity（PCA 2D KDE）---
    plot_kde(axes[1], feats_pca2d,
             f"Uniformity = {uniformity:.3f}\n(Feature Distribution, PCA 2D)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    ckpts = sorted(glob.glob(os.path.join(args.checkpoints_dir, "pcl_epoch*.pth")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {args.checkpoints_dir}")

    if args.epochs is not None:
        ckpts = [c for c in ckpts
                 if int(os.path.basename(c).replace("pcl_epoch", "").replace(".pth", ""))
                 in args.epochs]

    # seq_len / stride をチェックポイントから取得
    first_args = torch.load(ckpts[0], map_location="cpu", weights_only=False).get("args", {})
    seq_len = first_args.get("seq_len", 512)
    stride  = first_args.get("stride", seq_len)

    aug = TimeSeriesAugmentation()
    pair_ds = TimeSeriesDataset(
        path=args.data_path, variables=args.variables,
        target_col=args.target_col, seq_len=seq_len,
        stride=stride, split="train",
        transform=TwoViewTransform(aug),
    )
    feat_ds = TimeSeriesDataset(
        path=args.data_path, variables=args.variables,
        target_col=args.target_col, seq_len=seq_len,
        stride=stride, split="train", transform=None,
    )
    pair_loader = DataLoader(pair_ds, batch_size=256, shuffle=True,
                             num_workers=args.workers)
    feat_loader = DataLoader(feat_ds, batch_size=512, shuffle=False,
                             num_workers=args.workers)

    for ckpt_path in ckpts:
        epoch = int(os.path.basename(ckpt_path).replace("pcl_epoch", "").replace(".pth", ""))
        print(f"epoch {epoch:4d} ...", end=" ", flush=True)

        model, _ = load_model(ckpt_path, device)

        dists = get_pos_pair_dists(model, pair_loader, device, args.n_pairs)
        feats = get_features(model, feat_loader, device, args.n_feats)
        feats_pca2d = project_pca2d(feats)
        uniformity = compute_uniformity(feats)

        del model

        out = os.path.join(output_dir, f"au_epoch{epoch:04d}.png")
        make_figure(epoch, dists, feats_pca2d, uniformity, out)

    print("Done.")


if __name__ == "__main__":
    main()
