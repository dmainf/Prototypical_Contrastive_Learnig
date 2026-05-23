"""
UMAP visualization of learned time-series representations.
色は時系列上の時刻位置を表す（青=始め → 赤=終わり）。

Usage (単一チェックポイント):
  python3 visualize_umap.py --checkpoint ./checkpoints/pcl_epoch0200.pth

Usage (全チェックポイント):
  python3 visualize_umap.py --checkpoints-dir ./checkpoints
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from torch.utils.data import DataLoader

from pcl.model import PCL
from pcl.dataset import TimeSeriesDataset


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--checkpoint", default="")
    g.add_argument("--checkpoints-dir", default="./checkpoints")
    p.add_argument("--data-path", default="./datasets/electricity/electricity.csv")
    p.add_argument("--variables", default="individuals", choices=["univariate", "multivariate", "individuals"])
    p.add_argument("--target-col", default="OT")
    p.add_argument("--seq-len", default=None, type=int, help="省略時はチェックポイントから自動取得")
    p.add_argument("--stride", default=None, type=int, help="省略時はチェックポイントから自動取得")
    p.add_argument("--output-dir", default="./umap_all")
    p.add_argument("--workers", default=0, type=int)
    p.add_argument("--n-neighbors", default=15, type=int)
    p.add_argument("--min-dist", default=0.1, type=float)
    return p.parse_args()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    model = PCL(
        encoder_type=a.get("encoder", "transformer"),
        in_channels=a.get("in_channels", 1),
        seq_len=a.get("seq_len", 96),
        d_model=a.get("d_model", 64),
        nhead=a.get("nhead", 4),
        num_layers=a.get("num_layers", 2),
        dim=a.get("dim", 128),
        chronos_model_name=a.get("chronos_model", "amazon/chronos-bolt-small"),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def extract_features(model, loader, device):
    feats = []
    for x, _ in loader:
        f = nn.functional.normalize(model.encoder_k(x.to(device)), dim=1)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats)


def plot_umap(emb, n_samples, epoch, output_path):
    time_pos = np.linspace(0, 1, n_samples)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=time_pos, cmap="coolwarm", s=3, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="time position (0=start, 1=end)")
    ax.set_title(f"UMAP  epoch={epoch}  ({n_samples} windows)", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if args.checkpoint:
        ckpts = [args.checkpoint]
    else:
        ckpts = sorted(glob.glob(os.path.join(args.checkpoints_dir, "pcl_epoch*.pth")))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints in {args.checkpoints_dir}")

    # seq_len / stride をチェックポイントから自動取得
    first_ckpt_args = torch.load(ckpts[0], map_location="cpu", weights_only=False).get("args", {})
    seq_len = args.seq_len or first_ckpt_args.get("seq_len", 512)
    stride = args.stride or first_ckpt_args.get("stride", seq_len)

    ds = TimeSeriesDataset(
        path=args.data_path, variables=args.variables,
        target_col=args.target_col, seq_len=seq_len,
        stride=stride, split="train", transform=None,
    )
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=args.workers)
    print(f"Dataset: {len(ds)} windows  (seq_len={seq_len}, stride={stride})")

    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                        n_components=2, n_jobs=-1, verbose=False)

    for ckpt_path in ckpts:
        epoch = int(os.path.basename(ckpt_path).replace("pcl_epoch", "").replace(".pth", ""))
        print(f"epoch {epoch:4d} — extracting features...", end=" ", flush=True)
        model = load_model(ckpt_path, device)
        feats = extract_features(model, loader, device)
        del model
        print(f"UMAP...", end=" ", flush=True)
        emb = reducer.fit_transform(feats)
        out = os.path.join(args.output_dir, f"umap_epoch{epoch:04d}.png")
        plot_umap(emb, len(feats), epoch, out)


if __name__ == "__main__":
    main()
