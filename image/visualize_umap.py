"""
UMAP visualization of learned representations across training epochs.

Usage:
  python3 visualize_umap.py --checkpoints-dir ./checkpoints
  python3 visualize_umap.py --checkpoint ./checkpoints/pcl_epoch0200.pth
"""

import argparse
import gc
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from torchvision import datasets
import umap

from pcl.dataset import cifar10_eval_transform, imagenet_eval_transform


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--checkpoint", default="", help="単一チェックポイント")
    g.add_argument("--checkpoints-dir", default="./checkpoints", help="ディレクトリ内の全チェックポイント（デフォルト: ./checkpoints）")
    p.add_argument("--dataset", default="cifar10", choices=["imagenet", "cifar10"])
    p.add_argument("--data-path", default="./data")
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--output-dir", default="./umap_all")
    p.add_argument("--workers", default=2, type=int)
    p.add_argument("--n-neighbors", default=15, type=int)
    p.add_argument("--min-dist", default=0.1, type=float)
    return p.parse_args()


def load_encoder(ckpt_path, arch, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    dim = saved_args.get("dim", 128)
    use_mlp = saved_args.get("use_mlp", False)

    encoder = getattr(models, arch)(weights=None)
    feat_dim = encoder.fc.in_features
    if use_mlp:
        encoder.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, dim)
        )
    else:
        encoder.fc = nn.Linear(feat_dim, dim)

    state = {
        k.replace("encoder_k.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("encoder_k.")
    }
    encoder.load_state_dict(state, strict=True)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


@torch.no_grad()
def extract_features(encoder, loader, device):
    feats, labels = [], []
    for images, targets in loader:
        f = F.normalize(encoder(images.to(device)), dim=1)
        feats.append(f.cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def plot_umap(embeddings, labels, num_classes, class_names, title, output_path):
    colors = cm.tab20(np.linspace(0, 1, num_classes))
    fig, ax = plt.subplots(figsize=(10, 8))
    for c in range(num_classes):
        mask = labels == c
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   s=2, alpha=0.5, color=colors[c],
                   label=class_names[c] if c < len(class_names) else str(c))
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=4, fontsize=8, loc="upper right",
              bbox_to_anchor=(1.18, 1.0))
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

    if args.dataset == "cifar10":
        ds = datasets.CIFAR10(args.data_path, train=True,
                              transform=cifar10_eval_transform(), download=True)
        class_names = CIFAR10_CLASSES
    else:
        ds = datasets.ImageFolder(os.path.join(args.data_path, "train"),
                                  transform=imagenet_eval_transform())
        class_names = [d[0] for d in ds.classes]

    num_classes = len(class_names)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=args.workers)
    print(f"Dataset: {len(ds)} samples, {num_classes} classes")

    if args.checkpoint:
        ckpts = [args.checkpoint]
    else:
        ckpts = sorted(glob.glob(os.path.join(args.checkpoints_dir, "pcl_epoch*.pth")))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints in {args.checkpoints_dir}")

    # Step 1: 全チェックポイントの特徴抽出
    print(f"\n=== Step 1: Feature extraction ({len(ckpts)} checkpoints) ===")
    all_feats = {}
    labels = None
    for ckpt_path in ckpts:
        epoch_num = int(os.path.basename(ckpt_path).replace("pcl_epoch", "").replace(".pth", ""))
        print(f"  epoch {epoch_num:4d} ...", end=" ", flush=True)
        encoder = load_encoder(ckpt_path, args.arch, device)
        feats, lbs = extract_features(encoder, loader, device)
        all_feats[epoch_num] = feats
        if labels is None:
            labels = lbs
        del encoder
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"done  shape={feats.shape}")

    # Step 2: 全チェックポイントにUMAP適用
    print(f"\n=== Step 2: UMAP ({len(all_feats)} checkpoints) ===")
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch_num, feats in all_feats.items():
        print(f"  epoch {epoch_num:4d} ...", end=" ", flush=True)
        reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                            n_components=2, n_jobs=-1, verbose=False)
        emb = reducer.fit_transform(feats)
        title = f"UMAP  epoch={epoch_num}  ({num_classes} classes, {len(feats)} samples)"
        out = os.path.join(args.output_dir, f"umap_epoch{epoch_num:04d}.png")
        plot_umap(emb, labels, num_classes, class_names, title, out)


if __name__ == "__main__":
    main()
