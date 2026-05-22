"""
t-SNE visualization of learned representations (cf. Figure 4 in the PCL paper).

Usage:
  # チェックポイントから可視化
  python3 visualize_tsne.py \
      --checkpoint ./checkpoints/pcl_epoch0200.pth \
      --dataset cifar10 --data-path ./data \
      --arch resnet18 --output ./tsne_epoch200.png

  # クラス数・サンプル数を絞る
  python3 visualize_tsne.py \
      --checkpoint ./checkpoints/pcl_epoch0200.pth \
      --dataset cifar10 --data-path ./data \
      --arch resnet18 --num-classes 10 --samples-per-class 200
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.manifold import TSNE

from pcl.dataset import cifar10_eval_transform, imagenet_eval_transform


# CIFAR-10クラス名
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="cifar10", choices=["imagenet", "cifar10"])
    p.add_argument("--data-path", default="./data")
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--num-classes", default=10, type=int,
                   help="可視化するクラス数 (先頭N クラス)")
    p.add_argument("--samples-per-class", default=200, type=int,
                   help="クラスあたりのサンプル数 (t-SNEは重いので絞る)")
    p.add_argument("--perplexity", default=30.0, type=float)
    p.add_argument("--output", default="./tsne.png")
    p.add_argument("--workers", default=2, type=int)
    p.add_argument("--title", default="", type=str)
    return p.parse_args()


def load_encoder(checkpoint_path: str, arch: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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


def build_subset(dataset, num_classes: int, samples_per_class: int):
    """先頭num_classesクラスからsamples_per_class枚ずつ選ぶ。"""
    targets = np.array(dataset.targets if hasattr(dataset, "targets") else dataset.labels)
    indices = []
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        idx = idx[:samples_per_class]
        indices.extend(idx.tolist())
    return Subset(dataset, indices), targets[indices]


@torch.no_grad()
def extract_features(encoder, loader, device):
    feats, labels = [], []
    for images, targets in loader:
        f = encoder(images.to(device))
        f = nn.functional.normalize(f, dim=1)
        feats.append(f.cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    class_names: list,
    title: str,
    output_path: str,
):
    colors = cm.tab20(np.linspace(0, 1, num_classes))

    fig, ax = plt.subplots(figsize=(10, 8))
    for c in range(num_classes):
        mask = labels == c
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            s=5, alpha=0.6,
            color=colors[c],
            label=class_names[c] if c < len(class_names) else str(c),
        )

    ax.set_title(title or "t-SNE of learned representations", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        markerscale=3, fontsize=8,
        loc="upper right", bbox_to_anchor=(1.18, 1.0),
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # エンコーダ読み込み
    encoder = load_encoder(args.checkpoint, args.arch, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # データセット
    if args.dataset == "cifar10":
        base_ds = datasets.CIFAR10(
            args.data_path, train=True,
            transform=cifar10_eval_transform(), download=True,
        )
        class_names = CIFAR10_CLASSES
    else:
        base_ds = datasets.ImageFolder(
            os.path.join(args.data_path, "train"),
            transform=imagenet_eval_transform(),
        )
        class_names = [d[0] for d in base_ds.classes]

    num_classes = min(args.num_classes, len(class_names))
    subset, subset_labels = build_subset(base_ds, num_classes, args.samples_per_class)
    loader = DataLoader(subset, batch_size=256, shuffle=False,
                        num_workers=args.workers)

    total = num_classes * args.samples_per_class
    print(f"Classes: {num_classes} | Samples: {total} | Perplexity: {args.perplexity}")

    # 特徴抽出
    print("Extracting features...")
    feats, labels = extract_features(encoder, loader, device)

    # t-SNE
    print("Running t-SNE (this may take a minute)...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        max_iter=1000,
        random_state=42,
        verbose=1,
    )
    embeddings = tsne.fit_transform(feats)

    # 描画・保存
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    title = args.title or f"t-SNE  [{ckpt_name}]  {num_classes} classes × {args.samples_per_class} samples"
    plot_tsne(embeddings, labels, num_classes, class_names, title, args.output)


if __name__ == "__main__":
    main()
