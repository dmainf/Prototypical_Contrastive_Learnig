"""
Prototypical Contrastive Learning — Training Script
Paper: https://arxiv.org/abs/2005.04966  (ICLR 2021)

Usage (CIFAR-10 quick test):
  python3 train.py --dataset cifar10 --data-path ./data \
      --arch resnet18 --num-clusters 50 200 500 \
      --queue-size 4096 --r 500 --epochs 200

Usage (ImageNet):
  python3 train.py --dataset imagenet --data-path /path/to/imagenet \
      --arch resnet50 --num-clusters 25000 50000 100000 \
      --queue-size 65536 --r 16000 --epochs 200
"""

import argparse
import os
import time

import resource
import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _hard), _hard))
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.manifold import TSNE

from pcl.model import PCL
from pcl.loss import ProtoNCELoss
from pcl.clustering import cluster_features
from pcl.dataset import (
    TwoViewTransform, IndexedDataset,
    imagenet_train_transform, imagenet_eval_transform,
    cifar10_train_transform, cifar10_eval_transform,
)


def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--dataset", default="cifar10", choices=["imagenet", "cifar10"])
    p.add_argument("--data-path", default="./data")
    # Model
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--dim", default=128, type=int, help="feature dimension")
    p.add_argument("--use-mlp", action="store_true", help="use MLP projection head (PCL v2)")
    # Contrastive
    p.add_argument("--queue-size", default=65536, type=int)
    p.add_argument("--momentum", default=0.999, type=float, help="momentum encoder EMA")
    p.add_argument("--tau", default=0.1, type=float, help="temperature")
    p.add_argument("--r", default=16000, type=int, help="negative prototypes sampled per step")
    # Clustering (E-step)
    p.add_argument("--num-clusters", nargs="+", type=int, default=[50, 200, 500],
                   help="cluster granularities K = {k_1, ..., k_M}")
    p.add_argument("--alpha", default=10.0, type=float, help="concentration smoothing parameter")
    # Training
    p.add_argument("--epochs", default=200, type=int)
    p.add_argument("--warm-up-epochs", default=20, type=int,
                   help="epochs using InfoNCE only (no clustering)")
    p.add_argument("--batch-size", default=256, type=int)
    p.add_argument("--lr", default=0.03, type=float)
    p.add_argument("--weight-decay", default=1e-4, type=float)
    p.add_argument("--workers", default=4, type=int)
    # Misc
    p.add_argument("--save-freq", default=10, type=int)
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--resume", default="", type=str)
    # t-SNE
    p.add_argument("--tsne-freq", default=0, type=int,
                   help="t-SNEをNエポックごとに保存 (0=無効)")
    p.add_argument("--tsne-classes", default=10, type=int,
                   help="t-SNEで描画するクラス数")
    p.add_argument("--tsne-samples", default=200, type=int,
                   help="クラスあたりのサンプル数")
    return p.parse_args()


def build_loaders(args, pin_memory: bool = False):
    if args.dataset == "cifar10":
        train_tf = TwoViewTransform(cifar10_train_transform())
        cluster_tf = cifar10_eval_transform()
        train_ds = datasets.CIFAR10(args.data_path, train=True,
                                    transform=train_tf, download=True)
        cluster_ds = datasets.CIFAR10(args.data_path, train=True,
                                      transform=cluster_tf, download=True)
        n = len(train_ds)
        args.num_clusters = [k for k in args.num_clusters if k < n]
        if not args.num_clusters:
            args.num_clusters = [10, 50, 200]
        args.r = min(args.r, min(args.num_clusters) - 1)
        args.queue_size = min(args.queue_size, n)
    else:
        train_tf = TwoViewTransform(imagenet_train_transform(args.use_mlp))
        cluster_tf = imagenet_eval_transform()
        train_ds = datasets.ImageFolder(
            os.path.join(args.data_path, "train"), transform=train_tf)
        cluster_ds = datasets.ImageFolder(
            os.path.join(args.data_path, "train"), transform=cluster_tf)

    indexed_ds = IndexedDataset(train_ds)
    train_loader = DataLoader(
        indexed_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin_memory, drop_last=True,
    )
    cluster_loader = DataLoader(
        cluster_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=pin_memory,
    )
    return train_loader, cluster_loader


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@torch.no_grad()
def save_tsne(model, args, epoch: int, device: torch.device):
    """学習中のMomentumエンコーダでt-SNEを生成して保存する。"""
    if args.dataset == "cifar10":
        base_ds = datasets.CIFAR10(args.data_path, train=True,
                                   transform=cifar10_eval_transform(), download=False)
        class_names = CIFAR10_CLASSES
    else:
        base_ds = datasets.ImageFolder(
            os.path.join(args.data_path, "train"), transform=imagenet_eval_transform())
        class_names = [c[0] for c in base_ds.classes]

    num_classes = min(args.tsne_classes, len(class_names))
    targets = np.array(base_ds.targets if hasattr(base_ds, "targets") else base_ds.labels)
    indices = []
    for c in range(num_classes):
        idx = np.where(targets == c)[0][:args.tsne_samples]
        indices.extend(idx.tolist())
    subset = Subset(base_ds, indices)
    subset_labels = targets[indices]

    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=2)

    model.encoder_k.eval()
    feats = []
    for images, _ in loader:
        f = nn.functional.normalize(model.encoder_k(images.to(device)), dim=1)
        feats.append(f.cpu().numpy())
    model.encoder_k.train()
    feats = np.concatenate(feats)

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, verbose=0)
    emb = tsne.fit_transform(feats)

    colors = cm.tab20(np.linspace(0, 1, num_classes))
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in range(num_classes):
        mask = subset_labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.6,
                   color=colors[c], label=class_names[c])
    ax.set_title(f"t-SNE  epoch={epoch}  ({num_classes} classes × {args.tsne_samples} samples)",
                 fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=3, fontsize=7, loc="upper right",
              bbox_to_anchor=(1.18, 1.0))
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, f"tsne_epoch{epoch+1:04d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → t-SNE saved: {out_path}")


def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"  → saved {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    train_loader, cluster_loader = build_loaders(args, pin_memory=(device.type == "cuda"))
    print(f"Dataset: {args.dataset} | "
          f"Train batches: {len(train_loader)} | "
          f"Cluster sizes: {args.num_clusters}")

    model = PCL(
        base_encoder=args.arch,
        dim=args.dim,
        queue_size=args.queue_size,
        momentum=args.momentum,
        use_mlp=args.use_mlp,
    ).to(device)

    optimizer = optim.SGD(
        model.encoder_q.parameters(),
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
    )
    # Learning rate drops by 0.1 at epochs 120 and 160 (paper §4.1)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[120, 160], gamma=0.1
    )
    criterion = ProtoNCELoss(tau=args.tau, r=args.r)

    start_epoch = 0
    cluster_results = None

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        warm_up = epoch < args.warm_up_epochs
        t0 = time.time()

        # ── E-step: clustering ───────────────────────────────────────────────
        if not warm_up:
            print(f"[Epoch {epoch}] E-step: extracting features for clustering...")
            features = model.get_features(cluster_loader, device)
            cluster_results = cluster_features(
                features.numpy(), args.num_clusters, args.alpha, args.tau
            )
            # Transfer centroids and phi to device once per epoch; keep assignments on device
            cluster_results = [
                (c.to(device), a.to(device), p.to(device))
                for c, a, p in cluster_results
            ]

        # ── M-step: one epoch of gradient updates ────────────────────────────
        model.train()
        total_loss = 0.0

        for i, (images, _, indices) in enumerate(train_loader):
            im_q = images[0].to(device)
            im_k = images[1].to(device)

            q, k = model(im_q, im_k)

            loss = criterion(
                q, k,
                model.queue,
                cluster_results,
                indices.to(device),
                warm_up=warm_up,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0:
                tag = "(warm-up)" if warm_up else ""
                print(f"  [{epoch}/{args.epochs}][{i}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.5f} {tag}")

        scheduler.step()
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] avg_loss={total_loss/len(train_loader):.4f}  "
              f"time={elapsed:.1f}s")

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args),
                },
                os.path.join(args.output_dir, f"pcl_epoch{epoch+1:04d}.pth"),
            )

        if args.tsne_freq > 0 and ((epoch + 1) % args.tsne_freq == 0 or epoch == args.epochs - 1):
            print(f"[Epoch {epoch}] Generating t-SNE...")
            save_tsne(model, args, epoch, device)


if __name__ == "__main__":
    main()
