"""
Prototypical Contrastive Learning — 時系列版 Training Script
Paper: https://arxiv.org/abs/2005.04966  (ICLR 2021)

Usage (デフォルト / ETTh1 OTのみ):
  python3 train.py

Usage (多変量 / CNNエンコーダ):
  python3 train.py --variables multivariate --encoder cnn
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE

from pcl.model import PCL
from pcl.loss import ProtoNCELoss
from pcl.clustering import cluster_features
from pcl.dataset import (
    TimeSeriesDataset, TwoViewTransform, IndexedDataset,
    TimeSeriesAugmentation,
)


def parse_args():
    p = argparse.ArgumentParser()

    # データ
    p.add_argument("--data-path", default="./datasets/ETT-small/ETTh1.csv",
                   help="CSVファイルのパス")
    p.add_argument("--variables", default="univariate",
                   choices=["univariate", "multivariate"],
                   help="univariate: target-colのみ / multivariate: 全数値カラム")
    p.add_argument("--target-col", default="OT",
                   help="univariateのときに使うカラム名")
    p.add_argument("--seq-len", default=96, type=int,
                   help="1サンプルのウィンドウ長")
    p.add_argument("--stride", default=1, type=int,
                   help="スライディングウィンドウのストライド")

    # エンコーダ
    p.add_argument("--encoder", default="transformer",
                   choices=["transformer", "cnn"],
                   help="エンコーダアーキテクチャ")
    p.add_argument("--d-model", default=64, type=int,
                   help="Transformerの埋め込み次元（--encoder transformer のみ有効）")
    p.add_argument("--nhead", default=4, type=int,
                   help="Transformerのアテンションヘッド数")
    p.add_argument("--num-layers", default=2, type=int,
                   help="Transformerのレイヤー数")
    p.add_argument("--dim", default=128, type=int,
                   help="エンコーダ出力の特徴量次元数")

    # Augmentation
    p.add_argument("--jitter-sigma", default=0.03, type=float,
                   help="Jitterのノイズ強度")
    p.add_argument("--slice-ratio", default=0.9, type=float,
                   help="WindowSlicingの切り取り割合（0〜1）")
    p.add_argument("--no-jitter", action="store_true",
                   help="Jitterを無効化する")
    p.add_argument("--no-slicing", action="store_true",
                   help="WindowSlicingを無効化する")

    # 対照学習
    p.add_argument("--queue-size", default=4096, type=int)
    p.add_argument("--momentum", default=0.999, type=float)
    p.add_argument("--tau", default=0.1, type=float)
    p.add_argument("--r", default=500, type=int,
                   help="1ステップでサンプルする負例プロトタイプ数")

    # クラスタリング
    p.add_argument("--num-clusters", nargs="+", type=int, default=[50, 200, 500],
                   help="クラスタ数（複数指定で階層的プロトタイプ）")
    p.add_argument("--alpha", default=10.0, type=float)

    # 学習
    p.add_argument("--epochs", default=200, type=int)
    p.add_argument("--warm-up-epochs", default=20, type=int)
    p.add_argument("--batch-size", default=256, type=int)
    p.add_argument("--lr", default=1e-4, type=float,
                   help="学習率（Transformerはresnetより小さい値が安定）")
    p.add_argument("--weight-decay", default=1e-4, type=float)
    p.add_argument("--workers", default=2, type=int)

    # 保存
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--save-freq", default=10, type=int)
    p.add_argument("--resume", default="", type=str)

    # t-SNE
    p.add_argument("--tsne-freq", default=0, type=int,
                   help="Nエポックごとにt-SNE画像を保存（0=無効）")
    p.add_argument("--tsne-samples", default=500, type=int,
                   help="t-SNEで使うサンプル数")

    return p.parse_args()


def build_loaders(args, pin_memory: bool = False):
    aug = TimeSeriesAugmentation(
        jitter_sigma=args.jitter_sigma,
        slice_ratio=args.slice_ratio,
        use_jitter=not args.no_jitter,
        use_slicing=not args.no_slicing,
    )

    train_ds = TimeSeriesDataset(
        path=args.data_path, variables=args.variables,
        target_col=args.target_col, seq_len=args.seq_len,
        stride=args.stride, split="train",
        transform=TwoViewTransform(aug),
    )
    cluster_ds = TimeSeriesDataset(
        path=args.data_path, variables=args.variables,
        target_col=args.target_col, seq_len=args.seq_len,
        stride=args.stride, split="train",
        transform=None,
    )

    n = len(train_ds)
    args.num_clusters = [k for k in args.num_clusters if k < n]
    if not args.num_clusters:
        args.num_clusters = [10, 50, 100]
    args.r = min(args.r, min(args.num_clusters) - 1)
    args.queue_size = min(args.queue_size, n)

    indexed_ds = IndexedDataset(train_ds)
    train_loader = DataLoader(indexed_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin_memory, drop_last=True)
    cluster_loader = DataLoader(cluster_ds, batch_size=args.batch_size * 2, shuffle=False,
                                num_workers=args.workers, pin_memory=pin_memory)
    return train_loader, cluster_loader, train_ds.in_channels


def save_tsne(model, args, epoch: int, device: torch.device):
    """学習中のMomentumエンコーダでt-SNEを生成して保存する。"""
    ds = TimeSeriesDataset(
        path=args.data_path, variables=args.variables,
        target_col=args.target_col, seq_len=args.seq_len,
        stride=args.stride, split="train", transform=None,
    )
    n = len(ds)
    samples = min(args.tsne_samples, n)
    idx = np.linspace(0, n - 1, samples, dtype=int)
    subset = Subset(ds, idx.tolist())
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=2)

    model.encoder_k.eval()
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            f = nn.functional.normalize(model.encoder_k(x.to(device)), dim=1)
            feats.append(f.cpu().numpy())
    model.encoder_k.train()
    feats = np.concatenate(feats)

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, verbose=0)
    emb = tsne.fit_transform(feats)

    # 時刻位置で色付け（青=系列の始め → 赤=終わり）
    time_pos = np.linspace(0, 1, samples)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=time_pos, cmap="coolwarm", s=8, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="time position (0=start, 1=end)")
    ax.set_title(f"t-SNE  epoch={epoch}  ({samples} windows, colored by time)", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
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

    train_loader, cluster_loader, in_channels = build_loaders(
        args, pin_memory=(device.type == "cuda")
    )
    print(f"Data: {args.data_path} | variables={args.variables} | "
          f"in_channels={in_channels} | seq_len={args.seq_len} | "
          f"train_windows={len(train_loader.dataset)} | "
          f"num_clusters={args.num_clusters}")

    model = PCL(
        encoder_type=args.encoder,
        in_channels=in_channels,
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim=args.dim,
        queue_size=args.queue_size,
        momentum=args.momentum,
    ).to(device)

    optimizer = optim.Adam(model.encoder_q.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = ProtoNCELoss(tau=args.tau, r=args.r)

    start_epoch = 0
    cluster_results = None

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        warm_up = epoch < args.warm_up_epochs
        t0 = time.time()

        # E-step
        if not warm_up:
            print(f"[Epoch {epoch}] E-step: extracting features...")
            features = model.get_features(cluster_loader, device)
            cluster_results = cluster_features(
                features.numpy().astype("float32"),
                args.num_clusters, args.alpha, args.tau,
            )

        # M-step
        model.train()
        total_loss = 0.0

        for i, (views, _, indices) in enumerate(train_loader):
            x_q = views[0].to(device)
            x_k = views[1].to(device)

            q, k = model(x_q, x_k)
            loss = criterion(q, k, model.queue, cluster_results, indices, warm_up=warm_up)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 50 == 0:
                tag = "(warm-up)" if warm_up else ""
                print(f"  [{epoch}/{args.epochs}][{i}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e} {tag}")

        scheduler.step()
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] avg_loss={total_loss/len(train_loader):.4f}  time={elapsed:.1f}s")

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                {"epoch": epoch, "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "args": vars(args)},
                os.path.join(args.output_dir, f"pcl_epoch{epoch+1:04d}.pth"),
            )

        if args.tsne_freq > 0 and ((epoch + 1) % args.tsne_freq == 0 or epoch == args.epochs - 1):
            print(f"[Epoch {epoch}] Generating t-SNE...")
            save_tsne(model, args, epoch, device)


if __name__ == "__main__":
    main()
