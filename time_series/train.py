"""
Prototypical Contrastive Learning — 時系列版 Training Script
Paper: https://arxiv.org/abs/2005.04966  (ICLR 2021)

Usage (デフォルト / ETTh1 OTのみ):
  python3 train.py

Usage (多変量 / CNNエンコーダ):
  python3 train.py --variables multivariate --encoder cnn
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from pcl.model import PCL
from pcl.loss import HybridContrastiveLoss
from pcl.clustering import cluster_features
from pcl.dataset import (
    TimeSeriesDataset, TwoViewTransform, IndexedDataset,
    TimeSeriesAugmentation,
)


def parse_args():
    p = argparse.ArgumentParser()

    # データ
    p.add_argument("--data-path", default="./datasets/electricity/electricity.csv",
                   help="CSVファイルのパス")
    p.add_argument("--variables", default="individuals",
                   choices=["univariate", "multivariate", "individuals"],
                   help="univariate: target-colのみ / multivariate: 全チャンネルを1サンプル / individuals: 列ごとに独立した個体")
    p.add_argument("--target-col", default="OT",
                   help="univariateのときに使うカラム名")
    p.add_argument("--seq-len", default=512, type=int,
                   help="1サンプルのウィンドウ長")
    p.add_argument("--stride", default=None, type=int,
                   help="スライディングウィンドウのストライド（デフォルト: seq-lenと同じ）")

    # エンコーダ
    p.add_argument("--encoder", default="chronos",
                   choices=["transformer", "cnn", "chronos"],
                   help="エンコーダアーキテクチャ")
    p.add_argument("--chronos-model", default="amazon/chronos-bolt-small",
                   help="Chronos-Bolt のモデル名（--encoder chronos のみ有効）"
                        " 例: amazon/chronos-bolt-mini, amazon/chronos-bolt-small, amazon/chronos-bolt-base")
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
    p.add_argument("--scale-range", nargs=2, type=float, default=[0.8, 1.2],
                   metavar=("LOW", "HIGH"),
                   help="Scalingの振幅スケール範囲")
    p.add_argument("--mask-ratio", default=0.2, type=float,
                   help="ContinuousMaskingでマスクする区間長の割合（0〜1）")
    p.add_argument("--no-jitter", action="store_true",
                   help="Jitterを無効化する")
    p.add_argument("--no-scaling", action="store_true",
                   help="Scalingを無効化する")
    p.add_argument("--no-masking", action="store_true",
                   help="ContinuousMaskingを無効化する")

    # 対照学習
    p.add_argument("--queue-size", default=4096, type=int)
    p.add_argument("--momentum", default=0.999, type=float)
    p.add_argument("--tau", default=0.1, type=float)
    # Loss の設計（直交する2軸）
    p.add_argument("--base-loss", default="align_uniform",
                   choices=["infonce", "align_uniform"],
                   help="ベース損失: infonce (MoCo方式) / align_uniform (Wang & Isola, ICML 2020)")
    p.add_argument("--use-proto", action="store_true",
                   help="ProtoNCEをベース損失に加算する（EMクラスタリングを有効化）")
    # Alignment / Uniformity パラメータ（--base-loss align_uniform のみ有効）
    p.add_argument("--align-alpha", default=2.0, type=float,
                   help="L_align のべき乗パラメータ alpha")
    p.add_argument("--uniform-t", default=2.0, type=float,
                   help="L_uniform のガウスカーネルパラメータ t")
    p.add_argument("--lam", default=1.0, type=float,
                   help="L_uniform の重み λ: loss = L_align + λ * L_uniform")
    # クラスタリング
    p.add_argument("--num-clusters", nargs="+", type=int, default=[50, 200, 500],
                   help="クラスタ数（複数指定で階層的プロトタイプ）")
    p.add_argument("--alpha", default=10.0, type=float)

    # 学習
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--warm-up-epochs", default=10, type=int)
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--lr", default=1e-4, type=float,
                   help="学習率（Transformerはresnetより小さい値が安定）")
    p.add_argument("--weight-decay", default=1e-4, type=float)
    p.add_argument("--workers", default=2, type=int)

    # 保存
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--save-freq", default=10, type=int)
    p.add_argument("--resume", default="", type=str)

    args = p.parse_args()
    if args.stride is None:
        args.stride = args.seq_len
    return args


def build_loaders(args, pin_memory: bool = False):
    aug = TimeSeriesAugmentation(
        jitter_sigma=args.jitter_sigma,
        scale_range=tuple(args.scale_range),
        mask_ratio=args.mask_ratio,
        use_jitter=not args.no_jitter,
        use_scaling=not args.no_scaling,
        use_masking=not args.no_masking,
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
    args.queue_size = min(args.queue_size, n)

    indexed_ds = IndexedDataset(train_ds)
    train_loader = DataLoader(indexed_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin_memory, drop_last=True)
    cluster_loader = DataLoader(cluster_ds, batch_size=args.batch_size * 2, shuffle=False,
                                num_workers=args.workers, pin_memory=pin_memory)
    return train_loader, cluster_loader, train_ds.in_channels


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
        chronos_model_name=args.chronos_model,
    ).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.encoder_q.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = HybridContrastiveLoss(
        base_type=args.base_loss,
        use_proto=args.use_proto,
        alpha=args.align_alpha,
        t=args.uniform_t,
        lam=args.lam,
        tau=args.tau,
    )

    start_epoch = 0
    cluster_results = None
    loss_history = []
    loss_rows = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    if start_epoch == 0:
        save_checkpoint(
            {"epoch": -1, "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "args": vars(args)},
            os.path.join(args.output_dir, "pcl_epoch0000.pth"),
        )

    for epoch in range(start_epoch, args.epochs):
        warm_up = epoch < args.warm_up_epochs
        t0 = time.time()

        # E-step
        if args.use_proto and not warm_up:
            print(f"[Epoch {epoch}] E-step: extracting features...")
            features = model.get_features(cluster_loader, device)
            features_np = features.numpy().astype("float32")
            nan_mask = np.isfinite(features_np).all(axis=1)
            if not nan_mask.all():
                print(f"  [warning] {(~nan_mask).sum()} NaN features zeroed before clustering")
                features_np[~nan_mask] = 0.0
            cluster_results = [
                (c.to(device), a, p.to(device))
                for c, a, p in cluster_features(
                    features_np,
                    args.num_clusters, args.alpha, args.tau,
                )
            ]

        # M-step
        model.train()
        total_loss = 0.0
        total_base = 0.0
        total_proto_nce = 0.0

        for i, (views, _, indices) in enumerate(train_loader):
            x_q = views[0].to(device)
            x_k = views[1].to(device)

            q, k = model(x_q, x_k)
            loss, breakdown = criterion(q, k, model.queue, cluster_results, indices, is_warmup=warm_up)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder_q.parameters(), max_norm=1.0)
            optimizer.step()

            if args.base_loss == "infonce":
                model.enqueue(k)

            total_loss += loss.item()
            total_base += breakdown["base_loss"]
            total_proto_nce += breakdown.get("proto_nce", 0.0)
            if i % 50 == 0:
                tag = "(warm-up)" if warm_up else ""
                if "info_nce" in breakdown:
                    base_str = f"info={breakdown['info_nce']:.4f}"
                else:
                    base_str = f"align={breakdown['align']:.4f} uniform={breakdown['uniform']:.4f}"
                proto_str = f" proto={breakdown['proto_nce']:.4f}" if "proto_nce" in breakdown else ""
                print(f"  [{epoch}/{args.epochs}][{i}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} {base_str}{proto_str} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e} {tag}")

        scheduler.step()
        elapsed = time.time() - t0
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_base = total_base / n_batches
        avg_proto = total_proto_nce / n_batches
        loss_history.append(avg_loss)
        loss_rows.append({"epoch": epoch, "loss": round(avg_loss, 6),
                          "base": round(avg_base, 6), "proto": round(avg_proto, 6)})
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f} "
              f"(base={avg_base:.4f} proto={avg_proto:.4f})  time={elapsed:.1f}s")

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                {"epoch": epoch, "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "args": vars(args)},
                os.path.join(args.output_dir, f"pcl_epoch{epoch+1:04d}.pth"),
            )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(start_epoch, start_epoch + len(loss_history)), loss_history)
    ax.axvline(x=args.warm_up_epochs - 1, color="gray", linestyle="--", label="warm-up end")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Loss")
    ax.set_title("Training Loss Curve")
    ax.legend()
    plt.tight_layout()
    loss_path = os.path.join(args.output_dir, "loss_curve.png")
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve saved: {loss_path}")

    csv_path = os.path.join(args.output_dir, "loss.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "base", "proto"])
        writer.writeheader()
        writer.writerows(loss_rows)
    print(f"Loss CSV saved: {csv_path}")

if __name__ == "__main__":
    main()
