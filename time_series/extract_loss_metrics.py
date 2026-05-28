"""
各チェックポイントの InfoNCE (in-batch) / ProtoNCE を計算して CSV に出力する。
  align_uniform/infonce.csv  : 行=モード, 列=epoch
  align_uniform/protonce.csv : 行=モード, 列=epoch

Usage:
  python3 extract_loss_metrics.py
"""

import csv
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pcl.model import PCL
from pcl.clustering import cluster_features
from pcl.dataset import TimeSeriesDataset, TwoViewTransform, TimeSeriesAugmentation

CONFIGS = [
    ("Align_Uniform",       "./checkpoints_Align-Uniform"),
    ("Align_Uniform+Proto", "./checkpoints_Align-Uniform+ProtoNCE"),
    ("InfoNCE",             "./checkpoints_InfoNCE"),
    ("InfoNCE+Proto",       "./checkpoints_InfoNCE+ProtoNCE"),
]

DATA_PATH    = "./datasets/electricity/electricity.csv"
TAU          = 0.1
N_FEATS      = 3000
N_NCE_BATCH  = 10   # InfoNCE 計算に使うバッチ数
WORKERS      = 0
BATCH        = 256
OUT_DIR      = "./align_uniform"


@torch.no_grad()
def get_infonce(model, pair_loader, device, tau, n_batches):
    """In-batch 負例による InfoNCE"""
    total = 0.0
    count = 0
    for views, _ in pair_loader:
        x1, x2 = views[0].to(device), views[1].to(device)
        q = F.normalize(model.encoder_q(x1), dim=1)
        k = F.normalize(model.encoder_k(x2), dim=1)
        N = q.shape[0]
        sim = torch.mm(q, k.t()) / tau          # N×N
        labels = torch.arange(N, device=device)
        total += F.cross_entropy(sim, labels).item()
        count += 1
        if count >= n_batches:
            break
    return total / count


@torch.no_grad()
def extract_features(model, feat_loader, device, n_feats):
    feats = []
    for x, _ in feat_loader:
        f = F.normalize(model.encoder_q(x.to(device)), dim=1)
        feats.append(f.cpu())
        if sum(len(v) for v in feats) >= n_feats:
            break
    return torch.cat(feats)[:n_feats]


def get_protonc(model, feat_loader, device, num_clusters, alpha, tau, n_feats):
    """全特徴量でk-meansを走らせてProtoNCEを計算する"""
    feats = extract_features(model, feat_loader, device, n_feats)
    feats_np = feats.numpy().astype("float32")

    cluster_results = cluster_features(feats_np, num_clusters, alpha, tau)

    total = 0.0
    q = feats.to(device)
    for centroids, assignments, phi in cluster_results:
        centroids   = centroids.to(device)
        assignments = assignments.to(device)
        phi         = phi.to(device)
        logits = torch.mm(q, centroids.t()) / phi.clamp(min=tau).unsqueeze(0)
        total += F.cross_entropy(logits, assignments).item()
    return total / len(cluster_results)


def write_wide_csv(path, epochs, data):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode"] + [f"epoch_{e}" for e in epochs])
        for mode, vals in data.items():
            writer.writerow([mode] + [round(vals[e], 4) for e in epochs])
    print(f"Saved: {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    infonce_data  = {}
    protonc_data  = {}
    all_epochs    = set()

    for label, ckpt_dir in CONFIGS:
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "pcl_epoch*.pth")))
        if not ckpts:
            print(f"  [skip] no checkpoints in {ckpt_dir}")
            continue
        print(f"\n=== {label} ===")

        first_args  = torch.load(ckpts[0], map_location="cpu", weights_only=False).get("args", {})
        seq_len     = first_args.get("seq_len", 512)
        stride      = first_args.get("stride", seq_len)
        variables   = first_args.get("variables", "individuals")
        num_clusters = first_args.get("num_clusters", [50, 200, 500])
        alpha       = first_args.get("alpha", 10.0)

        aug = TimeSeriesAugmentation()
        pair_ds = TimeSeriesDataset(
            path=DATA_PATH, variables=variables,
            seq_len=seq_len, stride=stride, split="train",
            transform=TwoViewTransform(aug),
        )
        feat_ds = TimeSeriesDataset(
            path=DATA_PATH, variables=variables,
            seq_len=seq_len, stride=stride, split="train",
            transform=None,
        )
        pair_loader = DataLoader(pair_ds, batch_size=BATCH, shuffle=True,  num_workers=WORKERS)
        feat_loader = DataLoader(feat_ds, batch_size=BATCH, shuffle=False, num_workers=WORKERS)

        model = PCL(
            encoder_type=first_args.get("encoder", "transformer"),
            in_channels=first_args.get("in_channels", 1),
            seq_len=seq_len,
            d_model=first_args.get("d_model", 64),
            nhead=first_args.get("nhead", 4),
            num_layers=first_args.get("num_layers", 2),
            dim=first_args.get("dim", 128),
            chronos_model_name=first_args.get("chronos_model", "amazon/chronos-bolt-small"),
        ).to(device)

        infonce_data[label] = {}
        protonc_data[label] = {}

        for ckpt_path in ckpts:
            epoch = int(os.path.basename(ckpt_path)
                        .replace("pcl_epoch", "").replace(".pth", ""))
            all_epochs.add(epoch)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            nce  = get_infonce(model, pair_loader, device, TAU, N_NCE_BATCH)
            with torch.no_grad():
                pnce = get_protonc(model, feat_loader, device,
                                   num_clusters, alpha, TAU, N_FEATS)

            print(f"  epoch {epoch:4d}  infonce={nce:.4f}  protonc={pnce:.4f}")
            infonce_data[label][epoch] = nce
            protonc_data[label][epoch] = pnce

        del model

    epochs = sorted(all_epochs)
    write_wide_csv(os.path.join(OUT_DIR, "infonce.csv"),  epochs, infonce_data)
    write_wide_csv(os.path.join(OUT_DIR, "protonce.csv"), epochs, protonc_data)


if __name__ == "__main__":
    main()
