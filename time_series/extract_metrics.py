"""
各チェックポイントの Alignment / Uniformity を計算して CSV に出力する。
  alignment.csv : 行=モード(4行), 列=epoch
  uniformity.csv: 行=モード(4行), 列=epoch

Usage:
  python3 extract_metrics.py
"""

import csv
import glob
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pcl.model import PCL
from pcl.dataset import TimeSeriesDataset, TwoViewTransform, TimeSeriesAugmentation

CONFIGS = [
    ("Align_Uniform",       "./checkpoints_Align-Uniform"),
    ("Align_Uniform+Proto", "./checkpoints_Align-Uniform+ProtoNCE"),
    ("InfoNCE",             "./checkpoints_InfoNCE"),
    ("InfoNCE+Proto",       "./checkpoints_InfoNCE+ProtoNCE"),
]

DATA_PATH = "./datasets/electricity/electricity.csv"
N_PAIRS   = 3000
N_FEATS   = 5000
WORKERS   = 0
BATCH     = 512


@torch.no_grad()
def get_alignment(model, loader, device, n_pairs):
    dists = []
    for views, _ in loader:
        x1, x2 = views[0].to(device), views[1].to(device)
        f1 = F.normalize(model.encoder_k(x1), dim=1)
        f2 = F.normalize(model.encoder_k(x2), dim=1)
        dists.append((f1 - f2).norm(dim=1).cpu())
        if sum(len(d) for d in dists) >= n_pairs:
            break
    return torch.cat(dists)[:n_pairs].mean().item()


@torch.no_grad()
def get_uniformity(model, loader, device, n_feats, t=2.0):
    feats = []
    for x, _ in loader:
        f = F.normalize(model.encoder_k(x.to(device)), dim=1)
        feats.append(f.cpu())
        if sum(len(f) for f in feats) >= n_feats:
            break
    z = torch.cat(feats)[:n_feats]
    sq_norms = (z * z).sum(dim=1)
    sq_pdist = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * (z @ z.t())
    sq_pdist = sq_pdist.clamp(min=0)
    n = z.shape[0]
    mask = torch.ones(n, n, dtype=torch.bool).triu(diagonal=1)
    return sq_pdist[mask].mul(-t).exp().mean().log().item()


def write_wide_csv(path, epochs, data):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode"] + [f"epoch_{e}" for e in epochs])
        for mode, vals in data.items():
            writer.writerow([mode] + [round(vals[e], 4) for e in epochs])
    print(f"Saved: {path}")


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    align_data = {}
    unif_data  = {}
    all_epochs = set()

    for label, ckpt_dir in CONFIGS:
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "pcl_epoch*.pth")))
        if not ckpts:
            print(f"  [skip] no checkpoints in {ckpt_dir}")
            continue
        print(f"\n=== {label} ===")

        first_args = torch.load(ckpts[0], map_location="cpu", weights_only=False).get("args", {})
        seq_len   = first_args.get("seq_len", 512)
        stride    = first_args.get("stride", seq_len)
        variables = first_args.get("variables", "individuals")

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

        # モデルは1回だけ作成し、以降は state_dict だけ入れ替える
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

        align_data[label] = {}
        unif_data[label]  = {}

        for ckpt_path in ckpts:
            epoch = int(os.path.basename(ckpt_path)
                        .replace("pcl_epoch", "").replace(".pth", ""))
            all_epochs.add(epoch)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            align = get_alignment(model, pair_loader, device, N_PAIRS)
            unif  = get_uniformity(model, feat_loader, device, N_FEATS)
            print(f"  epoch {epoch:4d}  alignment={align:.4f}  uniformity={unif:.4f}")
            align_data[label][epoch] = align
            unif_data[label][epoch]  = unif

        del model

    epochs = sorted(all_epochs)
    write_wide_csv("alignment.csv",  epochs, align_data)
    write_wide_csv("uniformity.csv", epochs, unif_data)


if __name__ == "__main__":
    main()
