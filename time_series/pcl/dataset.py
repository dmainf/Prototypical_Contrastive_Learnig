import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ── Augmentation ──────────────────────────────────────────────────────────────

class Jitter:
    """ガウスノイズを付加する。"""
    def __init__(self, sigma: float = 0.03):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.sigma


class WindowSlicing:
    """
    ウィンドウをランダムにクロップして元の長さにリサイズする（画像のRandomCropに相当）。
    ratio: クロップするウィンドウ長の割合（デフォルト0.9 = 90%を切り取る）
    """
    def __init__(self, ratio: float = 0.9):
        self.ratio = ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[-1]
        slice_len = max(2, int(L * self.ratio))
        start = torch.randint(0, L - slice_len + 1, (1,)).item()
        sliced = x[:, start:start + slice_len]
        return F.interpolate(sliced.unsqueeze(0), size=L, mode="linear",
                             align_corners=False).squeeze(0)


class TimeSeriesAugmentation:
    """Jitter + WindowSlicing を組み合わせたAugmentation。"""
    def __init__(self, jitter_sigma: float = 0.03, slice_ratio: float = 0.9,
                 use_jitter: bool = True, use_slicing: bool = True):
        self.augments = []
        if use_slicing:
            self.augments.append(WindowSlicing(slice_ratio))
        if use_jitter:
            self.augments.append(Jitter(jitter_sigma))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for aug in self.augments:
            x = aug(x)
        return x


# ── Two-view transform ────────────────────────────────────────────────────────

class TwoViewTransform:
    """同じウィンドウに異なるAugmentationを2回適用して2つのviewを生成する。"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x: torch.Tensor):
        return [self.transform(x), self.transform(x)]


# ── Indexed dataset ───────────────────────────────────────────────────────────

class IndexedDataset(Dataset):
    """サンプルインデックスも返すようにラップするDataset。"""
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data, target, idx

    def __len__(self):
        return len(self.dataset)


# ── 時系列Dataset ─────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    CSVから時系列を読み込み、スライディングウィンドウでサンプルを生成する。

    引数:
        path       : CSVファイルのパス
        variables  : 'univariate' or 'multivariate'
        target_col : univariateのとき使うカラム名（デフォルト 'OT'）
        seq_len    : ウィンドウ長
        stride     : ウィンドウをずらすステップ数
        split      : 'train' / 'val' / 'test'
        transform  : Augmentation（Noneなら生データを返す）

    出力:
        x: (in_channels, seq_len) の Tensor
        0: ダミーラベル（ラベルなし）
    """

    SPLIT_RATIOS = {"train": (0.0, 0.6), "val": (0.6, 0.8), "test": (0.8, 1.0)}

    def __init__(
        self,
        path: str,
        variables: str = "univariate",
        target_col: str = "OT",
        seq_len: int = 96,
        stride: int = 1,
        split: str = "train",
        transform=None,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.variables = variables

        df = pd.read_csv(path)
        if "date" in df.columns:
            df = df.drop(columns=["date"])

        if variables == "univariate":
            raw = df[[target_col]].values.astype(np.float32)   # (T, 1)
        else:
            raw = df.select_dtypes(include=[np.number]).values.astype(np.float32)  # (T, C)

        # 分割
        n = len(raw)
        s, e = self.SPLIT_RATIOS[split]
        raw = raw[int(n * s):int(n * e)]

        # Z-score正規化（カラムごと）
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        raw = (raw - mean) / (std + 1e-8)

        if variables == "individuals":
            # 各列を独立した個体の単変量系列として扱う
            # self.data: (N_individuals, T)
            self.data = torch.tensor(raw.T, dtype=torch.float32)
            self.n_individuals = self.data.shape[0]
            self.n_windows_per_individual = (raw.shape[0] - seq_len) // stride + 1
            self.n_windows = self.n_individuals * self.n_windows_per_individual
        else:
            # (C, T) — 全チャンネルを1サンプルとして扱う通常モード
            self.data = torch.tensor(raw.T, dtype=torch.float32)
            self.n_windows = (raw.shape[0] - seq_len) // stride + 1

    @property
    def in_channels(self):
        return 1 if self.variables == "individuals" else self.data.shape[0]

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        if self.variables == "individuals":
            individual = idx // self.n_windows_per_individual
            window_idx = idx % self.n_windows_per_individual
            start = window_idx * self.stride
            x = self.data[individual, start:start + self.seq_len].unsqueeze(0)  # (1, seq_len)
            label = individual
        else:
            start = idx * self.stride
            x = self.data[:, start:start + self.seq_len]   # (C, seq_len)
            label = 0

        if self.transform:
            x = self.transform(x.clone())
        return x, label
