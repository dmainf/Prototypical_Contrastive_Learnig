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
        self.transform = transform

        df = pd.read_csv(path)
        if "date" in df.columns:
            df = df.drop(columns=["date"])

        if variables == "univariate":
            data = df[[target_col]].values.astype(np.float32)
        else:
            data = df.select_dtypes(include=[np.number]).values.astype(np.float32)

        # 分割
        n = len(data)
        s, e = self.SPLIT_RATIOS[split]
        data = data[int(n * s):int(n * e)]

        # Z-score正規化（std=0のカラムはそのまま）
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        data = (data - mean) / (std + 1e-8)

        self.data = data          # (T, C)
        self.stride = stride
        self.indices = list(range(0, len(data) - seq_len + 1, stride))

    @property
    def in_channels(self):
        return self.data.shape[1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        window = self.data[start:start + self.seq_len]      # (seq_len, C)
        x = torch.tensor(window.T, dtype=torch.float32)    # (C, seq_len)
        if self.transform:
            x = self.transform(x)
        return x, 0
