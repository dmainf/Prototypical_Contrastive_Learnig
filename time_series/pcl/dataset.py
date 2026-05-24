import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ── Augmentation ──────────────────────────────────────────────────────────────

class Jitter:
    """ガウスノイズを付加する。波形の大局的な周期性・位相を保持する。"""
    def __init__(self, sigma: float = 0.03):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.sigma


class Scaling:
    """
    信号全体にランダムなスカラーを乗算する。
    振幅の絶対値ではなく波形ダイナミクスの形状に着目させる。
    """
    def __init__(self, scale_range: tuple = (0.8, 1.2)):
        self.low, self.high = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(1).uniform_(self.low, self.high).item()
        return x * scale


class ContinuousMasking:
    """
    系列の中間にある連続した区間をゼロでマスクする。
    時間軸を伸縮させないため周波数特性を完全に保持する。
    mask_ratio: マスクする区間長の割合（デフォルト0.2 = 20%をマスク）
    """
    def __init__(self, mask_ratio: float = 0.2):
        self.mask_ratio = mask_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[-1]
        mask_len = max(1, int(L * self.mask_ratio))
        start = torch.randint(0, L - mask_len + 1, (1,)).item()
        out = x.clone()
        out[:, start:start + mask_len] = 0.0
        return out


class TimeSeriesAugmentation:
    """Scaling + ContinuousMasking + Jitter を組み合わせたAugmentation。"""
    def __init__(
        self,
        jitter_sigma: float = 0.03,
        scale_range: tuple = (0.8, 1.2),
        mask_ratio: float = 0.2,
        use_jitter: bool = True,
        use_scaling: bool = True,
        use_masking: bool = True,
    ):
        self.augments = []
        if use_scaling:
            self.augments.append(Scaling(scale_range))
        if use_masking:
            self.augments.append(ContinuousMasking(mask_ratio))
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
