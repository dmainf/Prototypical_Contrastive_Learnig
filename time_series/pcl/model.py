import torch
import torch.nn as nn


# ── Encoder 定義 ──────────────────────────────────────────────────────────────

class _TransformerEncoder(nn.Module):
    """
    Transformerベースのエンコーダ。
    入力: (batch, in_channels, seq_len)
    出力: (batch, dim)
    """
    def __init__(self, in_channels=1, seq_len=96, d_model=64,
                 nhead=4, num_layers=2, dim=128, dropout=0.1):
        super().__init__()
        self.out_dim = dim
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, dim)

    def forward(self, x):
        # x: (batch, in_channels, seq_len) → (batch, seq_len, in_channels)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x) + self.pos_enc
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


class _CNN1DEncoder(nn.Module):
    """
    1D CNNベースのエンコーダ（軽量・高速）。
    入力: (batch, in_channels, seq_len)
    出力: (batch, dim)
    """
    def __init__(self, in_channels=1, dim=128):
        super().__init__()
        self.out_dim = dim
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, dim)

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.fc(x)


class ChronosBoltEncoder(nn.Module):
    """
    Chronos-Bolt をバックボーンとするエンコーダ。
    backbone は完全凍結。proj のみ学習される。
    入力: (batch, in_channels, seq_len)
    出力: (batch, dim)

    時間軸は平均プーリングで集約（末尾マスク時の劣化を回避）。
    チャンネルは flatten して結合（変量間の相関を proj に渡す）。
    """
    def __init__(self, chronos_model, in_channels: int, dim: int = 128):
        super().__init__()
        d = chronos_model.config.d_model
        self.out_dim = dim
        self.chronos = chronos_model
        self.proj = nn.Sequential(
            nn.Linear(d * in_channels, d),
            nn.ReLU(),
            nn.Linear(d, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, C, L = x.shape
        context = x.view(batch * C, L).float()
        with torch.no_grad():
            hidden, _, _, _ = self.chronos.encode(context)
        # (batch*C, T, d) → mean over T → (batch*C, d) → (batch, C*d)
        feat = hidden.mean(dim=1).view(batch, C * hidden.shape[-1])
        return self.proj(feat)


def _load_chronos_backbone(model_name: str):
    from chronos import ChronosBoltPipeline
    pipe = ChronosBoltPipeline.from_pretrained(model_name, device_map="cpu", dtype=torch.float32)
    backbone = pipe.model
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def build_encoder(encoder_type: str, in_channels: int, seq_len: int,
                  d_model: int, nhead: int, num_layers: int, dim: int):
    if encoder_type == "transformer":
        return _TransformerEncoder(in_channels, seq_len, d_model, nhead, num_layers, dim)
    elif encoder_type == "cnn":
        return _CNN1DEncoder(in_channels, dim)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Choose 'transformer', 'cnn', or 'chronos'.")


# ── PCL モデル ────────────────────────────────────────────────────────────────

class PCL(nn.Module):
    """
    Prototypical Contrastive Learning — 時系列版。
    画像版と同じインターフェース (forward / get_features / queue)。
    """

    def __init__(
        self,
        encoder_type: str = "transformer",
        in_channels: int = 1,
        seq_len: int = 96,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim: int = 128,
        queue_size: int = 4096,
        momentum: float = 0.999,
        chronos_model_name: str = "amazon/chronos-bolt-small",
    ):
        super().__init__()
        self.momentum = momentum
        self.queue_size = queue_size

        if encoder_type == "chronos":
            print(f"Loading Chronos-Bolt backbone: {chronos_model_name} ...")
            backbone = _load_chronos_backbone(chronos_model_name)
            def _make():
                return ChronosBoltEncoder(backbone, in_channels, dim)
        else:
            def _make():
                return build_encoder(encoder_type, in_channels, seq_len, d_model, nhead, num_layers, dim)

        self.encoder_q = _make()
        self.encoder_k = _make()

        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

        feat_dim = self.encoder_q.out_dim
        self.register_buffer("queue", torch.randn(feat_dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _update_momentum_encoder(self):
        m = self.momentum
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = m * p_k.data + (1.0 - m) * p_q.data

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        end = min(ptr + batch_size, self.queue_size)
        actual = end - ptr
        self.queue[:, ptr:end] = keys[:actual].T
        if actual < batch_size:
            self.queue[:, :batch_size - actual] = keys[actual:].T
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor):
        """
        x_q, x_k: (batch, in_channels, seq_len)
        Returns q, k — caller must call enqueue(k) AFTER computing the loss.
        """
        q = nn.functional.normalize(self.encoder_q(x_q), dim=1)
        with torch.no_grad():
            self._update_momentum_encoder()
            k = nn.functional.normalize(self.encoder_k(x_k), dim=1)
        return q, k

    def enqueue(self, keys: torch.Tensor):
        self._enqueue(keys.detach())

    @torch.no_grad()
    def get_features(self, loader, device: torch.device) -> torch.Tensor:
        """全データのMomentumエンコーダ特徴量を抽出（クラスタリング用）。"""
        self.encoder_k.eval()
        all_features = []
        for batch in loader:
            x = batch[0]
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            feats = nn.functional.normalize(self.encoder_k(x), dim=1)
            all_features.append(feats.cpu())
        self.encoder_k.train()
        return torch.cat(all_features, dim=0)
