import torch
import torch.nn as nn
import torchvision.models as models


class PCL(nn.Module):
    """
    Prototypical Contrastive Learning model.
    Consists of a query encoder, a momentum (key) encoder, and a queue
    for instance-wise contrastive negatives.
    """

    def __init__(
        self,
        base_encoder: str = "resnet50",
        dim: int = 128,
        queue_size: int = 65536,
        momentum: float = 0.999,
        use_mlp: bool = False,
    ):
        super().__init__()
        self.momentum = momentum
        self.queue_size = queue_size

        encoder_cls = getattr(models, base_encoder)
        self.encoder_q = encoder_cls(weights=None)
        self.encoder_k = encoder_cls(weights=None)

        feat_dim = self.encoder_q.fc.in_features
        if use_mlp:
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, dim)
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, dim)
            )
        else:
            self.encoder_q.fc = nn.Linear(feat_dim, dim)
            self.encoder_k.fc = nn.Linear(feat_dim, dim)

        # Initialize momentum encoder with same weights; freeze it
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

        # Queue stores negative key features: shape (dim, queue_size)
        self.register_buffer("queue", torch.randn(dim, queue_size))
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
            self.queue[:, : batch_size - actual] = keys[actual:].T
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor):
        """
        im_q: augmented view 1 (N, C, H, W)
        im_k: augmented view 2 (N, C, H, W)
        Returns:
            q: L2-normalized query features (N, dim)
            k: L2-normalized key features from momentum encoder (N, dim)
        """
        q = nn.functional.normalize(self.encoder_q(im_q), dim=1)

        with torch.no_grad():
            self._update_momentum_encoder()
            k = nn.functional.normalize(self.encoder_k(im_k), dim=1)

        self._enqueue(k.detach())
        return q, k

    @torch.no_grad()
    def get_features(self, loader, device: torch.device) -> torch.Tensor:
        """Extract L2-normalized features from all data using the momentum encoder."""
        self.encoder_k.eval()
        all_features = []
        for batch in loader:
            images = batch[0]
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            feats = nn.functional.normalize(self.encoder_k(images), dim=1)
            all_features.append(feats.cpu())
        self.encoder_k.train()
        return torch.cat(all_features, dim=0)
