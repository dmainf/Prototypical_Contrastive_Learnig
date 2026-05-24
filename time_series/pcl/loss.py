import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNCELoss(nn.Module):
    """
    ProtoNCE loss (Eq. 11 in the paper).

    Combines:
      1. Instance-wise InfoNCE (MoCo-style, using a queue of negative keys)
      2. Prototypical contrastive loss across M granularity levels
    """

    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def _info_nce(
        self, q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor
    ) -> torch.Tensor:
        """Instance-wise InfoNCE (Eq. 1 / first term of Eq. 11)."""
        N = q.shape[0]
        l_pos = (q * k).sum(dim=1, keepdim=True) / self.tau       # (N, 1)
        l_neg = torch.mm(q, queue.detach()) / self.tau             # (N, K)
        logits = torch.cat([l_pos, l_neg], dim=1)                  # (N, 1+K)
        labels = torch.zeros(N, dtype=torch.long, device=q.device)
        return F.cross_entropy(logits, labels)

    def _proto_nce_single(
        self,
        q: torch.Tensor,
        prototypes: torch.Tensor,
        assignments: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        """Prototypical contrastive loss for one granularity level (Eq. 10 in the paper)."""
        # (N, D) x (D, K) -> (N, K), scaled by per-prototype concentration
        logits = torch.mm(q, prototypes.t()) / phi.clamp(min=self.tau).unsqueeze(0)
        return F.cross_entropy(logits, assignments)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        queue: torch.Tensor,
        cluster_results: list | None,
        sample_indices: torch.Tensor,
        warm_up: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        q              : (N, D) query features
        k              : (N, D) key features (momentum encoder)
        queue          : (D, queue_size) negative key queue
        cluster_results: list of (centroids, assignments_full, phi) or None
        sample_indices : (N,) indices of current batch within the full dataset
        warm_up        : if True, skip prototypical term (only InfoNCE)

        Returns (loss, breakdown) where breakdown is a dict for logging.
        """
        info_nce = self._info_nce(q, k, queue)
        loss = info_nce
        breakdown = {"info_nce": info_nce.item()}

        if not warm_up and cluster_results:
            proto_loss = torch.tensor(0.0, device=q.device)
            for centroids, assignments_all, phi in cluster_results:
                batch_assign = assignments_all[sample_indices].to(q.device)
                proto_loss = proto_loss + self._proto_nce_single(q, centroids, batch_assign, phi)
            proto_loss = proto_loss / len(cluster_results)
            loss = loss + proto_loss
            breakdown["proto_nce"] = proto_loss.item()

        return loss, breakdown
