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

    def __init__(self, tau: float = 0.1, r: int = 16000):
        super().__init__()
        self.tau = tau
        self.r = r  # number of negative prototypes sampled per forward pass

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
        """Prototypical contrastive loss for one granularity level (second term of Eq. 11)."""
        N = q.shape[0]
        device = q.device
        K = prototypes.shape[0]

        prototypes = prototypes.to(device)
        phi = phi.to(device)

        # Positive: each sample's assigned prototype
        pos_proto = prototypes[assignments]                          # (N, D)
        pos_phi = phi[assignments]                                   # (N,)
        pos_sim = (q * pos_proto).sum(dim=1) / pos_phi              # (N,)

        # Sample r random negative prototypes
        r = min(self.r, K)
        neg_idx = torch.randint(0, K, (r,), device=device)
        neg_proto = prototypes[neg_idx]                              # (r, D)
        neg_phi = phi[neg_idx]                                       # (r,)
        neg_sim = torch.mm(q, neg_proto.t()) / neg_phi.unsqueeze(0) # (N, r)

        # Cross-entropy: positive is at column 0
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (N, 1+r)
        labels = torch.zeros(N, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        queue: torch.Tensor,
        cluster_results: list | None,
        sample_indices: torch.Tensor,
        warm_up: bool = False,
    ) -> torch.Tensor:
        """
        q              : (N, D) query features
        k              : (N, D) key features (momentum encoder)
        queue          : (D, queue_size) negative key queue
        cluster_results: list of (centroids, assignments_full, phi) or None
        sample_indices : (N,) indices of current batch within the full dataset
        warm_up        : if True, skip prototypical term (only InfoNCE)
        """
        loss = self._info_nce(q, k, queue)

        if not warm_up and cluster_results:
            proto_loss = torch.tensor(0.0, device=q.device)
            for centroids, assignments_all, phi in cluster_results:
                batch_assign = assignments_all[sample_indices.cpu()].to(q.device)
                proto_loss = proto_loss + self._proto_nce_single(q, centroids, batch_assign, phi)
            loss = loss + proto_loss / len(cluster_results)

        return loss
