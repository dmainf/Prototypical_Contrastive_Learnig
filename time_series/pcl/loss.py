import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridContrastiveLoss(nn.Module):
    """
    Unified contrastive loss supporting 4 modes via base_type × use_proto:

      base_type="infonce",      use_proto=False  →  pure InfoNCE (MoCo-style)
      base_type="infonce",      use_proto=True   →  PCL (InfoNCE + ProtoNCE)
      base_type="align_uniform", use_proto=False →  pure Align & Uniform
      base_type="align_uniform", use_proto=True  →  hybrid (Align & Uniform + ProtoNCE)
    """

    def __init__(
        self,
        base_type: str = "align_uniform",
        use_proto: bool = True,
        alpha: float = 2.0,
        t: float = 2.0,
        lam: float = 1.0,
        tau: float = 0.1,
    ):
        super().__init__()
        self.base_type = base_type
        self.use_proto = use_proto
        self.alpha = alpha
        self.t = t
        self.lam = lam
        self.tau = tau

    def _align(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).norm(dim=1).pow(self.alpha).mean()

    def _uniform(self, x: torch.Tensor) -> torch.Tensor:
        sq_pdist = torch.pdist(x, p=2).pow(2)
        return sq_pdist.mul(-self.t).exp().mean().log()

    def _info_nce(self, q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor) -> torch.Tensor:
        N = q.shape[0]
        l_pos = (q * k).sum(dim=1, keepdim=True) / self.tau
        l_neg = torch.mm(q, queue.detach()) / self.tau
        logits = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=q.device)
        return F.cross_entropy(logits, labels)

    def _proto_nce_single(
        self,
        q: torch.Tensor,
        prototypes: torch.Tensor,
        assignments: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.mm(q, prototypes.t()) / phi.clamp(min=self.tau).unsqueeze(0)
        return F.cross_entropy(logits, assignments)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        queue: torch.Tensor,
        cluster_results: list | None,
        sample_indices: torch.Tensor,
        is_warmup: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        breakdown = {}

        if self.base_type == "align_uniform":
            l_align = self._align(q, k)
            l_uniform = (self._uniform(q) + self._uniform(k)) / 2.0
            base_loss = l_align + self.lam * l_uniform
            breakdown["align"] = l_align.item()
            breakdown["uniform"] = l_uniform.item()
        else:
            base_loss = self._info_nce(q, k, queue)
            breakdown["info_nce"] = base_loss.item()

        breakdown["base_loss"] = base_loss.item()
        loss = base_loss

        if self.use_proto and not is_warmup and cluster_results:
            proto_loss = torch.tensor(0.0, device=q.device)
            for centroids, assignments_all, phi in cluster_results:
                batch_assign = assignments_all[sample_indices].to(q.device)
                proto_loss = proto_loss + self._proto_nce_single(q, centroids, batch_assign, phi)
            proto_loss = proto_loss / len(cluster_results)
            loss = loss + proto_loss
            breakdown["proto_nce"] = proto_loss.item()

        return loss, breakdown
