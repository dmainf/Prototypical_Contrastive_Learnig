import numpy as np
import torch

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import faiss
    _FAISS = True
except ImportError:
    _FAISS = False


def _kmeans_faiss(features: np.ndarray, k: int, niter: int = 20) -> tuple:
    n, d = features.shape
    kmeans = faiss.Kmeans(d, k, niter=niter, verbose=False, gpu=False)
    kmeans.train(features)
    _, assignments = kmeans.index.search(features, 1)
    return kmeans.centroids, assignments.squeeze(1)


def _kmeans_sklearn(features: np.ndarray, k: int) -> tuple:
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=k, n_init=3, max_iter=100, random_state=42)
    assignments = km.fit_predict(features)
    return km.cluster_centers_.astype(np.float32), assignments


def _run_kmeans(features: np.ndarray, k: int, backend: str = "auto") -> tuple:
    """backend: 'faiss' | 'sklearn' | 'auto' (faiss on Linux, sklearn on macOS)"""
    import platform
    use_faiss = (
        _FAISS
        and backend == "faiss"
        or (backend == "auto" and _FAISS and platform.system() != "Darwin")
    )
    if use_faiss:
        return _kmeans_faiss(features, k)
    return _kmeans_sklearn(features, k)


def _concentration(
    features: np.ndarray,
    centroids: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Per-prototype concentration φ (Eq. 12 in the paper)."""
    k = centroids.shape[0]
    dists = np.linalg.norm(features - centroids[assignments], axis=1)
    dist_sum = np.zeros(k, dtype=np.float64)
    np.add.at(dist_sum, assignments, dists)
    sizes = np.bincount(assignments, minlength=k).astype(np.float64)
    phi = np.ones(k, dtype=np.float64)
    nonzero = sizes > 0
    phi[nonzero] = dist_sum[nonzero] / (sizes[nonzero] * np.log(sizes[nonzero] + alpha))
    return phi.astype(np.float32)


def cluster_features(
    features: np.ndarray,
    k_list: list,
    alpha: float = 10.0,
    tau: float = 0.1,
    backend: str = "auto",
) -> list:
    """
    Run k-means for each k in k_list and compute concentration estimates.

    Returns a list of (centroids, assignments, phi) tensors per granularity level.
      centroids  : (k, D) float32
      assignments: (N,)   int64
      phi        : (k,)   float32  — normalized so mean(phi) == tau
    """
    results = []
    for k in k_list:
        print(f"  k-means k={k} ...", end=" ", flush=True)
        centroids, assignments = _run_kmeans(features, k, backend=backend)
        phi = _concentration(features, centroids, assignments, alpha)
        # Normalize so mean(phi) = tau (paper section 3.3)
        mean_phi = phi.mean()
        if mean_phi > 0:
            phi = phi / mean_phi * tau
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.maximum(norms, 1e-8)
        results.append((
            torch.tensor(centroids, dtype=torch.float32),
            torch.tensor(assignments, dtype=torch.long),
            torch.tensor(phi, dtype=torch.float32),
        ))
        print("done")
    return results
