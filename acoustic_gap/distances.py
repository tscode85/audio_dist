# file: acoustic_gap/distances.py
"""Distribution distances between real and simulated embedding sets.

Implements three complementary metrics, all *lower-is-better*:

* **FAD** (Fréchet Audio Distance) — Gaussian (Fréchet) distance between the two
  embedding distributions. Computed in closed form from means/covariances here,
  and (optionally) directly via ``fadtk`` for a directory-vs-directory reference
  number in :func:`fad_via_fadtk`.
* **MMD** (Maximum Mean Discrepancy) — kernel two-sample distance. Uses an RBF
  kernel with the median-heuristic bandwidth by default (a robust, standard,
  parameter-free choice). Returns MMD^2 (unbiased estimator).
* **Wasserstein** — optimal-transport distance. We report both a per-dimension
  averaged 1-D Wasserstein (``scipy.stats.wasserstein_distance``) and a
  multivariate estimate (exact EMD via POT, or a sliced-Wasserstein approximation
  that scales to large sample counts).

Tradeoff notes are inline at each function.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from scipy import linalg
from scipy.stats import wasserstein_distance

from .config import DistanceConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _subsample(x: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if max_n and x.shape[0] > max_n:
        idx = rng.choice(x.shape[0], size=max_n, replace=False)
        return x[idx]
    return x


# ---------------------------------------------------------------------------
# Fréchet distance (FAD closed-form)
# ---------------------------------------------------------------------------
def frechet_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet distance between Gaussians fit to ``x`` and ``y``.

    ``|mu_x - mu_y|^2 + tr(Cx + Cy - 2 sqrt(Cx Cy))``. This is the FAD formula and
    matches how fadtk / frechet_audio_distance score embeddings.
    """
    mu1, mu2 = x.mean(axis=0), y.mean(axis=0)
    c1 = np.cov(x, rowvar=False)
    c2 = np.cov(y, rowvar=False)
    c1 = np.atleast_2d(c1)
    c2 = np.atleast_2d(c2)
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(c1 @ c2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(c1.shape[0]) * eps
        covmean = linalg.sqrtm((c1 + offset) @ (c2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = float(diff @ diff + np.trace(c1) + np.trace(c2) - 2.0 * np.trace(covmean))
    return max(fd, 0.0)


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------
def _median_bandwidth(z: np.ndarray, rng: np.random.Generator, cap: int = 1000) -> float:
    z = _subsample(z, cap, rng)
    # Pairwise squared distances (median heuristic).
    sq = np.sum(z ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (z @ z.T)
    d2 = d2[np.triu_indices_from(d2, k=1)]
    d2 = d2[d2 > 0]
    if d2.size == 0:
        return 1.0
    med = np.median(d2)
    return float(np.sqrt(med / 2.0)) or 1.0


def _rbf_kernel(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    sa = np.sum(a ** 2, axis=1)[:, None]
    sb = np.sum(b ** 2, axis=1)[None, :]
    d2 = sa + sb - 2.0 * (a @ b.T)
    return np.exp(-np.maximum(d2, 0.0) / (2.0 * sigma ** 2))


def mmd_rbf(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth="median",
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Unbiased squared MMD with an RBF kernel.

    ``bandwidth`` may be ``"median"`` (median heuristic on the pooled sample) or a
    float sigma. Returns MMD^2, clamped at 0.
    """
    rng = rng or np.random.default_rng(0)
    if bandwidth == "median":
        sigma = _median_bandwidth(np.vstack([x, y]), rng)
    else:
        sigma = float(bandwidth)
    kxx = _rbf_kernel(x, x, sigma)
    kyy = _rbf_kernel(y, y, sigma)
    kxy = _rbf_kernel(x, y, sigma)
    m, n = x.shape[0], y.shape[0]
    # Unbiased: exclude diagonal self-similarities.
    sum_xx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1)) if m > 1 else 0.0
    sum_yy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) if n > 1 else 0.0
    sum_xy = kxy.mean()
    return max(float(sum_xx + sum_yy - 2.0 * sum_xy), 0.0)


# ---------------------------------------------------------------------------
# Wasserstein
# ---------------------------------------------------------------------------
def per_dimension_wasserstein(x: np.ndarray, y: np.ndarray) -> float:
    """Mean of per-dimension 1-D Wasserstein-1 distances.

    Cheap, robust marginal-level gap. Ignores cross-dimension coupling (that is
    what the multivariate estimate below captures).
    """
    dims = x.shape[1]
    vals = [wasserstein_distance(x[:, d], y[:, d]) for d in range(dims)]
    return float(np.mean(vals))


def sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    n_projections: int = 128,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Sliced-Wasserstein-1: average 1-D W1 over random projections.

    Scales linearly in sample count (unlike exact EMD's O(n^3)); the standard
    choice for large embedding sets. Approximation quality improves with
    ``n_projections``.
    """
    rng = rng or np.random.default_rng(0)
    dim = x.shape[1]
    projs = rng.standard_normal((dim, n_projections))
    projs /= np.linalg.norm(projs, axis=0, keepdims=True) + 1e-12
    xp = x @ projs
    yp = y @ projs
    vals = [wasserstein_distance(xp[:, k], yp[:, k]) for k in range(n_projections)]
    return float(np.mean(vals))


def wasserstein_pot(
    x: np.ndarray,
    y: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    max_n: int = 2000,
) -> float:
    """Exact multivariate Wasserstein-2 via POT's EMD on empirical samples.

    O(n^3) in the sample count, so inputs are subsampled to ``max_n``. Uses
    uniform marginals and squared-Euclidean cost, returning sqrt(EMD) = W2.
    """
    try:
        import ot  # POT
    except Exception as exc:  # pragma: no cover
        raise ImportError("POT (pip install pot) is required for the 'pot' backend.") from exc
    rng = rng or np.random.default_rng(0)
    x = _subsample(x, max_n, rng)
    y = _subsample(y, max_n, rng)
    a = np.full(x.shape[0], 1.0 / x.shape[0])
    b = np.full(y.shape[0], 1.0 / y.shape[0])
    M = ot.dist(x, y, metric="sqeuclidean")
    cost = ot.emd2(a, b, M)
    return float(np.sqrt(max(cost, 0.0)))


# ---------------------------------------------------------------------------
# FAD via fadtk (directory reference)
# ---------------------------------------------------------------------------
def fad_via_fadtk(
    real_dir: str, sim_dir: str, model_name: str = "vggish"
) -> float:
    """Directory-vs-directory FAD computed directly by ``fadtk`` (offline).

    This is the "official" FAD number; the closed-form :func:`frechet_distance`
    on our own embeddings is used elsewhere for parity across backbones/metrics.
    """
    try:
        from fadtk import FrechetAudioDistance  # type: ignore
        from fadtk.model_loader import get_all_models  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("fadtk is required for fad_via_fadtk.") from exc
    models = {m.name: m for m in get_all_models()}
    model = models[model_name]
    fad = FrechetAudioDistance(model, audio_load_worker=1, load_model=True)
    return float(fad.score(real_dir, sim_dir))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def compute_distances(
    real_emb: np.ndarray,
    sim_emb: np.ndarray,
    cfg: DistanceConfig,
) -> Dict[str, float]:
    """Compute all configured metrics between two embedding sets.

    Returns a dict with keys among ``fad``, ``mmd``, ``wasserstein_1d`` and
    ``wasserstein_mv``. Empty/degenerate inputs yield ``nan`` for the affected
    metric with a warning rather than raising.
    """
    rng = np.random.default_rng(cfg.random_seed)
    out: Dict[str, float] = {}
    if real_emb.shape[0] < 2 or sim_emb.shape[0] < 2:
        logger.warning("Too few samples (real=%d sim=%d) for stable distances",
                       real_emb.shape[0], sim_emb.shape[0])

    r = _subsample(real_emb, cfg.max_samples, rng)
    s = _subsample(sim_emb, cfg.max_samples, rng)

    if "fad" in cfg.metrics:
        try:
            out["fad"] = frechet_distance(real_emb, sim_emb)
        except Exception as exc:  # pragma: no cover
            logger.warning("FAD failed: %s", exc)
            out["fad"] = float("nan")

    if "mmd" in cfg.metrics:
        try:
            out["mmd"] = mmd_rbf(r, s, bandwidth=cfg.mmd_bandwidth, rng=rng)
        except Exception as exc:  # pragma: no cover
            logger.warning("MMD failed: %s", exc)
            out["mmd"] = float("nan")

    if "wasserstein" in cfg.metrics:
        try:
            out["wasserstein_1d"] = per_dimension_wasserstein(r, s)
        except Exception as exc:  # pragma: no cover
            logger.warning("1D Wasserstein failed: %s", exc)
            out["wasserstein_1d"] = float("nan")
        try:
            if cfg.wasserstein_backend == "pot":
                out["wasserstein_mv"] = wasserstein_pot(r, s, rng=rng, max_n=cfg.max_samples)
            else:
                out["wasserstein_mv"] = sliced_wasserstein(
                    r, s, n_projections=cfg.sliced_n_projections, rng=rng
                )
        except Exception as exc:  # pragma: no cover
            logger.warning("Multivariate Wasserstein failed: %s", exc)
            out["wasserstein_mv"] = float("nan")
        # Convenience alias used by the reporter's headline metric.
        out["wasserstein"] = out.get("wasserstein_mv", float("nan"))

    return out
