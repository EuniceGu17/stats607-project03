"""Optimized, deterministic evaluation of multiple gamma values for one family.

The DGP and random-number stream remain unchanged from Unit 2.
"""
import numpy as np


def evaluate_family_fast(p, is_null, gammas, lam, cap_pi0=False):
    """Return a list of (FDR estimates, fixed metrics, BH metrics, threshold).

    Results correspond to the order of ``gammas``. For the same p-values,
    these calculations reproduce the Unit 2 definitions exactly, up to
    floating-point arithmetic precision.
    """
    p = np.asarray(p, dtype=float)
    is_null = np.asarray(is_null, dtype=bool)
    gammas = np.asarray(gammas, dtype=float)
    if p.ndim != 1 or is_null.shape != p.shape or not p.size:
        raise ValueError("p and is_null must be matching nonempty 1D arrays")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values")
    if gammas.ndim != 1 or not len(gammas) or not np.isfinite(gammas).all() or np.any((gammas <= 0) | (gammas >= 1)):
        raise ValueError("gammas must be a nonempty 1D array in (0,1)")
    if not 0 < lam < 1:
        raise ValueError("lam must lie in (0,1)")

    m = len(p)
    m0 = int(is_null.sum())
    m1 = m - m0
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    null_cumsum = np.cumsum(is_null[order], dtype=np.int64)
    ranks = np.arange(1, m + 1)

    w = int(np.count_nonzero(p > lam))
    pi0_raw = w / (m * (1 - lam))
    pi0_used = min(1.0, pi0_raw) if cap_pi0 else pi0_raw

    fixed_k = np.searchsorted(sorted_p, gammas, side="right")
    fdr_raw = pi0_used * m * gammas / np.maximum(fixed_k, 1)
    fdr_hat = np.minimum(1.0, fdr_raw)

    # Identical mathematical criterion to Unit 2, across all thresholds.
    eligible = sorted_p[:, None] <= fdr_hat[None, :] * ranks[:, None] / m
    bh_k = np.max(np.where(eligible, ranks[:, None], 0), axis=0)

    def metrics(k):
        k = int(k)
        v = int(null_cumsum[k - 1]) if k else 0
        s = k - v
        return {"m0": m0, "m1": m1, "R": k, "V": v, "S": s,
                "FDP": v / max(k, 1), "power": s / m1 if m1 else np.nan}

    results = []
    for i in range(len(gammas)):
        estimates = {"W": w, "pi0_hat_raw": pi0_raw,
                     "pi0_hat_used": pi0_used, "fdr_hat_raw": float(fdr_raw[i]),
                     "fdr_hat": float(fdr_hat[i])}
        k_bh = int(bh_k[i])
        threshold = float(sorted_p[k_bh - 1]) if k_bh else 0.0
        results.append((estimates, metrics(fixed_k[i]), metrics(k_bh), threshold))
    return results
