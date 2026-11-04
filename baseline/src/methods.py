"""Methods see p-values only; truth labels are reserved for evaluation."""

import numpy as np


def validate_pvalues(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not len(p) or not np.all(np.isfinite(p)):
        raise ValueError("p-values must be a nonempty finite one-dimensional array")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0,1]")
    return p


def reject_fixed(pvalues, gamma):
    p = validate_pvalues(pvalues)
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0,1]")
    return p <= gamma


def estimate_fdr(pvalues, gamma, lam=0.5, cap_pi0=False):
    p = validate_pvalues(pvalues)
    if not 0 < lam < 1 or not 0 < gamma < 1:
        raise ValueError("Require lambda and gamma in (0,1)")
    m = len(p)
    w = int(np.count_nonzero(p > lam))
    r = int(np.count_nonzero(p <= gamma))
    pi0_raw = w / (m * (1 - lam))
    pi0_used = min(1.0, pi0_raw) if cap_pi0 else pi0_raw
    raw = pi0_used * m * gamma / max(r, 1)
    return {"W": w, "pi0_hat_raw": pi0_raw, "pi0_hat_used": pi0_used,
            "fdr_hat_raw": raw, "fdr_hat": min(1.0, raw)}


def reject_bh(pvalues, alpha, order=None):
    p = validate_pvalues(pvalues)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("BH alpha must lie in [0,1]")
    # The driver reuses the sort for both gamma values in a family.
    order = np.argsort(p, kind="stable") if order is None else np.asarray(order)
    sorted_p = p[order]
    eligible = np.flatnonzero(sorted_p <= alpha * np.arange(1, len(p) + 1) / len(p))
    rejected = np.zeros(len(p), dtype=bool)
    if not len(eligible):
        return rejected, 0.0
    k = int(eligible[-1]) + 1
    rejected[order[:k]] = True
    return rejected, float(sorted_p[k - 1])
