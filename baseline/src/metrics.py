"""Replicate metrics and independent, DGP-specific analytic benchmarks."""

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def evaluate_rejections(rejected, is_null):
    rejected = np.asarray(rejected, dtype=bool)
    is_null = np.asarray(is_null, dtype=bool)
    if rejected.ndim != 1 or rejected.shape != is_null.shape:
        raise ValueError("Rejection and truth vectors must have matching 1D shapes")
    m0 = int(is_null.sum())
    m1 = len(is_null) - m0
    r = int(rejected.sum())
    v = int(np.count_nonzero(rejected & is_null))
    s = r - v
    return {"m0": m0, "m1": m1, "R": r, "V": v, "S": s,
            "FDP": v / max(r, 1), "power": s / m1 if m1 else np.nan}


def alternative_power(gamma, mu):
    return float(norm.sf(norm.isf(gamma) - mu))


def analytic_targets(m, pi0, mu, gamma, label_mode="fixed", lam=0.5):
    """Fixed-count FDR is not replaced by the iid-mixture pFDR formula."""
    g = alternative_power(gamma, mu)
    if label_mode == "fixed":
        m0 = round(m * pi0)
        m1 = m - m0
        rejection_prob = float(-np.expm1(m0 * np.log1p(-gamma) + m1 * np.log1p(-g)))
        if m0 == 0:
            fdr, error = 0.0, 0.0
        else:
            def integrand(t):
                return np.exp((m0 - 1) * np.log1p(-gamma * (1 - t))
                              + m1 * np.log1p(-g * (1 - t)))
            integral, error = quad(integrand, 0, 1, epsabs=1e-12, epsrel=1e-11)
            fdr = m0 * gamma * integral
            error *= m0 * gamma
        power = g if m1 else np.nan
    elif label_mode == "bernoulli":
        q = pi0 * gamma + (1 - pi0) * g
        rejection_prob = float(-np.expm1(m * np.log1p(-q)))
        fdr = pi0 * gamma / q * rejection_prob if q > 0 else 0.0
        error = 0.0
        power = g if pi0 < 1 else np.nan
    else:
        raise ValueError("Unknown label mode")
    expected_pi0_raw = pi0 + (1 - pi0) * norm.cdf(norm.isf(lam) - mu) / (1 - lam)
    return {"true_fdr": float(fdr), "true_power": power,
            "true_rejection_probability": rejection_prob,
            "true_pfdr": fdr / rejection_prob if rejection_prob else np.nan,
            "expected_pi0_hat_raw": float(expected_pi0_raw),
            "quadrature_error": float(error)}


def mean_mcse(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, 0
    return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(len(x)))
                            if len(x) > 1 else np.nan), len(x)
