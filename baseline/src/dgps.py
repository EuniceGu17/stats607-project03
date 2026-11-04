"""Generate one family of independent one-sided normal tests."""

import struct
import numpy as np
from scipy.stats import norm


def family_rng(master_seed, m, pi0, mu, replicate):
    """Stable across loop order, method, gamma, and requested replicate count."""
    words = [int(master_seed), int(m), int(replicate)]
    for value in [pi0, mu]:
        words.extend(struct.unpack("<II", struct.pack("<d", float(value))))
    return np.random.default_rng(np.random.SeedSequence(words))


def generate_family(m, pi0, mu, rng, label_mode="fixed"):
    if m < 1 or not 0 <= pi0 <= 1 or not np.isfinite(mu):
        raise ValueError("Invalid DGP parameters")
    if label_mode == "fixed":
        if not np.isclose(m * pi0, round(m * pi0), rtol=0, atol=1e-8):
            raise ValueError("Fixed labels require integer m*pi0")
        is_null = np.arange(m) < round(m * pi0)
    elif label_mode == "bernoulli":
        is_null = rng.random(m) < pi0
    else:
        raise ValueError("Unknown label mode")
    z = rng.normal(size=m) + (~is_null) * mu
    # sf avoids the cancellation error in 1 - cdf(z) for large positive z.
    return norm.sf(z), is_null
