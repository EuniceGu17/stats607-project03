"""Audit regular and extreme simulation settings for numerical warnings/errors."""
import argparse
import csv
from pathlib import Path
import warnings
import numpy as np
from .dgps import family_rng, generate_family
from .fast_kernel import evaluate_family_fast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/performance"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in (100, 1000, 10000):
        for mu in (0., 2., 5., 8.):
            for pi0 in (0.1, 0.5, 0.9):
                for replicate in range(10):
                    case = {"m": m, "mu": mu, "pi0": pi0, "replicate": replicate,
                            "warnings": 0, "nonfinite": 0, "exception": ""}
                    try:
                        rng = family_rng(20261006, m, pi0, mu, replicate)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            p, truth = generate_family(m, pi0, mu, rng)
                            results = evaluate_family_fast(p, truth, [1e-8, 1e-5, .001, .01], .5)
                        case["warnings"] = len(caught)
                        case["nonfinite"] = int(not np.isfinite(p).all())
                        for estimates, fixed, bh, threshold in results:
                            for name in ("W", "pi0_hat_raw", "pi0_hat_used", "fdr_hat_raw", "fdr_hat"):
                                case["nonfinite"] += int(not np.isfinite(estimates[name]))
                            for result in (fixed, bh):
                                for name in ("FDP", "power"):
                                    case["nonfinite"] += int(not np.isfinite(result[name]))
                    except Exception as exc:
                        case["exception"] = repr(exc)
                    rows.append(case)
    with (args.out / "stability.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Cases={len(rows)}, warnings={sum(r['warnings'] for r in rows)}, "
          f"nonfinite={sum(r['nonfinite'] for r in rows)}, "
          f"exceptions={sum(bool(r['exception']) for r in rows)}")
    if any(r["warnings"] or r["nonfinite"] or r["exception"] for r in rows):
        raise SystemExit("Review stability.csv: at least one check did not pass")

if __name__ == "__main__":
    main()
