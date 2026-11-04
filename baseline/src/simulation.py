"""Run the study: generate -> apply both methods -> save replicate-level rows."""

import csv
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd

from .common import (ROOT, arguments, cache_valid, design_rows, file_hash, load_config,
                     simulation_signature, versions, write_frame, write_json)
from .dgps import family_rng, generate_family
from .methods import estimate_fdr, reject_bh, reject_fixed
from .metrics import evaluate_rejections

LOG = logging.getLogger(__name__)


def run_simulation(config, output, force=False):
    output = Path(output)
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    signature = simulation_signature(config)
    if not force and cache_valid(output, signature):
        LOG.info("Verified cached results: %s", raw / "replicates.csv")
        return raw / "replicates.csv"
    start = time.perf_counter()
    # Invalidate the old completion marker before replacing any file.
    write_json(raw / "metadata.json", {"status": "running", "signature": signature})
    design = pd.DataFrame(design_rows(config))
    write_frame(raw / "design_matrix.csv", design)
    columns = ["scenario_id", "experiment", "family_id", "replicate_id", "m", "pi0",
               "mu", "gamma", "lambda", "label_mode", "method", "m0", "m1", "R",
               "V", "S", "FDP", "power", "W", "pi0_hat_raw", "pi0_hat_used",
               "fdr_hat_raw", "fdr_hat", "alpha_rep", "bh_threshold"]
    fd, temporary = tempfile.mkstemp(dir=raw, suffix=".csv.tmp")
    diagnostic = []
    rows_written = 0
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for mu in sorted(config["mu_values"]):
                for pi0 in sorted(config["pi0_values"]):
                    conditions = design[(design.mu == mu) & (design.pi0 == pi0)]
                    for replicate in range(config["replicates"]):
                        rng = family_rng(config["master_seed"], config["m"], pi0, mu, replicate)
                        p, is_null = generate_family(config["m"], pi0, mu, rng, config["label_mode"])
                        order = np.argsort(p, kind="stable")
                        family = f"p{pi0:g}_mu{mu:g}_r{replicate}"
                        # Preselected replicate 0 is the diagnostic; no extra randomness.
                        if replicate == 0 and pi0 == min(config["pi0_values"]):
                            diagnostic.extend({"mu": mu, "pi0": pi0, "index": i,
                                               "is_null": bool(null), "pvalue": value}
                                              for i, (value, null) in enumerate(zip(p, is_null)))
                        for condition in conditions.to_dict("records"):
                            gamma = condition["gamma"]
                            estimates = estimate_fdr(p, gamma, config["lambda"], config["cap_pi0"])
                            alpha = estimates["fdr_hat"]
                            fixed = reject_fixed(p, gamma)
                            bh, threshold = reject_bh(p, alpha, order)
                            for method, rejected in [("fixed", fixed), ("bh", bh)]:
                                row = {k: v for k, v in condition.items() if k != "replicates"}
                                row.update(family_id=family, replicate_id=replicate, method=method,
                                           alpha_rep=alpha, bh_threshold=threshold)
                                row.update(estimates)
                                row.update(evaluate_rejections(rejected, is_null))
                                writer.writerow(row)
                                rows_written += 1
                    LOG.info("Completed mu=%g, pi0=%g (%d families)", mu, pi0,
                             config["replicates"])
        os.replace(temporary, raw / "replicates.csv")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    write_frame(raw / "diagnostic.csv", pd.DataFrame(diagnostic))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        commit = None
    expected = len(design) * config["replicates"] * 2
    if rows_written != expected:
        raise RuntimeError(f"Wrote {rows_written} rows, expected {expected}")
    metadata = {"status": "complete", "signature": signature, "config": config,
                "versions": versions(), "platform": platform.platform(), "git_commit": commit,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "simulation_seconds": time.perf_counter() - start,
                "conditions": len(design), "rows": rows_written,
                "files": {name: file_hash(raw / name) for name in
                          ["replicates.csv", "design_matrix.csv", "diagnostic.csv"]}}
    write_json(raw / "metadata.json", metadata)
    LOG.info("Saved %d rows in %.2f seconds", rows_written, metadata["simulation_seconds"])
    return raw / "replicates.csv"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = arguments(__doc__)
    run_simulation(load_config(args.config), args.output, args.force)
