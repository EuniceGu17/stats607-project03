"""Run optimized Storey simulation, optionally across independent (mu, pi0) blocks."""
import argparse
import csv
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .common import (ROOT, cache_valid, design_rows, file_hash, load_config,
                     simulation_signature, versions, write_frame, write_json)
from .dgps import family_rng, generate_family
from .fast_kernel import evaluate_family_fast

LOG = logging.getLogger(__name__)
COLUMNS = ["scenario_id", "experiment", "family_id", "replicate_id", "m", "pi0",
           "mu", "gamma", "lambda", "label_mode", "method", "m0", "m1", "R",
           "V", "S", "FDP", "power", "W", "pi0_hat_raw", "pi0_hat_used",
           "fdr_hat_raw", "fdr_hat", "alpha_rep", "bh_threshold"]


def _simulate_block(task):
    """Return one complete block; no shared RNG state or shared writer."""
    config, mu, pi0, conditions = task
    result_rows = []
    diagnostic = []
    gammas = [c["gamma"] for c in conditions]
    for replicate in range(config["replicates"]):
        rng = family_rng(config["master_seed"], config["m"], pi0, mu, replicate)
        p, is_null = generate_family(config["m"], pi0, mu, rng, config["label_mode"])
        family = f"p{pi0:g}_mu{mu:g}_r{replicate}"
        if replicate == 0 and pi0 == min(config["pi0_values"]):
            diagnostic.extend({"mu": mu, "pi0": pi0, "index": i,
                               "is_null": bool(null), "pvalue": value}
                              for i, (value, null) in enumerate(zip(p, is_null)))
        evaluations = evaluate_family_fast(p, is_null, gammas, config["lambda"],
                                           config["cap_pi0"])
        for condition, (estimates, fixed_metrics, bh_metrics, threshold) in zip(conditions, evaluations):
            for method, method_metrics in (("fixed", fixed_metrics), ("bh", bh_metrics)):
                row = {k: v for k, v in condition.items() if k != "replicates"}
                row.update(family_id=family, replicate_id=replicate, method=method,
                           alpha_rep=estimates["fdr_hat"], bh_threshold=threshold)
                row.update(estimates)
                row.update(method_metrics)
                result_rows.append(row)
    return result_rows, diagnostic


def run_simulation(config, output, force=False, workers=1):
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    output = Path(output)
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    signature = simulation_signature(config)
    if not force and cache_valid(output, signature):
        LOG.info("Verified cached results: %s", raw / "replicates.csv")
        return raw / "replicates.csv"
    start = time.perf_counter()
    write_json(raw / "metadata.json", {"status": "running", "signature": signature})
    design = pd.DataFrame(design_rows(config))
    write_frame(raw / "design_matrix.csv", design)

    # Preserve Unit 2 ordering: mu, pi0, replicate, gamma, method.
    tasks = []
    for mu in sorted(config["mu_values"]):
        for pi0 in sorted(config["pi0_values"]):
            conditions = design[(design.mu == mu) & (design.pi0 == pi0)].to_dict("records")
            tasks.append((config, mu, pi0, conditions))

    fd, temporary = tempfile.mkstemp(dir=raw, suffix=".csv.tmp")
    diagnostic = []
    rows_written = 0
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            if workers == 1:
                block_results = map(_simulate_block, tasks)
                pool = None
            else:
                pool = ProcessPoolExecutor(max_workers=workers)
                block_results = pool.map(_simulate_block, tasks)
            try:
                for (rows, diag), (_, mu, pi0, _) in zip(block_results, tasks):
                    writer.writerows(rows)
                    rows_written += len(rows)
                    diagnostic.extend(diag)
                    LOG.info("Completed mu=%g, pi0=%g (%d families)",
                             mu, pi0, config["replicates"])
            finally:
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
        expected = len(design) * config["replicates"] * 2
        if rows_written != expected:
            raise RuntimeError(f"Wrote {rows_written} rows, expected {expected}")
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
    metadata = {"status": "complete", "signature": signature, "config": config,
                "versions": versions(), "platform": platform.platform(),
                "git_commit": commit, "workers": workers,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/full.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    conf_path = args.config.resolve()
    destination = (args.output or ROOT / "results" / conf_path.stem).resolve()
    run_simulation(load_config(conf_path), destination, args.force, args.workers)
