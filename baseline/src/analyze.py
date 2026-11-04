"""Validate raw rows, then summarize their performance and paper differences."""

import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .common import (ROOT, arguments, cache_valid, design_rows, file_hash, load_config,
                     simulation_signature, write_frame, write_json)
from .metrics import analytic_targets, mean_mcse

LOG = logging.getLogger(__name__)
KEYS = ["scenario_id", "method"]


def validate_raw(raw, config):
    design = pd.DataFrame(design_rows(config)).set_index("scenario_id")
    expected_rows = len(design) * config["replicates"] * 2
    if len(raw) != expected_rows or raw.duplicated(["scenario_id", "replicate_id", "method"]).any():
        raise ValueError("Incorrect row count or duplicate replicate keys")
    if set(raw.scenario_id) != set(design.index) or set(raw.method) != {"fixed", "bh"}:
        raise ValueError("Missing or unexpected scenarios/methods")
    if not np.all(np.isfinite(raw.select_dtypes(include="number").drop(columns="power"))):
        raise ValueError("Unexpected non-finite raw values")
    for (scenario, _), group in raw.groupby(KEYS):
        if set(group.replicate_id) != set(range(config["replicates"])):
            raise ValueError("Missing replicates")
        expected = design.loc[scenario]
        for column in ["m", "pi0", "mu", "gamma", "lambda"]:
            if not np.allclose(group[column], expected[column], rtol=0, atol=1e-12):
                raise ValueError(f"Configuration mismatch in {column}")
        if not group.label_mode.eq(config["label_mode"]).all():
            raise ValueError("Label mode mismatch")
        if not group.experiment.eq(expected.experiment).all():
            raise ValueError("Experiment label mismatch")
    for column in ["R", "V", "S", "m0", "m1", "W"]:
        values = raw[column].to_numpy()
        if np.any(values < 0) or np.any(values > config["m"]) or np.any(values != np.floor(values)):
            raise ValueError(f"Invalid count: {column}")
    if not (raw.R.eq(raw.V + raw.S).all() and raw.m0.add(raw.m1).eq(config["m"]).all()
            and raw.V.le(raw.m0).all() and raw.S.le(raw.m1).all()):
        raise ValueError("Count identities failed")
    if not np.allclose(raw.FDP, raw.V / raw.R.clip(lower=1), atol=1e-12):
        raise ValueError("FDP calculation failed")
    positive = raw.m1 > 0
    if not np.allclose(raw.loc[positive, "power"],
                       raw.loc[positive, "S"] / raw.loc[positive, "m1"], atol=1e-12):
        raise ValueError("Power calculation failed")
    if raw.loc[~positive, "power"].notna().any():
        raise ValueError("Power must be missing when m1=0")
    if not np.allclose(raw.pi0_hat_raw, raw.W / (raw.m * (1 - raw["lambda"]))):
        raise ValueError("Null proportion estimate mismatch")
    expected_pi = raw.pi0_hat_raw.clip(upper=1) if config["cap_pi0"] else raw.pi0_hat_raw
    if not np.allclose(raw.pi0_hat_used, expected_pi):
        raise ValueError("Null proportion cap mismatch")
    fixed = raw[raw.method == "fixed"].set_index(["scenario_id", "replicate_id"])
    bh = raw[raw.method == "bh"].set_index(["scenario_id", "replicate_id"])
    fields = ["m0", "m1", "W", "pi0_hat_raw", "fdr_hat_raw", "fdr_hat", "alpha_rep"]
    if not fixed[fields].equals(bh[fields]):
        raise ValueError("Methods are not paired on shared families")
    expected_raw = fixed.pi0_hat_used * fixed.m * fixed.gamma / fixed.R.clip(lower=1)
    if not np.allclose(fixed.fdr_hat_raw, expected_raw):
        raise ValueError("Fixed-threshold FDR estimate mismatch")
    if not np.allclose(raw.fdr_hat, raw.fdr_hat_raw.clip(upper=1)):
        raise ValueError("FDR cap mismatch")
    if not np.allclose(raw.alpha_rep, raw.fdr_hat):
        raise ValueError("BH comparison level mismatch")
    if config["label_mode"] == "fixed":
        if not np.allclose(raw.m0, config["m"] * raw.pi0):
            raise ValueError("Incorrect fixed null counts")
    if not ((raw.FDP.between(0, 1)).all() and raw.alpha_rep.between(0, 1).all()
            and raw.bh_threshold.between(0, 1).all()):
        raise ValueError("Probability outside [0,1]")
    return {"rows": len(raw), "conditions": len(design), "status": "passed"}


def summarize(raw, config):
    summary_rows, paired_rows = [], []
    for condition in design_rows(config):
        scenario = raw[raw.scenario_id == condition["scenario_id"]]
        targets = analytic_targets(config["m"], condition["pi0"], condition["mu"],
                                   condition["gamma"], config["label_mode"], config["lambda"])
        for method, group in scenario.groupby("method"):
            row = dict(condition, method=method)
            row.update(targets)
            for source, label in [("FDP", "fdr"), ("power", "power"),
                                  ("pi0_hat_raw", "pi0_hat"), ("fdr_hat", "fdr_hat"),
                                  ("fdr_hat_raw", "fdr_hat_raw"),
                                  ("bh_threshold", "bh_threshold")]:
                mean, se, count = mean_mcse(group[source])
                row[label], row[label + "_mcse"], row[label + "_n"] = mean, se, count
            row["rejection_probability"], row["rejection_probability_mcse"], _ = mean_mcse(group.R > 0)
            row["pfdr"], row["pfdr_mcse"], row["pfdr_n"] = mean_mcse(group.loc[group.R > 0, "FDP"])
            pi_errors = group.pi0_hat_raw.to_numpy() - condition["pi0"]
            row["pi0_bias"] = float(pi_errors.mean())
            row["pi0_rmse"] = float(np.sqrt(np.mean(pi_errors**2)))
            for source, prefix in [("fdr_hat_raw", "fdr_est_raw"), ("fdr_hat", "fdr_est")]:
                errors = group[source].to_numpy() - targets["true_fdr"]
                row[prefix + "_bias"] = float(errors.mean()) if method == "fixed" else np.nan
                row[prefix + "_rmse"] = float(np.sqrt(np.mean(errors**2))) if method == "fixed" else np.nan
            summary_rows.append(row)
        paired = scenario.pivot(index="replicate_id", columns="method", values="power")
        mean, se, count = mean_mcse(paired["fixed"] - paired["bh"])
        paired_rows.append(dict(condition, power_difference=mean, mcse=se, n=count,
                                ci_low=mean - 1.96 * se, ci_high=mean + 1.96 * se))
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def compare_paper(summary, config):
    reference = pd.read_csv(ROOT / "references/table2.csv")
    rows = []
    if config["m"] != 1000 or config["lambda"] != 0.5:
        return pd.DataFrame(columns=["pi0", "gamma", "metric", "paper", "simulation", "mcse", "difference"])
    mappings = {"paper_fdr": ("fixed", "fdr"),
                "paper_power_fixed": ("fixed", "power"),
                "paper_power_bh": ("bh", "power"),
                "paper_fdr_hat": ("fixed", "fdr_hat"),
                "paper_pi0_hat": ("fixed", "pi0_hat"),
                "paper_bh_threshold": ("bh", "bh_threshold")}
    for ref in reference.to_dict("records"):
        baseline = summary[(summary.mu == 2) & np.isclose(summary.pi0, ref["pi0"])
                           & np.isclose(summary.gamma, ref["gamma"])]
        if baseline.empty:
            continue
        for metric, (method, column) in mappings.items():
            current = baseline[baseline.method == method].iloc[0]
            rows.append({"pi0": ref["pi0"], "gamma": ref["gamma"], "metric": metric,
                         "paper": ref[metric], "simulation": current[column],
                         "mcse": current[column + "_mcse"],
                         "difference": current[column] - ref[metric]})
    return pd.DataFrame(rows, columns=["pi0", "gamma", "metric", "paper", "simulation", "mcse", "difference"])


def analyze(config, output):
    start = time.perf_counter()
    output = Path(output)
    if not cache_valid(output, simulation_signature(config)):
        raise ValueError("Raw results are missing, stale, or changed. Run make simulate first.")
    raw = pd.read_csv(output / "raw/replicates.csv")
    validation = validate_raw(raw, config)
    summary, paired = summarize(raw, config)
    directory = output / "summaries"
    write_frame(directory / "summary.csv", summary)
    write_frame(directory / "paired_power.csv", paired)
    write_frame(directory / "comparison_table2.csv", compare_paper(summary, config))
    write_json(directory / "validation.json", validation)
    write_json(directory / "metadata.json", {
        "raw_sha256": file_hash(output / "raw/replicates.csv"),
        "summary_sha256": file_hash(directory / "summary.csv"),
        "analysis_seconds": time.perf_counter() - start,
        "note": "MCSEs quantify our simulation uncertainty; original MCSEs are unavailable."})
    LOG.info("Validated %d rows; saved %d method-condition summaries", len(raw), len(summary))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = arguments(__doc__)
    analyze(load_config(args.config), args.output)
