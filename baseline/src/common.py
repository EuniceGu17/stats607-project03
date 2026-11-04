"""Configuration and small, atomic file operations shared by the CLI stages."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {"m", "pi0_values", "mu_values", "gamma_values", "lambda",
                "replicates", "master_seed", "label_mode", "cap_pi0", "cap_fdr"}
    if set(config) != required:
        raise ValueError(f"Configuration keys must be exactly {sorted(required)}")
    for key in ["m", "replicates", "master_seed"]:
        if type(config[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if config["m"] < 1 or config["replicates"] < 2 or config["master_seed"] < 0:
        raise ValueError("Require m>=1, replicates>=2, master_seed>=0")
    if config["label_mode"] not in {"fixed", "bernoulli"}:
        raise ValueError("label_mode must be fixed or bernoulli")
    if any(type(config[k]) is not bool for k in ["cap_pi0", "cap_fdr"]):
        raise ValueError("cap_pi0 and cap_fdr must be booleans")
    import math
    for key in ["pi0_values", "mu_values", "gamma_values"]:
        values = config[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{key} must be a nonempty list without duplicates")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError(f"{key} must contain finite numbers")
    if any(not 0 <= p <= 1 for p in config["pi0_values"]):
        raise ValueError("pi0 must lie in [0,1]")
    if any(not 0 < g < 1 for g in config["gamma_values"]):
        raise ValueError("gamma must lie in (0,1)")
    if type(config["lambda"]) not in (int, float) or not 0 < config["lambda"] < 1:
        raise ValueError("lambda must lie in (0,1)")
    if config["label_mode"] == "fixed":
        if any(not math.isclose(config["m"] * p, round(config["m"] * p), abs_tol=1e-8)
               for p in config["pi0_values"]):
            raise ValueError("Fixed labels require integer m*pi0")
    if not config["cap_fdr"]:
        raise ValueError("The BH comparison requires cap_fdr=true to keep alpha in [0,1]")
    return config


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/full.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true", help="Rebuild this stage")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.output = (args.output or ROOT / "results" / args.config.stem).resolve()
    return args


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions():
    import platform
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name)
        for name in ["numpy", "scipy", "pandas", "matplotlib", "pytest"]}}


def simulation_signature(config):
    files = ["common.py", "dgps.py", "methods.py", "metrics.py", "simulation.py"]
    source = {name: file_hash(ROOT / "src" / name) for name in files}
    payload = {"config": config, "source": source, "versions": versions()}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_frame(path, frame):
    atomic_text(path, frame.to_csv(index=False, float_format="%.12g"))


def design_rows(config):
    rows = []
    for mu in sorted(config["mu_values"]):
        for pi0 in sorted(config["pi0_values"]):
            for gamma in sorted(config["gamma_values"]):
                rows.append({
                    "scenario_id": f"p{pi0:g}_mu{mu:g}_g{gamma:g}",
                    "experiment": "baseline" if mu == 2 else "extension",
                    "m": config["m"], "pi0": pi0, "mu": mu, "gamma": gamma,
                    "lambda": config["lambda"], "replicates": config["replicates"],
                    "label_mode": config["label_mode"],
                })
    return rows


def cache_valid(output, signature):
    raw = Path(output) / "raw"
    metadata_file = raw / "metadata.json"
    if not metadata_file.exists():
        return False
    try:
        metadata = json.loads(metadata_file.read_text())
        return (metadata["status"] == "complete" and metadata["signature"] == signature
                and all(file_hash(raw / name) == expected
                        for name, expected in metadata["files"].items()))
    except (OSError, KeyError, ValueError):
        return False
