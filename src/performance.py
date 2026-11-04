"""Reproducible profiling, runtime benchmarks, complexity and parallel scaling.

Run with `python -m src.performance {profile,benchmark,complexity,parallel}`.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run_engine(engine, config, output, workers=1, profile=None):
    module = "baseline.src.simulation" if engine == "baseline" else "src.simulation"
    command = [sys.executable]
    if profile is not None:
        command += ["-m", "cProfile", "-o", str(profile)]
    command += ["-m", module, "--config", str(config), "--output", str(output), "--force"]
    if engine == "optimized":
        command += ["--workers", str(workers)]
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    start = time.perf_counter()
    subprocess.run(command, cwd=ROOT, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - start


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profile(out, config):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        for engine in ("baseline", "optimized"):
            filename = out / f"{engine}.prof"
            seconds = run_engine(engine, config, Path(temp) / engine, profile=filename)
            import pstats
            with (out / f"{engine}_profile.txt").open("w") as f:
                pstats.Stats(str(filename), stream=f).strip_dirs().sort_stats("cumtime").print_stats(40)
            rows.append({"engine": engine, "elapsed_seconds": seconds,
                         "profile": str(filename)})
    save_csv(out / "profile_times.csv", rows)
    print(rows)


def benchmark(out, config, repeats, workers_list):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    specs = [("baseline", 1), ("optimized", 1)] + [("optimized", w) for w in workers_list if w != 1]
    with tempfile.TemporaryDirectory() as temp:
        for engine, workers in specs:
            samples = []
            for i in range(repeats):
                t = run_engine(engine, config, Path(temp) / f"{engine}_{workers}", workers)
                samples.append(t)
                rows.append({"engine": engine, "workers": workers, "repeat": i + 1,
                             "seconds": t, "config": str(config)})
            print(engine, workers, "median", round(float(np.median(samples)), 4), "s")
    save_csv(out / "benchmark.csv", rows)
    medians = [(e, w, np.median([r["seconds"] for r in rows if r["engine"] == e and r["workers"] == w]))
               for e, w in specs]
    baseline = next(t for e, w, t in medians if e == "baseline")
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{e}\n{w} worker(s)" for e, w, t in medians]
    ax.bar(labels, [t for e, w, t in medians])
    ax.set_ylabel("Median wall-clock time (seconds)")
    ax.set_title("Unit 2 baseline vs optimized simulation")
    for i, (_, _, t) in enumerate(medians):
        ax.text(i, t, f"{baseline / t:.2f}x", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out / "benchmark.png", dpi=180)
    plt.close(fig)
    serial = next(t for e, w, t in medians if e == "optimized" and w == 1)
    parallel_points = [(w, serial / t) for e, w, t in medians if e == "optimized"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([w for w, s in parallel_points], [s for w, s in parallel_points], "o-", label="Observed")
    ax.plot([w for w, s in parallel_points], [w for w, s in parallel_points], "--", label="Ideal")
    ax.set_xlabel("Number of worker processes")
    ax.set_ylabel("Speedup relative to optimized serial")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "parallel_speedup.png", dpi=180)
    plt.close(fig)


def complexity(out, repeats, m_values, n_replicates):
    out.mkdir(parents=True, exist_ok=True)
    base_conf = json.loads((ROOT / "configs" / "pilot.json").read_text())
    # Use fixed-count labels valid for all tested m; one family design block.
    base_conf.update(pi0_values=[0.5], mu_values=[2.0], replicates=n_replicates)
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        for m in m_values:
            if m % 2 != 0:
                raise ValueError("Select even m for pi0=0.5 fixed counts")
            conf = dict(base_conf, m=m)
            cfg = temp / f"m{m}.json"
            cfg.write_text(json.dumps(conf))
            for engine in ("baseline", "optimized"):
                for i in range(repeats):
                    elapsed = run_engine(engine, cfg, temp / f"{engine}_m{m}")
                    rows.append({"engine": engine, "m": m, "repeat": i + 1,
                                 "seconds": elapsed, "seconds_per_family": elapsed / n_replicates})
                print(m, engine, "done")
    save_csv(out / "complexity.csv", rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    slopes = {}
    for engine in ("baseline", "optimized"):
        x = np.array(m_values, dtype=float)
        y = np.array([np.median([r["seconds_per_family"] for r in rows
                                 if r["engine"] == engine and r["m"] == m]) for m in m_values])
        slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
        slopes[engine] = float(slope)
        ax.loglog(x, y, "o", label=f"{engine} median")
        ax.loglog(x, np.exp(intercept) * x ** slope, "--", label=f"{engine} fit: exponent {slope:.2f}")
    ax.set_xlabel("Number of tests per family (m)")
    ax.set_ylabel("Wall time per family (seconds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "complexity.png", dpi=180)
    plt.close(fig)
    (out / "complexity_slopes.json").write_text(json.dumps(slopes, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["profile", "benchmark", "complexity", "parallel"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/performance")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--m-values", type=int, nargs="+", default=[250, 500, 1000, 2000, 4000, 8000])
    parser.add_argument("--replicates", type=int, default=300)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "environment.json").write_text(json.dumps({"python": sys.version,
        "platform": platform.platform(), "cpu_count": os.cpu_count()}, indent=2) + "\n")
    if args.mode == "profile":
        profile(args.out, args.config.resolve())
    elif args.mode == "complexity":
        complexity(args.out, args.repeats, args.m_values, args.replicates)
    elif args.mode == "parallel":
        benchmark(args.out, args.config.resolve(), args.repeats, args.workers)
    else:
        benchmark(args.out, args.config.resolve(), args.repeats, [1])

if __name__ == "__main__":
    main()
