"""Diagnostic, baseline reproduction, extension, and FDR calibration plots."""

import json
import logging
from pathlib import Path
import time

from .common import ROOT, arguments, cache_valid, file_hash, load_config, simulation_signature, write_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
COLORS = {"fixed": "#0072B2", "bh": "#D55E00"}
LABELS = {"fixed": "Fixed p-value threshold", "bh": "BH at estimated FDR level"}


def save(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def note(config):
    return (f"m={config['m']}; K={config['replicates']} per condition; "
            f"lambda={config['lambda']}; {config['label_mode']} labels. "
            "Error bars: approximate 95% Monte Carlo intervals.")


def power_grid(summary, config, mus, directory, stem, title):
    gammas = sorted(config["gamma_values"])
    selected = summary[summary.mu.isin(mus)]
    ymax = min(1.025, max(.05, 1.18 * float((selected.power + 1.96 * selected.power_mcse).max())))
    fig, axes = plt.subplots(len(mus), len(gammas), figsize=(5 * len(gammas), 3.1 * len(mus)),
                             squeeze=False, sharex=True, sharey=True, layout="constrained")
    for i, mu in enumerate(mus):
        for j, gamma in enumerate(gammas):
            ax = axes[i, j]
            subset = summary[np.isclose(summary.mu, mu) & np.isclose(summary.gamma, gamma)]
            for method in ["fixed", "bh"]:
                group = subset[subset.method == method].sort_values("pi0")
                ax.errorbar(group.pi0, group.power, yerr=1.96 * group.power_mcse,
                            color=COLORS[method], marker="o" if method == "fixed" else "s",
                            linestyle="-" if method == "fixed" else "--", capsize=2,
                            label=LABELS[method])
            fixed = subset[subset.method == "fixed"].sort_values("pi0")
            ax.plot(fixed.pi0, fixed.true_power, color="#333333", linestyle=":", label="Analytic fixed-rule power")
            if stem == "02_baseline_power" and config["m"] == 1000 and config["lambda"] == .5:
                reference = pd.read_csv(ROOT / "references/table2.csv")
                reference = reference[np.isclose(reference.gamma, gamma)
                                      & reference.pi0.isin(config["pi0_values"])]
                ax.scatter(reference.pi0, reference.paper_power_bh, color="#333333",
                           marker="x", s=38, zorder=5, label="Published BH power (Table 2)")
            ax.set(title=f"mu={mu:g}, gamma={gamma:g}", ylim=(-.01, ymax))
            ax.grid(alpha=0.2)
            if i == len(mus) - 1:
                ax.set_xlabel("True null proportion (pi0)")
            if j == 0:
                ax.set_ylabel("Mean power")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(title + "\n" + note(config), fontsize=11)
    save(fig, directory, stem)


def make_figures(config, output):
    start = time.perf_counter()
    output = Path(output)
    if not cache_valid(output, simulation_signature(config)):
        raise ValueError("Raw results do not match this configuration; run make all")
    summary_file = output / "summaries/summary.csv"
    metadata = json.loads((output / "summaries/metadata.json").read_text())
    if (metadata["raw_sha256"] != file_hash(output / "raw/replicates.csv")
            or metadata["summary_sha256"] != file_hash(summary_file)):
        raise ValueError("Analysis is stale; run make analyze")
    summary = pd.read_csv(summary_file)
    diagnostic = pd.read_csv(output / "raw/diagnostic.csv")
    directory = output / "figures"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    mus = sorted(config["mu_values"])
    fig, axes = plt.subplots(1, len(mus), figsize=(4.3 * len(mus), 3.4), squeeze=False, layout="constrained")
    for ax, mu in zip(axes[0], mus):
        subset = diagnostic[np.isclose(diagnostic.mu, mu)]
        for is_null, color, label in [(True, "#0072B2", "True null"), (False, "#D55E00", "Alternative")]:
            values = subset.loc[subset.is_null == is_null, "pvalue"]
            if len(values):
                ax.hist(values, bins=np.linspace(0, 1, 21), density=True, histtype="step",
                        linewidth=1.8, color=color, label=label)
        ax.axhline(1, color="#333333", linestyle=":", label="Uniform null density")
        ax.set(title=f"mu={mu:g}", xlabel="One-sided p-value", ylabel="Density", xlim=(0, 1))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("DGP diagnostic: preselected replicate 0\n"
                 f"m={config['m']}; pi0={min(config['pi0_values']):g}; "
                 f"{config['label_mode']} labels; densities conditional on null/alternative", fontsize=11)
    save(fig, directory, "01_dgp_diagnostic")
    if 2 in mus:
        power_grid(summary, config, [2], directory, "02_baseline_power", "Reproduction of the Section 5 power comparison")
    power_grid(summary, config, mus, directory, "03_signal_extension", "Power across signal strengths")
    gammas = sorted(config["gamma_values"])
    fig, axes = plt.subplots(len(mus), len(gammas), figsize=(5 * len(gammas), 3.1 * len(mus)),
                             squeeze=False, sharex=True, layout="constrained")
    for i, mu in enumerate(mus):
        for j, gamma in enumerate(gammas):
            ax = axes[i, j]
            subset = summary[np.isclose(summary.mu, mu) & np.isclose(summary.gamma, gamma)]
            for method in ["fixed", "bh"]:
                group = subset[subset.method == method].sort_values("pi0")
                ax.errorbar(group.pi0, group.fdr, yerr=1.96 * group.fdr_mcse,
                            color=COLORS[method], marker="o", capsize=2, label=LABELS[method] + ": achieved FDR")
            fixed = subset[subset.method == "fixed"].sort_values("pi0")
            ax.plot(fixed.pi0, fixed.true_fdr, ":", color="#333333", label="Fixed-rule analytic FDR")
            ax.errorbar(fixed.pi0, fixed.fdr_hat, yerr=1.96 * fixed.fdr_hat_mcse,
                        color="#009E73", linestyle="--", marker="^", capsize=2, label="Mean estimated fixed-rule FDR")
            upper = max(float((subset.fdr + 1.96 * subset.fdr_mcse).max()),
                        float((fixed.fdr_hat + 1.96 * fixed.fdr_hat_mcse).max()),
                        float(fixed.true_fdr.max()))
            ax.set(title=f"mu={mu:g}, gamma={gamma:g}",
                   ylim=(-0.002, min(1.02, max(.02, 1.18 * upper))))
            ax.grid(alpha=0.2)
            if i == len(mus) - 1:
                ax.set_xlabel("True null proportion (pi0)")
            if j == 0:
                ax.set_ylabel("False discovery rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Achieved FDR and estimation calibration\n" + note(config), fontsize=11)
    save(fig, directory, "04_fdr_calibration")
    write_json(directory / "metadata.json", {"summary_sha256": file_hash(summary_file),
                                             "figures_seconds": time.perf_counter() - start})
    LOG.info("Saved figures to %s", directory)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = arguments(__doc__)
    make_figures(load_config(args.config), args.output)
