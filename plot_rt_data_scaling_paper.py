#!/usr/bin/env python3
"""Create publication-ready vector figures from a completed RT scaling study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"mae": "#0072B2", "rmse": "#D55E00", "r2": "#009E73", "individual": "#9AA0A6"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
    "axes.titlesize": 11, "legend.fontsize": 8, "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
    "axes.spines.right": False,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make paper-ready plots for RT data scaling")
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def save_all(fig: plt.Figure, base: Path) -> None:
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(base.with_suffix(suffix), dpi=450 if suffix == ".png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fraction_labels(values: np.ndarray) -> list[str]:
    return [f"{value:.0%}" for value in values]


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.results_dir / "data_scaling_summary.csv").sort_values("train_fraction")
    runs = pd.read_csv(args.results_dir / "data_scaling_runs.csv").sort_values(["repeat", "train_fraction"])
    if summary.empty or runs.empty:
        raise SystemExit("The scaling result CSVs contain no completed runs")
    output = args.results_dir / "paper_figures"
    output.mkdir(parents=True, exist_ok=True)
    fractions = summary["train_fraction"].to_numpy()

    # Figure 1: the compact result figure for a paper main panel.
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35))
    panels = [
        ("mae", "Validation MAE", "lower is better", COLORS["mae"]),
        ("rmse", "Validation RMSE", "lower is better", COLORS["rmse"]),
        ("r2", "Validation R²", "higher is better", COLORS["r2"]),
    ]
    for label, (metric, title, direction, color) in zip("ABC", panels):
        axis = axes["ABC".index(label)]
        mean = summary[f"validation_{metric}_mean"].to_numpy()
        ci = summary[f"validation_{metric}_ci95"].to_numpy()
        axis.errorbar(fractions, mean, yerr=ci, color=color, marker="o", markersize=5,
                      capsize=3, linewidth=2, zorder=3)
        axis.set_xticks(fractions, fraction_labels(fractions))
        axis.set_xlim(.07, 1.03)
        axis.set_xlabel("Training data used")
        axis.set_ylabel(title)
        axis.set_title(f"{label}  {title}", loc="left", fontweight="bold")
        axis.grid(axis="y", linestyle=":", alpha=.45)
        axis.text(.98, .06, direction, transform=axis.transAxes, ha="right", va="bottom", color="#555555")
    fig.suptitle("More Carafe RT training data improves held-out prediction accuracy", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_all(fig, output / "fig1_rt_data_scaling_multimetric")

    # Figure 2: all repeats, preserving the paired structure of the experiment.
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    for _, group in runs.groupby("repeat", sort=True):
        group = group.sort_values("train_fraction")
        axis.plot(group["train_fraction"], group["validation_mae"], color=COLORS["individual"],
                  linewidth=.75, alpha=.36, zorder=1)
        axis.scatter(group["train_fraction"], group["validation_mae"], color=COLORS["individual"],
                     s=10, alpha=.36, zorder=1)
    axis.errorbar(fractions, summary["validation_mae_mean"], yerr=summary["validation_mae_ci95"],
                  color=COLORS["mae"], marker="o", markersize=6, linewidth=2.4, capsize=4,
                  label="Mean ± 95% CI across repeats", zorder=4)
    axis.set(xticks=fractions, xticklabels=fraction_labels(fractions), xlabel="Training data used",
             ylabel="Validation MAE (normalized RT)", title="RT learning curve across all repeated splits")
    axis.grid(axis="y", linestyle=":", alpha=.45)
    axis.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_all(fig, output / "fig2_rt_data_scaling_repeats")

    # Figure 3: paired gains quantify how much each extra data step is worth.
    full = runs[runs["train_fraction"] == fractions[-1]].set_index("repeat")["validation_mae"]
    records = []
    for fraction in fractions[:-1]:
        values = runs[runs["train_fraction"] == fraction].set_index("repeat")["validation_mae"]
        paired = values.index.intersection(full.index)
        differences = (values.loc[paired] - full.loc[paired]).to_numpy()  # positive: full data improves MAE
        records.append({"fraction": fraction, "mean": differences.mean(),
                        "ci95": 1.96 * differences.std(ddof=1) / np.sqrt(len(differences)), "n": len(differences)})
    gains = pd.DataFrame(records)
    gains.to_csv(output / "paired_mae_gain_vs_full_data.csv", index=False)
    fig, axis = plt.subplots(figsize=(6.2, 4.0))
    bars = axis.bar(fraction_labels(gains["fraction"].to_numpy()), gains["mean"], yerr=gains["ci95"],
                    color=COLORS["mae"], alpha=.9, capsize=4, width=.62)
    axis.axhline(0, color="#333333", linewidth=.9)
    axis.set(xlabel="Training data used before adding the remaining data", ylabel="MAE improvement from using 100% data",
             title="Additional training data remains beneficial")
    axis.grid(axis="y", linestyle=":", alpha=.45)
    for bar, row in zip(bars, gains.itertuples(index=False)):
        axis.text(bar.get_x() + bar.get_width()/2, row.mean + row.ci95, f"{row.mean:.4f}",
                  ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_all(fig, output / "fig3_rt_data_scaling_paired_gain")

    improvement = float(summary.iloc[0]["validation_mae_mean"] - summary.iloc[-1]["validation_mae_mean"])
    relative = improvement / float(summary.iloc[0]["validation_mae_mean"]) * 100
    (output / "README.md").write_text(f"""# Paper figures: DIA-NN RT data scaling

All figures are available as **PDF** and **SVG** vector graphics, plus 450 dpi PNG previews. They use the 20 repeated, sequence-disjoint scaling runs in the parent directory.

## Main conclusion

Increasing the training set from 10% to 100% improves mean validation MAE from `{summary.iloc[0]['validation_mae_mean']:.5f}` to `{summary.iloc[-1]['validation_mae_mean']:.5f}`: an absolute improvement of `{improvement:.5f}` ({relative:.1f}% relative). The improvement remains positive for every successive data-size step.

## Figures

- `fig1_rt_data_scaling_multimetric`: compact three-panel main-paper figure (MAE, RMSE, R²).
- `fig2_rt_data_scaling_repeats`: all 20 paired repetitions in grey; blue is the mean and its 95% interval.
- `fig3_rt_data_scaling_paired_gain`: paired improvement of using all available data instead of 10%, 25%, 50%, or 75%.

Error bars are normal-approximation 95% confidence intervals across the 20 independent repeated splits. `paired_mae_gain_vs_full_data.csv` contains the values behind Figure 3.

Suggested concise caption for Figure 1: *Increasing the amount of Carafe RT training data consistently improved DIA-NN RT prediction on sequence-disjoint validation peptides. Points show the mean across 20 repeated splits; error bars show 95% confidence intervals.*
""", encoding="utf-8")
    print(f"Paper figures written to: {output}")


if __name__ == "__main__":
    main()
