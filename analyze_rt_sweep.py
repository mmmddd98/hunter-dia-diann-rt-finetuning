#!/usr/bin/env python3
"""Create a reproducible local analysis report for a DIA-NN RT sweep."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HEAVY_NTERM = "Dimethyl:2H(4)13C(2)@N-term"
HEAVY_LYSINE = "Dimethyl:2H(4)13C(2)@K"
PLOT_DPI = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze completed DIA-NN RT sweep runs")
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: <sweep-dir>/analysis",
    )
    parser.add_argument("--baseline-dir", type=Path, help="Optional prior reference run")
    parser.add_argument("--reported-runs", type=int, help="Optional W&B trial count for integrity notes")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_metrics(frame: pd.DataFrame) -> Tuple[Dict[str, float], np.ndarray]:
    target = frame["rt_norm"].to_numpy(dtype=float)
    prediction = frame["rt_pred_finetuned"].to_numpy(dtype=float)
    absolute_error = np.abs(prediction - target)
    mods = frame["mods"].fillna("").astype(str)
    heavy_nterm = mods.str.contains(HEAVY_NTERM, regex=False).to_numpy()
    heavy_lysine = mods.str.contains(HEAVY_LYSINE, regex=False).to_numpy()
    masks = {
        "heavy": heavy_nterm | heavy_lysine,
        "not_heavy": ~(heavy_nterm | heavy_lysine),
        "heavy_nterm_only": heavy_nterm & ~heavy_lysine,
        "heavy_lysine_only": heavy_lysine & ~heavy_nterm,
        "heavy_both": heavy_nterm & heavy_lysine,
    }
    result: Dict[str, float] = {}
    for name, mask in masks.items():
        result[f"{name}_rows"] = int(mask.sum())
        result[f"{name}_mae"] = float(absolute_error[mask].mean()) if mask.any() else float("nan")
    return result, absolute_error


def load_runs(sweep_dir: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
    records = []
    errors: Dict[str, np.ndarray] = {}
    validation_hashes: Dict[str, int] = {}
    missing = {"metadata": [], "history": [], "predictions": [], "legacy_subgroups": []}
    run_dirs = sorted(path for path in sweep_dir.glob("run.*") if path.is_dir())
    for run_dir in run_dirs:
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.exists():
            missing["metadata"].append(run_dir.name)
            continue
        history_path = run_dir / "training_history.csv"
        predictions_path = run_dir / "validation_predictions.tsv"
        validation_path = run_dir / "validation_split.tsv"
        if not history_path.exists():
            missing["history"].append(run_dir.name)
        if not predictions_path.exists():
            missing["predictions"].append(run_dir.name)
            continue
        if not (run_dir / "subgroup_metrics.csv").exists():
            missing["legacy_subgroups"].append(run_dir.name)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result = metadata["result"]
        parameters = metadata["parameters"]
        metrics = result["finetuned_metrics"]
        predictions = pd.read_csv(predictions_path, sep="\t")
        subgroup, absolute_error = _prediction_metrics(predictions)
        errors[run_dir.name] = absolute_error
        if validation_path.exists():
            digest = _sha256(validation_path)
            validation_hashes[digest] = validation_hashes.get(digest, 0) + 1
        record = {
            "run": run_dir.name,
            "run_dir": str(run_dir.resolve()),
            **parameters,
            "strategy": result.get("strategy", parameters.get("trainable")),
            "best_epoch": result["best_epoch"],
            "elapsed_seconds": result["elapsed_seconds"],
            "trainable_parameters": result.get("trainable_parameters"),
            **metrics,
            **subgroup,
        }
        records.append(record)
    frame = pd.DataFrame(records)
    integrity = {
        "sweep_dir": str(sweep_dir.resolve()),
        "discovered_run_directories": len(run_dirs),
        "complete_analyzed_runs": len(frame),
        "unique_validation_split_hashes": len(validation_hashes),
        "validation_split_hash_counts": validation_hashes,
        "missing": missing,
    }
    return frame, errors, integrity


def grouped_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        frame.groupby(column, dropna=False)
        .agg(
            runs=("mae", "size"),
            median_mae=("mae", "median"),
            q25_mae=("mae", lambda values: values.quantile(0.25)),
            best_mae=("mae", "min"),
            median_rmse=("rmse", "median"),
            median_heavy_mae=("heavy_mae", "median"),
            median_heavy_both_mae=("heavy_both_mae", "median"),
        )
        .reset_index()
        .sort_values("median_mae")
    )


def pareto_front(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("mae")
    records = []
    best_double = float("inf")
    for _, row in ordered.iterrows():
        if row["heavy_both_mae"] < best_double:
            records.append(row)
            best_double = row["heavy_both_mae"]
    return pd.DataFrame(records)


def paired_bootstrap(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    mask: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> Dict[str, float]:
    delta = errors_a[mask] - errors_b[mask]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means[start : start + count] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "rows": len(delta),
        "mean_delta_mae": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_a_better": float((means < 0).mean()),
    }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_strategy_distribution(frame: pd.DataFrame, path: Path) -> None:
    order = frame.groupby("strategy")["mae"].median().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    views = [(axes[0], order, "All policies"), (axes[1], order[:5], "Zoom: competitive policies")]
    for axis, selected, title in views:
        values = [frame.loc[frame.strategy == strategy, "mae"] for strategy in selected]
        axis.boxplot(
            values,
            labels=[f"{name}\n(n={len(value)})" for name, value in zip(selected, values)],
            showfliers=True,
        )
        axis.set(title=title, ylabel="Validation MAE", xlabel="Layer policy")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    axes[1].set_ylim(0.0215, 0.045)
    fig.suptitle("Validation MAE by trainable model blocks")
    _save_figure(fig, path)


def plot_loss_within_all(frame: pd.DataFrame, path: Path) -> None:
    subset = frame[frame.strategy == "all"]
    order = subset.groupby("loss")["mae"].median().sort_values().index.tolist()
    values = [subset.loc[subset.loss == loss, "mae"] for loss in order]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.boxplot(values, labels=[f"{name}\n(n={len(value)})" for name, value in zip(order, values)], showfliers=True)
    axis.set(title="Loss comparison within full fine-tuning", ylabel="Validation MAE", xlabel="Loss")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def plot_learning_rate(frame: pd.DataFrame, path: Path) -> None:
    subset = frame[frame.strategy == "all"]
    colors = {"mae": "tab:blue", "huber": "tab:orange", "mse": "tab:green"}
    fig, axis = plt.subplots(figsize=(8, 5))
    for loss, group in subset.groupby("loss"):
        axis.scatter(group.learning_rate, group.mae, s=45, alpha=0.8, label=loss, color=colors.get(loss))
    axis.set_xscale("log")
    axis.set(title="Main learning rate within full fine-tuning", xlabel="Learning rate (log scale)", ylabel="Validation MAE")
    axis.grid(alpha=0.25)
    axis.legend(title="Loss")
    _save_figure(fig, path)


def plot_hyperparameters(frame: pd.DataFrame, path: Path) -> None:
    subset = frame[frame.strategy == "all"]
    columns = [
        ("batch_size", "Batch size"),
        ("warmup_epochs", "Warmup epochs"),
        ("weight_decay", "Weight decay"),
        ("heavy_nterm_weight", "Heavy N-term weight"),
        ("heavy_lysine_weight", "Heavy lysine weight"),
        ("heavy_both_weight", "Heavy both-sites weight"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for axis, (column, title) in zip(axes.flat, columns):
        summary = subset.groupby(column).mae.median().sort_index()
        labels = [f"{value:g}" if isinstance(value, (int, float)) else str(value) for value in summary.index]
        axis.bar(labels, summary.values, color="tab:blue", alpha=0.8)
        axis.set(title=title, ylabel="Median validation MAE")
        axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def plot_pareto(frame: pd.DataFrame, candidates: pd.DataFrame, path: Path) -> None:
    colors = {
        "all": "tab:blue", "discriminative": "tab:orange", "gru_head": "tab:green",
        "cnn_head": "tab:red", "rt_head": "tab:purple", "head": "tab:brown",
        "embeddings": "tab:gray",
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    offsets = [(8, 18), (8, -18), (8, 2)]
    for axis in axes:
        for strategy, group in frame.groupby("strategy"):
            axis.scatter(
                group.mae, group.heavy_both_mae, s=22, alpha=0.5,
                label=strategy, color=colors.get(strategy),
            )
        for offset, (_, row) in zip(offsets, candidates.iterrows()):
            axis.scatter(
                row.mae, row.heavy_both_mae, s=140, marker="*",
                edgecolor="black", linewidth=0.8, zorder=5,
            )
            axis.annotate(
                row.candidate, (row.mae, row.heavy_both_mae), xytext=offset,
                textcoords="offset points", fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
        axis.set(xlabel="Global validation MAE", ylabel="True double-heavy MAE (28 rows)")
        axis.grid(alpha=0.25)
    axes[0].set_title("All runs")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_title("Zoom: competitive region")
    axes[1].set_xlim(0.0215, 0.04)
    axes[1].set_ylim(0.0445, 0.056)
    fig.suptitle("Trade-off: global MAE vs true Heavy N-term + Heavy K MAE")
    _save_figure(fig, path)


def plot_candidate_metrics(candidates: pd.DataFrame, path: Path) -> None:
    metrics = ["mae", "heavy_mae", "heavy_nterm_only_mae", "heavy_lysine_only_mae", "heavy_both_mae"]
    labels = ["Global", "All heavy", "Heavy N-term", "Heavy K", "Both heavy"]
    x = np.arange(len(metrics))
    width = 0.8 / len(candidates)
    fig, axis = plt.subplots(figsize=(11, 5.5))
    for index, (_, row) in enumerate(candidates.iterrows()):
        axis.bar(x - 0.4 + width / 2 + index * width, [row[m] for m in metrics], width, label=row.candidate)
    axis.set_xticks(x, labels)
    axis.set(title="Selected candidate models across validation subgroups", ylabel="MAE")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    _save_figure(fig, path)


def plot_candidate_curves(sweep_dir: Path, candidates: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for _, row in candidates.iterrows():
        history = pd.read_csv(sweep_dir / row.run / "training_history.csv")
        axis.plot(history.epoch, history.validation_mae, label=row.candidate)
    axis.set(title="Validation MAE across epochs for selected candidates", xlabel="Epoch", ylabel="Validation MAE")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    _save_figure(fig, path)


def _markdown_table(frame: pd.DataFrame, columns: Iterable[str], digits: int = 6) -> str:
    shown = frame[list(columns)].copy()
    for column in shown.select_dtypes(include=["float"]).columns:
        shown[column] = shown[column].map(lambda value: f"{value:.{digits}f}")
    return shown.to_markdown(index=False)


def build_readme(
    output_dir: Path,
    frame: pd.DataFrame,
    integrity: Dict[str, object],
    strategy_summary: pd.DataFrame,
    loss_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    bootstrap: pd.DataFrame,
    baseline_name: str | None,
    reported_runs: int | None,
) -> None:
    best_mae = candidates[candidates.candidate == "Best global MAE"].iloc[0]
    best_huber = candidates[candidates.candidate == "Best Huber / recommended"].iloc[0]
    best_rmse = candidates[candidates.candidate == "Best RMSE/R²"].iloc[0]
    delta = best_mae.mae - best_huber.mae
    strategy_table = _markdown_table(
        strategy_summary,
        ["strategy", "runs", "median_mae", "best_mae", "median_heavy_mae", "median_heavy_both_mae"],
    )
    loss_table = _markdown_table(
        loss_summary,
        ["loss", "runs", "median_mae", "best_mae", "median_rmse", "median_heavy_mae", "median_heavy_both_mae"],
    )
    candidate_table = _markdown_table(
        candidates,
        [
            "candidate", "run", "strategy", "loss", "mae", "rmse", "r2", "heavy_mae",
            "heavy_nterm_only_mae", "heavy_lysine_only_mae", "heavy_both_mae",
            "learning_rate", "batch_size", "warmup_epochs", "weight_decay", "best_epoch",
        ],
    )
    bootstrap_text = _markdown_table(
        bootstrap,
        ["comparison", "subset", "rows", "mean_delta_mae", "ci95_low", "ci95_high", "probability_a_better"],
    ) if not bootstrap.empty else "Kein Baseline-Vergleich verfügbar."
    text = f"""# DIA-NN RT sweep summary

Completed local runs: {len(frame)}. Distinct validation split hashes:
{integrity['unique_validation_split_hashes']}.

## Layer policies

{strategy_table}

## Losses within full fine-tuning

{loss_table}

## Selected configurations

{candidate_table}

## Paired bootstrap comparisons

{bootstrap_text}

These are validation comparisons. Model selection and repeated use of the
same partition do not establish performance on an independent final test set.
See the generated CSV tables and plots for the per-run metrics and provenance.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def build_html(output_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    images = [
        ("Layer policies", "01_strategy_mae_distribution.png"),
        ("Loss within full fine-tuning", "02_loss_within_full_finetuning.png"),
        ("Learning rate", "03_learning_rate_within_full_finetuning.png"),
        ("Other hyperparameters", "04_hyperparameters_within_full_finetuning.png"),
        ("Global vs true double-heavy", "05_global_vs_true_double_heavy.png"),
        ("Candidate subgroup metrics", "06_candidate_subgroup_metrics.png"),
        ("Candidate validation curves", "07_candidate_validation_curves.png"),
    ]
    body = "".join(
        f"<h2>{html.escape(title)}</h2><img src='plots/{filename}' style='max-width:100%'>"
        for title, filename in images
    )
    table_html = "".join(
        f"<h2>{html.escape(title)}</h2>{frame.to_html(index=False, float_format=lambda value: f'{value:.6g}')}"
        for title, frame in tables.items()
    )
    report = f"""<!doctype html><html><head><meta charset="utf-8">
<title>DIA-NN RT sweep analysis</title><style>
body{{font-family:sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;font-size:.86rem}}th,td{{border:1px solid #ccc;padding:.3rem}}
th{{background:#eee;position:sticky;top:0}}code{{background:#eee;padding:.15rem}}
</style></head><body><h1>DIA-NN RT sweep analysis</h1>
<p>Die Tabellen und Hinweise zur Validation stehen in <code>README.md</code>.</p>
{table_html}{body}</body></html>"""
    (output_dir / "report.html").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    sweep_dir = args.sweep_dir.resolve()
    output_dir = (args.output_dir or sweep_dir / "analysis").resolve()
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    frame, errors, integrity = load_runs(sweep_dir)
    if frame.empty:
        raise SystemExit(f"No complete sweep runs found below {sweep_dir}")
    frame = frame.sort_values("mae").reset_index(drop=True)
    frame.insert(0, "global_mae_rank", np.arange(1, len(frame) + 1))
    frame.to_csv(output_dir / "run_summary.csv", index=False)
    (output_dir / "data_integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    strategy_summary = grouped_summary(frame, "strategy")
    strategy_summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    full = frame[frame.strategy == "all"].copy()
    loss_summary = grouped_summary(full, "loss")
    loss_summary.to_csv(output_dir / "loss_summary_full_finetuning.csv", index=False)
    hyper_records = []
    for parameter in (
        "batch_size", "warmup_epochs", "weight_decay", "heavy_nterm_weight",
        "heavy_lysine_weight", "heavy_both_weight",
    ):
        summary = grouped_summary(full, parameter).rename(columns={parameter: "value"})
        summary.insert(0, "parameter", parameter)
        hyper_records.append(summary)
    hyper_summary = pd.concat(hyper_records, ignore_index=True)
    hyper_summary.to_csv(output_dir / "hyperparameter_summary_full_finetuning.csv", index=False)

    top_global = frame.head(30).copy()
    top_global.to_csv(output_dir / "top_runs_global_mae.csv", index=False)
    top_double = frame.sort_values("heavy_both_mae").head(30).copy()
    top_double.to_csv(output_dir / "top_runs_true_double_heavy.csv", index=False)
    pareto = pareto_front(frame)
    pareto.to_csv(output_dir / "pareto_global_vs_double_heavy.csv", index=False)

    best_global = frame.iloc[0].copy()
    best_huber = frame[frame.loss == "huber"].iloc[0].copy()
    best_rmse = frame.sort_values("rmse").iloc[0].copy()
    candidate_rows = []
    seen = set()
    for label, row in (
        ("Best global MAE", best_global),
        ("Best Huber / recommended", best_huber),
        ("Best RMSE/R²", best_rmse),
    ):
        if row.run in seen:
            continue
        row = row.copy()
        row["candidate"] = label
        candidate_rows.append(row)
        seen.add(row.run)
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(output_dir / "selected_candidates.csv", index=False)

    prediction_reference = pd.read_csv(sweep_dir / best_global.run / "validation_predictions.tsv", sep="\t")
    mods = prediction_reference.mods.fillna("").astype(str)
    heavy_nterm = mods.str.contains(HEAVY_NTERM, regex=False).to_numpy()
    heavy_lysine = mods.str.contains(HEAVY_LYSINE, regex=False).to_numpy()
    masks = {
        "all": np.ones(len(prediction_reference), dtype=bool),
        "all_heavy": heavy_nterm | heavy_lysine,
        "true_double_heavy": heavy_nterm & heavy_lysine,
    }
    comparisons = []

    def add_comparison(name_a: str, run_a: str, name_b: str, run_b: str, errors_b: np.ndarray) -> None:
        for offset, (subset, mask) in enumerate(masks.items()):
            result = paired_bootstrap(
                errors[run_a], errors_b, mask,
                samples=args.bootstrap_samples, seed=20240727 + len(comparisons) + offset,
            )
            comparisons.append({"comparison": f"{name_a} - {name_b}", "subset": subset, **result})

    add_comparison("Best MAE", best_global.run, "Best Huber", best_huber.run, errors[best_huber.run])
    baseline_name = None
    if args.baseline_dir is not None:
        baseline_predictions = pd.read_csv(args.baseline_dir / "validation_predictions.tsv", sep="\t")
        _, baseline_errors = _prediction_metrics(baseline_predictions)
        baseline_name = args.baseline_dir.name
        add_comparison("Best MAE", best_global.run, baseline_name, baseline_name, baseline_errors)
        add_comparison("Best Huber", best_huber.run, baseline_name, baseline_name, baseline_errors)
    bootstrap = pd.DataFrame(comparisons)
    bootstrap.to_csv(output_dir / "paired_bootstrap.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_strategy_distribution(frame, plot_dir / "01_strategy_mae_distribution.png")
    plot_loss_within_all(frame, plot_dir / "02_loss_within_full_finetuning.png")
    plot_learning_rate(frame, plot_dir / "03_learning_rate_within_full_finetuning.png")
    plot_hyperparameters(frame, plot_dir / "04_hyperparameters_within_full_finetuning.png")
    plot_pareto(frame, candidates, plot_dir / "05_global_vs_true_double_heavy.png")
    plot_candidate_metrics(candidates, plot_dir / "06_candidate_subgroup_metrics.png")
    plot_candidate_curves(sweep_dir, candidates, plot_dir / "07_candidate_validation_curves.png")

    build_readme(
        output_dir, frame, integrity, strategy_summary, loss_summary, candidates,
        bootstrap, baseline_name, args.reported_runs,
    )
    build_html(
        output_dir,
        {
            "Selected candidates": candidates,
            "Strategy summary": strategy_summary,
            "Loss summary within full fine-tuning": loss_summary,
            "Paired bootstrap": bootstrap,
        },
    )
    print(f"Analyzed {len(frame)} runs")
    print(f"Recommended run: {best_huber.run}")
    print(f"Report: {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
