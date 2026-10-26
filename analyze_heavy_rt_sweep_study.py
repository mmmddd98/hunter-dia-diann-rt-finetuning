#!/usr/bin/env python3
"""Evaluate the four targeted Heavy-Dimethyl RT sweeps reproducibly.

The script reads local prediction and configuration files, creates one analysis
folder inside every sweep, and builds a joint study report on the exact common
Carafe2/DIA-NN validation rows.  W&B is deliberately not required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HEAVY_NTERM = "Dimethyl:2H(4)13C(2)@N-term"
HEAVY_LYSINE = "Dimethyl:2H(4)13C(2)@K"
EXPECTED_SEEDS = tuple(range(2024, 2034))
RT_SCALE_MINUTES = 44.104345704269
PLOT_DPI = 300
COLORS = {
    "carafe2_oversampling": "#0072B2",
    "carafe2_loss_weighting": "#56B4E9",
    "diann_oversampling": "#D55E00",
    "diann_loss_weighting": "#E69F00",
}
SHORT_LABELS = {
    "carafe2_oversampling": "Carafe2\noversampling",
    "carafe2_loss_weighting": "Carafe2\nloss weighting",
    "diann_oversampling": "DIA-NN\noversampling",
    "diann_loss_weighting": "DIA-NN\nloss weighting",
}
METRIC_GROUPS = (
    "global",
    "normal",
    "heavy_dimethyl",
    "heavy_nterm_only",
    "heavy_lysine_only",
    "heavy_both",
)


@dataclass(frozen=True)
class SweepSpec:
    key: str
    title: str
    model: str
    intervention: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Carafe2 and DIA-NN Heavy-Dimethyl RT sweeps"
    )
    parser.add_argument("--carafe2-oversampling", type=Path, required=True)
    parser.add_argument("--carafe2-loss-weighting", type=Path, required=True)
    parser.add_argument("--diann-oversampling", type=Path, required=True)
    parser.add_argument("--diann-loss-weighting", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--rt-scale-minutes", type=float, default=RT_SCALE_MINUTES)
    return parser.parse_args()


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_loss(value: Any) -> str:
    text = str(value).lower()
    return "mae" if text in {"l1", "mae"} else text


def stable_validation_hash(frame: pd.DataFrame) -> str:
    columns = ["sequence", "mods", "mod_sites", "rt_norm", "nAA"]
    normalized = frame[columns].fillna("").astype(str).sort_values(columns)
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def add_row_key(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    columns = ["sequence", "mods", "mod_sites", "rt_norm", "nAA"]
    text = result[columns].fillna("").astype(str)
    result["_row_key"] = text.agg("\x1f".join, axis=1)
    if result["_row_key"].duplicated().any():
        duplicates = int(result["_row_key"].duplicated(keep=False).sum())
        raise ValueError(f"Validation set contains {duplicates} duplicate row keys")
    return result


def subgroup_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    mods = frame["mods"].fillna("").astype(str)
    nterm = mods.str.contains(HEAVY_NTERM, regex=False).to_numpy()
    lysine = mods.str.contains(HEAVY_LYSINE, regex=False).to_numpy()
    return {
        "global": np.ones(len(frame), dtype=bool),
        "normal": ~(nterm | lysine),
        "heavy_dimethyl": nterm | lysine,
        "heavy_nterm_only": nterm & ~lysine,
        "heavy_lysine_only": lysine & ~nterm,
        "heavy_both": nterm & lysine,
    }


def regression_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    target = frame["rt_norm"].to_numpy(dtype=float)
    prediction = frame[prediction_column].to_numpy(dtype=float)
    output: dict[str, float] = {}
    for group, mask in subgroup_masks(frame).items():
        y = target[mask]
        pred = prediction[mask]
        residual = pred - y
        output[f"{group}_n"] = int(mask.sum())
        output[f"{group}_mae"] = float(np.mean(np.abs(residual)))
        output[f"{group}_rmse"] = float(np.sqrt(np.mean(np.square(residual))))
        denominator = float(np.sum(np.square(y - y.mean())))
        output[f"{group}_r2"] = (
            float(1.0 - np.sum(np.square(residual)) / denominator)
            if denominator > 0
            else float("nan")
        )
    return output


def normalized_config(spec: SweepSpec, raw: dict[str, Any]) -> dict[str, Any]:
    prefix = "rt_" if spec.model == "carafe2" else ""

    def get(name: str, default: Any = None) -> Any:
        return raw.get(prefix + name.replace("-", "_"), raw.get(name, default))

    if spec.model == "carafe2":
        learning_rate = raw.get("rt_lr")
        batch_size = raw.get("rt_batch_size")
        epochs = raw.get("rt_epoch")
        warmup = raw.get("rt_warmup_epoch")
        loss = raw.get("rt_loss")
        huber_delta = raw.get("rt_huber_delta")
    else:
        learning_rate = raw.get("learning-rate")
        batch_size = raw.get("batch-size")
        epochs = raw.get("epochs")
        warmup = raw.get("warmup-epochs")
        loss = raw.get("loss")
        huber_delta = raw.get("huber-delta")

    suffix = "oversample" if spec.intervention == "oversampling" else "weight"
    if spec.model == "carafe2":
        nterm = raw.get(f"rt_heavy_nterm_{suffix}", 1.0)
        lysine = raw.get(f"rt_heavy_lysine_{suffix}", 1.0)
        both = raw.get(f"rt_heavy_both_{suffix}", 1.0)
    else:
        nterm = raw.get(f"heavy-nterm-{suffix}", 1.0)
        lysine = raw.get(f"heavy-lysine-{suffix}", 1.0)
        both = raw.get(f"heavy-both-{suffix}", 1.0)

    return {
        "loss": canonical_loss(loss),
        "huber_delta": float(huber_delta),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "warmup_epochs": int(warmup),
        "trainable": str(raw.get("trainable", "all")),
        "heavy_nterm_factor": float(nterm),
        "heavy_lysine_factor": float(lysine),
        "heavy_both_factor": float(both),
    }


def discover_seed_dirs(spec: SweepSpec) -> list[Path]:
    pattern = "trial.*/seed.*" if spec.model == "carafe2" else "runs/trial.*/seed.*"
    return sorted(path for path in spec.path.glob(pattern) if path.is_dir())


def load_predictions(spec: SweepSpec, seed_dir: Path) -> tuple[pd.DataFrame, str | None]:
    if spec.model == "carafe2":
        frame = pd.read_csv(seed_dir / "rt_test.tsv", sep="\t")
        frame = frame.rename(columns={"rt_pred": "rt_pred_finetuned"})
        pretrained_path = seed_dir / "rt_test_pretrained.tsv"
        pretrained_column = None
        if pretrained_path.exists():
            pretrained = pd.read_csv(pretrained_path, sep="\t").rename(
                columns={"rt_pred": "rt_pred_pretrained"}
            )
            keys = ["sequence", "mods", "mod_sites", "rt_norm", "nAA"]
            frame = frame.merge(
                pretrained[keys + ["rt_pred_pretrained"]], on=keys, how="left", validate="one_to_one"
            )
            pretrained_column = "rt_pred_pretrained"
        return add_row_key(frame), pretrained_column
    frame = pd.read_csv(seed_dir / "validation_predictions.tsv", sep="\t")
    return add_row_key(frame), "rt_pred_pretrained"


def best_epoch(spec: SweepSpec, seed_dir: Path, metadata: dict[str, Any] | None) -> int | None:
    if spec.model == "diann":
        return int(metadata["result"]["best_epoch"]) if metadata else None
    history_path = seed_dir / "rt_history.tsv"
    if not history_path.exists():
        return None
    history = pd.read_csv(history_path, sep="\t")
    return int(history.loc[history["test_loss"].idxmin(), "epoch"])


def load_sweep(spec: SweepSpec) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    predictions: dict[str, pd.DataFrame] = {}
    missing: list[dict[str, str]] = []
    hashes: dict[str, int] = {}
    seed_dirs = discover_seed_dirs(spec)
    for seed_dir in seed_dirs:
        command_path = seed_dir / "command.json"
        metric_path = (
            seed_dir / "model_evaluation_metrics.json"
            if spec.model == "carafe2"
            else seed_dir / "run_metadata.json"
        )
        prediction_path = (
            seed_dir / "rt_test.tsv"
            if spec.model == "carafe2"
            else seed_dir / "validation_predictions.tsv"
        )
        absent = [
            path.name
            for path in (command_path, metric_path, prediction_path)
            if not path.exists()
        ]
        if absent:
            missing.append({"seed_dir": str(seed_dir), "missing": ",".join(absent)})
            continue
        command = json_load(command_path)
        metadata = json_load(metric_path) if spec.model == "diann" else None
        frame, pretrained_column = load_predictions(spec, seed_dir)
        digest = stable_validation_hash(frame)
        hashes[digest] = hashes.get(digest, 0) + 1
        trial = seed_dir.parent.name
        seed = int(seed_dir.name.split(".", 1)[1])
        run_uid = f"{spec.key}/{trial}/{seed_dir.name}"
        record: dict[str, Any] = {
            "run_uid": run_uid,
            "sweep": spec.key,
            "model": spec.model,
            "intervention": spec.intervention,
            "configuration": trial,
            "seed": seed,
            "seed_dir": str(seed_dir.resolve()),
            "validation_hash": digest,
            "validation_rows": len(frame),
            **normalized_config(spec, command["wandb_config"]),
            **regression_metrics(frame, "rt_pred_finetuned"),
        }
        record["best_epoch"] = best_epoch(spec, seed_dir, metadata)
        if pretrained_column and frame[pretrained_column].notna().all():
            pretrained_metrics = regression_metrics(frame, pretrained_column)
            record.update({f"pretrained_{key}": value for key, value in pretrained_metrics.items()})
        records.append(record)
        predictions[run_uid] = frame

    runs = pd.DataFrame(records)
    trial_seed_counts = (
        runs.groupby("configuration")["seed"].agg(lambda values: sorted(set(values))).to_dict()
        if not runs.empty
        else {}
    )
    incomplete = {
        trial: {
            "observed": seeds,
            "missing": sorted(set(EXPECTED_SEEDS) - set(seeds)),
            "unexpected": sorted(set(seeds) - set(EXPECTED_SEEDS)),
        }
        for trial, seeds in trial_seed_counts.items()
        if tuple(seeds) != EXPECTED_SEEDS
    }
    duplicate_configs = 0
    if not runs.empty:
        config_columns = [
            "loss",
            "huber_delta",
            "learning_rate",
            "batch_size",
            "epochs",
            "warmup_epochs",
            "trainable",
            "heavy_nterm_factor",
            "heavy_lysine_factor",
            "heavy_both_factor",
        ]
        unique_configs = runs.groupby("configuration")[config_columns].first()
        duplicate_configs = int(unique_configs.duplicated().sum())
    metric_columns = [column for column in runs.columns if column.endswith(("_mae", "_rmse", "_r2"))]
    nonfinite = int((~np.isfinite(runs[metric_columns].to_numpy(dtype=float))).sum()) if metric_columns else 0
    quality = {
        "sweep": spec.key,
        "path": str(spec.path.resolve()),
        "discovered_seed_directories": len(seed_dirs),
        "complete_seed_runs": len(runs),
        "configurations": int(runs["configuration"].nunique()) if not runs.empty else 0,
        "expected_seeds_per_configuration": list(EXPECTED_SEEDS),
        "incomplete_configurations": incomplete,
        "missing_required_files": missing,
        "unique_validation_hashes": len(hashes),
        "validation_hash_counts": hashes,
        "duplicate_hyperparameter_configurations": duplicate_configs,
        "nonfinite_metric_values": nonfinite,
        "status": "pass" if not missing and not incomplete and len(hashes) == 1 and nonfinite == 0 else "review",
    }
    return runs, predictions, quality


def t_critical_975(n: int) -> float:
    values = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
    }
    return values.get(n, 1.96)


def summarize_configurations(runs: pd.DataFrame) -> pd.DataFrame:
    parameter_columns = [
        "sweep",
        "model",
        "intervention",
        "loss",
        "huber_delta",
        "learning_rate",
        "batch_size",
        "epochs",
        "warmup_epochs",
        "trainable",
        "heavy_nterm_factor",
        "heavy_lysine_factor",
        "heavy_both_factor",
    ]
    metric_columns = [
        column
        for column in runs.columns
        if column.endswith(("_mae", "_rmse", "_r2", "_n"))
        and not column.startswith("pretrained_")
    ]
    records: list[dict[str, Any]] = []
    for configuration, group in runs.groupby("configuration", sort=False):
        row: dict[str, Any] = {"configuration": configuration}
        for column in parameter_columns:
            row[column] = group[column].iloc[0]
        seeds = sorted(group["seed"].unique().tolist())
        row["n_seeds"] = len(seeds)
        row["seeds"] = ",".join(str(seed) for seed in seeds)
        row["complete_10_seeds"] = tuple(seeds) == EXPECTED_SEEDS
        row["best_epoch_mean"] = float(group["best_epoch"].mean())
        for column in metric_columns:
            values = group[column].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half = t_critical_975(len(values)) * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = std
            row[f"{column}_ci95_low"] = mean - half
            row[f"{column}_ci95_high"] = mean + half
        records.append(row)
    summary = pd.DataFrame(records)
    summary = summary.sort_values(
        ["complete_10_seeds", "heavy_dimethyl_mae_mean", "global_mae_mean"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    summary.insert(0, "heavy_mae_rank", np.arange(1, len(summary) + 1))
    return summary


def add_common_metrics(
    runs: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    common_keys: set[str],
) -> pd.DataFrame:
    records = []
    for _, row in runs.iterrows():
        frame = predictions[row["run_uid"]]
        subset = frame[frame["_row_key"].isin(common_keys)].copy()
        if len(subset) != len(common_keys):
            raise ValueError(f"{row['run_uid']} has only {len(subset)} common validation rows")
        metrics = regression_metrics(subset, "rt_pred_finetuned")
        records.append({f"common_{key}": value for key, value in metrics.items()})
    return pd.concat([runs.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def parameter_effect_summary(configs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for parameter in (
        "heavy_nterm_factor",
        "heavy_lysine_factor",
        "heavy_both_factor",
        "loss",
        "batch_size",
        "epochs",
        "warmup_epochs",
    ):
        for value, group in configs.groupby(parameter, dropna=False):
            records.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "configurations": len(group),
                    "heavy_mae_mean_across_configurations": float(group["heavy_dimethyl_mae_mean"].mean()),
                    "heavy_mae_median_across_configurations": float(group["heavy_dimethyl_mae_mean"].median()),
                    "global_mae_mean_across_configurations": float(group["global_mae_mean"].mean()),
                }
            )
    return pd.DataFrame(records)


def paired_seed_comparison(
    runs: pd.DataFrame,
    configuration_a: str,
    configuration_b: str,
    metric: str,
    *,
    label: str,
) -> dict[str, Any]:
    """Compare two configurations after pairing identical training seeds."""
    columns = ["seed", metric]
    left = runs.loc[runs.configuration == configuration_a, columns].rename(columns={metric: "a"})
    right = runs.loc[runs.configuration == configuration_b, columns].rename(columns={metric: "b"})
    paired = left.merge(right, on="seed", how="inner", validate="one_to_one")
    delta = paired.a - paired.b
    n = len(delta)
    mean = float(delta.mean())
    std = float(delta.std(ddof=1)) if n > 1 else 0.0
    half = t_critical_975(n) * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "comparison": label,
        "configuration_a": configuration_a,
        "configuration_b": configuration_b,
        "metric": metric,
        "direction": "A minus B; negative favors A",
        "paired_seeds": n,
        "mean_delta": mean,
        "std_delta": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "all_seed_deltas_favor_a": bool((delta < 0).all()),
        "seed_deltas": ";".join(
            f"{seed}:{value:.8f}" for seed, value in zip(paired.seed, delta)
        ),
    }


def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    keep = np.ones(len(x), dtype=bool)
    for index in range(len(x)):
        dominated = (x <= x[index]) & (y <= y[index]) & ((x < x[index]) | (y < y[index]))
        if dominated.any():
            keep[index] = False
    return keep


def style_plots() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": PLOT_DPI,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(path_without_suffix.with_suffix(f".{suffix}"), dpi=PLOT_DPI)
    plt.close(fig)


def add_minutes_axis(axis: plt.Axes, scale: float, location: str = "top") -> None:
    secondary = axis.secondary_xaxis(location, functions=(lambda x: x * scale, lambda x: x / scale))
    secondary.set_xlabel("Approximate MAE (minutes)")


def plot_config_ranking(configs: pd.DataFrame, plot_dir: Path, scale: float) -> None:
    selected = configs.head(min(15, len(configs))).sort_values("heavy_dimethyl_mae_mean", ascending=False)
    y = np.arange(len(selected))
    mean = selected["heavy_dimethyl_mae_mean"].to_numpy()
    low = selected["heavy_dimethyl_mae_ci95_low"].to_numpy()
    high = selected["heavy_dimethyl_mae_ci95_high"].to_numpy()
    labels = [f"{value}  ({loss})" for value, loss in zip(selected.configuration, selected.loss)]
    fig, axis = plt.subplots(figsize=(9.2, max(4.3, 0.42 * len(selected) + 1.6)))
    axis.errorbar(mean, y, xerr=[mean - low, high - mean], fmt="o", color="#1B4965", capsize=3)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Heavy-Dimethyl validation MAE (normalized RT; mean ± 95% CI, 10 seeds)")
    axis.set_title("Configuration ranking")
    add_minutes_axis(axis, scale)
    save_figure(fig, plot_dir / "01_configuration_ranking")


def plot_tradeoff(configs: pd.DataFrame, plot_dir: Path) -> None:
    x = configs["global_mae_mean"].to_numpy()
    y = configs["heavy_dimethyl_mae_mean"].to_numpy()
    front = pareto_mask(x, y)
    losses = sorted(configs.loss.unique())
    palette = {loss: color for loss, color in zip(losses, ("#0072B2", "#D55E00", "#009E73", "#CC79A7"))}
    fig, axis = plt.subplots(figsize=(7.4, 5.6))
    for loss, group in configs.groupby("loss"):
        axis.scatter(
            group["global_mae_mean"],
            group["heavy_dimethyl_mae_mean"],
            s=58,
            alpha=0.82,
            label=loss.upper(),
            color=palette[loss],
            edgecolor="white",
            linewidth=0.5,
        )
    axis.scatter(x[front], y[front], s=120, facecolors="none", edgecolors="black", linewidths=1.3, label="Pareto")
    for _, row in configs.head(min(3, len(configs))).iterrows():
        axis.annotate(row.configuration.replace("trial.", ""), (row.global_mae_mean, row.heavy_dimethyl_mae_mean), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set(
        xlabel="Global validation MAE (normalized RT)",
        ylabel="Heavy-Dimethyl validation MAE (normalized RT)",
        title="Global performance versus Heavy-Dimethyl objective",
    )
    axis.legend(frameon=True)
    save_figure(fig, plot_dir / "02_global_vs_heavy_tradeoff")


def draw_factor_panel(axis: plt.Axes, configs: pd.DataFrame, column: str, title: str) -> None:
    levels = sorted(configs[column].unique())
    rng = np.random.default_rng(20260730)
    for index, level in enumerate(levels):
        values = configs.loc[configs[column] == level, "heavy_dimethyl_mae_mean"].to_numpy()
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        axis.scatter(np.full(len(values), index) + jitter, values, s=35, alpha=0.75, color="#0072B2")
        axis.plot([index - 0.18, index + 0.18], [np.median(values)] * 2, color="#D55E00", linewidth=2.2)
    axis.set_xticks(range(len(levels)), [f"{value:g}" for value in levels])
    axis.set(title=title, xlabel="Factor", ylabel="Heavy-Dimethyl MAE")


def plot_intervention_effects(configs: pd.DataFrame, plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    draw_factor_panel(axes[0], configs, "heavy_nterm_factor", "Heavy N-terminus")
    draw_factor_panel(axes[1], configs, "heavy_lysine_factor", "Heavy lysine")
    draw_factor_panel(axes[2], configs, "heavy_both_factor", "Heavy at both sites")
    fig.suptitle("Marginal intervention-factor associations (configuration means)")
    save_figure(fig, plot_dir / "03_intervention_factor_effects")


def plot_top_subgroups(configs: pd.DataFrame, plot_dir: Path) -> None:
    selected = configs.head(min(3, len(configs)))
    groups = ["global", "normal", "heavy_nterm_only", "heavy_lysine_only", "heavy_both"]
    labels = ["Global", "Normal", "Heavy N-term", "Heavy K", "Both heavy"]
    x = np.arange(len(groups))
    width = 0.8 / len(selected)
    fig, axis = plt.subplots(figsize=(10.2, 5.2))
    for index, (_, row) in enumerate(selected.iterrows()):
        means = [row[f"{group}_mae_mean"] for group in groups]
        lows = [row[f"{group}_mae_ci95_low"] for group in groups]
        highs = [row[f"{group}_mae_ci95_high"] for group in groups]
        positions = x - 0.4 + width / 2 + index * width
        axis.bar(positions, means, width, label=row.configuration.replace("trial.", ""), alpha=0.88)
        axis.errorbar(positions, means, yerr=[np.array(means) - lows, np.array(highs) - means], fmt="none", color="black", capsize=2, linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Validation MAE (normalized RT)")
    axis.set_title("Top configurations across validation subgroups")
    axis.legend(title="Configuration", ncol=min(3, len(selected)))
    save_figure(fig, plot_dir / "04_top_configuration_subgroups")


def plot_loss_lr(configs: pd.DataFrame, plot_dir: Path) -> None:
    losses = sorted(configs.loss.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    values = [configs.loc[configs.loss == loss, "heavy_dimethyl_mae_mean"] for loss in losses]
    axes[0].boxplot(values, labels=[loss.upper() for loss in losses], showfliers=True)
    rng = np.random.default_rng(20260730)
    for index, group in enumerate(values, start=1):
        axes[0].scatter(np.full(len(group), index) + rng.uniform(-0.07, 0.07, len(group)), group, s=28, alpha=0.7)
    axes[0].set(xlabel="Loss", ylabel="Heavy-Dimethyl MAE", title="Loss association")
    palette = {loss: color for loss, color in zip(losses, ("#0072B2", "#D55E00", "#009E73", "#CC79A7"))}
    for loss, group in configs.groupby("loss"):
        axes[1].scatter(group.learning_rate, group.heavy_dimethyl_mae_mean, label=loss.upper(), s=45, alpha=0.8, color=palette[loss])
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Learning rate (log scale)", ylabel="Heavy-Dimethyl MAE", title="Learning-rate association")
    axes[1].legend()
    save_figure(fig, plot_dir / "05_loss_and_learning_rate")


def markdown_table(frame: pd.DataFrame, columns: Iterable[str], digits: int = 5) -> str:
    columns = list(columns)
    headers = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for _, row in frame[columns].iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.{digits}f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([headers, separator, *rows])


def config_description(row: pd.Series, intervention: str) -> str:
    noun = "oversampling factors" if intervention == "oversampling" else "loss weights"
    return (
        f"loss={row.loss}, LR={row.learning_rate:.3g}, batch={int(row.batch_size)}, "
        f"epochs={int(row.epochs)}, warmup={int(row.warmup_epochs)}, {noun} "
        f"N-term/K/both={row.heavy_nterm_factor:g}/{row.heavy_lysine_factor:g}/{row.heavy_both_factor:g}"
    )


def write_sweep_readme(
    spec: SweepSpec,
    configs: pd.DataFrame,
    quality: dict[str, Any],
    analysis_dir: Path,
    scale: float,
    top_comparison: dict[str, Any] | None,
) -> None:
    best = configs.iloc[0]
    baseline = configs[
        (configs.heavy_nterm_factor == 1)
        & (configs.heavy_lysine_factor == 1)
        & (configs.heavy_both_factor == 1)
    ]
    median = float(configs.heavy_dimethyl_mae_mean.median())
    gain = 100.0 * (median - best.heavy_dimethyl_mae_mean) / median
    if baseline.empty:
        baseline_text = (
            "Es wurde keine Konfiguration mit 1/1/1 für alle drei Faktoren gezogen. "
            "Damit kann dieser Random Sweep die kausale Wirkung des Eingriffs nicht isolieren."
        )
    else:
        reference = baseline.iloc[0]
        delta = best.heavy_dimethyl_mae_mean - reference.heavy_dimethyl_mae_mean
        baseline_text = (
            f"Eine ungewichtete 1/1/1-Konfiguration ist vorhanden (`{reference.configuration}`); "
            f"der beste Run liegt um {delta:+.5f} normalized MAE relativ dazu. Da zugleich Loss, "
            "Learning Rate und Batchgröße abweichen, ist das nur eine unkontrollierte Referenz."
        )
    top = configs.head(min(5, len(configs))).copy()
    top["MAE_min"] = top.heavy_dimethyl_mae_mean * scale
    top_table = markdown_table(
        top,
        [
            "configuration",
            "loss",
            "heavy_nterm_factor",
            "heavy_lysine_factor",
            "heavy_both_factor",
            "heavy_dimethyl_mae_mean",
            "heavy_dimethyl_mae_std",
            "global_mae_mean",
            "MAE_min",
        ],
    )
    sparse_note = (
        "Mit nur drei getesteten Konfigurationen ist dies eine Pilot-Auswertung; Faktor- und "
        "Loss-Effekte sind nicht belastbar schätzbar."
        if len(configs) <= 3
        else "Die Faktorplots sind deskriptiv: Der Random Sweep variiert mehrere Hyperparameter gleichzeitig."
    )
    if top_comparison is None:
        top_comparison_text = "Kein Runner-up für einen gepaarten Vergleich vorhanden."
    else:
        interval_excludes_zero = (
            top_comparison["ci95_high"] < 0 or top_comparison["ci95_low"] > 0
        )
        interpretation = (
            "Das 95%-Intervall schließt null nicht ein."
            if interval_excludes_zero
            else "Das 95%-Intervall schließt null ein; die Kandidaten sind über Seeds nicht klar getrennt."
        )
        top_comparison_text = (
            f"Gegenüber dem Runner-up `{top_comparison['configuration_b']}` beträgt die gepaarte "
            f"Seed-Differenz (Gewinner minus Runner-up) {top_comparison['mean_delta']:+.5f} "
            f"normalized MAE (95%-CI {top_comparison['ci95_low']:+.5f} bis "
            f"{top_comparison['ci95_high']:+.5f}; {top_comparison['mean_delta'] * scale:+.3f} min). "
            f"{interpretation}"
        )
    text = f"""# {spec.title}: validation summary

## Configuration metrics

{top_table}

The tables summarize the supplied validation runs. Seed variation describes
training variability, not generalization across independent experiments.
Configuration selection requires a separate held-out final evaluation.
"""
    (analysis_dir / "README.md").write_text(text, encoding="utf-8")


def prepare_analysis_dir(path: Path) -> tuple[Path, Path]:
    analysis = path / "analysis"
    plots = analysis / "plots"
    analysis.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    return analysis, plots


def study_best_configs(all_configs: pd.DataFrame) -> pd.DataFrame:
    return (
        all_configs.sort_values(["common_heavy_dimethyl_mae_mean", "common_global_mae_mean"])
        .groupby("sweep", as_index=False, sort=False)
        .first()
        .sort_values("common_heavy_dimethyl_mae_mean")
        .reset_index(drop=True)
    )


def plot_study_best(best: pd.DataFrame, plot_dir: Path, scale: float) -> None:
    ordered = best.sort_values("common_heavy_dimethyl_mae_mean", ascending=False)
    y = np.arange(len(ordered))
    mean = ordered.common_heavy_dimethyl_mae_mean.to_numpy()
    low = ordered.common_heavy_dimethyl_mae_ci95_low.to_numpy()
    high = ordered.common_heavy_dimethyl_mae_ci95_high.to_numpy()
    colors = [COLORS[value] for value in ordered.sweep]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.errorbar(mean, y, xerr=[mean - low, high - mean], fmt="none", color="black", capsize=3, linewidth=1)
    axis.scatter(mean, y, s=90, c=colors, zorder=3)
    axis.set_yticks(y, [SHORT_LABELS[value].replace("\n", " ") for value in ordered.sweep])
    axis.set_xlabel("Heavy-Dimethyl MAE on 815 common validation rows (mean ± 95% CI)")
    axis.set_title("Best configuration found in each targeted sweep")
    add_minutes_axis(axis, scale)
    save_figure(fig, plot_dir / "01_best_configuration_comparison")


def plot_study_tradeoff(all_configs: pd.DataFrame, plot_dir: Path) -> None:
    x = all_configs.common_global_mae_mean.to_numpy()
    y = all_configs.common_heavy_dimethyl_mae_mean.to_numpy()
    front = pareto_mask(x, y)
    fig, axis = plt.subplots(figsize=(8.2, 6.1))
    for sweep, group in all_configs.groupby("sweep"):
        axis.scatter(
            group.common_global_mae_mean,
            group.common_heavy_dimethyl_mae_mean,
            label=SHORT_LABELS[sweep].replace("\n", " "),
            color=COLORS[sweep],
            s=52,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.5,
        )
    axis.scatter(x[front], y[front], s=125, facecolors="none", edgecolors="black", linewidths=1.2, label="Overall Pareto")
    axis.set(
        xlabel="Global MAE on common hold-out (normalized RT)",
        ylabel="Heavy-Dimethyl MAE on common hold-out (normalized RT)",
        title="All 54 configurations on the common validation set",
    )
    axis.legend(frameon=True)
    save_figure(fig, plot_dir / "02_all_configurations_tradeoff")


def plot_study_subgroups(best: pd.DataFrame, plot_dir: Path) -> None:
    groups = ["common_global", "common_normal", "common_heavy_nterm_only", "common_heavy_lysine_only", "common_heavy_both"]
    labels = ["Global", "Normal", "Heavy N-term", "Heavy K", "Both heavy"]
    x = np.arange(len(groups))
    width = 0.19
    fig, axis = plt.subplots(figsize=(11.5, 5.4))
    for index, (_, row) in enumerate(best.sort_values("sweep").iterrows()):
        means = np.array([row[f"{group}_mae_mean"] for group in groups])
        lows = np.array([row[f"{group}_mae_ci95_low"] for group in groups])
        highs = np.array([row[f"{group}_mae_ci95_high"] for group in groups])
        positions = x - 1.5 * width + index * width
        axis.bar(positions, means, width, color=COLORS[row.sweep], label=SHORT_LABELS[row.sweep].replace("\n", " "), alpha=0.9)
        axis.errorbar(positions, means, yerr=[means - lows, highs - means], fmt="none", color="black", capsize=2, linewidth=0.7)
    axis.set_xticks(x, labels)
    axis.set_ylabel("MAE on common hold-out (normalized RT)")
    axis.set_title("Best sweep configurations by peptide subgroup")
    axis.legend(ncol=2)
    save_figure(fig, plot_dir / "03_best_configuration_subgroups")


def plot_study_distributions(all_configs: pd.DataFrame, plot_dir: Path) -> None:
    order = ["carafe2_oversampling", "carafe2_loss_weighting", "diann_oversampling", "diann_loss_weighting"]
    values = [all_configs.loc[all_configs.sweep == sweep, "common_heavy_dimethyl_mae_mean"] for sweep in order]
    fig, axis = plt.subplots(figsize=(9.6, 5.2))
    boxes = axis.boxplot(values, labels=[SHORT_LABELS[sweep] for sweep in order], patch_artist=True, showfliers=True)
    for patch, sweep in zip(boxes["boxes"], order):
        patch.set_facecolor(COLORS[sweep])
        patch.set_alpha(0.55)
    rng = np.random.default_rng(20260730)
    for index, (group, sweep) in enumerate(zip(values, order), start=1):
        axis.scatter(np.full(len(group), index) + rng.uniform(-0.08, 0.08, len(group)), group, color=COLORS[sweep], s=28, alpha=0.75)
    axis.set(ylabel="Heavy-Dimethyl MAE (configuration mean)", title="Search-result distributions and unequal search coverage")
    for index, group in enumerate(values, start=1):
        axis.text(index, axis.get_ylim()[1], f"n={len(group)} configs", ha="center", va="top", fontsize=8)
    save_figure(fig, plot_dir / "04_search_result_distributions")


def plot_study_loss(all_configs: pd.DataFrame, plot_dir: Path) -> None:
    models = ["carafe2", "diann"]
    losses = ["mae", "huber", "mse"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True)
    rng = np.random.default_rng(20260730)
    for axis, model in zip(axes, models):
        subset = all_configs[all_configs.model == model]
        available = [loss for loss in losses if loss in set(subset.loss)]
        values = [subset.loc[subset.loss == loss, "common_heavy_dimethyl_mae_mean"] for loss in available]
        axis.boxplot(values, labels=[loss.upper() for loss in available], showfliers=True)
        for index, group in enumerate(values, start=1):
            axis.scatter(np.full(len(group), index) + rng.uniform(-0.07, 0.07, len(group)), group, s=28, alpha=0.72)
        axis.set(title=model.upper(), xlabel="Loss")
    axes[0].set_ylabel("Heavy-Dimethyl MAE (configuration mean)")
    fig.suptitle("Descriptive loss association across targeted sweeps")
    save_figure(fig, plot_dir / "05_loss_association_by_model")


def plot_study_mechanism_differences(
    comparisons: pd.DataFrame,
    plot_dir: Path,
    scale: float,
) -> None:
    shown = comparisons[comparisons.metric == "common_heavy_dimethyl_mae"].copy()
    shown = shown.sort_values("model")
    y = np.arange(len(shown))
    mean = shown.mean_delta.to_numpy()
    low = shown.ci95_low.to_numpy()
    high = shown.ci95_high.to_numpy()
    colors = ["#56B4E9" if model == "carafe2" else "#E69F00" for model in shown.model]
    fig, axis = plt.subplots(figsize=(8.2, 4.3))
    axis.axvline(0, color="black", linewidth=1, linestyle="--")
    axis.errorbar(
        mean,
        y,
        xerr=[mean - low, high - mean],
        fmt="none",
        color="black",
        capsize=3,
    )
    axis.scatter(mean, y, s=90, c=colors, zorder=3)
    axis.set_yticks(y, ["DIA-NN" if model == "diann" else "Carafe2" for model in shown.model])
    axis.set_xlabel("Best loss weighting − best oversampling (Heavy-Dimethyl MAE)")
    axis.set_title("Paired seed comparison of the best observed mechanisms")
    add_minutes_axis(axis, scale)
    save_figure(fig, plot_dir / "06_best_mechanism_paired_difference")


def write_figure_captions(plot_dir: Path, scale: float) -> None:
    text = f"""# Figure captions and LaTeX use

All figures are available as `.pdf`, `.svg`, and `.png`. Normalized RT errors can be converted
to minutes with `error × {scale:.9f}`.

## Figure 1

**Best Heavy-Dimethyl RT performance per targeted sweep.** Points show the mean MAE across ten
training seeds and error bars show 95% t confidence intervals. All metrics were recomputed on the
same 815 validation rows shared by Carafe2 and DIA-NN.

## Figure 2

**Global-versus-targeted RT trade-off for all 54 hyperparameter configurations.** Each point is a
ten-seed configuration mean. Open circles mark the overall non-dominated Pareto front. Search
budgets were unequal (3/3/23/25 configurations), so best-of-sweep comparisons are descriptive.

## Figure 3

**Subgroup MAE of the best configuration from each sweep.** The Heavy N-terminal and both-heavy
groups are much smaller than Heavy-lysine and therefore have wider sampling uncertainty than the
seed intervals alone express.

## Figure 4

**Distribution of configuration-level Heavy-Dimethyl MAE.** This visualizes both robustness across
the sampled search space and the substantially smaller Carafe2 search coverage.

## Figure 5

**Descriptive association between loss function and Heavy-Dimethyl MAE.** Loss was not randomized
independently of learning rate, batch size, epochs, warmup, and intervention strength; the plot is
hypothesis-generating rather than causal.

## Figure 6

**Paired seed difference between the best loss-weighting and oversampling configuration within
each model.** Negative values favor loss weighting. Pairing uses identical seed numbers and error
bars show 95% t confidence intervals. Because the best configurations differ in several other
hyperparameters, this remains a best-candidate comparison rather than an isolated causal effect.

Example:

```latex
\\begin{{figure}}[t]
  \\centering
  \\includegraphics[width=0.9\\linewidth]{{01_best_configuration_comparison.pdf}}
  \\caption{{Best Heavy-Dimethyl RT performance per targeted sweep.}}
  \\label{{fig:heavy-rt-sweeps}}
\\end{{figure}}
```
"""
    (plot_dir.parent / "FIGURE_CAPTIONS.md").write_text(text, encoding="utf-8")


def write_study_readme(
    study_dir: Path,
    all_runs: pd.DataFrame,
    all_configs: pd.DataFrame,
    best: pd.DataFrame,
    quality: list[dict[str, Any]],
    common_manifest: dict[str, Any],
    scale: float,
    comparisons: pd.DataFrame,
) -> None:
    top = best.copy()
    top["sweep_label"] = top.sweep.map(lambda value: SHORT_LABELS[value].replace("\n", " "))
    top["heavy_MAE_min"] = top.common_heavy_dimethyl_mae_mean * scale
    table = markdown_table(
        top,
        [
            "sweep_label",
            "configuration",
            "loss",
            "heavy_nterm_factor",
            "heavy_lysine_factor",
            "heavy_both_factor",
            "common_heavy_dimethyl_mae_mean",
            "common_heavy_dimethyl_mae_std",
            "common_global_mae_mean",
            "heavy_MAE_min",
        ],
    )
    overall_best = best.sort_values("common_heavy_dimethyl_mae_mean").iloc[0]
    model_best = best.sort_values("common_heavy_dimethyl_mae_mean").groupby("model", as_index=False).first()
    carafe_best = model_best[model_best.model == "carafe2"].iloc[0]
    diann_best = model_best[model_best.model == "diann"].iloc[0]
    relative = 100.0 * (
        carafe_best.common_heavy_dimethyl_mae_mean - diann_best.common_heavy_dimethyl_mae_mean
    ) / carafe_best.common_heavy_dimethyl_mae_mean
    mechanism = comparisons[
        comparisons.metric == "common_heavy_dimethyl_mae"
    ].set_index("model")
    carafe_mechanism = mechanism.loc["carafe2"]
    diann_mechanism = mechanism.loc["diann"]
    data_quality_table = markdown_table(
        pd.DataFrame(quality),
        ["sweep", "complete_seed_runs", "configurations", "unique_validation_hashes", "nonfinite_metric_values", "status"],
        digits=0,
    )
    text = f"""# Heavy-dimethyl RT fine-tuning comparison

## Configuration summary

{table}

## Input consistency

{data_quality_table}

Common rows: {common_manifest['common_rows']}; Carafe2-only rows:
{common_manifest['carafe2_only_rows']}; DIA-NN-only rows:
{common_manifest['diann_only_rows']}.

Inspect the CSV tables and generated plots for configuration-level and
seed-level errors. These validation comparisons do not establish independent
test performance or a causal advantage of weighting over oversampling.
"""
    (study_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    specs = [
        SweepSpec("carafe2_oversampling", "Carafe2 RT Heavy oversampling", "carafe2", "oversampling", args.carafe2_oversampling.resolve()),
        SweepSpec("carafe2_loss_weighting", "Carafe2 RT Heavy loss weighting", "carafe2", "loss_weighting", args.carafe2_loss_weighting.resolve()),
        SweepSpec("diann_oversampling", "DIA-NN RT Heavy oversampling", "diann", "oversampling", args.diann_oversampling.resolve()),
        SweepSpec("diann_loss_weighting", "DIA-NN RT Heavy loss weighting", "diann", "loss_weighting", args.diann_loss_weighting.resolve()),
    ]
    for spec in specs:
        if not spec.path.is_dir():
            raise SystemExit(f"Missing sweep directory: {spec.path}")
    study_dir = args.study_dir.resolve()
    study_plot_dir = study_dir / "plots"
    study_plot_dir.mkdir(parents=True, exist_ok=True)
    style_plots()

    loaded: dict[str, tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]] = {}
    for spec in specs:
        loaded[spec.key] = load_sweep(spec)

    carafe_reference = next(iter(loaded["carafe2_oversampling"][1].values()))
    diann_reference = next(iter(loaded["diann_oversampling"][1].values()))
    carafe_keys = set(carafe_reference["_row_key"])
    diann_keys = set(diann_reference["_row_key"])
    common_keys = carafe_keys & diann_keys
    common_manifest = {
        "comparison_basis": "exact sequence/mods/mod_sites/rt_norm/nAA row key",
        "common_rows": len(common_keys),
        "carafe2_rows": len(carafe_keys),
        "diann_rows": len(diann_keys),
        "carafe2_only_rows": len(carafe_keys - diann_keys),
        "diann_only_rows": len(diann_keys - carafe_keys),
        "rt_scale_minutes": args.rt_scale_minutes,
    }

    all_runs_parts = []
    all_config_parts = []
    qualities = []
    for spec in specs:
        runs, predictions, quality = loaded[spec.key]
        runs = add_common_metrics(runs, predictions, common_keys)
        configs = summarize_configurations(runs)
        analysis_dir, plot_dir = prepare_analysis_dir(spec.path)
        runs.sort_values(["configuration", "seed"]).to_csv(analysis_dir / "run_summary.csv", index=False)
        configs.to_csv(analysis_dir / "configuration_summary.csv", index=False)
        parameter_effect_summary(configs).to_csv(analysis_dir / "parameter_effect_summary.csv", index=False)
        top_comparison = None
        if len(configs) > 1:
            top_comparison = paired_seed_comparison(
                runs,
                configs.iloc[0].configuration,
                configs.iloc[1].configuration,
                "heavy_dimethyl_mae",
                label="winner minus runner-up",
            )
            pd.DataFrame([top_comparison]).to_csv(
                analysis_dir / "top_vs_runner_up.csv", index=False
            )
        (analysis_dir / "data_quality.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        plot_config_ranking(configs, plot_dir, args.rt_scale_minutes)
        plot_tradeoff(configs, plot_dir)
        plot_intervention_effects(configs, plot_dir)
        plot_top_subgroups(configs, plot_dir)
        plot_loss_lr(configs, plot_dir)
        write_sweep_readme(
            spec,
            configs,
            quality,
            analysis_dir,
            args.rt_scale_minutes,
            top_comparison,
        )
        all_runs_parts.append(runs)
        all_config_parts.append(configs)
        qualities.append(quality)

    all_runs = pd.concat(all_runs_parts, ignore_index=True)
    all_configs = pd.concat(all_config_parts, ignore_index=True)
    best = study_best_configs(all_configs)
    best_by_sweep = best.set_index("sweep")
    comparisons = []
    for model in ("carafe2", "diann"):
        weighting = best_by_sweep.loc[f"{model}_loss_weighting"]
        oversampling = best_by_sweep.loc[f"{model}_oversampling"]
        for metric in (
            "common_heavy_dimethyl_mae",
            "common_global_mae",
            "common_normal_mae",
            "common_heavy_nterm_only_mae",
            "common_heavy_lysine_only_mae",
            "common_heavy_both_mae",
        ):
            comparison = paired_seed_comparison(
                all_runs[all_runs.model == model],
                weighting.configuration,
                oversampling.configuration,
                metric,
                label="best loss weighting minus best oversampling",
            )
            comparison["model"] = model
            comparisons.append(comparison)
    comparison_frame = pd.DataFrame(comparisons)
    all_runs.to_csv(study_dir / "all_seed_runs.csv", index=False)
    all_configs.to_csv(study_dir / "all_configuration_summary.csv", index=False)
    best.to_csv(study_dir / "best_configurations.csv", index=False)
    comparison_frame.to_csv(study_dir / "best_pairwise_comparisons.csv", index=False)
    pd.DataFrame(qualities).drop(columns=["incomplete_configurations", "missing_required_files", "validation_hash_counts"]).to_csv(
        study_dir / "data_quality_summary.csv", index=False
    )
    (study_dir / "common_holdout_manifest.json").write_text(json.dumps(common_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "analysis_script": str(Path(__file__).resolve()),
        "source_sweeps": {spec.key: str(spec.path) for spec in specs},
        "expected_seeds": list(EXPECTED_SEEDS),
        "primary_metric": "common_heavy_dimethyl_mae",
        "confidence_interval": "two-sided 95% t interval across training seeds",
        **common_manifest,
    }
    (study_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_study_best(best, study_plot_dir, args.rt_scale_minutes)
    plot_study_tradeoff(all_configs, study_plot_dir)
    plot_study_subgroups(best, study_plot_dir)
    plot_study_distributions(all_configs, study_plot_dir)
    plot_study_loss(all_configs, study_plot_dir)
    plot_study_mechanism_differences(comparison_frame, study_plot_dir, args.rt_scale_minutes)
    write_figure_captions(study_plot_dir, args.rt_scale_minutes)
    write_study_readme(
        study_dir,
        all_runs,
        all_configs,
        best,
        qualities,
        common_manifest,
        args.rt_scale_minutes,
        comparison_frame,
    )

    print(f"Analyzed {len(all_runs)} seed runs in {len(all_configs)} configurations")
    print(f"Common cross-model validation rows: {len(common_keys)}")
    print(f"Study report: {study_dir / 'README.md'}")


if __name__ == "__main__":
    main()
