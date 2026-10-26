"""Carafe-style plots and reports for individual runs and campaigns."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .training import regression_metrics


PLOT_DPI = 180


def _line_plot(frame: pd.DataFrame, columns: Iterable[str], path: Path, title: str, ylabel: str) -> None:
    columns = [column for column in columns if column in frame.columns]
    if not columns:
        return
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for column in columns:
        axis.plot(frame["epoch"], frame[column], marker="o", markersize=2.5, label=column)
    axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def _scatter_plot(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    target = frame["rt_norm"].to_numpy()
    low = float(min(target.min(), frame["rt_pred_pretrained"].min(), frame["rt_pred_finetuned"].min()))
    high = float(max(target.max(), frame["rt_pred_pretrained"].max(), frame["rt_pred_finetuned"].max()))
    for axis, column, label in (
        (axes[0], "rt_pred_pretrained", "Pretrained"),
        (axes[1], "rt_pred_finetuned", "Fine-tuned"),
    ):
        metrics = regression_metrics(target, frame[column].to_numpy())
        axis.scatter(target, frame[column], s=8, alpha=0.45)
        axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        axis.set_title(f"{label}\nMAE={metrics['mae']:.4f}, r={metrics['pearson']:.4f}")
        axis.set_xlabel("Observed normalized RT")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted normalized RT")
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def _add_strata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    mods = result["mods"].fillna("").astype(str)
    heavy_nterm = mods.str.contains("Dimethyl:2H(4)13C(2)@N-term", regex=False)
    heavy_lysine = mods.str.contains("Dimethyl:2H(4)13C(2)@K", regex=False)
    heavy = heavy_nterm | heavy_lysine
    result["heavy_dimethyl"] = np.where(heavy, "heavy", "not_heavy")
    result["heavy_site"] = "none"
    result.loc[heavy_nterm & ~heavy_lysine, "heavy_site"] = "n_term"
    result.loc[heavy_lysine & ~heavy_nterm, "heavy_site"] = "lysine"
    result.loc[heavy_nterm & heavy_lysine, "heavy_site"] = "n_term_and_lysine"
    lengths = result["sequence"].str.len()
    result["length_bin"] = pd.cut(
        lengths, bins=[0, 10, 20, np.inf], labels=["7-10", "11-20", "21+"]
    ).astype(str)
    return result


def subgroup_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    annotated = _add_strata(frame)
    records: List[Dict[str, object]] = []
    for stratum in ("heavy_dimethyl", "heavy_site", "length_bin"):
        for group, subset in annotated.groupby(stratum, dropna=False):
            if len(subset) < 2:
                continue
            for model, column in (
                ("pretrained", "rt_pred_pretrained"),
                ("finetuned", "rt_pred_finetuned"),
            ):
                records.append(
                    {
                        "stratum": stratum,
                        "group": str(group),
                        "model": model,
                        "rows": len(subset),
                        **regression_metrics(
                            subset["rt_norm"].to_numpy(), subset[column].to_numpy()
                        ),
                    }
                )
    return pd.DataFrame(records)


def _subgroup_plot(metrics: pd.DataFrame, path: Path) -> None:
    subset = metrics[(metrics["stratum"] == "heavy_dimethyl")]
    if subset.empty:
        return
    pivot = subset.pivot(index="group", columns="model", values="mae")
    axis = pivot.plot(kind="bar", figsize=(7, 4.5), rot=0)
    axis.set(title="Validation MAE by heavy-dimethyl status", ylabel="MAE", xlabel="")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=PLOT_DPI)
    plt.close(axis.figure)


def _final_heavy_comparison_plot(predictions: pd.DataFrame, path: Path) -> None:
    """Compare final MAE for all, any heavy dimethyl, and N-terminal heavy."""
    mods = predictions["mods"].fillna("").astype(str)
    nterm = mods.str.contains("Dimethyl:2H(4)13C(2)@N-term", regex=False)
    lysine = mods.str.contains("Dimethyl:2H(4)13C(2)@K", regex=False)
    subsets = {
        "All validation": np.ones(len(predictions), dtype=bool),
        "Any heavy\ndimethyl": (nterm | lysine).to_numpy(),
        "N-term heavy\ndimethyl": nterm.to_numpy(),
    }
    rows = []
    for label, mask in subsets.items():
        if not mask.any():
            continue
        targets = predictions.loc[mask, "rt_norm"].to_numpy()
        for model, column in (("Pretrained", "rt_pred_pretrained"), ("Fine-tuned", "rt_pred_finetuned")):
            rows.append({
                "subset": label,
                "model": model,
                "mae": regression_metrics(targets, predictions.loc[mask, column].to_numpy())["mae"],
            })
    if not rows:
        return
    pivot = pd.DataFrame(rows).pivot(index="subset", columns="model", values="mae")
    axis = pivot.plot(kind="bar", figsize=(8, 4.8), rot=0, color=["#9aa0a6", "#1f77b4"])
    axis.set(title="Final RT MAE by Heavy-Dimethyl subset", ylabel="Normalized RT MAE", xlabel="")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=PLOT_DPI)
    plt.close(axis.figure)


def generate_run_diagnostics(output_dir: Path) -> Dict[str, str]:
    """Generate plots, subgroup tables, and a self-contained index report."""
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(output_dir / "training_history.csv")
    predictions = pd.read_csv(output_dir / "validation_predictions.tsv", sep="\t")
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))

    plots = {
        "loss": plot_dir / "loss_comparison.png",
        "train_loss": plot_dir / "train_loss.png",
        "validation_loss": plot_dir / "validation_loss.png",
        "validation_mae": plot_dir / "validation_mae.png",
        "validation_rmse": plot_dir / "validation_rmse.png",
        "validation_r2": plot_dir / "validation_r2.png",
        "validation_pearson": plot_dir / "validation_pearson.png",
        "validation_spearman": plot_dir / "validation_spearman.png",
        "learning_rates": plot_dir / "learning_rates.png",
        "predictions": plot_dir / "rt_predictions.png",
        "heavy": plot_dir / "heavy_dimethyl_mae.png",
        "heavy_losses": plot_dir / "heavy_dimethyl_losses.png",
        "final_heavy_comparison": plot_dir / "final_heavy_dimethyl_comparison.png",
    }
    _line_plot(history, ["train_loss", "validation_loss"], plots["loss"], "Training objective and validation MAE", "Loss")
    _line_plot(history, ["train_loss", "train_mae"], plots["train_loss"], "Training objective and MAE", "Loss")
    _line_plot(history, ["validation_loss"], plots["validation_loss"], "Validation loss", "MAE")
    for metric in ("mae", "rmse", "r2", "pearson", "spearman"):
        _line_plot(
            history,
            [f"validation_{metric}"],
            plots[f"validation_{metric}"],
            f"Validation {metric.upper()}",
            metric.upper(),
        )
    _line_plot(
        history,
        [column for column in history.columns if column.startswith("lr_")],
        plots["learning_rates"],
        "Learning-rate schedule",
        "Learning rate",
    )
    _line_plot(
        history,
        [
            "train_heavy_dimethyl_loss",
            "validation_heavy_dimethyl_loss",
            "train_heavy_nterm_dimethyl_loss",
            "validation_heavy_nterm_dimethyl_loss",
        ],
        plots["heavy_losses"],
        "Heavy-Dimethyl objective by subset",
        "Unweighted objective loss",
    )
    _scatter_plot(predictions, plots["predictions"])
    _final_heavy_comparison_plot(predictions, plots["final_heavy_comparison"])
    groups = subgroup_metrics(predictions)
    groups_path = output_dir / "subgroup_metrics.csv"
    groups.to_csv(groups_path, index=False)
    _subgroup_plot(groups, plots["heavy"])

    result = metadata["result"]
    image_tags = "\n".join(
        f'<h3>{html.escape(name.replace("_", " ").title())}</h3><img src="plots/{path.name}" style="max-width:100%">'
        for name, path in plots.items()
        if path.exists()
    )
    report = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DIA-NN RT fine-tuning report</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse}}th,td{{border:1px solid #ccc;padding:.35rem}}code{{background:#eee;padding:.15rem}}</style></head>
<body><h1>DIA-NN RT fine-tuning report</h1>
<p>Strategy: <code>{html.escape(str(result.get('strategy')))}</code>; best epoch: {result['best_epoch']}</p>
<h2>Final metrics</h2>{pd.DataFrame([result['pretrained_metrics'], result['finetuned_metrics']], index=['pretrained','finetuned']).to_html()}
{image_tags}
<h2>Subgroup metrics</h2>{groups.to_html(index=False)}
</body></html>"""
    report_path = output_dir / "training_report.html"
    report_path.write_text(report, encoding="utf-8")
    return {"report": str(report_path), "subgroups": str(groups_path)}


def generate_campaign_diagnostics(campaign_dir: Path, run_names: Iterable[str]) -> Path:
    """Aggregate completed runs into comparison CSV, plots, and HTML."""
    campaign_dir = Path(campaign_dir)
    records: List[Dict[str, object]] = []
    histories: Dict[str, pd.DataFrame] = {}
    for name in run_names:
        run_dir = campaign_dir / name
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result = metadata["result"]
        record: Dict[str, object] = {
            "experiment": name,
            "strategy": result.get("strategy"),
            "loss": result.get("loss", {}).get("name", "mae"),
            "huber_delta": result.get("loss", {}).get("huber_delta"),
            "heavy_nterm_weight": result.get("loss", {}).get("heavy_nterm_weight", 1.0),
            "heavy_lysine_weight": result.get("loss", {}).get("heavy_lysine_weight", 1.0),
            "heavy_both_weight": result.get("loss", {}).get("heavy_both_weight", 1.0),
            "best_epoch": result["best_epoch"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
        for stage in ("pretrained", "finetuned"):
            for metric, value in result[f"{stage}_metrics"].items():
                record[f"{stage}_{metric}"] = value
        records.append(record)
        histories[name] = pd.read_csv(run_dir / "training_history.csv")
    summary = pd.DataFrame(records)
    summary_path = campaign_dir / "campaign_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_dir = campaign_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        axis = summary.set_index("experiment")[["pretrained_mae", "finetuned_mae"]].plot(
            kind="bar", figsize=(9, 5), rot=20
        )
        axis.set(title="RT validation MAE by strategy", ylabel="MAE", xlabel="")
        axis.grid(axis="y", alpha=0.25)
        axis.figure.tight_layout()
        axis.figure.savefig(plot_dir / "strategy_mae.png", dpi=PLOT_DPI)
        plt.close(axis.figure)

        fig, axis = plt.subplots(figsize=(9, 5))
        for name, history in histories.items():
            axis.plot(history["epoch"], history["validation_loss"], label=name)
        axis.set(title="Validation MAE across strategies", xlabel="Epoch", ylabel="MAE")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "strategy_validation_curves.png", dpi=PLOT_DPI)
        plt.close(fig)
    report_path = campaign_dir / "campaign_report.html"
    report_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>DIA-NN campaign</title></head><body>"
        "<h1>DIA-NN RT fine-tuning campaign</h1>"
        + summary.to_html(index=False)
        + "<h2>MAE comparison</h2><img src='plots/strategy_mae.png' style='max-width:100%'>"
        + "<h2>Validation curves</h2><img src='plots/strategy_validation_curves.png' style='max-width:100%'>"
        + "</body></html>",
        encoding="utf-8",
    )
    return report_path
