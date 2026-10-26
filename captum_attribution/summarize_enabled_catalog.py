"""Build a compact, sweep-style report from completed catalog evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

DEFAULT_ROOT = Path("rt_model_evaluations")
COLORS = {"aligned_or_combined": "#2878b5", "source_specific": "#d95f02"}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _run_dir(root: Path, row: pd.Series) -> Path:
    return root / row.family / f"{row.model_name}__{row.dataset_name}__v1"


def collect(root: Path) -> pd.DataFrame:
    catalog = pd.read_csv(
        root / "registry/evaluation_catalog.tsv", sep="\t", keep_default_na=False
    )
    enabled = catalog[
        catalog.enabled.astype(str).str.lower().isin(["true", "1", "yes"])
    ].copy()
    records = []
    for _, row in enabled.iterrows():
        run_dir = _run_dir(root, row)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        attrs = run_dir / "attributions.parquet"
        dimethyl_summary = run_dir / "dimethyl_peptide_attribution_summary.parquet"
        if dimethyl_summary.exists():
            dimethyl = pd.read_parquet(dimethyl_summary)
        elif attrs.exists():
            # Work with the final stored attribution table, so the catalog
            # report remains reproducible even when per-run plots were not
            # regenerated yet.
            tokens = pd.read_parquet(
                attrs,
                columns=[
                    "peptide_id",
                    "dimethyl_class",
                    "predictor_token_string",
                    "absolute_attribution",
                ],
            )
            tokens = tokens[~tokens.predictor_token_string.isin(["<", ">"])]
            tokens["is_dimethyl"] = tokens.predictor_token_string.astype(
                str
            ).str.contains("Dimethyl", case=False, na=False)
            dimethyl = tokens.groupby(
                ["peptide_id", "dimethyl_class"], as_index=False
            ).agg(
                total_absolute_attribution=("absolute_attribution", "sum"),
                dimethyl_absolute_attribution=(
                    "absolute_attribution",
                    lambda x: x[tokens.loc[x.index, "is_dimethyl"]].sum(),
                ),
            )
            dimethyl["dimethyl_attribution_share"] = np.divide(
                dimethyl.dimethyl_absolute_attribution,
                dimethyl.total_absolute_attribution,
                out=np.zeros(len(dimethyl)),
                where=dimethyl.total_absolute_attribution.to_numpy() != 0,
            )
        else:
            dimethyl = pd.DataFrame()
        modified = (
            dimethyl[
                dimethyl.dimethyl_class.isin(
                    ["heavy_nterm", "heavy_lysine", "heavy_both"]
                )
            ]
            if not dimethyl.empty
            else dimethyl
        )
        records.append(
            {
                "model_name": row.model_name,
                "model_id": row.model_id,
                "dataset_name": row.dataset_name,
                "family": row.family,
                "training_scope": row.training_scope,
                "training_scope_kind": row.training_scope_kind,
                "alignment_target": row.alignment_target,
                "n_attributed_peptides": int(
                    pd.read_parquet(attrs, columns=["peptide_id"]).peptide_id.nunique()
                )
                if attrs.exists()
                else 0,
                "mean_dimethyl_attribution_share_modified": float(
                    modified.dimethyl_attribution_share.mean()
                )
                if not modified.empty
                else np.nan,
                "median_dimethyl_attribution_share_modified": float(
                    modified.dimethyl_attribution_share.median()
                )
                if not modified.empty
                else np.nan,
                "run_dir": str(run_dir),
                **metrics,
            }
        )
    return pd.DataFrame(records)


def ranking_plot(summary: pd.DataFrame, plot_dir: Path) -> None:
    groups = list(summary.groupby("dataset_name", sort=True))
    if not groups:
        return
    fig, axes = plt.subplots(
        len(groups), 1, figsize=(10, max(3.4, 0.45 * len(summary) + 1.8)), squeeze=False
    )
    for axis, (dataset, group) in zip(axes[:, 0], groups):
        ordered = group.sort_values("mae", ascending=True).reset_index(drop=True)
        y = np.arange(len(ordered))
        colors = [COLORS.get(value, "#7570b3") for value in ordered.training_scope_kind]
        axis.barh(y, ordered.mae, color=colors, alpha=0.9)
        axis.set_yticks(
            y, [f"{x.model_id}  ({x.training_scope})" for x in ordered.itertuples()]
        )
        axis.invert_yaxis()
        axis.set(title=dataset, xlabel="MAE (normalisierte RT)")
        axis.grid(axis="x", alpha=0.25)
        for index, value in enumerate(ordered.mae):
            axis.text(value, index, f"  {value:.5f}", va="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS.values()]
    fig.legend(
        handles,
        ["combined/aligned", "source-specific"],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    fig.suptitle("RT-Modellranking innerhalb desselben Testdatensatzes", y=1.01)
    _save(fig, plot_dir / "01_model_ranking_by_test_dataset")


def quality_plot(summary: pd.DataFrame, plot_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.6))
    for kind, group in summary.groupby("training_scope_kind", sort=True):
        axis.scatter(
            group.mae,
            group.pearson,
            s=64,
            alpha=0.9,
            label=kind.replace("_", " "),
            color=COLORS.get(kind, "#7570b3"),
        )
        for row in group.itertuples():
            axis.annotate(
                row.model_id,
                (row.mae, row.pearson),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    axis.set(
        xlabel="MAE (normalisierte RT; kleiner ist besser)",
        ylabel="Pearson r (größer ist besser)",
        title="Genauigkeit der abgeschlossenen Evaluierungen",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    _save(fig, plot_dir / "02_mae_vs_pearson")


def dimethyl_reliance_plot(summary: pd.DataFrame, plot_dir: Path) -> None:
    usable = summary.dropna(subset=["mean_dimethyl_attribution_share_modified"])
    if usable.empty:
        return
    groups = list(usable.groupby("dataset_name", sort=True))
    fig, axes = plt.subplots(
        len(groups), 1, figsize=(10, max(3.4, 0.45 * len(usable) + 1.8)), squeeze=False
    )
    for axis, (dataset, group) in zip(axes[:, 0], groups):
        ordered = group.sort_values("mean_dimethyl_attribution_share_modified")
        y = np.arange(len(ordered))
        colors = [COLORS.get(value, "#7570b3") for value in ordered.training_scope_kind]
        axis.barh(y, ordered.mean_dimethyl_attribution_share_modified, color=colors)
        axis.set_yticks(
            y,
            [f"{row.model_id} ({row.training_scope})" for row in ordered.itertuples()],
        )
        axis.set(
            title=dataset,
            xlabel="Mean share of |attribution| assigned to dimethyl tokens",
        )
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlim(left=0)
    fig.suptitle(
        "Model reliance on heavy-dimethyl tokens (modified peptides only)", y=1.01
    )
    _save(fig, plot_dir / "03_dimethyl_attribution_share_by_model")


def write_report(summary: pd.DataFrame, output: Path) -> None:
    display = summary.sort_values(["dataset_name", "mae"]).copy()
    display["mae"] = display.mae.map("{:.6f}".format)
    display["rmse"] = display.rmse.map("{:.6f}".format)
    display["pearson"] = display.pearson.map("{:.6f}".format)
    columns = [
        "dataset_name",
        "model_id",
        "training_scope",
        "alignment_target",
        "mae",
        "rmse",
        "pearson",
        "r2",
        "n",
        "n_attributed_peptides",
        "mean_dimethyl_attribution_share_modified",
    ]
    table = (
        display[columns]
        .rename(columns={"n": "peptides", "n_attributed_peptides": "Captum peptides"})
        .to_html(index=False, escape=True)
    )
    (output / "README.md").write_text(
        "# RT evaluation catalog report\n\n"
        "This report ranks models **only within the same held-out test table**; "
        "metrics on differently aligned RT scales must not be pooled.\n\n"
        "- [Model ranking](plots/01_model_ranking_by_test_dataset.pdf)\n"
        "- [MAE versus Pearson](plots/02_mae_vs_pearson.pdf)\n\n"
        "- [Dimethyl attribution share](plots/03_dimethyl_attribution_share_by_model.pdf)\n\n"
        "`n_attributed_peptides` is a live progress indicator: it may be below `peptides` while Captum is still running.\n",
        encoding="utf-8",
    )
    (output / "report.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>RT evaluation catalog</title>"
        "<style>body{font-family:system-ui;margin:2rem;max-width:1200px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:.45rem;text-align:left}th{background:#2878b5;color:#fff}tr:nth-child(even){background:#f6f8fa}img{max-width:100%;margin:1rem 0}</style>"
        "</head><body><h1>RT evaluation catalog</h1>"
        "<p>Rankings are valid only within an identical held-out test table and RT alignment scale.</p>"
        "<img src='plots/01_model_ranking_by_test_dataset.png'><img src='plots/02_mae_vs_pearson.png'><img src='plots/03_dimethyl_attribution_share_by_model.png'><h2>Metrics</h2>"
        + table
        + "</body></html>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    output = args.root / "catalog_summary"
    plot_dir = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    summary = collect(args.root)
    if summary.empty:
        raise SystemExit("No completed enabled evaluations found.")
    summary.sort_values(["dataset_name", "mae"]).to_parquet(
        output / "evaluation_summary.parquet", index=False
    )
    summary.sort_values(["dataset_name", "mae"]).to_csv(
        output / "evaluation_summary.csv", index=False
    )
    ranking_plot(summary, plot_dir)
    quality_plot(summary, plot_dir)
    dimethyl_reliance_plot(summary, plot_dir)
    write_report(summary, output)
    print(output)


if __name__ == "__main__":
    main()
