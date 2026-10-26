#!/usr/bin/env python3
"""Repeated, sequence-disjoint data-scaling experiment for DIA-NN RT fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from diann_finetune.data import (
    load_carafe_rt,
    nested_stratified_sequence_subset,
    stratified_sequence_split,
)
from diann_finetune.catalog import DEFAULT_CATALOG, resolve_catalog_build


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "rt_data_scaling.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure DIA-NN RT performance as a function of training-data size"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--in-dir", type=Path, help="Carafe output or completed catalog build")
    source.add_argument("--dataset-id", help="Build ID from datasets.yaml; created automatically if absent")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--model-dir", type=Path, required=True, help="DIA-NN models directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repeats", type=int, help="Override configured repeat count (recommend 10 or 15)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true", help="Rerun completed individual trainings")
    parser.add_argument("--smoke", action="store_true", help="One repeat, one epoch; only for checking setup")
    parser.add_argument("--max-rows", type=int, help="Development-only source row limit")
    parser.add_argument("--wandb-config", type=Path, help="Enable W&B tracking for every run")
    parser.add_argument("--wandb-group", help="W&B group (default: output directory name)")
    return parser.parse_args()


def run_name(repeat: int, fraction: float) -> str:
    return f"repeat_{repeat:02d}/train_{round(fraction * 100):03d}pct"


def run_and_tee(command: List[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_command(
    args: argparse.Namespace, defaults: Dict[str, object], train_path: Path,
    validation_path: Path, run_dir: Path, repeat: int, fraction: float,
) -> List[str]:
    epochs = 1 if args.smoke else int(defaults["epochs"])
    warmup = 0 if args.smoke else int(defaults["warmup_epochs"])
    command = [
        sys.executable, str(ROOT / "train_carafe_rt.py"),
        "--train-split", str(train_path), "--validation-split", str(validation_path),
        "--model-dir", str(args.model_dir), "--output-dir", str(run_dir),
        "--device", args.device, "--trainable", str(defaults["trainable"]),
        "--epochs", str(epochs), "--warmup-epochs", str(warmup),
        "--batch-size", str(defaults["batch_size"]),
        "--learning-rate", str(defaults["learning_rate"]),
        "--weight-decay", str(defaults["weight_decay"]), "--seed", str(repeat),
        "--loss", str(defaults["loss"]), "--huber-delta", str(defaults["huber_delta"]),
        "--heavy-nterm-weight", str(defaults.get("heavy_nterm_weight", 1.0)),
        "--heavy-lysine-weight", str(defaults.get("heavy_lysine_weight", 1.0)),
    ]
    if defaults.get("heavy_both_weight") is not None:
        command.extend(["--heavy-both-weight", str(defaults["heavy_both_weight"])])
    if args.wandb_config:
        command.extend([
            "--wandb-config", str(args.wandb_config),
            "--wandb-run-name", run_name(repeat, fraction).replace("/", "_"),
            "--wandb-group", args.wandb_group or args.output_dir.name,
        ])
    return command


def write_report(output_dir: Path, records: List[Dict[str, object]], config: Dict[str, object]) -> Path:
    raw = pd.DataFrame(records).sort_values(["train_fraction", "repeat"])
    raw.to_csv(output_dir / "data_scaling_runs.csv", index=False)
    metric_columns = ["validation_mae", "validation_rmse", "validation_r2", "validation_pearson"]
    aggregate = raw.groupby("train_fraction", as_index=False).agg(
        training_rows_mean=("training_rows", "mean"),
        training_unique_sequences_mean=("training_unique_sequences", "mean"),
        repeats=("repeat", "count"),
        **{f"{metric}_mean": (metric, "mean") for metric in metric_columns},
        **{f"{metric}_std": (metric, "std") for metric in metric_columns},
    )
    for metric in metric_columns:
        aggregate[f"{metric}_sem"] = aggregate[f"{metric}_std"] / aggregate["repeats"].pow(0.5)
        aggregate[f"{metric}_ci95"] = 1.96 * aggregate[f"{metric}_sem"]
    aggregate.to_csv(output_dir / "data_scaling_summary.csv", index=False)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = aggregate["training_unique_sequences_mean"]
    axes[0].errorbar(x, aggregate["validation_mae_mean"], yerr=aggregate["validation_mae_ci95"], marker="o", capsize=4, label="MAE")
    axes[0].errorbar(x, aggregate["validation_rmse_mean"], yerr=aggregate["validation_rmse_ci95"], marker="o", capsize=4, label="RMSE")
    axes[0].set(xlabel="Training sequences", ylabel="Validation error", title="Data scaling: error (95% CI)")
    axes[0].grid(alpha=.25); axes[0].legend()
    axes[1].errorbar(x, aggregate["validation_r2_mean"], yerr=aggregate["validation_r2_ci95"], marker="o", capsize=4, color="tab:green")
    axes[1].set(xlabel="Training sequences", ylabel="Validation R²", title="Data scaling: explained variance (95% CI)")
    axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(plot_dir / "data_scaling_curve.png", dpi=180); plt.close(fig)

    full = aggregate.iloc[-1]
    prior = aggregate.iloc[-2] if len(aggregate) > 1 else full
    mae_delta = float(prior["validation_mae_mean"] - full["validation_mae_mean"])
    decision = (
        "Die Kurve verbessert sich noch von 75% auf 100%; zusätzliche Daten sind wahrscheinlich nützlich. "
        "Bestätigt ist das erst, wenn die 95%-Intervalle und die Wiederholungen konsistent sind."
        if mae_delta > 0 else
        "Zwischen 75% und 100% zeigt sich im Mittel keine weitere MAE-Verbesserung; priorisiere gezieltere "
        "Heavy-Dimethyl-Daten oder Modell-/Loss-Änderungen statt nur mehr Gesamtmenge."
    )
    readme = f"""# DIA-NN RT Data-Scaling-Experiment

## Ziel

Wir messen mit einer **eingefrorenen** Modellkonfiguration, ob mehr Carafe2-Trainingsdaten die RT-Genauigkeit weiter verbessern. Es wird keine Hyperparameter-Suche durchgeführt; damit ist die Datenmenge die zentrale veränderte Größe.

## Design

- Trainingsanteile: {', '.join(f'{value:.0%}' for value in config['fractions'])}
- Wiederholungen: {int(aggregate['repeats'].max())}
- Pro Wiederholung: stratifizierter, sequenz-disjunkter 90/10-Train/Test-Split.
- Innerhalb einer Wiederholung sind die kleineren Trainingsmengen verschachtelte, stratifizierte **Sequenz**-Teilmengen der 90%-Trainingsmenge.
- Strata: keine schwere Dimethylierung, schwere N-terminale Dimethylierung, schwere Lysin-Dimethylierung, beide schweren Stellen. Ein Peptid kann deshalb nicht gleichzeitig in Training und Test vorkommen.
- Konfiguration: `all` layers, Huber (`delta=0.03`), LR `1.13155e-4`, Batch 512, 40 Epochen, 5 Warm-up-Epochen, keine Heavy-Gewichtung. Dies ist die robuste Huber-Kandidatin aus dem Zufalls-Sweep.

## Ergebnis lesen

`data_scaling_summary.csv` enthält Mittelwert, Standardabweichung und 95%-Intervall über die Wiederholungen. `plots/data_scaling_curve.png` zeigt dieselben Werte. Entscheidend ist primär Validation-MAE; RMSE und R² sind ergänzende Checks.

Vergleich 75% → 100%: mittlere MAE-Änderung = `{mae_delta:+.6f}` (positiv heißt: 100% ist besser).

**Vorläufige Einordnung:** {decision}

## Wichtige Einschränkung

Diese Wiederholungen beantworten die Frage nach der *Dateneffizienz* robust. Sie sind aber kein endgültiger, unabhängiger Leistungsnachweis: Die bisherigen Daten und die Modellkonfiguration wurden bereits im Sweep verwendet. Für eine finale Aussage brauchen wir später neue, bislang unberührte Runs/Peptide als externen Test.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    report = output_dir / "data_scaling_report.html"
    report.write_text(
        "<!doctype html><meta charset='utf-8'><title>DIA-NN RT data scaling</title>"
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:.35rem}</style>"
        "<h1>DIA-NN RT data-scaling experiment</h1><p>Repeated, stratified, sequence-disjoint splits; error bars are 95% normal-approximation intervals across repeats.</p>"
        + aggregate.to_html(index=False, float_format=lambda value: f"{value:.6f}")
        + "<h2>Learning curve</h2><img src='plots/data_scaling_curve.png' style='max-width:100%'>"
        + "<p>See <code>README.md</code> for interpretation and limitations.</p>", encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    if args.dataset_id:
        args.in_dir = resolve_catalog_build(args.dataset_id, args.catalog)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fractions = [float(value) for value in config["fractions"]]
    repeats = 1 if args.smoke else (args.repeats or int(config["repeats"]))
    if repeats < 2 and not args.smoke:
        raise SystemExit("Use at least two repeats (10 is the recommended default)")
    if fractions != sorted(set(fractions)) or any(not 0 < value <= 1 for value in fractions):
        raise SystemExit("fractions must be unique, sorted values in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_rt = args.in_dir / "diann_rt_input" / "rt_train_data.tsv"
    source = catalog_rt if catalog_rt.is_file() else args.in_dir / "rt_train_data.tsv"
    prepared = load_carafe_rt(source, max_rows=256 if args.smoke else args.max_rows)
    records: List[Dict[str, object]] = []
    manifest: Dict[str, object] = {"config": config, "input": str(source.resolve()), "runs": []}
    manifest_path = args.output_dir / "data_scaling_manifest.json"
    total = repeats * len(fractions)
    ordinal = 0

    for repeat in range(1, repeats + 1):
        train_pool, validation, split_info = stratified_sequence_split(
            prepared.frame, validation_fraction=float(config.get("validation_fraction", .1)), random_state=repeat
        )
        split_dir = args.output_dir / "splits" / f"repeat_{repeat:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)
        validation_path = split_dir / "validation.tsv"
        validation.to_csv(validation_path, sep="\t", index=False)
        for fraction in fractions:
            ordinal += 1
            name = run_name(repeat, fraction)
            run_dir = args.output_dir / name
            metadata_path = run_dir / "run_metadata.json"
            train_path = split_dir / f"train_{round(fraction * 100):03d}pct.tsv"
            train = nested_stratified_sequence_subset(train_pool, fraction, random_state=repeat)
            train.to_csv(train_path, sep="\t", index=False)
            if metadata_path.exists() and not args.force:
                print(f"[{ordinal}/{total}] {name}: already complete, skipping")
                status = "reused"
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                command = build_command(args, config["defaults"], train_path, validation_path, run_dir, repeat, fraction)
                print(f"[{ordinal}/{total}] {name}: {fraction:.0%} of repeat-{repeat:02d} training sequences")
                print("command:", " ".join(command))
                run_and_tee(command, run_dir / "training.log")
                status = "complete"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metrics = metadata["result"]["finetuned_metrics"]
            split = metadata["split"]
            records.append({
                "repeat": repeat, "train_fraction": fraction, "run": name, "status": status,
                "training_rows": split["used_training_rows"],
                "training_unique_sequences": split["used_training_unique_sequences"],
                "validation_rows": split["validation_rows"],
                "validation_unique_sequences": split["validation_unique_sequences"],
                "validation_sha256": sha256(validation_path), "best_epoch": metadata["result"]["best_epoch"],
                **{f"validation_{key}": value for key, value in metrics.items()},
            })
            manifest["runs"].append({"run": name, "status": status, "split": split_info})
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = write_report(args.output_dir, records, config)
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["report"] = str(report)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Data-scaling experiment complete: {report}")


if __name__ == "__main__":
    main()
