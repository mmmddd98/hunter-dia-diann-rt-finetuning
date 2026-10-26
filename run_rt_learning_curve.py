#!/usr/bin/env python3
"""Run a resumable DIA-NN RT learning curve on nested training subsets."""

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

from diann_finetune.catalog import DEFAULT_CATALOG, resolve_catalog_build


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "rt_learning_curve.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DIA-NN RT data-size learning curve")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--in-dir", type=Path, help="Carafe output or completed catalog build")
    source.add_argument("--dataset-id", help="Build ID from datasets.yaml; created automatically if absent")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--model-dir", type=Path, required=True, help="DIA-NN models directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Learning-curve output directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true", help="Rerun completed fractions")
    parser.add_argument("--max-rows", type=int, help="Development-only input row limit")
    parser.add_argument("--smoke", action="store_true", help="Use one epoch and at most 256 input rows")
    parser.add_argument("--wandb-config", type=Path, help="Enable W&B for every fraction")
    parser.add_argument("--wandb-group", help="Shared W&B group (default: output directory name)")
    return parser.parse_args()


def fraction_name(fraction: float) -> str:
    return f"train_{round(fraction * 100):03d}pct"


def build_command(
    args: argparse.Namespace, defaults: Dict[str, object], fraction: float, run_dir: Path
) -> List[str]:
    epochs = 1 if args.smoke else int(defaults["epochs"])
    warmup = 0 if args.smoke else int(defaults["warmup_epochs"])
    command = [
        sys.executable,
        str(ROOT / "train_carafe_rt.py"),
        "--in-dir", str(args.in_dir),
        "--model-dir", str(args.model_dir),
        "--output-dir", str(run_dir),
        "--device", args.device,
        "--trainable", str(defaults["trainable"]),
        "--epochs", str(epochs),
        "--warmup-epochs", str(warmup),
        "--batch-size", str(defaults["batch_size"]),
        "--learning-rate", str(defaults["learning_rate"]),
        "--weight-decay", str(defaults["weight_decay"]),
        "--seed", str(defaults["seed"]),
        "--train-fraction", str(fraction),
        "--loss", str(defaults.get("loss", "mae")),
        "--huber-delta", str(defaults.get("huber_delta", 0.03)),
        "--heavy-nterm-weight", str(defaults.get("heavy_nterm_weight", 1.0)),
        "--heavy-lysine-weight", str(defaults.get("heavy_lysine_weight", 1.0)),
    ]
    if defaults.get("heavy_both_weight") is not None:
        command.extend(["--heavy-both-weight", str(defaults["heavy_both_weight"])])
    if defaults.get("backbone_learning_rate") is not None:
        command.extend(["--backbone-learning-rate", str(defaults["backbone_learning_rate"])])
    if defaults.get("embedding_learning_rate") is not None:
        command.extend(["--embedding-learning-rate", str(defaults["embedding_learning_rate"])])
    if args.wandb_config is not None:
        command.extend([
            "--wandb-config", str(args.wandb_config),
            "--wandb-run-name", fraction_name(fraction),
            "--wandb-group", args.wandb_group or args.output_dir.name,
        ])
    max_rows = 256 if args.smoke else args.max_rows
    if max_rows is not None:
        command.extend(["--max-rows", str(max_rows)])
    return command


def run_and_tee(command: List[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(output_dir: Path, records: List[Dict[str, object]]) -> Path:
    summary = pd.DataFrame(records).sort_values("train_fraction")
    summary_path = output_dir / "learning_curve_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "learning_curve.png"

    fig, left = plt.subplots(figsize=(8, 5))
    x = summary["training_rows"]
    left.plot(x, summary["validation_mae"], marker="o", label="Validation MAE")
    left.plot(x, summary["validation_rmse"], marker="o", label="Validation RMSE")
    left.set(xlabel="Training rows", ylabel="Error (lower is better)", title="DIA-NN RT learning curve")
    left.grid(alpha=0.25)
    right = left.twinx()
    right.plot(x, summary["validation_r2"], marker="s", color="tab:green", label="Validation R²")
    right.set_ylabel("R² (higher is better)")
    lines = left.get_lines() + right.get_lines()
    left.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    report_path = output_dir / "learning_curve_report.html"
    report_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>DIA-NN RT learning curve</title>"
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:.35rem}</style></head><body>"
        "<h1>DIA-NN RT learning curve</h1><p>All runs use nested training subsets and the same "
        "sequence-disjoint validation split.</p>"
        + summary.to_html(index=False)
        + "<h2>Learning curve</h2><img src='plots/learning_curve.png' style='max-width:100%'>"
        + "</body></html>",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    args = parse_args()
    if args.dataset_id:
        args.in_dir = resolve_catalog_build(args.dataset_id, args.catalog)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fractions = [float(value) for value in config["fractions"]]
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise SystemExit("Every configured fraction must be in (0, 1]")
    if fractions != sorted(set(fractions)):
        raise SystemExit("Fractions must be unique and sorted in ascending order")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, object] = {
        "config": str(args.config.resolve()),
        "input": str(args.in_dir.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": [],
    }
    manifest_path = args.output_dir / "learning_curve_manifest.json"
    validation_hash = None
    records: List[Dict[str, object]] = []

    for position, fraction in enumerate(fractions, start=1):
        name = fraction_name(fraction)
        run_dir = args.output_dir / name
        metadata_path = run_dir / "run_metadata.json"
        if metadata_path.exists() and not args.force:
            print(f"[{position}/{len(fractions)}] {name}: already complete, skipping")
            status = "reused"
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            command = build_command(args, config["defaults"], fraction, run_dir)
            print(f"[{position}/{len(fractions)}] {name}: training on {fraction:.0%} of the fixed split")
            print("command:", " ".join(command))
            run_and_tee(command, run_dir / "training.log")
            status = "complete"

        current_validation_hash = sha256(run_dir / "validation_split.tsv")
        if validation_hash is None:
            validation_hash = current_validation_hash
        elif current_validation_hash != validation_hash:
            raise RuntimeError(f"{name} did not use the same validation split")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics = metadata["result"]["finetuned_metrics"]
        split = metadata["split"]
        records.append({
            "experiment": name,
            "train_fraction": fraction,
            "training_rows": split["used_training_rows"],
            "training_unique_sequences": split["used_training_unique_sequences"],
            "validation_rows": split["validation_rows"],
            "best_epoch": metadata["result"]["best_epoch"],
            **{f"validation_{key}": value for key, value in metrics.items()},
        })
        manifest["runs"].append({
            "name": name,
            "fraction": fraction,
            "status": status,
            "validation_sha256": current_validation_hash,
        })
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = write_report(args.output_dir, records)
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["report"] = str(report)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Learning curve complete: {report}")


if __name__ == "__main__":
    main()
