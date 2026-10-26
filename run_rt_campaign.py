#!/usr/bin/env python3
"""Run a resumable sequence of DIA-NN RT fine-tuning experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from diann_finetune.diagnostics import generate_campaign_diagnostics
from diann_finetune.catalog import DEFAULT_CATALOG, resolve_catalog_build


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "rt_finetune_campaign.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modular DIA-NN RT fine-tuning campaign")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--in-dir", type=Path, help="Carafe output or completed catalog build")
    source.add_argument("--dataset-id", help="Build ID from datasets.yaml; created automatically if absent")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--model-dir", type=Path, required=True, help="DIA-NN models directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Campaign output directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--only", nargs="*", help="Run only named experiments")
    parser.add_argument("--force", action="store_true", help="Rerun completed experiments")
    parser.add_argument("--max-rows", type=int, help="Development-only data limit")
    parser.add_argument("--wandb-config", type=Path, help="Enable W&B for every experiment")
    parser.add_argument("--wandb-group", help="Shared W&B group (default: output directory name)")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Override every experiment to 1 epoch and at most 256 rows",
    )
    return parser.parse_args()


def _command(
    args: argparse.Namespace,
    defaults: Dict[str, object],
    experiment: Dict[str, object],
    run_dir: Path,
) -> List[str]:
    settings = {**defaults, **experiment}
    epochs = 1 if args.smoke else int(settings["epochs"])
    warmup = 0 if args.smoke else int(settings["warmup_epochs"])
    command = [
        sys.executable,
        str(ROOT / "train_carafe_rt.py"),
        "--in-dir",
        str(args.in_dir),
        "--model-dir",
        str(args.model_dir),
        "--output-dir",
        str(run_dir),
        "--device",
        args.device,
        "--trainable",
        str(settings["trainable"]),
        "--epochs",
        str(epochs),
        "--warmup-epochs",
        str(warmup),
        "--batch-size",
        str(settings["batch_size"]),
        "--learning-rate",
        str(settings["learning_rate"]),
        "--weight-decay",
        str(settings["weight_decay"]),
        "--seed",
        str(settings["seed"]),
        "--loss",
        str(settings.get("loss", "mae")),
        "--huber-delta",
        str(settings.get("huber_delta", 0.03)),
        "--heavy-nterm-weight",
        str(settings.get("heavy_nterm_weight", 1.0)),
        "--heavy-lysine-weight",
        str(settings.get("heavy_lysine_weight", 1.0)),
    ]
    if settings.get("heavy_both_weight") is not None:
        command.extend(["--heavy-both-weight", str(settings["heavy_both_weight"])])
    if settings.get("backbone_learning_rate") is not None:
        command.extend(["--backbone-learning-rate", str(settings["backbone_learning_rate"])])
    if settings.get("embedding_learning_rate") is not None:
        command.extend(["--embedding-learning-rate", str(settings["embedding_learning_rate"])])
    if args.wandb_config is not None:
        command.extend([
            "--wandb-config", str(args.wandb_config),
            "--wandb-run-name", str(experiment["name"]),
            "--wandb-group", args.wandb_group or args.output_dir.name,
        ])
    max_rows = 256 if args.smoke else args.max_rows
    if max_rows is not None:
        command.extend(["--max-rows", str(max_rows)])
    return command


def _run_and_tee(command: List[str], log_path: Path) -> None:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.dataset_id:
        args.in_dir = resolve_catalog_build(args.dataset_id, args.catalog)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiments = config["experiments"]
    if args.only:
        requested = set(args.only)
        experiments = [experiment for experiment in experiments if experiment["name"] in requested]
        missing = requested - {experiment["name"] for experiment in experiments}
        if missing:
            raise SystemExit(f"Unknown experiment names: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": str(args.config.resolve()),
        "input": str(args.in_dir.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "device": args.device,
        "smoke": args.smoke,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": [],
    }
    manifest_path = args.output_dir / "campaign_manifest.json"
    expected_split_hashes = None

    for position, experiment in enumerate(experiments, start=1):
        name = experiment["name"]
        run_dir = args.output_dir / name
        metadata_path = run_dir / "run_metadata.json"
        if metadata_path.exists() and not args.force:
            print(f"[{position}/{len(experiments)}] {name}: already complete, skipping")
            run_record = {"name": name, "status": "reused"}
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            command = _command(args, config["defaults"], experiment, run_dir)
            print(f"[{position}/{len(experiments)}] {name}: {experiment['description']}")
            print("command:", " ".join(command))
            run_started = time.monotonic()
            try:
                _run_and_tee(command, run_dir / "training.log")
            except Exception:
                manifest["runs"].append({"name": name, "status": "failed"})
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                raise
            run_record = {
                "name": name,
                "status": "complete",
                "elapsed_seconds": time.monotonic() - run_started,
            }
        split_hashes = {
            "train": _sha256(run_dir / "train_split.tsv"),
            "validation": _sha256(run_dir / "validation_split.tsv"),
        }
        if expected_split_hashes is None:
            expected_split_hashes = split_hashes
        elif split_hashes != expected_split_hashes:
            raise RuntimeError(f"Experiment {name} did not use the same train/validation split")
        run_record["split_sha256"] = split_hashes
        manifest["runs"].append(run_record)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    names = [experiment["name"] for experiment in experiments]
    report = generate_campaign_diagnostics(args.output_dir, names)
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["report"] = str(report)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Campaign complete: {report}")


if __name__ == "__main__":
    main()
