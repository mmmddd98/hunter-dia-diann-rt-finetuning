#!/usr/bin/env python3
"""Create a W&B sweep and run complete trials within a wall-time budget."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SWEEP_CONFIG = ROOT / "configs" / "rt_wandb_sweep.json"
DEFAULT_WANDB_CONFIG = ROOT / "configs" / "wandb.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a time-budgeted DIA-NN RT W&B sweep")
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--max-runs", type=int, default=1000)
    parser.add_argument("--sweep-config", type=Path, default=DEFAULT_SWEEP_CONFIG)
    parser.add_argument("--wandb-config", type=Path, default=DEFAULT_WANDB_CONFIG)
    parser.add_argument("--project", help="Override project from wandb.json")
    parser.add_argument("--entity", help="Override entity from wandb.json")
    parser.add_argument("--group", default="rt-hyperparameter-sweep")
    parser.add_argument("--sweep-id", help="Continue an existing sweep instead of creating one")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print without contacting W&B")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> tuple[dict, dict]:
    if args.hours <= 0 or args.max_runs < 1:
        raise SystemExit("--hours and --max-runs must be positive")
    rt_data = args.in_dir / "rt_train_data.tsv"
    if not rt_data.is_file():
        raise SystemExit(f"Missing training data: {rt_data}")
    for filename in ("rt.d0.pt", "dict.txt"):
        path = args.model_dir / filename
        if not path.is_file():
            raise SystemExit(f"Missing DIA-NN model input: {path}")
    sweep = json.loads(args.sweep_config.read_text(encoding="utf-8"))
    wandb_settings = json.loads(args.wandb_config.read_text(encoding="utf-8"))
    if sweep.get("metric", {}).get("name") != "validation/mae":
        raise SystemExit("Sweep metric must be validation/mae")
    return sweep, wandb_settings


def main() -> None:
    args = parse_args()
    sweep_config, wandb_settings = validate(args)
    project = args.project or str(wandb_settings.get("project", "diann-rt-finetune"))
    entity = args.entity or wandb_settings.get("entity")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb_controller_dir = args.output_dir / "wandb_controller"
    wandb_controller_dir.mkdir(parents=True, exist_ok=True)
    # This process creates the sweep itself, so configure it before importing
    # W&B as well as passing the setting to the child agent processes below.
    os.environ["WANDB_DIR"] = str(wandb_controller_dir.resolve())
    environment = os.environ.copy()
    environment.update(
        {
            "DIANN_SWEEP_IN_DIR": str(args.in_dir.resolve()),
            "DIANN_SWEEP_MODEL_DIR": str(args.model_dir.resolve()),
            "DIANN_SWEEP_OUTPUT_DIR": str(args.output_dir.resolve()),
            "DIANN_SWEEP_WANDB_CONFIG": str(args.wandb_config.resolve()),
            "DIANN_SWEEP_GROUP": args.group,
            # Keep the controller/agent bookkeeping with the sweep, not in the
            # source checkout's local ./wandb directory.
            "WANDB_DIR": str(wandb_controller_dir.resolve()),
        }
    )
    print(f"project={project} entity={entity or '<default>'} budget={args.hours:g}h")
    print(f"outputs={args.output_dir.resolve()}")
    print(json.dumps(sweep_config["parameters"], indent=2))
    if args.dry_run:
        print("Dry run complete; no W&B sweep was created.")
        return

    try:
        import wandb
    except ImportError as error:
        raise SystemExit("Install W&B in the active environment before starting a sweep") from error
    resolved_entity = entity or getattr(wandb.Api(), "default_entity", None)
    if not resolved_entity:
        raise SystemExit("Could not determine W&B entity; set entity in configs/wandb.json")
    sweep_id = args.sweep_id or wandb.sweep(
        sweep=sweep_config, entity=resolved_entity, project=project
    )
    sweep_path = f"{resolved_entity}/{project}/{sweep_id}"
    (args.output_dir / "sweep_id.txt").write_text(sweep_path + "\n", encoding="utf-8")
    print(f"Sweep ID: {sweep_path}")

    deadline = time.monotonic() + args.hours * 3600
    completed = 0
    while completed < args.max_runs and time.monotonic() < deadline:
        remaining_minutes = (deadline - time.monotonic()) / 60
        print(f"Starting trial {completed + 1}; nominal budget remaining: {remaining_minutes:.1f} min")
        subprocess.run(
            [sys.executable, "-m", "wandb", "agent", "--count", "1", sweep_path],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        completed += 1
    print(
        f"Sweep agent stopped after {completed} complete trial(s). "
        "A final trial may extend beyond the nominal wall-time budget."
    )


if __name__ == "__main__":
    main()
