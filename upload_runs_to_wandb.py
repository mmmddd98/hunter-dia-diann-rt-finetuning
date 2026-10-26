#!/usr/bin/env python3
"""Upload already completed local DIA-NN RT runs to W&B without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from diann_finetune.tracking import WandbTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload completed DIA-NN RT runs to W&B")
    parser.add_argument("--runs-dir", type=Path, required=True, help="Campaign or output directory")
    parser.add_argument("--wandb-config", type=Path, required=True)
    parser.add_argument("--wandb-group", help="Shared group (default: runs directory name)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_paths = sorted(args.runs_dir.glob("**/run_metadata.json"))
    if not metadata_paths:
        raise SystemExit(f"No completed run_metadata.json files found below {args.runs_dir}")
    group = args.wandb_group or args.runs_dir.name
    for position, metadata_path in enumerate(metadata_paths, start=1):
        run_dir = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"[{position}/{len(metadata_paths)}] uploading {run_dir.name}")
        tracker = WandbTracker(
            args.wandb_config,
            run_name=run_dir.name,
            group=group,
            run_config={**metadata.get("parameters", {}), **metadata.get("split", {})},
            output_dir=run_dir,
        )
        history_path = run_dir / "training_history.csv"
        if history_path.exists():
            for record in pd.read_csv(history_path).to_dict(orient="records"):
                tracker.log_epoch(record)
        tracker.finish(metadata, run_dir)
    print(f"Uploaded {len(metadata_paths)} completed runs to W&B group {group!r}")


if __name__ == "__main__":
    main()
