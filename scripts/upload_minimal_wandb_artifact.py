#!/usr/bin/env python3
"""Upload only the small provenance files for a selected DIA-NN RT run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WANDB_CONFIG = ROOT / "configs" / "wandb.json"
DEFAULT_FILES = (
    "run_metadata.json",
    "command.json",
    "sweep_id.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a minimal W&B artifact for one selected run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed seed/run directory")
    parser.add_argument("--wandb-config", type=Path, default=DEFAULT_WANDB_CONFIG)
    parser.add_argument("--artifact-name", help="Default: <run-dir-name>-minimal-summary")
    parser.add_argument("--alias", action="append", default=["best"], help="Artifact alias; can be repeated")
    parser.add_argument("--extra-file", type=Path, action="append", default=[], help="Additional small file to include")
    parser.add_argument("--run-name", help="W&B run name for the upload event")
    parser.add_argument("--group", help="W&B group for the upload event")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "minimal-run-summary"


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def discover_files(run_dir: Path, wandb_config: Path, extra_files: list[Path]) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for relative in DEFAULT_FILES:
        path = run_dir / relative
        if path.is_file():
            files.append((path, relative))

    command_path = run_dir / "command.json"
    if command_path.is_file():
        command = json.loads(command_path.read_text(encoding="utf-8"))
        values = command.get("command", [])
        if isinstance(values, list) and "--wandb-config" in values:
            index = values.index("--wandb-config")
            if index + 1 < len(values):
                configured = Path(str(values[index + 1]))
                if configured.is_file():
                    files.append((configured, "wandb_config.json"))

    parent_sweep = first_existing([parent / "sweep_id.txt" for parent in [run_dir, *run_dir.parents]])
    if parent_sweep is not None:
        files.append((parent_sweep, "sweep_id.txt"))

    if wandb_config.is_file():
        files.append((wandb_config, "upload_wandb_config.json"))
    for path in extra_files:
        if not path.is_file():
            raise SystemExit(f"Extra file does not exist: {path}")
        files.append((path, f"extra/{path.name}"))

    deduped: list[tuple[Path, str]] = []
    seen_paths: set[Path] = set()
    for path, artifact_name in files:
        resolved = path.resolve()
        if resolved not in seen_paths:
            deduped.append((path, artifact_name))
            seen_paths.add(resolved)
    return deduped


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    if not args.wandb_config.is_file():
        raise SystemExit(f"W&B config does not exist: {args.wandb_config}")

    files = discover_files(run_dir, args.wandb_config, args.extra_file)
    if not files:
        raise SystemExit(f"No minimal artifact files found in {run_dir}")
    artifact_name = clean_name(args.artifact_name or f"{run_dir.name}-minimal-summary")
    print(f"artifact={artifact_name}")
    for path, artifact_path in files:
        print(f"{artifact_path}: {path}")
    if args.dry_run:
        return

    import wandb

    settings = json.loads(args.wandb_config.read_text(encoding="utf-8"))
    run = wandb.init(
        project=settings.get("project", "diann-rt-finetune"),
        entity=settings.get("entity"),
        mode=settings.get("mode", "online"),
        name=args.run_name or f"upload-{artifact_name}",
        group=args.group,
        job_type="minimal-artifact-upload",
        config={"source_run_dir": str(run_dir), "artifact_files": [name for _, name in files]},
    )
    artifact = wandb.Artifact(artifact_name, type="run-summary")
    for path, artifact_path in files:
        artifact.add_file(str(path), name=artifact_path)
    run.log_artifact(artifact, aliases=args.alias)
    run.finish()


if __name__ == "__main__":
    main()
