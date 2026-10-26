#!/usr/bin/env python3
"""Run a time-budgeted, multi-seed DIA-NN RT W&B sweep."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train_carafe_rt.py"
EVALUATE_SCRIPT = ROOT / "evaluate_organism_tests.py"
DEFAULT_WANDB_CONFIG = ROOT / "configs" / "wandb.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a targeted multi-seed DIA-NN RT W&B sweep")
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-dir", type=Path,
        help="Optional directory containing combined.tsv plus celegans/triplemix/human.tsv",
    )
    parser.add_argument(
        "--rt-alignment", type=Path,
        help="Training-only RT alignment JSON (default: <validation-dir>/rt_alignment.json)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sweep-config", type=Path, required=True)
    parser.add_argument("--configuration-map", type=Path,
                        help="Optional JSON map from a sweep configuration name to fixed parameters")
    parser.add_argument("--wandb-config", type=Path, default=DEFAULT_WANDB_CONFIG)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", default="2024,2025,2026,2027,2028,2029,2030,2031,2032,2033")
    parser.add_argument("--max-runs", type=int, default=1000)
    parser.add_argument("--group")
    parser.add_argument("--sweep-id", help="Resume an existing W&B sweep")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_rt_data(in_dir: Path) -> Path:
    direct = in_dir / "rt_train_data.tsv"
    catalog = in_dir / "diann_rt_input" / "rt_train_data.tsv"
    if direct.is_file():
        return direct
    if catalog.is_file():
        return catalog
    raise SystemExit(f"Missing training data: {direct} (also checked {catalog})")


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise SystemExit("--seeds must be a comma-separated list of integers") from error
    if not seeds or len(seeds) != len(set(seeds)):
        raise SystemExit("--seeds must contain at least one unique integer")
    return seeds


def main() -> None:
    args = parse_args()
    if args.hours <= 0 or args.max_runs < 1:
        raise SystemExit("--hours and --max-runs must be positive")
    rt_data = resolve_rt_data(args.in_dir)
    validation_split = None
    rt_alignment = None
    if args.validation_dir:
        validation_split = args.validation_dir / "combined.tsv"
        validation_tests = [
            args.validation_dir / f"{name}.tsv"
            for name in ("celegans", "triplemix", "human")
        ]
        rt_alignment = args.rt_alignment or args.validation_dir / "rt_alignment.json"
        missing_validation = [
            str(path)
            for path in (validation_split, *validation_tests, rt_alignment)
            if not path.is_file()
        ]
        if missing_validation:
            raise SystemExit("Missing sweep validation input(s):\n" + "\n".join(missing_validation))
    elif args.rt_alignment:
        raise SystemExit("--rt-alignment requires --validation-dir")
    for filename in ("rt.d0.pt", "dict.txt"):
        if not (args.model_dir / filename).is_file():
            raise SystemExit(f"Missing DIA-NN model input: {args.model_dir / filename}")
    if not Path(args.python).is_file():
        raise SystemExit(f"Python interpreter does not exist: {args.python}")
    seeds = parse_seeds(args.seeds)
    sweep_definition = json.loads(args.sweep_config.read_text(encoding="utf-8"))
    configuration_map = {}
    if args.configuration_map:
        configuration_map = json.loads(args.configuration_map.read_text(encoding="utf-8"))
    wandb_settings = json.loads(args.wandb_config.read_text(encoding="utf-8"))
    expected_metric = "final/rt/validation_composite_mae"
    if sweep_definition.get("metric", {}).get("name") == expected_metric and not args.validation_dir:
        raise SystemExit(f"Sweep metric {expected_metric} requires --validation-dir")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    controller_dir = args.output_dir / "wandb_controller"
    controller_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(controller_dir.resolve())
    project = str(wandb_settings.get("project", "diann-rt-finetune"))
    entity = wandb_settings.get("entity")
    group = args.group or args.output_dir.name
    print(f"input={rt_data.resolve()}")
    print(f"model_dir={args.model_dir.resolve()}")
    print(f"output={args.output_dir.resolve()}")
    print(f"budget={args.hours:g}h seeds={','.join(map(str, seeds))}")
    print(json.dumps(sweep_definition["parameters"], indent=2))
    if args.dry_run:
        print("Dry run complete; no W&B sweep was created.")
        return

    try:
        import wandb
    except ImportError as error:
        raise SystemExit(f"W&B is not installed in {args.python}") from error
    try:
        from wandb.errors import CommError
    except ImportError:  # W&B < 0.19
        from wandb.errors.errors import CommError
    resolved_entity = entity or getattr(wandb.Api(), "default_entity", None)
    if not resolved_entity:
        raise SystemExit("Could not determine W&B entity; set entity in configs/wandb.json")
    supplied_sweep_id = args.sweep_id or wandb.sweep(
        sweep_definition, project=project, entity=resolved_entity
    )
    sweep_id = supplied_sweep_id.rsplit("/", 1)[-1]
    sweep_path = (
        supplied_sweep_id if supplied_sweep_id.count("/") == 2
        else f"{resolved_entity}/{project}/{sweep_id}"
    )
    (args.output_dir / "sweep_id.txt").write_text(sweep_path + "\n", encoding="utf-8")
    deadline = time.monotonic() + args.hours * 3600
    completed = 0

    def run_seed(config: dict, parent_id: str, parent_name: str, seed: int) -> dict:
        run_dir = args.output_dir / f"trial.{parent_id}" / f"seed.{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        child_id = wandb.util.generate_id()
        child_name = f"{parent_name}.seed{seed}"
        command = [
            args.python,
            "-u",
            str(TRAIN_SCRIPT),
            "--train-split",
            str(rt_data),
            "--model-dir",
            str(args.model_dir),
            "--output-dir",
            str(run_dir),
            "--device",
            "auto",
            "--seed",
            str(seed),
            "--wandb-config",
            str(args.wandb_config),
            "--wandb-run-id",
            child_id,
            "--wandb-run-name",
            child_name,
            "--wandb-group",
            group,
        ]
        if validation_split is not None:
            command.extend(["--validation-split", str(validation_split)])
        options = (
            "epochs", "warmup-epochs", "batch-size", "learning-rate",
            "backbone-learning-rate", "embedding-learning-rate", "weight-decay",
            "loss", "huber-delta", "trainable", "heavy-nterm-weight",
            "heavy-lysine-weight", "heavy-both-weight", "heavy-nterm-oversample",
            "heavy-lysine-oversample", "heavy-both-oversample",
        )
        if "backbone-lr-ratio" in config and "backbone-learning-rate" not in config:
            config["backbone-learning-rate"] = float(config["learning-rate"]) * float(config["backbone-lr-ratio"])
        for option in options:
            if option in config:
                command.extend([f"--{option}", str(config[option])])
        (run_dir / "command.json").write_text(
            json.dumps({"wandb_config": config, "seed": seed, "command": command}, indent=2) + "\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        for key in (
            "WANDB_SWEEP_ID", "WANDB_RUN_ID", "WANDB_RUN_GROUP", "WANDB_CONFIG",
            "WANDB_NAME", "WANDB_PROJECT", "WANDB_ENTITY",
        ):
            environment.pop(key, None)
        environment["WANDB_DIR"] = str((run_dir / "wandb").resolve())
        print(f"[{completed + 1}; seed {seed}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"Missing run metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result = {**metadata["result"]["finetuned_metrics"]}
        if args.validation_dir is None:
            return result
        subprocess.run([
            args.python, str(EVALUATE_SCRIPT), "--run-dir", str(run_dir),
            "--tests-dir", str(args.validation_dir), "--device", "auto",
            "--rt-alignment", str(rt_alignment),
        ], cwd=ROOT, env=environment, check=True)
        shutil.copy2(run_dir / "organism_test_metrics.json", run_dir / "validation_metrics.json")
        organism_metrics = json.loads((run_dir / "organism_test_metrics.json").read_text(encoding="utf-8"))
        components = []
        for organism in ("celegans", "triplemix", "human"):
            item = organism_metrics[organism]
            components.append(
                0.4 * float(item["mae"])
                + 0.3 * float(item["heavy_dimethyl_mae"])
                + 0.3 * float(item["heavy_nterm_dimethyl_mae"])
            )
        return {
            **result,
            "validation_composite_mae": statistics.fmean(components),
            "organism_metrics": organism_metrics,
        }

    def trial() -> None:
        nonlocal completed
        parent = wandb.init()
        config = dict(parent.config)
        selected_configuration = config.get("configuration")
        if selected_configuration:
            if selected_configuration not in configuration_map:
                raise RuntimeError(f"Unknown mapped configuration: {selected_configuration}")
            config.update(configuration_map[selected_configuration])
        parent_id = parent.id
        parent_name = parent.name or parent_id
        parent.finish()
        try:
            seed_metrics = [
                run_seed(config, parent_id, parent_name, seed) for seed in seeds
            ]
            aggregate: dict[str, float] = {}
            metrics = (
                "mae", "rmse", "r2", "heavy_dimethyl_mae", "heavy_nterm_dimethyl_mae",
                "heavy_nterm_mae", "heavy_lysine_mae", "heavy_both_mae", "normal_mae",
                "validation_composite_mae",
            )
            for metric in metrics:
                values = [float(item[metric]) for item in seed_metrics if metric in item]
                if values:
                    aggregate[f"final/rt/mean_seed_{metric}"] = statistics.fmean(values)
                    if len(values) > 1:
                        aggregate[f"final/rt/std_seed_{metric}"] = statistics.stdev(values)
            composite_key = "final/rt/mean_seed_validation_composite_mae"
            if composite_key in aggregate:
                aggregate["final/rt/validation_composite_mae"] = aggregate[composite_key]
            if args.validation_dir:
                for organism in ("celegans", "triplemix", "human"):
                    for metric in ("mae", "heavy_dimethyl_mae", "heavy_nterm_dimethyl_mae"):
                        values = [
                            float(item["organism_metrics"][organism][metric])
                            for item in seed_metrics
                        ]
                        aggregate[f"final/rt/{organism}_{metric}"] = statistics.fmean(values)
            parent = wandb.init(
                id=parent_id, resume="allow", project=project, entity=resolved_entity,
                config=config,
            )
            parent.log(aggregate)
            parent.summary["seed_count"] = len(seed_metrics)
            parent.finish()
        except Exception:
            failed = wandb.init(
                id=parent_id, resume="allow", project=project, entity=resolved_entity
            )
            failed.finish(exit_code=1)
            raise
        completed += 1

    while completed < args.max_runs and time.monotonic() < deadline:
        try:
            wandb.agent(
                sweep_id, function=trial, count=1, project=project, entity=resolved_entity
            )
        except CommError as error:
            if "is not running" in str(error):
                print(
                    f"Sweep {sweep_path} is no longer running; "
                    f"stopping after {completed} completed configuration(s)."
                )
                break
            raise
    print(f"Sweep complete: {completed} complete multi-seed configuration(s); {sweep_path}")


if __name__ == "__main__":
    main()
