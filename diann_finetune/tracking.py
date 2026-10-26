"""Optional Weights & Biases tracking without coupling core training to W&B."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


class WandbTracker:
    """Thin lazy-loaded W&B adapter used by individual training processes."""

    def __init__(
        self,
        config_path: Path | None,
        *,
        run_name: str,
        group: str | None,
        run_config: Dict[str, object],
        output_dir: Path,
        run_id: str | None = None,
    ) -> None:
        self.enabled = config_path is not None
        self.run = None
        self.wandb = None
        self.finished = False
        self.settings: Dict[str, object] = {}
        if not self.enabled:
            return
        self.settings = json.loads(config_path.read_text(encoding="utf-8"))
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "W&B tracking was requested but `wandb` is not installed. "
                "Run `pip install 'wandb>=0.17,<1'`."
            ) from error
        self.wandb = wandb
        init_kwargs = {
            "project": self.settings.get("project", "diann-rt-finetune"),
            "entity": self.settings.get("entity"),
            "name": run_name,
            "group": group or self.settings.get("group"),
            "job_type": self.settings.get("job_type", "rt-finetune"),
            "tags": self.settings.get("tags", ["diann", "rt", "carafe2"]),
            "notes": self.settings.get("notes"),
            "mode": os.environ.get("WANDB_MODE", self.settings.get("mode", "online")),
            "config": run_config,
            "dir": str(output_dir),
            "id": run_id,
            "resume": "allow" if run_id else None,
        }
        self.run = wandb.init(**{key: value for key, value in init_kwargs.items() if value is not None})
        self.run.define_metric("epoch")
        self.run.define_metric("train/*", step_metric="epoch")
        self.run.define_metric("validation/*", step_metric="epoch")
        self.run.define_metric("learning_rate/*", step_metric="epoch")

    def log_epoch(self, record: Dict[str, float]) -> None:
        if not self.enabled or self.run is None:
            return
        payload: Dict[str, float] = {"epoch": record["epoch"]}
        mapping = {
            "train_loss": "train/objective",
            "train_mae": "train/mae",
            "validation_mae": "validation/mae",
            "validation_rmse": "validation/rmse",
            "validation_r2": "validation/r2",
            "validation_pearson": "validation/pearson",
            "validation_spearman": "validation/spearman",
            "learning_rate": "learning_rate/max",
        }
        for source, destination in mapping.items():
            if source in record:
                payload[destination] = record[source]
        for key, value in record.items():
            if key.startswith("lr_"):
                payload[f"learning_rate/{key[3:]}"] = value
            elif key.startswith("train_") and key not in mapping and not key.endswith("_rows"):
                payload[f"train/{key.removeprefix('train_')}"] = value
            elif key.startswith("validation_") and key not in mapping and not key.endswith("_rows"):
                payload[f"validation/{key.removeprefix('validation_')}"] = value
        self.run.log(payload)

    def finish(self, metadata: Dict[str, object], output_dir: Path, *, exit_code: int = 0) -> None:
        if not self.enabled or self.run is None or self.wandb is None or self.finished:
            return
        self.finished = True
        result = metadata.get("result", {})
        if isinstance(result, dict):
            for stage in ("pretrained_metrics", "finetuned_metrics"):
                metrics = result.get(stage, {})
                if isinstance(metrics, dict):
                    prefix = stage.removesuffix("_metrics")
                    for metric, value in metrics.items():
                        self.run.summary[f"final/{prefix}_{metric}"] = value
            for key in (
                "best_epoch", "elapsed_seconds", "trainable_parameters",
                "optimizer_parameter_elements", "total_parameters",
            ):
                if key in result:
                    self.run.summary[f"final/{key}"] = result[key]
            for key in ("strategy", "trainable_parameter_tensors", "trainable_embedding_rows"):
                if key in result:
                    self.run.summary[f"model/{key}"] = result[key]
        split = metadata.get("split", {})
        if isinstance(split, dict):
            for key, value in split.items():
                self.run.summary[f"data/{key}"] = value

        if bool(self.settings.get("log_media", True)):
            subgroup_path = output_dir / "subgroup_metrics.csv"
            if subgroup_path.exists():
                groups = pd.read_csv(subgroup_path)
                self.run.log({"subgroups/metrics": self.wandb.Table(dataframe=groups)})
            for image_name in ("loss_comparison.png", "rt_predictions.png", "heavy_dimethyl_mae.png"):
                image_path = output_dir / "plots" / image_name
                if image_path.exists():
                    self.run.log({f"plots/{image_path.stem}": self.wandb.Image(str(image_path))})

        if bool(self.settings.get("log_artifacts", True)):
            artifact = self.wandb.Artifact(
                name=f"{self.run.name}-outputs",
                type="diann-rt-model",
                metadata={"workflow_version": metadata.get("workflow_version")},
            )
            files: Iterable[str] = (
                "run_metadata.json",
                "training_history.csv",
                "validation_predictions.tsv",
                "validation_split.tsv",
                "subgroup_metrics.csv",
                "training_report.html",
                "dict.txt",
            )
            if bool(self.settings.get("log_models", True)):
                files = (*files, "rt.d0.finetuned.pt")
            for filename in files:
                path = output_dir / filename
                if path.exists():
                    artifact.add_file(str(path), name=filename)
            self.run.log_artifact(artifact)
        self.run.finish(exit_code=exit_code)
