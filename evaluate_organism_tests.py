#!/usr/bin/env python3
"""Evaluate one fine-tuned DIA-NN RT model on all organism hold-outs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ALIGNMENT_MODULE_DIR = Path(
    os.environ.get(
        "CARAFE_DATA_COLLECT_DIR",
        Path(__file__).resolve().parents[1] / "carafe2" / "ma" / "data_collect",
    )
).expanduser()
if str(ALIGNMENT_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(ALIGNMENT_MODULE_DIR))
from rt_alignment import predictions_to_observed_rt

from diann_finetune.data import load_carafe_rt
from diann_finetune.encoding import TokenEncoder, read_tokens
from diann_finetune.training import (
    evaluate,
    heavy_stratified_regression_metrics,
    load_torchscript_model,
    prepare_examples,
    resolve_device,
)


def observed_rt_metrics(frame: pd.DataFrame, predictions: np.ndarray, alignment: dict) -> dict:
    """Evaluate in observed minutes, the common scale for all target modes."""
    missing = {"rt", "source_organism"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Observed-RT evaluation requires {sorted(missing)}; rerun "
            "prepare_organism_scaling_study.py to refresh legacy test inputs"
        )
    source = frame["source_organism"].astype(str)
    targets = pd.to_numeric(frame["rt"], errors="raise").to_numpy(float)
    predicted_rt = predictions_to_observed_rt(predictions, source, alignment)
    delta = predicted_rt - targets
    result = {
        "raw_mae_minutes": float(np.mean(np.abs(delta))),
        "raw_rmse_minutes": float(np.sqrt(np.mean(delta ** 2))),
        "raw_bias_minutes": float(np.mean(delta)),
    }
    raw_frame = frame.copy()
    raw_frame["rt_norm"] = targets
    heavy = heavy_stratified_regression_metrics(raw_frame, predicted_rt)
    result.update({f"raw_{key}_minutes" if key.endswith(("mae", "rmse")) else f"raw_{key}": value
                   for key, value in heavy.items()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tests-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "gpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--rt-alignment", type=Path, required=True,
                        help="Training-only target-to-observed RT calibration JSON")
    args = parser.parse_args()

    model_path = args.run_dir / "rt.d0.finetuned.pt"
    dictionary_path = args.run_dir / "dict.txt"
    if not model_path.is_file() or not dictionary_path.is_file():
        raise SystemExit(f"Incomplete DIA-NN run: {args.run_dir}")
    device = resolve_device("cuda" if args.device == "gpu" else args.device)
    model = load_torchscript_model(model_path, device)
    encoder = TokenEncoder(read_tokens(dictionary_path))
    all_metrics = {}
    alignment = json.loads(args.rt_alignment.read_text(encoding="utf-8"))
    for organism in ("celegans", "triplemix", "human"):
        test_path = args.tests_dir / f"{organism}.tsv"
        frame = load_carafe_rt(test_path).frame
        encoded, targets = prepare_examples(frame, encoder, 2)
        metrics, predictions = evaluate(model, encoded, targets, args.batch_size, device)
        metrics.update(heavy_stratified_regression_metrics(frame, predictions))
        metrics.update(observed_rt_metrics(frame, predictions, alignment))
        metrics["rows"] = len(frame)
        metrics["unique_sequences"] = int(frame["sequence"].nunique())
        all_metrics[organism] = metrics
        output = frame.copy()
        output["rt_pred"] = predictions
        output.to_csv(args.run_dir / f"test_predictions_{organism}.tsv", sep="\t", index=False)
    (args.run_dir / "organism_test_metrics.json").write_text(
        json.dumps(all_metrics, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
