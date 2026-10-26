#!/usr/bin/env bash
set -euo pipefail

: "${DIANN_SWEEP_IN_DIR:?DIANN_SWEEP_IN_DIR is required}"
: "${DIANN_SWEEP_MODEL_DIR:?DIANN_SWEEP_MODEL_DIR is required}"
: "${DIANN_SWEEP_OUTPUT_DIR:?DIANN_SWEEP_OUTPUT_DIR is required}"
: "${DIANN_SWEEP_WANDB_CONFIG:?DIANN_SWEEP_WANDB_CONFIG is required}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
mkdir -p "$DIANN_SWEEP_OUTPUT_DIR"
run_dir=$(mktemp -d "$DIANN_SWEEP_OUTPUT_DIR/run.XXXXXXXX")

python3 "$repo_dir/train_carafe_rt.py" \
    --in-dir "$DIANN_SWEEP_IN_DIR" \
    --model-dir "$DIANN_SWEEP_MODEL_DIR" \
    --output-dir "$run_dir" \
    --device auto \
    --wandb-config "$DIANN_SWEEP_WANDB_CONFIG" \
    --wandb-run-name "$(basename "$run_dir")" \
    --wandb-group "${DIANN_SWEEP_GROUP:-rt-hyperparameter-sweep}" \
    "$@"
