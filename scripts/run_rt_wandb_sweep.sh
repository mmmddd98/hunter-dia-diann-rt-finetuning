#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <hours> <carafe-output-dir> <sweep-output-dir> [diann-model-dir]" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
hours=$1
carafe_output=$2
sweep_output=$3
model_dir=${4:-${DIANN_MODEL_DIR:-"$repo_dir/models"}}

python3 "$repo_dir/run_wandb_sweep.py" \
    --hours "$hours" \
    --in-dir "$carafe_output" \
    --model-dir "$model_dir" \
    --output-dir "$sweep_output" \
    --wandb-config "$repo_dir/configs/wandb.json"
