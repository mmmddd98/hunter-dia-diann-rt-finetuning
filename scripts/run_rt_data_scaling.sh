#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
    echo "Usage: $0 <carafe-output-dir> <data-scaling-output-dir> [diann-model-dir] [wandb-config] [repeats]" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
carafe_output=$1
scaling_output=$2
model_dir=${3:-${DIANN_MODEL_DIR:-"$repo_dir/models"}}
wandb_config=${4:-}
repeats=${5:-}

if [[ ! -f "$carafe_output/rt_train_data.tsv" ]]; then
    echo "Missing training data: $carafe_output/rt_train_data.tsv" >&2
    exit 1
fi

command=(python3 "$repo_dir/run_rt_data_scaling.py" --in-dir "$carafe_output" --model-dir "$model_dir" --output-dir "$scaling_output" --device auto)
if [[ -n "$wandb_config" ]]; then command+=(--wandb-config "$wandb_config"); fi
if [[ -n "$repeats" ]]; then command+=(--repeats "$repeats"); fi
"${command[@]}"
