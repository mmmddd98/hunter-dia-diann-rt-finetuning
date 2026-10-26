#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <carafe-output-dir> <finetune-output-dir> [diann-model-dir]" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
carafe_output=$1
finetune_output=$2
model_dir=${3:-${DIANN_MODEL_DIR:-"$repo_dir/models"}}

python3 "$repo_dir/train_carafe_rt.py" \
    --in-dir "$carafe_output" \
    --model-dir "$model_dir" \
    --output-dir "$finetune_output" \
    --device auto \
    --epochs 40 \
    --warmup-epochs 10 \
    --batch-size 1024 \
    --learning-rate 0.0001
