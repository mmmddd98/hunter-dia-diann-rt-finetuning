#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 <carafe-output-dir> <loss-campaign-output-dir> [diann-model-dir] [wandb-config]" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
carafe_output=$1
campaign_output=$2
model_dir=${3:-${DIANN_MODEL_DIR:-"$repo_dir/models"}}
wandb_config=${4:-}

command=(python3 "$repo_dir/run_rt_campaign.py" \
    --in-dir "$carafe_output" \
    --model-dir "$model_dir" \
    --output-dir "$campaign_output" \
    --config "$repo_dir/configs/rt_loss_campaign.json" \
    --device auto)
if [[ -n "$wandb_config" ]]; then
    command+=(--wandb-config "$wandb_config")
fi
"${command[@]}"
