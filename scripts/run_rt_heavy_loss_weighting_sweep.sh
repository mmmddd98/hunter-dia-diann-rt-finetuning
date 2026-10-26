#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 7 ]]; then
    echo "Usage: $0 <hours> <training-data> [output] [model-dir] [seeds] [python] [wandb-config]" >&2
    exit 2
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
default_root=${DIANN_FINETUNE_OUTPUT_ROOT:-"$repo_dir/finetune_output"}
output=${3:-"$default_root/rt_heavy_loss_weighting_sweep_$(date +%Y%m%d_%H%M%S)"}
model_dir=${4:-${DIANN_MODEL_DIR:-"$repo_dir/models"}}
seeds=${5:-2024,2025,2026,2027,2028,2029,2030,2031,2032,2033}
python_bin=${6:-"$repo_dir/.venv-diann/bin/python"}
wandb_config=${7:-"$repo_dir/configs/wandb.json"}

exec "$python_bin" "$repo_dir/run_targeted_wandb_sweep.py" \
    --hours "$1" --in-dir "$2" --model-dir "$model_dir" --output-dir "$output" \
    --python "$python_bin" --seeds "$seeds" \
    --sweep-config "$repo_dir/configs/rt_heavy_loss_weighting_sweep.json" \
    --wandb-config "$wandb_config" --group "diann-rt-heavy-loss-weighting"
