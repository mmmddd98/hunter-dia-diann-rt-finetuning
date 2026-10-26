#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 7 ]]; then
  echo "Usage: $0 <training-data-dir> <gpu> [output-dir] [model-dir] [python] [seeds] [wandb-config]" >&2
  exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output_root=${DIANN_FINETUNE_OUTPUT_ROOT:-"$root/finetune_output"}
output=${3:-"$output_root/diann_rt_controlled_weights_$(date +%Y%m%d_%H%M%S)"}
model=${4:-${DIANN_MODEL_DIR:-"$root/models"}}
python_bin=${5:-$root/.venv-diann/bin/python}
seeds=${6:-2024,2025,2026,2027,2028,2029,2030,2031,2032,2033}
wandb=${7:-$root/configs/wandb.json}
exec env CUDA_VISIBLE_DEVICES="$2" "$python_bin" "$root/run_targeted_wandb_sweep.py" \
  --hours 12 --in-dir "$1" --model-dir "$model" --output-dir "$output" --python "$python_bin" \
  --seeds "$seeds" --sweep-config "$root/configs/rt_controlled_weights_sweep.json" \
  --configuration-map "$root/configs/rt_controlled_weights_map.json" \
  --wandb-config "$wandb" --group "diann-controlled-weights"
