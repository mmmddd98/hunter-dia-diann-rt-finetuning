#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 8 ]]; then
    echo "Usage: $0 <hours> <training-data> [queue-output] [model-dir] [python] [gpu-pair] [seeds] [wandb-config]" >&2
    echo "gpu-pair example: 0,1" >&2
    exit 2
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
default_root=${DIANN_FINETUNE_OUTPUT_ROOT:-"$repo_dir/finetune_output"}
queue_dir=${3:-"$default_root/diann_rt_targeted_sweeps_$(date +%Y%m%d_%H%M%S)"}
model_dir=${4:-${DIANN_MODEL_DIR:-"$repo_dir/models"}}
python_bin=${5:-"$repo_dir/.venv-diann/bin/python"}
gpu_pair=${6:-0,1}
seeds=${7:-2024,2025,2026,2027,2028,2029,2030,2031,2032,2033}
wandb_config=${8:-"$repo_dir/configs/wandb.json"}
IFS=',' read -r -a gpus <<<"$gpu_pair"
if [[ ${#gpus[@]} -ne 2 || -z ${gpus[0]} || -z ${gpus[1]} ]]; then
    echo "Exactly two GPU IDs are required, for example 0,1" >&2
    exit 2
fi
if [[ ${gpus[0]} == "${gpus[1]}" ]]; then
    echo "The two parallel sweeps must use different GPUs" >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Python interpreter is not executable: $python_bin" >&2
    exit 2
fi
if [[ ! -f "$2/rt_train_data.tsv" && ! -f "$2/diann_rt_input/rt_train_data.tsv" ]]; then
    echo "No DIA-NN rt_train_data.tsv found below: $2" >&2
    exit 2
fi
for required_model_file in rt.d0.pt dict.txt; do
    if [[ ! -f "$model_dir/$required_model_file" ]]; then
        echo "Missing DIA-NN model input: $model_dir/$required_model_file" >&2
        exit 2
    fi
done
if [[ ! -f "$wandb_config" ]]; then
    echo "Missing W&B configuration: $wandb_config" >&2
    exit 2
fi

mkdir -p "$queue_dir"
manifest="$queue_dir/queue.tsv"
printf 'job\tgpu\tpid\tstatus_file\tlog\toutput\n' >"$manifest"

submit_job() {
    local name=$1
    local gpu=$2
    local config=$3
    local hours=$4
    local training_data=$5
    local job_dir="$queue_dir/$name"
    local output_dir="$job_dir/runs"
    local status_file="$job_dir/status"
    local exit_file="$job_dir/exit_code"
    local log_file="$job_dir/sweep.log"
    mkdir -p "$job_dir"
    printf 'queued\n' >"$status_file"
    nohup env CUDA_VISIBLE_DEVICES="$gpu" \
        "$repo_dir/scripts/run_diann_rt_sweep_worker.sh" "$status_file" "$exit_file" \
        "$python_bin" "$repo_dir/run_targeted_wandb_sweep.py" \
        --hours "$hours" --in-dir "$training_data" --model-dir "$model_dir" --output-dir "$output_dir" \
        --python "$python_bin" --seeds "$seeds" --sweep-config "$config" \
        --wandb-config "$wandb_config" --group "diann-$name" \
        >"$log_file" 2>&1 </dev/null &
    local pid=$!
    printf '%s\n' "$pid" >"$job_dir/pid"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$gpu" "$pid" "$status_file" "$log_file" "$output_dir" >>"$manifest"
}

submit_job \
    rt_heavy_oversampling "${gpus[0]}" "$repo_dir/configs/rt_heavy_oversampling_sweep.json" \
    "$1" "$2"
submit_job \
    rt_heavy_loss_weighting "${gpus[1]}" "$repo_dir/configs/rt_heavy_loss_weighting_sweep.json" \
    "$1" "$2"

printf 'Queue started: %s\n' "$queue_dir"
printf 'Oversampling sweep: GPU %s\n' "${gpus[0]}"
printf 'Loss-weighting sweep: GPU %s\n' "${gpus[1]}"
printf 'Status: %s/scripts/status_diann_rt_sweep_queue.sh %s\n' "$repo_dir" "$queue_dir"
