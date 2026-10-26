#!/usr/bin/env bash
set -euo pipefail
# Bash may otherwise continue reading a modified script after waiting for hours
# on the shared GPU lock. Buffer the complete source into argv once at startup.
script_path=${ORGANISM_SCALING_SCRIPT_PATH:-$(readlink -f -- "${BASH_SOURCE[0]}")}
if [[ ${ORGANISM_SCALING_SCRIPT_BUFFERED:-0} != 1 ]]; then
  script_text=$(<"$script_path")
  exec env ORGANISM_SCALING_SCRIPT_BUFFERED=1 \
    ORGANISM_SCALING_SCRIPT_PATH="$script_path" \
    bash -c "$script_text" "$script_path" "$@"
fi
root=$(cd -- "$(dirname -- "$script_path")/.." && pwd)
data_collect_dir=${CARAFE_DATA_COLLECT_DIR:?Set CARAFE_DATA_COLLECT_DIR to the data_collect checkout}
study_root=${1:-"${DIANN_FINETUNE_OUTPUT_ROOT:-$root/finetune_output}/organism_scaling_v1"}
prep="$data_collect_dir/prepare_organism_scaling_study.py"
prepared="$study_root/prepared"
rt_target=${RT_TARGET:-global_60min}
has_column() {
  local path=$1 column=$2
  awk -F '\t' -v wanted="$column" 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == wanted) found = 1; exit !found }' "$path"
}
prepared_complete() {
  [[ -f "$prepared/study_manifest.json" ]] || return 1
  grep -q '"rt_target": "'"$rt_target"'"' "$prepared/study_manifest.json" || return 1
  local seed organism
  for seed in 2024 2025 2026 2027 2028; do
    for organism in celegans triplemix human; do
      [[ -s "$prepared/seed_${seed}/tests/${organism}.tsv" ]] || return 1
      has_column "$prepared/seed_${seed}/tests/${organism}.tsv" rt || return 1
      has_column "$prepared/seed_${seed}/tests/${organism}.tsv" source_organism || return 1
    done
  done
}
snapshot_tests() {
  local source_dir=$1 destination_dir=$2 organism
  mkdir -p "$destination_dir"
  for organism in celegans triplemix human; do
    [[ -s "$source_dir/${organism}.tsv" ]] || {
      echo "Missing prepared test input: $source_dir/${organism}.tsv" >&2
      return 1
    }
    cp -f "$source_dir/${organism}.tsv" "$destination_dir/${organism}.tsv"
  done
  cp -f "$(dirname "$source_dir")/rt_alignment.json" "$destination_dir/rt_alignment.json"
}

if ! prepared_complete; then
  python3 "$prep" --output-dir "$prepared" --rt-target "$rt_target"
fi
prepared_complete || { echo "Prepared organism test inputs are incomplete: $prepared" >&2; exit 1; }
gpu_lock=${STUDY_GPU_LOCK:-/tmp/rt_organism_scaling_all_gpus.lock}
exec 8>"$gpu_lock"
echo "Waiting for exclusive all-GPU study lock: $gpu_lock"
flock 8
python_bin=${DIANN_PYTHON:-python3}
dispatcher="$data_collect_dir/run_parallel_organism_scaling.py"
gpu_ids=${STUDY_GPUS:-0,1,2}
jobs_per_gpu=${STUDY_JOBS_PER_GPU:-2}
epochs=${STUDY_EPOCHS:-60}
warmup=${STUDY_WARMUP_EPOCHS:-10}
echo "DIA-NN will use GPUs $gpu_ids with $jobs_per_gpu independent runs per GPU"
python3 "$dispatcher" --family diann --study-root "$study_root" \
  --gpus "$gpu_ids" --jobs-per-gpu "$jobs_per_gpu" \
  --epochs "$epochs" --warmup-epochs "$warmup" --diann-python "$python_bin"
echo "DIA-NN organism-scaling study complete: $study_root"
# Comparison plotting is optional and uses an explicitly configured peer study.
carafe2_study=${CARAFE2_STUDY_ROOT:-}
if [[ -n "$carafe2_study" && -d "$carafe2_study/runs" ]]; then
  diann_results=$(find "$study_root/runs" -name organism_test_metrics.json -type f | wc -l)
  carafe2_results=$(find "$carafe2_study/runs" -name organism_test_metrics.json -type f | wc -l)
  if (( diann_results >= 70 && carafe2_results >= 70 )); then
    "$root/scripts/plot_organism_scaling_study.sh" "$study_root" "$carafe2_study" \
      "${ORGANISM_SCALING_PLOTS_DIR:-$study_root/plots}"
  fi
fi
