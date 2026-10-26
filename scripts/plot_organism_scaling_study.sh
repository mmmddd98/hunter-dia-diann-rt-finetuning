#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
data_collect_dir=${CARAFE_DATA_COLLECT_DIR:?Set CARAFE_DATA_COLLECT_DIR to the data_collect checkout}
diann=${1:-"${DIANN_FINETUNE_OUTPUT_ROOT:-$root/finetune_output}/organism_scaling_v1"}
carafe=${2:-"${CARAFE2_STUDY_ROOT:?Set CARAFE2_STUDY_ROOT to the prepared Carafe2 study}"}
out=${3:-"$diann/plots"}
mkdir -p "$out"
(
  flock -n 9 || { echo "Organism-scaling plots are already being generated: $out"; exit 0; }
  "${PLOT_PYTHON:-python3}" "$data_collect_dir/plot_organism_scaling_study.py" \
    --diann-dir "$diann" --carafe2-dir "$carafe" --output-dir "$out"
) 9>"$out/.plot.lock"
