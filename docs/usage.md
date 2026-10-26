# HUNTER DIA DIA-NN RT fine-tuning

Python training and evaluation of DIA-NN 2.5.0 retention-time TorchScript
models using Carafe2-format peptide observations. This is the external DIA-NN
RT branch linked from the [HUNTER DIA model-adaptation overview](https://github.com/mmmddd98/hunter-dia-model-adaptation).
It exports the tuned model together with the extended token dictionary needed
for heavy dimethylation (UniMod:510).

## Start here

| Task | Guide |
| --- | --- |
| Train a DIA-NN RT model from prepared partitions | [Quick start](#train-with-prepared-splits) |
| Understand input columns and outputs | [Data format](data-format.md) and [model contract](#model-and-output-contract) |
| Compare layers, losses, weights or dataset sizes | [Training recipes](training-recipes.md) |
| Evaluate a checkpoint or run attribution | [Evaluation guide](../captum_attribution/README.md) |
| Develop or validate the code | [Developer guide](development.md) |

This repository trains **DIA-NN TorchScript RT models**. The separate
[Carafe2 fork](https://github.com/mmmddd98/hunter-dia-carafe2) trains
AlphaPeptDeep/Carafe2 models. Both can consume prepared RT observations, but
their checkpoint formats and command-line options differ.

## Install

Use Python 3.11 (tested core: PyTorch 2.5.1, NumPy 2.0.2,
Pandas 2.2.3 and Matplotlib 3.9.2). Run commands from this repository root:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[test,evaluation]'
python -m pytest
```

Install a PyTorch build suited to your CPU/CUDA system. For optional tracking,
install `python -m pip install -e '.[tracking]'`. For evaluation and Captum,
install `python -m pip install -e '.[evaluation]'`, or use the separate
[Conda environment](../captum_attribution/environment/captum_eval.yml).

Obtain `rt.d0.pt` and `dict.txt` from your DIA-NN 2.5.0 installation.
Pretrained weights and executable distributions are external prerequisites;
they are not redistributed in the source tree. Pass `--model-dir` or set
`DIANN_MODEL_DIR`; the fallback is the ignored `models/` directory.

## Train with prepared splits

Input TSVs contain `sequence`, `mods`, `mod_sites`, and the explicit target
`rt_norm`; see the [column definitions and example](data-format.md). Unmodified peptides may have empty modification fields. Keep the
training and validation target scales identical and supply peptidoform-disjoint
partitions prepared before fitting alignment mappings.

```bash
python train_carafe_rt.py \
  --train-split inputs/train.tsv \
  --validation-split inputs/validation.tsv \
  --model-dir models \
  --output-dir outputs/rt_run \
  --epochs 40 --warmup-epochs 10 --batch-size 512 \
  --learning-rate 1e-4 --loss mae --seed 2024 --device cpu
```

Use `--device cuda` for GPU training. Repeat with seeds 2025 and 2026 for the
three-seed protocol. A prepared validation split is checked for peptidoform
overlap. `--in-dir` or `--rt-data` instead creates a sequence-disjoint validation
split internally; this is a separate baseline protocol. Batch size defaults
to 1024, so pass 512 explicitly for the main thesis protocol.

The main thesis comparisons used train-fitted RT targets, fixed peptidoform
partitions and a separate final test set. This trainer consumes those targets;
it does not fit cross-source alignment. Preparation and experiment definitions
are maintained in [hunter-dia-workflows/data_collect](https://github.com/mmmddd98/hunter-dia-workflows/tree/main/data_collect).
For catalog-based input, pass `--catalog` pointing to that checkout's
`finetune_catalog/datasets.yaml` or set `FINE_TUNE_CATALOG`. Prepared TSVs work
without the other repository.

## Model and output contract

Heavy N-terminal and lysine dimethylation use token IDs 40 and 41, initialised
from light-dimethyl IDs 38 and 39. Unsupported modifications raise an error
unless explicitly dropped. Deploy `rt.d0.finetuned.pt` **with the generated
`dict.txt`**. The source dictionary alone does not encode the new tokens.

Each run records best-validation and last-epoch checkpoints, training history,
split tables, predictions, subgroup metrics and `run_metadata.json`. Paths in
these generated manifests describe the user's local run. Outputs stay ignored.
Checkpoint selection uses validation metrics; final tests must remain separate.
Loss choices include MAE, MSE and Huber; layer policies, learning rates,
heavy-label weights, oversampling and optional fixed step budgets are listed
by `python train_carafe_rt.py --help`.

## Additional workflows

- [Training recipes and campaign configurations](training-recipes.md):
  baseline, layer controls, data scaling, loss comparisons and optional sweeps.
- [Evaluation and attribution](../captum_attribution/README.md): prediction,
  parity checks, metrics and optional Integrated Gradients.
- [Evaluation catalog](../captum_attribution/central_store_README.md): explicit
  model-package and held-out-data roots, opt-in selection and reporting.

W&B is opt-in. Copy `configs/wandb.example.json` to `configs/wandb.json` and
configure your account locally. Standard training needs no W&B account.
The older three-organism scaling helpers require `CARAFE_DATA_COLLECT_DIR`;
comparison plotting also requires `CARAFE2_STUDY_ROOT`. They are exploratory
campaign tools, distinct from the final four-source comparison.

## Repository layout

| Location | Responsibility |
| --- | --- |
| `train_carafe_rt.py` | Training CLI; prepared partitions or an internal validation split |
| `diann_finetune/data.py` | Loading, aggregation, splits and sampling |
| `diann_finetune/encoding.py` | Modification translation, token IDs and dictionary export |
| `diann_finetune/training.py` | Optimisation, validation and portable checkpoint export |
| `diann_finetune/diagnostics.py` | Run metrics, plots and HTML diagnostics |
| `configs/`, `scripts/` | Optional campaigns and launchers |
| `captum_attribution/` | Held-out evaluation and attribution |
| `tests/` | Synthetic regression fixtures |

Regression tests use synthetic fixtures and require no measurement files or
pretrained DIA-NN weights. RAW/mzML files, local environments, checkpoints,
logs, credentials and generated reports are excluded from the source tree.
