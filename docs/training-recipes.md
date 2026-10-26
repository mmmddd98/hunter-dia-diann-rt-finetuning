# Training recipes and exploratory campaigns

Run commands from the repository root. These recipes describe the included
configurations; they are not a complete frozen manifest of the final thesis runs.
For the thesis protocol and prepared-split entry point, see [the main README](../README.md).

## DIA-NN fine-tuning with Carafe2 data

This repository fine-tunes DIA-NN's bundled TorchScript predictors directly
from training tables produced by Carafe2. The first complete workflow targets
retention time (RT), because Carafe's normalized RT target maps directly to
DIA-NN's scalar `rt.d0.pt` output.

## What the RT workflow reuses from Carafe2

- input file: `rt_train_data.tsv` from a Carafe output directory;
- columns: `sequence`, `mods`, `mod_sites`, and `rt_norm`;
- median aggregation by peptide/modification form;
- seeded modification-aware sampling (`1337`), restricted to selected training
  sequences to prevent hold-out leakage;
- sequence-disjoint validation hold-out;
- hold-out size `max(1, min(1000, ceil(0.1 * rows) - 10))`;
- Adam, L1/MAE loss, cosine decay with warmup, gradient clipping;
- baseline defaults: 40 epochs, 10 warmup epochs, batch size 1024,
  learning rate `1e-4`, and run seed `2024`.

The workflow fine-tunes the original DIA-NN TorchScript parameters and saves
TorchScript modules that can be loaded again with `torch.jit.load`.

## Setup

```bash
cd /path/to/diann-fintune
python3 -m venv .venv-diann
. .venv-diann/bin/activate
pip install -r requirements.txt
pip install -e ".[test,evaluation]"
```

Pretrained DIA-NN weights are external inputs. Pass `--model-dir` pointing to
the `models` directory of a local DIA-NN 2.5.0 installation, for example:

```text
/path/to/DIA-NN-2.5.0/diann-2.5.0/models
```

## Run the Carafe-compatible RT baseline

Choose an output directory on your own system:

```bash
OUTPUT_ROOT=/path/to/diann_finetune_out
```

```bash
python3 train_carafe_rt.py \
  --in-dir /path/to/carafe_output \
  --model-dir /path/to/diann-2.5.0/models \
  --output-dir "$OUTPUT_ROOT/carafe_rt_baseline" \
  --device auto
```

The convenience wrapper exposes the same baseline:

```bash
scripts/run_carafe_rt_finetune.sh \
  /path/to/carafe_output \
  "$OUTPUT_ROOT/carafe_rt_baseline" \
  /path/to/diann-2.5.0/models
```

## Run the full comparison campaign

This exploratory comparison runs four independent experiments from the same
pretrained model and verifies identical SHA-256 hashes for all train and
validation split files:

| Step | Trainable layers | Epochs | Warmup | Learning rates |
| --- | --- | ---: | ---: | --- |
| `01_heavy_tokens_and_output` | UniMod 510 rows + final output | 30 | 5 | `1e-4` |
| `02_full_conservative` | all parameters | 40 | 5 | all `1e-5` |
| `03_discriminative` | custom rows + head + backbone | 40 | 5 | custom/head `1e-4`, CNN/GRU `1e-5` |
| `04_full_carafe_baseline` | all parameters | 40 | 10 | all `1e-4` |

```bash
scripts/run_full_rt_campaign.sh \
  /path/to/carafe_output \
  "$OUTPUT_ROOT/rt_campaign" \
  /path/to/diann-2.5.0/models
```

The campaign is resumable: completed experiment directories are reused. Use
`run_rt_campaign.py --force ...` only when every selected run should be
repeated. Individual steps can be selected with `--only <experiment-name>`.
The experiment list and all hyperparameters live in
`configs/rt_finetune_campaign.json`, so adding or changing a strategy does not
require editing the runner.

For a quick integration check, use `--max-rows`, one epoch, and CPU:

```bash
python3 train_carafe_rt.py \
  --in-dir /path/to/carafe_output \
  --model-dir /path/to/diann-2.5.0/models \
  --output-dir "$OUTPUT_ROOT/smoke" \
  --device cpu --epochs 1 --warmup-epochs 0 \
  --batch-size 16 --max-rows 64
```

## Outputs

Each run writes:

- `rt.d0.finetuned.pt`: best validation-MAE checkpoint;
- `rt.d0.last.pt`: final epoch checkpoint;
- `dict.txt`: extended DIA-NN dictionary required by both output models;
- `training_history.csv`: per-epoch loss, correlation, R², and RMSE;
- `train_split.tsv` and `validation_split.tsv`: reproducible input split;
- `validation_predictions.tsv`: pretrained and fine-tuned predictions;
- `run_metadata.json`: paths, parameters, split sizes, mappings, and metrics.
- `subgroup_metrics.csv`: heavy label, modification-site, and length strata;
- `plots/*.png`: separate loss, metric, LR, RT-scatter, and heavy-label plots;
- `training_report.html`: browsable report for one experiment.

A campaign additionally writes `campaign_summary.csv`, cross-strategy plots,
`campaign_manifest.json`, and `campaign_report.html`.

## Run the training-data learning curve

After choosing a training strategy, the learning-curve campaign trains fresh
copies of the pretrained model on deterministic, nested 25%, 50%, 75%, and
100% subsets of one fixed training split. Every run is evaluated on the exact
same sequence-disjoint validation set. The defaults use a full
Carafe-like baseline: all layers, 40 epochs, 10 warmup epochs, and learning
rate `1e-4`.

```bash
scripts/run_rt_learning_curve.sh \
  /path/to/carafe_output \
  "$OUTPUT_ROOT/rt_learning_curve" \
  /path/to/diann-2.5.0/models
```

The runner is resumable and writes `learning_curve_summary.csv`,
`plots/learning_curve.png`, `learning_curve_report.html`, and the complete
artifacts for each fraction. Change fractions or training settings in
`configs/rt_learning_curve.json`. Use `--force` with
`run_rt_learning_curve.py` to repeat completed fractions, or `--smoke` for a
one-epoch integration check.

## Select and compare loss functions

`train_carafe_rt.py` supports `--loss mae`, `--loss huber`, and `--loss mse`.
For Huber, `--huber-delta` controls where the objective changes from quadratic
to linear. Training rows can additionally be weighted by heavy-dimethyl site:

```bash
python3 train_carafe_rt.py \
  --in-dir /path/to/carafe_output \
  --model-dir /path/to/diann-2.5.0/models \
  --output-dir "$OUTPUT_ROOT/huber_weighted" \
  --loss huber --huber-delta 0.03 \
  --heavy-nterm-weight 2.0 \
  --heavy-lysine-weight 1.25 \
  --heavy-both-weight 3.0
```

Weights affect only the training objective. Validation metrics and checkpoint
selection remain unweighted, so MAE, RMSE, and R² stay directly comparable.
When `--heavy-both-weight` is omitted, rows with both sites receive the product
of the N-terminal and lysine weights.

The controlled loss campaign compares the existing MAE baseline, unweighted
Huber with delta 0.02, 0.03, and 0.05, plus a moderately heavy-weighted Huber
0.03 run with an otherwise identical unweighted control:

```bash
scripts/run_rt_loss_campaign.sh \
  /path/to/carafe_output \
  "$OUTPUT_ROOT/rt_loss_campaign" \
  /path/to/diann-2.5.0/models
```

Its experiment definitions are in `configs/rt_loss_campaign.json`. Each run
records the selected loss, resolved weights, and category row counts in
`run_metadata.json`.

## Track campaigns with Weights & Biases

W&B integration is optional. It logs every epoch, all validation metrics,
hyperparameters, data sizes, Heavy-Dimethyl subgroup tables, diagnostic plots,
the extended dictionary, and the best TorchScript model. Runs from one local
campaign are grouped together in the W&B project `diann-rt-finetune`.

Copy `configs/wandb.example.json` to the ignored local `configs/wandb.json`,
then install/login and edit `configs/wandb.json` if an explicit W&B team
(`entity`) or another project name should be used:

```bash
pip install 'wandb>=0.17,<1'
wandb login
```

Pass the W&B config as the optional fourth wrapper argument:

```bash
scripts/run_rt_learning_curve.sh \
  /path/to/carafe_output \
  "$OUTPUT_ROOT/rt_learning_curve_wandb" \
  /path/to/diann-2.5.0/models \
  configs/wandb.json
```

The same fourth argument works with `run_full_rt_campaign.sh` and
`run_rt_loss_campaign.sh`. For a local test without uploading anything, prefix
the command with `WANDB_MODE=offline`; the JSON mode is then overridden only
for that invocation. Without `configs/wandb.json`, no W&B package is imported
and training behaves exactly as before.

Already completed local runs can be uploaded without retraining:

```bash
python3 upload_runs_to_wandb.py \
  --runs-dir "$OUTPUT_ROOT/rt_campaign" \
  --wandb-config configs/wandb.json \
  --wandb-group rt-strategy-campaign
```

Run the historical uploader once per group; it creates W&B runs from the saved
history, metadata, plots, subgroup tables, and artifacts.

### Time-budgeted W&B hyperparameter sweep

The sweep uses random search and Hyperband pruning to explore loss type,
Huber delta, seven layer-freezing policies, separate embedding/head/backbone
learning rates, weight decay, batch size, warmup, and Heavy-Dimethyl sample weights. It minimizes the
same unweighted `validation/mae` used by the regular campaigns. A full grid
would contain thousands of combinations, so each trial independently samples
a random combination within the configured search space.

For a four-hour search:

```bash
scripts/run_rt_wandb_sweep.sh \
  4 \
  /path/to/carafe_output \
  "$OUTPUT_ROOT/rt_wandb_sweep" \
  /path/to/diann-2.5.0/models
```

The launcher checks its wall-time between trials and never deliberately cuts a
model off in the middle of an epoch sequence. Therefore the final trial may
finish slightly after the nominal deadline. Every trial receives a unique
local output directory and appears immediately in the W&B sweep dashboard.
Search ranges live in `configs/rt_wandb_sweep.json`; project/entity and artifact
settings remain in `configs/wandb.json`.

### Heavy-Dimethyl oversampling and loss-weighting sweeps

Two controlled multi-seed sweeps isolate the two ways of emphasizing the
Heavy-Dimethyl rows:

- `configs/rt_heavy_oversampling_sweep.json` varies how often Heavy-N-term,
  Heavy-K, and doubly-heavy training rows are presented. All loss weights are
  fixed to `1.0`.
- `configs/rt_heavy_loss_weighting_sweep.json` varies the per-peptide loss
  weights. All oversampling factors are fixed to `1.0`.

Both sweeps use the same fixed, sequence-disjoint validation logic, repeat
each sampled configuration with ten seeds, and minimize the mean unweighted
Heavy-Dimethyl validation MAE across those seeds. Global, Heavy-N-term,
Heavy-K, doubly-heavy, and normal metrics remain available separately.

Run the two six-hour sweeps concurrently on GPU 0 and GPU 1 with only the
dataset argument:

```bash
scripts/run_diann_rt_targeted_sweeps_parallel.sh \
  6 \
  /path/to/trainingdata/celegans_exploris_44min_rtms2_v1
```

The launcher returns after submitting both background jobs. Outputs, logs,
PIDs, exit codes, and W&B sweep IDs are stored below a timestamped directory
under `DIANN_FINETUNE_OUTPUT_ROOT`, or the ignored local `finetune_output/`
directory when that variable is unset.
The queue status command is printed after submission. Both processes receive
their own complete six-hour budget; one ten-seed configuration is never
deliberately interrupted at the deadline.

The sweeps can also be launched individually with
`scripts/run_rt_heavy_oversampling_sweep.sh` and
`scripts/run_rt_heavy_loss_weighting_sweep.sh`.

The layer policies are:

| Policy | Updated model blocks |
| --- | --- |
| `embeddings` | only UniMod 510 embedding rows 40/41 |
| `head` | rows 40/41 + final `Linear(128,1)` |
| `rt_head` | rows 40/41 + both RT linear layers |
| `cnn_head` | rows 40/41 + both CNNs + RT head; GRU frozen |
| `gru_head` | rows 40/41 + BiGRU + RT head; CNNs frozen |
| `discriminative` | all blocks; backbone, head, and custom rows use separate LRs |
| `all` | all parameters use the main learning rate |

For every trial, W&B records the policy, optimizer-group learning rates,
effective trainable parameter count, exact trainable tensor names, and whether
only embedding rows 40/41 or the complete embedding table can change. For
selective policies, the optimizer sees the full embedding tensor internally,
but a gradient mask guarantees that only the two custom rows update.

Validate paths and configuration without creating a cloud sweep:

```bash
python3 run_wandb_sweep.py \
  --hours 4 \
  --in-dir /path/to/carafe_output \
  --model-dir /path/to/diann-2.5.0/models \
  --output-dir "$OUTPUT_ROOT/rt_wandb_sweep" \
  --dry-run
```

### Analyze a completed sweep

The analyzer aggregates every complete local run, recomputes Heavy-N-term,
Heavy-K, and true double-heavy subgroups directly from prediction files,
creates summary CSVs and plots, performs paired bootstrap comparisons, and
writes a German interpretation into `<sweep-dir>/analysis/README.md`:

```bash
python3 analyze_rt_sweep.py \
  --sweep-dir /path/to/rt_wandb_sweep \
  --baseline-dir "$OUTPUT_ROOT/rt_campaign/04_full_carafe_baseline" \
  --reported-runs 359
```

The generated `report.html` provides a compact visual view; the README records
the methodology, conclusions, caveats, selected candidates, and recommended
follow-up experiments.

### Repeated data-scaling experiment

To answer whether more training data is worthwhile, run a fixed example
configuration (full fine-tuning, Huber delta 0.03) on 10 independent,
stratified and sequence-disjoint 90/10 splits. Each repeat compares nested
10%, 25%, 50%, 75%, and 100% sequence subsets, so there are 50 training runs:

```bash
./scripts/run_rt_data_scaling.sh \
  /path/to/carafe_output \
  "$OUTPUT_ROOT/rt_data_scaling" \
  /path/to/diann-2.5.0/models
```

Use 15 repeats when more precise intervals are worth the extra compute:

```bash
./scripts/run_rt_data_scaling.sh /path/to/carafe_output "$OUTPUT_ROOT/rt_data_scaling" \
  /path/to/diann-2.5.0/models "" 15
```

The resumable result directory contains `README.md`, an HTML report, the raw
per-run metrics, aggregated mean/standard deviation/95% intervals, the exact
split TSVs, and `plots/data_scaling_curve.png`. Optional W&B tracking is the
fourth shell argument, before the repeat count.

### Versioned Carafe training-data builds

The separate [data-collection repository](https://github.com/mmmddd98/hunter-dia-workflows/tree/main/data_collect) contains `finetune_catalog`, which
registers each completed Carafe export and builds traceable model inputs. Pass
the root of a completed build directly as `--in-dir`; this trainer
automatically resolves `<build>/diann_rt_input/rt_train_data.tsv`. The matching
Carafe2 launcher resolves `<build>/carafe2_input/`, so both model families use
the same checked and manifested RT source table.

## Modification handling

Carbamidomethyl C, oxidized M, N-terminal acetylation, and light dimethyl K/N
termini use DIA-NN's existing UniMod tokens. For heavy
`Dimethyl:2H(4)13C(2)` (UniMod 510), the workflow extends DIA-NN's vocabulary:

- token ID 40: N-terminal UniMod 510;
- token ID 41: lysine UniMod 510.

Their embedding rows are initialized from the corresponding light-dimethyl
embeddings (IDs 38 and 39), then trained independently on the heavy-labelled
Carafe examples. `run_metadata.json` records how many rows use each token and
the L2 update of each new embedding. The output model must always be deployed
together with the generated `dict.txt`; otherwise DIA-NN cannot produce the
new token IDs. Unsupported modifications fail loudly by default or can be
dropped explicitly with `--unknown-mods drop`.

## Current scope

RT is implemented end to end. Ion mobility can reuse the same adapter once the
Carafe CCS/mobility target normalization is validated against DIA-NN's `im`
output. Fragment fine-tuning remains intentionally separate until Carafe's
fragment rows are mapped and numerically validated against DIA-NN's dynamic
fragment-vector ordering.
