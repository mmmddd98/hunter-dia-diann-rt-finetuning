# RT model evaluation and Captum attribution

This is the reproducible Phase-1 evaluation pipeline for one DIA-NN RT model
and its matching dictionary against one labelled peptide table. It reuses
`diann_finetune.encoding.TokenEncoder` and `load_torchscript_model`, so its
input representation is the one already used for fine-tuning and organism
hold-out evaluation.

Create an isolated environment (it does not change `.venv-diann`):

```bash
conda env create -f captum_attribution/environment/captum_eval.yml
conda activate diann-captum-eval
```

Copy and edit `captum_attribution/configs/example_carafe_rt.yaml` from the
repository root. Model and dictionary are
deliberately separate required fields but always written together into the
manifest. Run a five-peptide, prediction-only check first:

```bash
python -m captum_attribution.evaluate_rt_model --config my_evaluation.yaml --prediction-only --smoke-test
```

Then run the full inference evaluation, or add Captum explicitly:

```bash
python -m captum_attribution.evaluate_rt_model --config my_evaluation.yaml --prediction-only
python -m captum_attribution.evaluate_rt_model --config my_evaluation.yaml --captum
```

`--plots-only` recreates plots from saved Parquet results. Captum is resumable:
completed peptide IDs in `attributions.parquet` are skipped on a later run.
If the input table has established DIA-NN predictions, configure
`reference_prediction_column` (and optionally its tolerance): the run writes
`inference_validation.json` and stops before attribution if Python inference
does not match it.

Each run writes `<model>__<dataset>__<evaluation_name>/` containing
`manifest.yaml`, `metrics.json`, `predictions.parquet`, `peptide_summary.parquet`,
`stratified_metrics.parquet`, optional `attributions.parquet`, attribution
summary Parquets, and vector PDF plus PNG plots under `plots/`.

`predictions.parquet` uses stable `peptide_id` values (a supplied ID or a hash
of sequence, DIA-NN predictor sequence, and charge). `attributions.parquet`
is long format with one model token position per row and both chemical
(`modified_sequence`) and predictor representations. These fields make later
model-to-model joins explicit rather than reconstructed from filenames.

Captum groups only equal-length encoded peptides and attributes each group in
GPU batches (`captum.peptide_batch_size`, default 32). This preserves DIA-NN's
native no-padding input representation while avoiding thousands of tiny
one-peptide GPU calls. `internal_batch_size` additionally batches the
Integrated-Gradients interpolation points. Checkpoints are written after the
configured number of completed peptides, so interrupted runs resume without
recomputing completed peptide IDs.

The exported DIA-NN model is TorchScript, which cannot accept Captum layer
hooks. Captum therefore uses ordinary `IntegratedGradients` on continuous
embedding vectors and an exact, parity-checked adapter for the original
CNN/GRU/output path. It differentiates embeddings, not integer token IDs. The default `boundary_only`
reference retains charge and `<`/`>` boundary tokens and places token ID 0 at
interior positions. It is a configurable reference, not an unspoken padding
assumption; `pad` and `zeros` are alternatives for sensitivity checks. The
manifest records token-zero aliases and the reference. Signed position scores
are the sum over embedding dimensions; absolute scores are their absolute
values. Captum's convergence delta is retained per peptide. The run stops
before writing attributions if the adapter fails to reproduce native DIA-NN
predictions.

Attribution is a contribution to this model output relative to its chosen IG
reference. It is not by itself a causal statement about peptide chemistry.

The command validates dictionary IDs against the model embedding table before
prediction. It also fails clearly when the model file is a Git-LFS pointer;
run `git lfs pull` or pass a real local DIA-NN model in that case. Numerical
parity should be checked against a known DIA-NN result before scientific use;
the smoke-test output is intended as the auditable input/output record for it.
