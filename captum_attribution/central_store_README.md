# Central RT evaluation store

Choose an evaluation output root; the default is `rt_model_evaluations/`
relative to the current directory. Source roots must be supplied explicitly:

```bash
python -m captum_attribution.build_evaluation_catalog \
  --cluster-root inputs/model_packages \
  --data-root inputs/prepared_targets \
  --output-root rt_model_evaluations
```

Packages are discovered at
`<cluster-root>/*/external_searches/*/*/model_package/model_manifest.json`.
Test tables use `<data-root>/<target>/test/diann_rt_input/rt_test_data.tsv`,
or `test/sources/<source>.tsv` for source-specific models. Source-native
packages additionally require `--source-native-root` containing the same
`test/` layout. Missing source-native configuration stops catalog generation.
Use the same `--root` for all following commands if you choose another output
root. Keep prepared source-specific tables in their matching source scale.

```text
rt_model_evaluations/
  registry/
    evaluation_catalog.tsv       # the sole selection/scheduling authority
    generated_configs/
  source_specific_models/
    <model-name>__<dataset-name>__v1/
  aligned_or_combined_models/
    <model-name>__<dataset-name>__v1/
```

The catalog is generated from packaged, externally deployed DIA-NN models in
`cluster_out`. It stores the exact `rt.d0.finetuned.pt`/`dict.txt` pair, hashes,
the source training run, alignment target, and proposed test table. Deployment
copies are deduplicated only when both model and dictionary hashes match. Every row
starts disabled. Set `enabled` to `true` only for models approved for analysis,
then run:

```bash
python -m captum_attribution.materialize_catalog_configs
```

The first analysis pass should use `--prediction-only`; run Captum after the
prediction outputs have passed review. Source-specific rows deliberately point
to their matching held-out source. They can later be duplicated with a new
dataset row for cross-organism analysis, but those results must retain the
source-specific alignment target and should not be pooled as raw RT quality
without a common RT scale.

For external DIA-NN validation, put the final cluster result path into
`external_diann_results_path` in the same catalog row. It belongs under the
corresponding evaluation run as `diann_external_validation/`; do not substitute
it for Python inference, as it is an independent validation record.

After approving rows, execute all and only those rows with:

```bash
# Activate the environment described in the evaluation README.
python -m captum_attribution.run_enabled_catalog --smoke-test
python -m captum_attribution.run_enabled_catalog             # prediction-only
python -m captum_attribution.run_enabled_catalog --captum    # resumed/full Captum
python -m captum_attribution.run_enabled_catalog --captum --gpu-workers 3
python -m captum_attribution.run_enabled_catalog --captum --devices cuda:0,cuda:1,cuda:2
```

`--gpu-workers 3` launches up to three independent runs in parallel, one per
GPU (logical IDs 0, 1, 2). Start with one model per GPU; do not use it if the
available GPU memory cannot hold three simultaneous Captum jobs.

For one large model×dataset run, put `devices: ["cuda:0", "cuda:1", "cuda:2"]`
under `runtime` in its generated YAML. The pipeline deterministically splits
peptides among model copies on those GPUs and merges results by `peptide_id`.
The runner's `--devices` option writes this setting automatically. Do not use
`--devices` and `--gpu-workers` together: the former is peptide-parallel within
one run; the latter is model-parallel across independent runs.

The default catalog configuration additionally batches 32 equal-length peptides
per Captum call and 256 Integrated-Gradients interpolation examples internally.
This changes neither the model prediction nor the attribution definition; it
only removes one-peptide GPU dispatch overhead. Each GPU writes its own
`attributions_gpu*.partial.parquet` checkpoint, so restarting with the same
`--devices` setting resumes the completed peptide IDs safely.

When the catalog runner finishes, it creates `catalog_summary/` with a sortable
CSV/Parquet overview, an HTML report, and PNG/PDF plots. Model rankings are
shown only within identical held-out test datasets, never across incompatible
RT alignment scales.
