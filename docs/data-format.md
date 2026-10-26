# RT tables and model packages

## Input schema

Training and validation files are UTF-8 TSVs. The prepared-split CLI accepts
one file per partition and checks exact `(sequence, mods, mod_sites)` overlap.

| Column | Required | Interpretation |
| --- | --- | --- |
| `sequence` | Yes | Amino-acid sequence; `peptide` is accepted as an alias |
| `rt_norm` | Yes | Finite numeric RT target in the coordinate shared by both partitions |
| `mods` | No | Semicolon-separated Carafe/AlphaBase modification names; defaults to empty |
| `mod_sites` | No | Matching positions; residues start at 1 and N-terminus is 0 |
| `source_organism` | No | Source label preserved during aggregation and used by organism sampling |
| `rt` | No | Observed RT in minutes, preserved separately during aggregation |

For example, the final line has heavy dimethylation at the N-terminus and at
lysine position 8:

```tsv
sequence	mods	mod_sites	rt_norm
PEPTIDEK			0.42
PEPTIDEK	Dimethyl:2H(4)13C(2)@N-term	0	0.44
PEPTIDEK	Dimethyl:2H(4)13C(2)@N-term;Dimethyl:2H(4)13C(2)@K	0;8	0.47
```

These rows illustrate the schema; they are not a training benchmark. Put an
identical peptidoform in only one partition. The internal sequence-split mode
requires at least two distinct sequences. Duplicate observations are reduced
to a median per peptidoform and, when present, source organism. Missing or
nonnumeric targets are excluded; infinite targets and empty usable datasets
raise an error. Unsupported modifications raise an error unless
`--unknown-mods drop` explicitly excludes them.

`rt_norm` names the target column; it does not imply every experiment uses a
0–1 scale. Preparation must fit alignment on training observations and reuse
the mapping for validation and final tests. Compare target-space errors only
within the same coordinate, or use the external evaluation tools to convert
to observed minutes.

## Model input and export

The input model directory must contain a real DIA-NN 2.5.0 `rt.d0.pt` and its
`dict.txt`. Obtain both externally. A Git-LFS pointer is not a model file.
The trainer adds heavy-dimethyl tokens 40/41 from light-dimethyl embeddings
38/39, and exports the extended dictionary with every tuned model.

Deploy these two files together:

```text
model_package/
  rt.d0.finetuned.pt
  dict.txt
```

`rt.d0.finetuned.pt` is selected using validation MAE;
`rt.d0.last.pt` contains the final epoch. Exported TorchScript tensors and the
scripted device attribute are saved on CPU for portable DIA-NN loading.
The package is a predictor replacement, not a complete DIA-NN installation.

## Run artifacts

Training writes split TSVs, `training_history.csv`, validation predictions,
`subgroup_metrics.csv` and `run_metadata.json`. PNG/HTML diagnostics are enabled
unless `--no-plots` is supplied. Prepared final test data must be evaluated
separately after checkpoint selection. Local paths in generated manifests
support reproducibility; review generated artifacts before publishing them.
