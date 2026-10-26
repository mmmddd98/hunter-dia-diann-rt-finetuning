# Development

Use the environment from the [main README](../README.md). Run all commands
from the repository root:

```bash
python -m pip install -e '.[test,evaluation]'
python -m pytest -q
python train_carafe_rt.py --help
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
```

The complete test suite includes evaluation/catalog tests and therefore needs
the evaluation extras. Training itself uses only the core dependencies; W&B
is imported only when requested. Fixtures are synthetic and require no DIA-NN
installation, measurement files or pretrained checkpoints. GPU training and
numerical parity with the external DIA-NN binary require a separate integration
run with those prerequisites.

## Working conventions

Keep data preparation separate from training. Preserve peptidoform split IDs,
RT coordinates and dictionary/model pairing when changing loaders or exports.
Validate model-specific changes with regression tests; changes to checkpoint
serialization should retain CPU-load compatibility. Keep optional evaluation
and tracking dependencies out of ordinary training imports.

Configure local paths through CLI flags or environment variables:

| Variable | Purpose |
| --- | --- |
| `DIANN_MODEL_DIR` | Directory containing `rt.d0.pt` and `dict.txt` |
| `FINE_TUNE_CATALOG` | External dataset catalog used with `--dataset-id` |
| `DIANN_FINETUNE_OUTPUT_ROOT` | Output root for optional campaign launchers |
| `CARAFE_DATA_COLLECT_DIR` | External preparation code for organism studies |
| `CARAFE2_STUDY_ROOT` | Matching Carafe2 study for comparison plots |

Never commit local account configuration, environments, checkpoints, runtime
tables or generated reports. `.gitignore` covers conventional output locations;
use an ignored directory or a location outside the repository for other runs.
Scientific source identifiers and public repository/citation links are part
of experiment provenance and should remain stable.

## Distribution

The wheel includes the training package, CLI modules and evaluation package.
Shell launchers and campaign configurations are available in the source tree;
run those workflows from a source checkout. Pretrained DIA-NN weights and
binaries are external prerequisites and are excluded from source snapshots.
