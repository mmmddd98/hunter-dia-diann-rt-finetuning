"""Resolve versioned Carafe fine-tuning builds from their catalog IDs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


DEFAULT_CATALOG = Path(
    os.environ.get(
        "FINE_TUNE_CATALOG",
        "datasets.yaml",
    )
).expanduser()


def resolve_catalog_build(dataset_id: str, catalog: Path | None = None) -> Path:
    """Create a registered build if necessary and return its root directory."""
    catalog_path = (catalog or DEFAULT_CATALOG).resolve()
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Fine-tuning catalog not found: {catalog_path}")
    builder = catalog_path.parent.parent / "fine_tune_catalog.py"
    if not builder.is_file():
        raise FileNotFoundError(
            "Catalog builder missing. Pass --catalog pointing to "
            "data_collect/finetune_catalog/datasets.yaml, set FINE_TUNE_CATALOG, "
            "or use --in-dir / --rt-data with already prepared inputs."
        )
    completed = subprocess.run(
        [
            sys.executable,
            str(builder),
            "--catalog",
            str(catalog_path),
            "ensure",
            "--build",
            dataset_id,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    lines = completed.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError("Catalog builder returned no build directory")
    # The builder prints its completed directory on the final output line.
    path = Path(lines[-1].rsplit(": ", 1)[-1])
    if not (path / "diann_rt_input" / "rt_train_data.tsv").is_file():
        raise RuntimeError(f"Catalog build did not produce DIA-NN RT input: {path}")
    return path
