"""Turn explicitly enabled catalog rows into one evaluation YAML per row."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml

DEFAULT_ROOT = Path("rt_model_evaluations")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    args.root = args.root.resolve()
    catalog = pd.read_csv(
        args.root / "registry/evaluation_catalog.tsv", sep="\t", keep_default_na=False
    )
    enabled = catalog[
        catalog.enabled.astype(str).str.lower().isin(["true", "1", "yes"])
    ]
    config_dir = args.root / "registry/generated_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for row in enabled.itertuples(index=False):
        config = {
            "evaluation_name": "v1",
            "model": {
                "name": row.model_name,
                "path": row.model_path,
                "token_dictionary": row.dictionary_path,
                "model_type": "rt",
                "training_scope": row.training_scope,
                "training_scope_kind": row.training_scope_kind,
                "alignment_target": row.alignment_target,
                "model_sha256": row.model_sha256,
            },
            "dataset": {
                "name": row.dataset_name,
                "path": row.dataset_path,
                "sequence_column": row.sequence_column,
                "observed_rt_column": row.observed_rt_column,
                "mods_column": row.mods_column,
                "mod_sites_column": row.mod_sites_column,
                "organism_column": row.organism_column,
                "semi_specific_column": row.semi_specific_column,
            },
            "output": {"root": str(args.root / row.family)},
            "runtime": {"device": "auto", "batch_size": 64},
            "captum": {
                "baseline": "boundary_only",
                "n_steps": 64,
                "peptide_batch_size": 32,
                "internal_batch_size": 256,
                "checkpoint_every": 100,
            },
        }
        (config_dir / f"{row.model_name}__{row.dataset_name}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    print(f"Materialized {len(enabled)} config(s) in {config_dir}")


if __name__ == "__main__":
    main()
