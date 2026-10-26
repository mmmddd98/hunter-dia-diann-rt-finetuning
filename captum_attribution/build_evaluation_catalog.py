"""Build a human-reviewable evaluation registry from external DIA-NN packages.

Nothing in the emitted registry is scheduled automatically: ``enabled`` is
always false. The registry records the packaged model and dictionary as a pair
and makes the training scope/alignment target visible before any attribution
job is launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path("rt_model_evaluations")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_dataset(scope: str | None) -> str | None:
    mapping = {
        "human_hunter": "markus_hunter_human_v1",
        "triplemix_fabian": "triplemix_fabian_entrap_neu_filtered_v1",
        "triplemix_paula": "triplemix_pd_hamilton_60min_v1",
        "celegans_exploris": "celegans_exploris_44min_v1",
    }
    return mapping.get(scope or "")


def dataset_path(
    target: str,
    source: str | None,
    data_root: Path,
    source_native_root: Path | None = None,
) -> Path:
    if target == "source_native":
        if source_native_root is None:
            raise ValueError(
                "Source-native models require --source-native-root with prepared per-source test tables"
            )
        split = source_native_root / "test"
    else:
        split = data_root / target / "test"
    return (
        split / "sources" / f"{source}.tsv"
        if source
        else split / "diann_rt_input" / "rt_test_data.tsv"
    )


def build(
    cluster_root: Path,
    output_root: Path,
    data_root: Path,
    source_native_root: Path | None = None,
) -> pd.DataFrame:
    # A model must remain paired with its dictionary, including across copies.
    canonical: dict[tuple[str, str], dict] = {}
    for manifest_path in sorted(
        cluster_root.glob("*/external_searches/*/*/model_package/model_manifest.json")
    ):
        manifest = json.loads(manifest_path.read_text())
        model = manifest_path.parent / "model" / "rt.d0.finetuned.pt"
        dictionary = manifest_path.parent / "model" / "dict.txt"
        if not model.is_file() or not dictionary.is_file():
            continue
        model_hash = digest(model)
        dictionary_hash = digest(dictionary)
        identity = (model_hash, dictionary_hash)
        if identity in canonical:
            continue
        cfg = manifest.get("configuration", {})
        target = cfg.get("target", cfg.get("mode", "unknown"))
        scope = cfg.get("scope")
        scope_kind = cfg.get("scope_kind", "aligned_or_combined")
        source = source_dataset(scope)
        # Source-specific models default to the matching source only. For
        # aligned/combined models the all-source held-out test is appropriate.
        if scope_kind == "source_specific" and source is None:
            raise ValueError(f"Unknown source-specific training scope: {scope!r}")
        evaluation_scope = (
            "in_scope_only"
            if scope_kind == "source_specific"
            else "all_heldout_sources"
        )
        ds = dataset_path(
            target,
            source if evaluation_scope == "in_scope_only" else None,
            data_root,
            source_native_root,
        )
        family = (
            "source_specific_models"
            if scope_kind == "source_specific"
            else "aligned_or_combined_models"
        )
        model_name = f"{manifest.get('model_id', model_hash[:12])}__{target}__{scope or 'all'}__{model_hash[:8]}_{dictionary_hash[:8]}"
        canonical[identity] = {
            "enabled": False,
            "model_name": model_name,
            "model_id": manifest.get("model_id"),
            "model_sha256": model_hash,
            "model_path": str(model),
            "dictionary_path": str(dictionary),
            "dictionary_sha256": dictionary_hash,
            "family": family,
            "training_scope": scope or "all",
            "training_scope_kind": scope_kind,
            "alignment_target": target,
            "source_sweep": manifest.get("source_sweep"),
            "source_run": manifest.get("source_run"),
            "package_name": manifest.get("package_name"),
            "package_path": str(manifest_path.parent),
            "default_evaluation_scope": evaluation_scope,
            "dataset_name": f"{target}__{source or 'all_heldout'}",
            "dataset_path": str(ds),
            "dataset_exists": ds.is_file(),
            "sequence_column": "sequence",
            "observed_rt_column": "rt_norm",
            "mods_column": "mods",
            "mod_sites_column": "mod_sites",
            "organism_column": "source_organisms",
            "semi_specific_column": "is_semi_tryptic",
            "external_diann_results_path": "",
            "notes": "Review enabled and test scope before scheduling.",
        }
    if not canonical:
        raise ValueError(f"No complete model packages found under {cluster_root}")
    frame = pd.DataFrame(canonical.values()).sort_values(
        ["family", "alignment_target", "training_scope", "model_name"]
    )
    registry_dir = output_root / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(registry_dir / "evaluation_catalog.tsv", sep="\t", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-native-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    frame = build(
        args.cluster_root.resolve(),
        args.output_root.resolve(),
        args.data_root.resolve(),
        args.source_native_root.resolve() if args.source_native_root else None,
    )
    print(
        f"Wrote {len(frame)} selectable models to {args.output_root / 'registry/evaluation_catalog.tsv'}"
    )


if __name__ == "__main__":
    main()
