"""Portable catalog inputs must preserve model identity and held-out scope."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from captum_attribution.build_evaluation_catalog import build
from diann_finetune.catalog import resolve_catalog_build


def package(root, name, dictionary="tokens", target="joint_anchor", scope="human_hunter"):
    directory = root / name / "external_searches" / "search" / "run" / "model_package"
    (directory / "model").mkdir(parents=True)
    (directory / "model" / "rt.d0.finetuned.pt").write_bytes(b"same-model")
    (directory / "model" / "dict.txt").write_text(dictionary)
    (directory / "model_manifest.json").write_text(json.dumps({
        "model_id": "test-model",
        "configuration": {"target": target, "scope": scope, "scope_kind": "source_specific"},
    }))


def test_registry_keeps_dictionary_pairs_and_matching_source(tmp_path):
    packages = tmp_path / "packages"
    package(packages, "first")
    package(packages, "copy")
    package(packages, "other_dictionary", dictionary="different tokens")
    targets = tmp_path / "targets"
    expected = targets / "joint_anchor/test/sources/markus_hunter_human_v1.tsv"
    expected.parent.mkdir(parents=True)
    expected.write_text("sequence\trt_norm\nPEPTIDE\t0.5\n")
    frame = build(packages, tmp_path / "evaluation", targets)
    assert len(frame) == 2  # Deployment copies merge; different dictionaries do not.
    assert frame.model_name.nunique() == 2
    assert frame.dataset_path.eq(str(expected)).all()
    assert frame.dataset_exists.all()
    assert not frame.enabled.any()


def test_source_native_requires_explicit_root_and_preserves_source_scope(tmp_path):
    packages = tmp_path / "packages"
    package(packages, "native", target="source_native")
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="source-native-root"):
        build(packages, output, tmp_path / "targets")
    assert not output.exists()
    native = tmp_path / "native_targets"
    frame = build(packages, output, tmp_path / "targets", native)
    assert frame.iloc[0].dataset_path == str(native / "test/sources/markus_hunter_human_v1.tsv")


def test_unknown_source_does_not_fall_back_to_combined_test(tmp_path):
    packages = tmp_path / "packages"
    package(packages, "unknown", scope="unrecognised")
    with pytest.raises(ValueError, match="Unknown source-specific"):
        build(packages, tmp_path / "evaluation", tmp_path / "targets")


def test_custom_catalog_selects_its_own_builder(tmp_path, monkeypatch):
    data_collect = tmp_path / "data_collect"
    catalog = data_collect / "finetune_catalog/datasets.yaml"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("{}")
    builder = data_collect / "fine_tune_catalog.py"
    builder.write_text("# external builder")
    result = tmp_path / "prepared"
    (result / "diann_rt_input").mkdir(parents=True)
    (result / "diann_rt_input/rt_train_data.tsv").write_text("sequence\trt_norm\n")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"Prepared: {result}\n")

    monkeypatch.setattr("diann_finetune.catalog.subprocess.run", run)
    assert resolve_catalog_build("example", catalog) == result
    assert Path(calls[0][1]) == builder


def test_missing_catalog_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="catalog not found"):
        resolve_catalog_build("example", tmp_path / "missing.yaml")
