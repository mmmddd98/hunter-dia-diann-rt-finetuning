import pandas as pd
import pytest

from diann_finetune.data import (
    carafe_sequence_split,
    heavy_dimethyl_row_strata,
    load_carafe_rt,
    nested_stratified_sequence_subset,
    nested_training_subset,
    oversample_heavy_dimethyl_rows,
    stratified_sequence_split,
)


def test_heavy_dimethyl_oversampling_is_stratified_and_deterministic():
    frame = pd.DataFrame(
        {
            "sequence": ["NORMAL", "NTERM", "LYSINE", "BOTH"],
            "mods": [
                "",
                "Dimethyl:2H(4)13C(2)@N-term",
                "Dimethyl:2H(4)13C(2)@K",
                "Dimethyl:2H(4)13C(2)@N-term;Dimethyl:2H(4)13C(2)@K",
            ],
        }
    )
    sampled, metadata = oversample_heavy_dimethyl_rows(
        frame, nterm_factor=2.0, lysine_factor=1.0, both_factor=3.0, random_state=7
    )
    assert heavy_dimethyl_row_strata(sampled).value_counts().to_dict() == {
        "heavy_both": 3,
        "heavy_nterm": 2,
        "normal": 1,
        "heavy_lysine": 1,
    }
    assert metadata["added_counts"] == {
        "normal": 0,
        "heavy_nterm": 1,
        "heavy_lysine": 0,
        "heavy_both": 2,
    }
    repeated, _ = oversample_heavy_dimethyl_rows(
        frame, nterm_factor=2.0, lysine_factor=1.0, both_factor=3.0, random_state=7
    )
    assert sampled.equals(repeated)


def test_load_and_sequence_disjoint_split(tmp_path):
    path = tmp_path / "rt_train_data.tsv"
    rows = []
    for index in range(40):
        rows.append(
            {
                "sequence": f"PEPTID{chr(65 + index % 20)}",
                "mods": "",
                "mod_sites": "",
                "rt_norm": index / 40,
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    prepared = load_carafe_rt(path)
    train, validation, metadata = carafe_sequence_split(prepared.frame)
    assert set(train.sequence).isdisjoint(set(validation.sequence))
    assert metadata["source_rows"] == len(prepared.frame)


@pytest.mark.parametrize("target", [float("inf"), float("-inf")])
def test_load_rejects_infinite_rt_targets(tmp_path, target):
    path = tmp_path / "targets.tsv"
    pd.DataFrame({"sequence": ["PEPTIDE"], "rt_norm": [target]}).to_csv(
        path, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="finite"):
        load_carafe_rt(path)


def test_load_rejects_empty_usable_table(tmp_path):
    path = tmp_path / "targets.tsv"
    path.write_text("sequence\trt_norm\nPEPTIDE\tmissing\n")
    with pytest.raises(ValueError, match="No usable"):
        load_carafe_rt(path)


def test_sequence_split_requires_two_distinct_sequences():
    frame = pd.DataFrame({"sequence": ["PEPTIDE", "PEPTIDE"], "rt_norm": [0.2, 0.3]})
    with pytest.raises(ValueError, match="two distinct sequences"):
        carafe_sequence_split(frame)


def test_load_preserves_source_organism_during_aggregation(tmp_path):
    path = tmp_path / "organism_test.tsv"
    pd.DataFrame(
        {
            "sequence": ["PEPTIDE", "PEPTIDE", "PEPTIDE"],
            "mods": ["", "", ""],
            "mod_sites": ["", "", ""],
            "rt_norm": [0.2, 0.4, 0.8],
            "source_organism": ["human", "human", "celegans"],
        }
    ).to_csv(path, sep="\t", index=False)

    prepared = load_carafe_rt(path)

    assert "source_organism" in prepared.frame.columns
    values = prepared.frame.set_index("source_organism")["rt_norm"]
    assert values["human"] == pytest.approx(0.3)
    assert values["celegans"] == pytest.approx(0.8)


def test_load_preserves_observed_rt_during_aggregation(tmp_path):
    path = tmp_path / "organism_test.tsv"
    pd.DataFrame(
        {
            "sequence": ["PEPTIDE", "PEPTIDE", "PEPTIDE"],
            "mods": ["", "", ""],
            "mod_sites": ["", "", ""],
            "rt_norm": [0.2, 0.4, 0.8],
            "rt": [12.0, 18.0, 30.0],
            "source_organism": ["human", "human", "celegans"],
        }
    ).to_csv(path, sep="\t", index=False)

    prepared = load_carafe_rt(path)

    values = prepared.frame.set_index("source_organism")["rt"]
    assert values["human"] == pytest.approx(15.0)
    assert values["celegans"] == pytest.approx(30.0)


def test_nested_training_subsets_are_reproducible_and_nested():
    frame = pd.DataFrame({"row": range(20)})
    quarter = nested_training_subset(frame, 0.25, random_state=2024)
    half = nested_training_subset(frame, 0.5, random_state=2024)
    full = nested_training_subset(frame, 1.0, random_state=2024)

    assert len(quarter) == 5
    assert len(half) == 10
    assert set(quarter["row"]).issubset(set(half["row"]))
    assert full["row"].tolist() == frame["row"].tolist()
    assert quarter.equals(nested_training_subset(frame, 0.25, random_state=2024))


def test_nested_training_subset_rejects_invalid_fraction():
    frame = pd.DataFrame({"row": range(5)})
    for fraction in (0.0, -0.5, 1.1):
        try:
            nested_training_subset(frame, fraction)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected fraction {fraction} to fail")


def test_stratified_sequence_split_and_nested_sequence_subsets():
    rows = []
    for index in range(40):
        mods = ""
        if index % 10 == 0:
            mods = "Dimethyl:2H(4)13C(2)@N-term"
        elif index % 10 == 1:
            mods = "Dimethyl:2H(4)13C(2)@K"
        elif index % 10 == 2:
            mods = "Dimethyl:2H(4)13C(2)@N-term;Dimethyl:2H(4)13C(2)@K"
        rows.append({"sequence": f"SEQ{index:03d}", "mods": mods, "rt_norm": index / 40})
    frame = pd.DataFrame(rows)
    train, validation, _ = stratified_sequence_split(frame, random_state=7)
    assert set(train.sequence).isdisjoint(set(validation.sequence))
    quarter = nested_stratified_sequence_subset(train, .25, random_state=7)
    half = nested_stratified_sequence_subset(train, .5, random_state=7)
    assert set(quarter.sequence).issubset(set(half.sequence))
    assert set(half.sequence).issubset(set(train.sequence))
