import pandas as pd
import torch

from captum_attribution.core import compatibility_check, metric_dict, prepare_dataset, stable_peptide_id
from diann_finetune.encoding import TokenEncoder


def test_stable_id_and_prediction_dataset_schema(tmp_path):
    data = tmp_path / "tiny.tsv"
    pd.DataFrame({"sequence": ["PEPTIDE", "KPEPTIDE"], "rt": [0.2, 0.4], "id": ["a", "b"]}).to_csv(data, sep="\t", index=False)
    result = prepare_dataset({"path": str(data), "sequence_column": "sequence", "observed_rt_column": "rt", "peptide_id_column": "id"})
    assert {"peptide_id", "predictor_sequence", "stripped_sequence", "peptide_length", "observed_rt"}.issubset(result)
    assert stable_peptide_id("PEP", "PEP", 2) == stable_peptide_id("PEP", "PEP", 2)
    assert metric_dict([1, 2], [1, 2])["mae"] == 0


def test_carafe_missing_modifications_are_not_tokenised_as_nan(tmp_path):
    data = tmp_path / "tiny.tsv"
    pd.DataFrame({"sequence": ["PEPTIDE"], "mods": [None], "sites": [None], "rt": [.2]}).to_csv(data, sep="\t", index=False)
    result = prepare_dataset({"path": str(data), "sequence_column": "sequence", "observed_rt_column": "rt", "mods_column": "mods", "mod_sites_column": "sites"})
    assert result.predictor_sequence.tolist() == ["PEPTIDE"]


def test_encoder_and_model_dictionary_compatibility():
    tokens = {"<": 1, ">": 2, "P": 3, "E": 4}
    assert TokenEncoder(tokens).tokenize("PEP") == [1, 3, 4, 3, 2]
    model = torch.nn.Module(); model.embedding = torch.nn.Embedding(8, 3)
    details = compatibility_check(model, tokens)
    assert details["dictionary_max_token_id"] == 4
