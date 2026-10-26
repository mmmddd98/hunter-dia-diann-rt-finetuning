"""Invalid external inputs must fail before creating run artifacts."""

import sys

import pytest

from train_carafe_rt import main


def test_missing_pretrained_model_does_not_create_run(tmp_path, monkeypatch):
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_carafe_rt.py",
            "--train-split",
            str(tmp_path / "train.tsv"),
            "--validation-split",
            str(tmp_path / "validation.tsv"),
            "--model-dir",
            str(tmp_path / "missing_models"),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ],
    )
    with pytest.raises(SystemExit, match="Required input file not found"):
        main()
    assert not output.exists()


def test_prepared_split_training_exports_model_dictionary_and_metadata(
    tmp_path, monkeypatch
):
    import json

    import pandas as pd
    import torch

    from diann_finetune.encoding import read_tokens

    class SyntheticRTModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(256, 4)
            self.out = torch.nn.Linear(4, 1)
            self.device = "cpu"

        def forward(self, tokens):
            return self.out(self.embedding(tokens[:, 1:]).mean(dim=1))

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    torch.jit.save(torch.jit.script(SyntheticRTModel()), str(model_dir / "rt.d0.pt"))
    tokens = {residue: index for index, residue in enumerate("ACDEFGHIKLMNPQRSTVWY", 1)}
    tokens.update({"<": 21, ">": 22, "<(UniMod:255)": 38, "K(UniMod:255)": 39})
    (model_dir / "dict.txt").write_text(
        "".join(f"{token}\t{index}\n" for token, index in tokens.items())
    )
    train = tmp_path / "train.tsv"
    validation = tmp_path / "validation.tsv"
    pd.DataFrame(
        {
            "sequence": ["PEPTIDEK", "PEPTIDER", "PEPTIDEM", "PEPTIDEC"],
            "rt_norm": [0.2, 0.4, 0.6, 0.8],
        }
    ).to_csv(train, sep="\t", index=False)
    pd.DataFrame(
        {
            "sequence": ["AEPTIDEK", "AEPTIDER", "AEPTIDEM", "AEPTIDEC"],
            "rt_norm": [0.25, 0.45, 0.65, 0.85],
        }
    ).to_csv(validation, sep="\t", index=False)
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_carafe_rt.py",
            "--train-split",
            str(train),
            "--validation-split",
            str(validation),
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--warmup-epochs",
            "0",
            "--batch-size",
            "2",
            "--no-plots",
        ],
    )
    main()
    model = torch.jit.load(str(output / "rt.d0.finetuned.pt"))
    assert next(model.parameters()).device.type == "cpu"
    assert model.device == "cpu"
    assert read_tokens(output / "dict.txt")["K(UniMod:510)"] == 41
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["split"]["split_source"] == "prepared_peptidoform_disjoint_files"
    assert (output / "validation_predictions.tsv").is_file()
    assert (output / "training_history.csv").is_file()
