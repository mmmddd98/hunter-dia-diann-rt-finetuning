import pytest

from diann_finetune.encoding import (
    TokenEncoder,
    augment_tokens_for_heavy_dimethyl,
    carafe_modified_sequence,
    read_tokens,
)


def test_carafe_modifications_convert_to_diann_tokens(tmp_path):
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text(
        "<\t1\n>\t2\nA\t3\nK(UniMod:255)\t39\n<(UniMod:1)\t29\n<(UniMod:255)\t38\n",
        encoding="utf-8",
    )
    modified, custom_tokens = carafe_modified_sequence(
        "AK", "Acetyl@N-term;Dimethyl:2H(4)13C(2)@K", "0;2"
    )
    assert modified == "(UniMod:1)AK(UniMod:510)"
    assert custom_tokens == ["UniMod:510 K token (ID 41)"]
    tokens, initializations = augment_tokens_for_heavy_dimethyl(read_tokens(dictionary))
    assert initializations == {40: 38, 41: 39}
    assert TokenEncoder(tokens).tokenize(modified) == [29, 3, 41, 2]


def test_invalid_modification_site_is_rejected():
    with pytest.raises(ValueError, match="expects K"):
        carafe_modified_sequence("AA", "Dimethyl:2H(4)13C(2)@K", "2")
