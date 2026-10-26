import pandas as pd

from diann_finetune.diagnostics import _add_strata


def test_heavy_site_groups_require_exact_heavy_modification_at_each_site():
    frame = pd.DataFrame(
        {
            "sequence": ["PEPK", "PEPK", "PEPK", "PEPK"],
            "mods": [
                "Dimethyl:2H(4)13C(2)@N-term",
                "Dimethyl:2H(4)13C(2)@K",
                "Dimethyl:2H(4)13C(2)@N-term;Dimethyl:2H(4)13C(2)@K",
                "Dimethyl:2H(4)13C(2)@N-term;Dimethyl@K",
            ],
            "mod_sites": ["0", "4", "0;4", "0;4"],
            "rt_norm": [0.1, 0.2, 0.3, 0.4],
            "rt_pred_pretrained": [0.11, 0.21, 0.31, 0.41],
            "rt_pred_finetuned": [0.1, 0.2, 0.3, 0.4],
        }
    )
    annotated = _add_strata(frame)
    assert annotated.heavy_site.tolist() == [
        "n_term",
        "lysine",
        "n_term_and_lysine",
        "n_term",
    ]
