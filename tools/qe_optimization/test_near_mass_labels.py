import pandas as pd

from diagnose_qe_near_mass_labels import physical_pairs


def test_numeric_near_equivalence_only_same_feature_channel_and_never_claims_formula():
    p = pd.DataFrame(
        dict(
            stream="NP_Neg",
            feat_feature_id=["a", "a", "a", "b", "a", "a"],
            name=[
                "BMP(34:1)",
                "PG(34:1)",
                "other",
                "third",
                "spike (IS)",
                "different ion",
            ],
            adduct=["[M-H]-"] * 5 + ["[M+HCOO]-"],
            ref_mz=[700.0, 700.00001, 700.001, 700.0, 700.0, 700.0],
            **{"class": ["BMP", "PG", "other", "third", "BMP", "PG"]},
            is_is=[False] * 4 + [True, False],
        )
    )
    result = physical_pairs(p)
    assert len(result) == 1 and result.numerical_scope.iloc[0] == "new_near"
    assert result.left_class.iloc[0] == "BMP" and result.right_class.iloc[0] == "PG"
    p.loc[1, "ref_mz"] = 700.0
    assert physical_pairs(p).numerical_scope.tolist() == ["exact"]
