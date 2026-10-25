import copy

import numpy as np
import pandas as pd
import pytest

from matlab_annotation.isomer_label_pairs import isomer_training_pairs


def record(project="a"):
    pool = pd.DataFrame(
        dict(
            stream=["RP_Pos"] * 7,
            name=[
                "PC-P(36:2)",
                "PC-O(36:3)",
                "PC-O(36:3)",
                "PC(34:3)",
                "PC-O(38:3)",
                "PC-P(40:2)",
                "IS: PC-O(36:3)",
            ],
            adduct=["H", "H", "H", "Na", "H", "H", "H"],
            **{"class": ["PC-P", "PC-O", "PC-O", "PC", "PC-O", "PC-P", "PC-O"]},
            feat_feature_id=["1", "1", "1", "1", "1", "2", "1"],
            ref_mz=[700.0, 700.0, 700.0, 700.0, 701.0, 750.0, 700.0],
            is_is=[False] * 7,
            training_positive=[True, False, False, False, False, False, False],
            training_label_known=[True, True, True, True, True, True, True],
        )
    )
    evidence = pd.DataFrame(
        dict(
            stream=pool.stream,
            **{"class": pool["class"]},
            rt_error=[1.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0],
        )
    )
    return dict(
        query_group=project,
        reference_training_groups=sorted({"a", "b", "c"} - {project}),
        reference_excluded_groups=[project, "z"],
        pool=pool,
        evidence=evidence,
    )


def pairs(records):
    return isomer_training_pairs(
        records, training_groups={"a", "b", "c"}, excluded_groups={"z"}
    )


def test_only_same_feature_adduct_mass_known_distinct_endogenous_labels_enter_pairs():
    data = record()
    original = copy.deepcopy(data)
    result = pairs({"a": data})
    assert result["positive"].tolist() == [1, 0]
    assert result["evidence"]["difference__rt_error"].tolist() == [-2, 2]
    assert result["class_pair_projects"] == {("PC[O]", "PC[P]"): ["a"]}
    assert result["features_by_project"] == {"a": 1}
    assert result["supervised_contributing_projects"] == ["a"]
    assert result["training_reference_permissions"] == ["a", "b", "c"]
    assert not any(
        "name" in c or "feature_id" in c or "training_" in c for c in result["evidence"]
    )
    pd.testing.assert_frame_equal(data["pool"], original["pool"], check_exact=True)
    pd.testing.assert_frame_equal(
        data["evidence"], original["evidence"], check_exact=True
    )


def test_unknown_alternatives_and_entirely_manual_absent_features_are_not_guessed():
    data = record()
    data["pool"].loc[[1, 2], "training_label_known"] = False
    with pytest.raises(ValueError, match="No permitted"):
        pairs({"a": data})


def test_duplicate_conflicts_multiple_positive_labels_and_wrong_scope_fail_closed():
    data = record()
    data["pool"].loc[2, "training_positive"] = True
    with pytest.raises(ValueError, match="duplicate"):
        pairs({"a": data})
    data = record()
    data["pool"].loc[3, "training_positive"] = True
    with pytest.raises(ValueError, match="Multiple positive"):
        pairs({"a": data})
    data = record("z")
    with pytest.raises(ValueError, match="not an allowed"):
        pairs({"z": data})


def test_feature_balance_then_equal_actual_project_totals_without_invented_groups():
    a, b = record("a"), record("b")
    b["pool"].loc[4, "ref_mz"] = 700.0
    result = pairs({"a": a, "b": b})
    assert result["pair_rows_by_project"] == {"a": 2, "b": 4}
    assert result["features_by_project"] == {"a": 1, "b": 1}
    assert result["supervised_contributing_projects"] == ["a", "b"]
    for g in ["a", "b"]:
        np.testing.assert_allclose(
            result["sample_weight"][result["groups"] == g].sum(), 3.0
        )
