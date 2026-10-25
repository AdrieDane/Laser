import copy

import numpy as np
import pandas as pd
import pytest

from matlab_annotation.conditional_peak_pairs import (
    conditional_training_pairs,
    pair_predictors,
)


def record(project="a"):
    pool = pd.DataFrame(
        dict(
            stream=["RP_Pos"] * 6,
            name=["PE(36:2)"] * 3 + ["PC(36:2)"] * 3,
            adduct=["H"] * 6,
            feat_feature_id=["1", "2", "3", "4", "5", "6"],
            training_positive=[True, False, False, False, False, False],
            training_label_known=[True, True, True, True, True, False],
        )
    )
    evidence = pd.DataFrame(
        dict(
            stream=["RP_Pos"] * 6,
            klass=["PE"] * 3 + ["PC"] * 3,
            rt_error=[1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        )
    )
    return dict(
        query_group=project,
        reference_training_groups=sorted({"a", "b", "c"} - {project}),
        reference_excluded_groups=[project, "z"],
        pool=pool,
        evidence=evidence,
    )


def test_mirrored_pairs_have_balanced_labels_and_no_identity_predictors():
    data = record()
    before = copy.deepcopy(data)
    result = conditional_training_pairs(
        {"a": data}, training_groups={"a", "b", "c"}, excluded_groups={"z"}
    )
    assert result["positive"].tolist() == [1, 1, 0, 0]
    x = result["evidence"]
    np.testing.assert_allclose(
        x["difference__rt_error"].to_numpy(), [-1, np.nan, 1, np.nan], equal_nan=True
    )
    assert x["missing_difference__rt_error"].tolist() == [0, -1, 0, 1]
    assert not any("name" in c or "feature_id" in c or "training_" in c for c in x)
    assert result["supervised_contributing_projects"] == ["a"]
    assert result["training_reference_permissions"] == ["a", "b", "c"]
    assert result["identities_by_project"] == {"a": 1}
    pd.testing.assert_frame_equal(data["pool"], before["pool"], check_exact=True)
    pd.testing.assert_frame_equal(
        data["evidence"], before["evidence"], check_exact=True
    )


def test_equal_project_weight_and_no_silent_invented_supervised_rows():
    a, b = record("a"), record("b")
    b["pool"].loc[2, "training_label_known"] = False
    result = conditional_training_pairs(
        {"a": a, "b": b}, training_groups={"a", "b", "c"}, excluded_groups={"z"}
    )
    g, w = result["groups"], result["sample_weight"]
    assert w[g == "a"].sum() == pytest.approx(3) and w[g == "b"].sum() == pytest.approx(
        3
    )
    assert result["pair_rows_by_project"] == {"a": 4, "b": 2}
    assert "c" not in result["supervised_contributing_projects"]


def test_unknowns_and_duplicate_physical_rows_do_not_multiply_pairs():
    data = record()
    data["pool"].loc[2, "feat_feature_id"] = "2"
    result = conditional_training_pairs(
        {"a": data}, training_groups={"a", "b", "c"}, excluded_groups={"z"}
    )
    assert len(result["positive"]) == 2
    data["pool"].loc[1, "training_label_known"] = False
    data["pool"].loc[2, "training_label_known"] = False
    with pytest.raises(ValueError, match="No permitted"):
        conditional_training_pairs(
            {"a": data}, training_groups={"a", "b", "c"}, excluded_groups={"z"}
        )


def test_query_scope_conflicting_positives_and_forbidden_predictors_rejected():
    data = record()
    with pytest.raises(ValueError, match="not an allowed"):
        conditional_training_pairs(
            {"a": data}, training_groups={"b", "c"}, excluded_groups={"a", "z"}
        )
    data["pool"].loc[1, "training_positive"] = True
    with pytest.raises(ValueError, match="multiple positive"):
        conditional_training_pairs(
            {"a": data}, training_groups={"a", "b", "c"}, excluded_groups={"z"}
        )
    a = pd.DataFrame({"feat_feature_id": ["1"]})
    with pytest.raises(ValueError, match="supervision"):
        pair_predictors(a, a)


def test_inconsistent_duplicate_labels_and_duplicate_row_indices_fail_closed():
    data = record()
    data["pool"].loc[2, "feat_feature_id"] = "2"
    data["pool"].loc[2, "training_label_known"] = False
    with pytest.raises(ValueError, match="duplicate physical"):
        conditional_training_pairs(
            {"a": data}, training_groups={"a", "b", "c"}, excluded_groups={"z"}
        )
    data = record()
    data["pool"].index = [0, 0, 2, 3, 4, 5]
    data["evidence"].index = data["pool"].index
    with pytest.raises(ValueError, match="indices must be unique"):
        conditional_training_pairs(
            {"a": data}, training_groups={"a", "b", "c"}, excluded_groups={"z"}
        )


def test_categorical_sides_swap_and_numeric_midpoints_are_symmetric():
    left = pd.DataFrame({"source": ["fitted"], "error": [1.0]})
    right = pd.DataFrame({"source": ["atlas"], "error": [3.0]})
    forward, reverse = pair_predictors(left, right), pair_predictors(right, left)
    assert (
        forward["left__source"].iloc[0] == reverse["right__source"].iloc[0] == "fitted"
    )
    assert (
        forward["right__source"].iloc[0] == reverse["left__source"].iloc[0] == "atlas"
    )
    assert forward["midpoint__error"].iloc[0] == reverse["midpoint__error"].iloc[0] == 2
    assert (
        forward["difference__error"].iloc[0]
        == -reverse["difference__error"].iloc[0]
        == -2
    )
