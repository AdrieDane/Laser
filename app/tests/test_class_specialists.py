import numpy as np
import pandas as pd
import pytest

from matlab_annotation.class_specialists import (
    cohort_balanced_weights,
    specialist_training_plan,
)


def test_balances_groups_without_upweighting_large_projects():
    groups = pd.Series(["a"] * 10 + ["b"] * 2)
    weights = cohort_balanced_weights(groups)
    assert np.mean(weights) == pytest.approx(1)
    assert weights[:10].sum() == pytest.approx(weights[10:].sum())
    assert len(cohort_balanced_weights(pd.Series(dtype=str))) == 0


def test_specialist_requires_multiple_positive_groups_and_both_outcomes():
    groups = pd.Series(list("abcdef") * 50)
    labels = pd.Series([True] * 150 + [False] * 150)
    classes = pd.Series(["RP_Pos|PE"] * 300)
    assert (
        specialist_training_plan(classes, labels, groups)["RP_Pos|PE"]["action"]
        == "specialist"
    )
    groups[labels] = "one"
    assert (
        specialist_training_plan(classes, labels, groups)["RP_Pos|PE"]["action"]
        == "global_fallback"
    )


def test_zero_positive_gate_requires_group_support_and_does_not_invent_keys():
    groups = pd.Series(list("abcdef"))
    classes = pd.Series(["NP_Neg|X"] * 6)
    labels = pd.Series([False] * 6)
    plan = specialist_training_plan(classes, labels, groups)
    assert plan["NP_Neg|X"]["action"] == "withhold_zero_positive_support"
    assert "RP_Pos|unseen" not in plan
    assert (
        specialist_training_plan(classes[:5], labels[:5], groups[:5])["NP_Neg|X"][
            "action"
        ]
        == "global_fallback"
    )


def test_unknown_labels_or_misaligned_rows_fail_closed():
    classes, groups = pd.Series(["x", "x"]), pd.Series(["a", "b"])
    with pytest.raises(ValueError):
        specialist_training_plan(classes, pd.Series([True, None]), groups)
    with pytest.raises(ValueError):
        specialist_training_plan(
            classes, pd.Series([True, False], index=[1, 2]), groups
        )
