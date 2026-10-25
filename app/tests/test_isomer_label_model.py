import copy

import numpy as np
import pytest

from matlab_annotation.isomer_label_model import (
    fit_isomer_label_model,
    predict_isomer_label_preferences,
)
from test_isomer_label_pairs import record


def model():
    return fit_isomer_label_model(
        {p: record(p) for p in ("a", "b", "c")},
        training_groups={"a", "b", "c"},
        excluded_groups={"z"},
        parameters=dict(
            max_iter=8,
            min_samples_leaf=1,
            max_leaf_nodes=3,
            early_stopping=False,
            random_state=1,
        ),
        minimum_pair_rows=2,
        minimum_contributing_projects=3,
        minimum_class_pair_projects=3,
    )


def test_numerical_preferences_antisymmetric_and_query_supervision_never_read():
    m, data = model(), record("z")
    a = predict_isomer_label_preferences(data, m, [0], [1])
    b = predict_isomer_label_preferences(data, m, [1], [0])
    np.testing.assert_allclose(a + b, 1.0)
    data["pool"]["training_positive"] = "NOT A LABEL"
    data["pool"]["training_label_known"] = None
    np.testing.assert_array_equal(
        a, predict_isomer_label_preferences(data, m, [0], [1])
    )
    assert m["experts"]["RP_Pos"]["supervised_contributing_projects"] == ["a", "b", "c"]
    assert not m["experts"]["NP_Pos"]["supported"]


def test_absent_classpair_support_returns_unknown_not_fabricated_preference():
    m, data = model(), record("z")
    m["minimum_class_pair_projects"] = 4
    assert np.isnan(predict_isomer_label_preferences(data, m, [0], [1])).all()


def test_other_features_mass_adduct_standard_and_reference_leak_rejected():
    m = model()
    for field, value, message in (
        ("feat_feature_id", "2", "Same originalfeature"),
        ("ref_mz", 701.0, "theoretical"),
        ("adduct", "Na", "Same originalfeature"),
        ("is_is", True, "standard"),
    ):
        data = record("z")
        data["pool"].loc[1, field] = value
        with pytest.raises(ValueError, match=message):
            predict_isomer_label_preferences(data, m, [0], [1])
    data = record("z")
    data["reference_training_groups"].append("z")
    with pytest.raises(ValueError, match="outside permitted"):
        predict_isomer_label_preferences(data, m, [0], [1])


def test_expert_scope_tampering_and_implicit_early_stopping_fail_closed():
    m, data = model(), record("z")
    broken = copy.deepcopy(m)
    broken["experts"]["RP_Pos"]["excluded_groups"] = ["other"]
    with pytest.raises(ValueError, match="scopes disagree"):
        predict_isomer_label_preferences(data, broken, [0], [1])
    with pytest.raises(ValueError, match="early stopping"):
        fit_isomer_label_model(
            {"a": record()},
            training_groups={"a", "b", "c"},
            excluded_groups={"z"},
            parameters={},
        )
