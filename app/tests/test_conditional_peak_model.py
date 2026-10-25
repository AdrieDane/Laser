import copy

import numpy as np
import pytest

from matlab_annotation.conditional_peak_model import (
    fit_conditional_peak_model,
    predict_peak_preferences,
)
from test_conditional_peak_pairs import record


def fixture():
    records = {g: record(g) for g in ["a", "b", "c"]}
    parameters = dict(
        max_iter=8,
        max_leaf_nodes=3,
        min_samples_leaf=1,
        early_stopping=False,
        random_state=1,
    )
    query = record("z")
    query["reference_training_groups"] = ["a", "b", "c"]
    query["reference_excluded_groups"] = ["z"]
    return records, parameters, query


def test_pair_predictions_antisymmetric_and_labels_never_used_at_query_time():
    records, params, query = fixture()
    model = fit_conditional_peak_model(
        records,
        training_groups={"a", "b", "c"},
        excluded_groups={"z"},
        parameters=params,
        minimum_pair_rows=2,
        minimum_contributing_projects=2,
    )
    assert model["experts"]["RP_Pos"]["pair_rows"] == 12
    assert model["experts"]["RP_Pos"]["supervised_contributing_projects"] == [
        "a",
        "b",
        "c",
    ]
    assert not model["experts"]["RP_Neg"]["supported"]
    p = predict_peak_preferences(query, model, [0, 1, 0], [1, 0, 0])
    assert p[0] > 0.5 and p[1] < 0.5 and p[2] == pytest.approx(0.5)
    assert p[0] + p[1] == pytest.approx(1)
    del query["pool"]["training_positive"], query["pool"]["training_label_known"]
    np.testing.assert_array_equal(
        p, predict_peak_preferences(query, model, [0, 1, 0], [1, 0, 0])
    )
    assert "name" not in model["experts"]["RP_Pos"]["columns"]


def test_sparse_mode_returns_unknown_not_fabricated_point_five():
    records, params, query = fixture()
    model = fit_conditional_peak_model(
        records,
        training_groups={"a", "b", "c"},
        excluded_groups={"z"},
        parameters=params,
        minimum_pair_rows=100,
        minimum_contributing_projects=3,
    )
    assert not model["experts"]["RP_Pos"]["supported"]
    assert np.isnan(predict_peak_preferences(query, model, [0], [1])).all()


def test_query_reference_leak_and_cross_identity_or_scope_mismatch_rejected():
    records, params, query = fixture()
    model = fit_conditional_peak_model(
        records,
        training_groups={"a", "b", "c"},
        excluded_groups={"z"},
        parameters=params,
        minimum_pair_rows=2,
        minimum_contributing_projects=2,
    )
    with pytest.raises(ValueError, match="same measurement"):
        predict_peak_preferences(query, model, [0], [3])
    broken = copy.deepcopy(model)
    broken["experts"]["RP_Pos"]["excluded_groups"] = ["other"]
    with pytest.raises(ValueError, match="scopes disagree"):
        predict_peak_preferences(query, broken, [0], [1])
    query["reference_training_groups"] = ["a", "z"]
    with pytest.raises(ValueError, match="permitted scoring"):
        predict_peak_preferences(query, model, [0], [1])


def test_implicit_validation_and_misaligned_mode_metadata_rejected():
    records, params, _ = fixture()
    params["early_stopping"] = True
    with pytest.raises(ValueError, match="early stopping"):
        fit_conditional_peak_model(
            records,
            training_groups={"a", "b", "c"},
            excluded_groups={"z"},
            parameters=params,
        )


def test_no_peak_preference_for_a_class_without_three_training_projects():
    records, params, query = fixture()
    records["c"]["pool"]["class"] = "PC"
    records["c"]["evidence"]["class"] = "PC"
    model = fit_conditional_peak_model(
        records,
        training_groups={"a", "b", "c"},
        excluded_groups={"z"},
        parameters=params,
        minimum_pair_rows=2,
        minimum_contributing_projects=2,
    )
    assert model["experts"]["RP_Pos"]["supported"]
    assert model["experts"]["RP_Pos"]["conditional_class_projects"]["PE"] == ["a", "b"]
    assert np.isnan(predict_peak_preferences(query, model, [0], [1])).all()


def test_physical_mode_metadata_cannot_disagree_with_predictor_mode():
    records, params, _ = fixture()
    params["early_stopping"] = False
    records["a"]["pool"].loc[0, "stream"] = "NP_Pos"
    with pytest.raises(ValueError, match="modes disagree"):
        fit_conditional_peak_model(
            records,
            training_groups={"a", "b", "c"},
            excluded_groups={"z"},
            parameters=params,
        )
