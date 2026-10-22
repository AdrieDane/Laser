import pandas as pd
import pytest

from matlab_annotation.evidence_training import (
    fit_grouped_evidence_model,
    predict_grouped_evidence,
    validate_feature_scope,
)


def data(query, references, excluded):
    pool = pd.DataFrame(
        dict(
            training_positive=[True, True, False, False],
            training_label_known=[True] * 4,
        )
    )
    evidence = pd.DataFrame({"class": ["PE"] * 4, "mass_error": [0.1, 0.2, 2.0, 2.5]})
    return dict(
        query_group=query,
        reference_training_groups=references,
        reference_excluded_groups=excluded,
        pool=pool,
        evidence=evidence,
    )


def fixture():
    groups = {"a", "b", "c"}
    records = {g: data(g, sorted(groups - {g}), [g, "z"]) for g in sorted(groups)}
    parameters = dict(
        max_iter=3,
        max_leaf_nodes=3,
        min_samples_leaf=1,
        early_stopping=False,
        random_state=1,
    )
    return records, groups, parameters


def test_grouped_fit_and_excluded_prediction_do_not_read_query_supervision():
    records, groups, parameters = fixture()
    model = fit_grouped_evidence_model(
        records, training_groups=groups, excluded_groups={"z"}, parameters=parameters
    )
    query = data("z", sorted(groups), ["z"])
    query["pool"] = query["pool"].drop(
        columns=["training_positive", "training_label_known"]
    )
    result = predict_grouped_evidence(query, model)
    assert len(result) == 4
    assert result.iloc[0] > result.iloc[-1]
    assert model["training_cohorts"] == sorted(groups)


def test_own_query_outer_test_or_inner_calibration_reference_leak_is_rejected():
    records, groups, parameters = fixture()
    for references in (["a", "b"], ["b", "z"], ["b", "validation"]):
        bad = data("a", references, ["a", "z", "validation"])
        with pytest.raises(ValueError, match="Reference|own project"):
            validate_feature_scope(
                bad, groups, {"z", "validation"}, query_is_training=True
            )


def test_identifier_predictors_and_implicit_row_validation_rejected():
    records, groups, parameters = fixture()
    records["a"]["evidence"]["dataset"] = "secret"
    with pytest.raises(ValueError, match="identities"):
        fit_grouped_evidence_model(
            records,
            training_groups=groups,
            excluded_groups={"z"},
            parameters=parameters,
        )


def test_new_standard_context_requires_its_own_query_excluded_scope():
    query = data("a", ["b", "c"], ["a", "z"])
    query["evidence"]["q_declared_standard_RT_difference_seconds"] = 1.0
    with pytest.raises(ValueError, match="standard-context reference scope"):
        validate_feature_scope(query, {"a", "b", "c"}, {"z"}, query_is_training=True)
    query["standard_reference_training_groups"] = ["b", "c"]
    query["standard_reference_excluded_groups"] = ["a", "z"]
    validate_feature_scope(query, {"a", "b", "c"}, {"z"}, query_is_training=True)
    query["standard_reference_training_groups"] = ["a", "b"]
    with pytest.raises(ValueError, match="own project"):
        validate_feature_scope(query, {"a", "b", "c"}, {"z"}, query_is_training=True)


def test_standard_context_may_not_reuse_outer_only_models_in_inner_calibration():
    query = data("validation", ["a", "b"], ["validation", "outer"])
    query["evidence"]["q_declared_standard_RT_difference_seconds"] = 1.0
    query["standard_reference_training_groups"] = ["a", "b", "validation"]
    query["standard_reference_excluded_groups"] = ["outer"]
    with pytest.raises(ValueError, match="excluded scoring groups"):
        validate_feature_scope(
            query, {"a", "b"}, {"validation", "outer"}, query_is_training=False
        )
    records, groups, parameters = fixture()
    parameters["early_stopping"] = True
    with pytest.raises(ValueError, match="row-wise"):
        fit_grouped_evidence_model(
            records,
            training_groups=groups,
            excluded_groups={"z"},
            parameters=parameters,
        )
