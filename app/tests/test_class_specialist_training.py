import numpy as np
import pytest

from matlab_annotation.class_specialist_training import fit_class_specialist
from matlab_annotation.evidence_training import (
    fit_grouped_evidence_model,
    predict_grouped_evidence,
)
from test_evidence_training import fixture, data


def test_all_contributing_projects_exactly_reproduce_original_fitter():
    records, groups, parameters = fixture()
    original = fit_grouped_evidence_model(
        records, training_groups=groups, excluded_groups={"z"}, parameters=parameters
    )
    specialist = fit_class_specialist(
        records, training_groups=groups, excluded_groups={"z"}, parameters=parameters
    )
    query = data("z", sorted(groups), ["z"])
    np.testing.assert_array_equal(
        predict_grouped_evidence(query, original),
        predict_grouped_evidence(query, specialist),
    )
    assert specialist["allowed_without_supervision"] == []


def test_zero_label_project_retains_reference_authorization_but_no_invented_labels():
    records, groups, parameters = fixture()
    groups = groups | {"empty"}
    for r in records.values():
        r["reference_training_groups"] += ["empty"]
    records["empty"] = data("empty", ["a", "b", "c"], ["empty", "z"])
    records["empty"]["pool"]["training_positive"] = False
    records["empty"]["pool"]["training_label_known"] = False
    model = fit_class_specialist(
        records, training_groups=groups, excluded_groups={"z"}, parameters=parameters
    )
    assert set(model["training_cohorts"]) == groups
    assert model["supervised_training_cohorts"] == ["a", "b", "c"]
    assert model["allowed_without_supervision"] == ["empty"]
    assert model["training_query_known_rows"]["empty"] == 0
    assert model["training_pairs"] == 12 and model["positives"] == 6
    query = data("z", sorted(groups), ["z"])
    assert np.isfinite(predict_grouped_evidence(query, model)).all()
    records["a"]["reference_training_groups"] += ["z"]
    with pytest.raises(ValueError, match="outside permitted"):
        fit_class_specialist(
            records,
            training_groups=groups,
            excluded_groups={"z"},
            parameters=parameters,
        )


def test_silently_dropping_a_project_or_unknown_positive_is_rejected():
    records, groups, parameters = fixture()
    records.pop("a")
    with pytest.raises(ValueError, match="represented explicitly"):
        fit_class_specialist(
            records,
            training_groups=groups,
            excluded_groups={"z"},
            parameters=parameters,
        )
    records, groups, parameters = fixture()
    records["a"]["pool"]["training_label_known"] = False
    with pytest.raises(ValueError, match="may not become unknown"):
        fit_class_specialist(
            records,
            training_groups=groups,
            excluded_groups={"z"},
            parameters=parameters,
        )
