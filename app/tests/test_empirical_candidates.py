import copy

import numpy as np
import pandas as pd
import pytest

from matlab_annotation.empirical_candidates import (
    prepare_empirical_candidates,
    empirical_retention_evidence,
    PREFIX,
)
from matlab_annotation.empirical_retention import predict_aligned_empirical_retention
from matlab_annotation.evidence_training import validate_feature_scope
from test_empirical_retention import fixture
from test_evidence_training import fixture as score_fixture, data as score_data


def inputs():
    _, _, atlas, model, _ = fixture()
    targets = pd.DataFrame(
        [
            dict(
                name="PE(34:2)",
                **{"class": "PE"},
                carbon=34,
                db=2,
                adduct="[M+H]+",
                ref_mz=800.0,
                ref_rt=5.8,
            )
        ]
    )
    auto = pd.DataFrame(columns=["feat_feature_id", "is_is"])
    predicted = (
        predict_aligned_empirical_retention(
            targets, auto, atlas, model, stream="RP_Pos", query_group="query"
        ).atlas_rt_seconds.iloc[0]
        / 60
    )
    features = pd.DataFrame(
        dict(
            feature_id=["good", "mass_out", "RT_out", "standard"],
            mz=[800.0001, 800.004, 800.0001, 800.0001],
            rt=[predicted + 1 / 60, predicted, predicted + 11 / 60, predicted],
            intensity=[1000, 2000, 3000, 4000],
        )
    )
    measurements = pd.DataFrame({"feat_feature_id": ["standard"]})
    declared = pd.DataFrame({"ref_mz": [900.0]})
    return targets, features, auto, declared, measurements, atlas, model


def call(values):
    return prepare_empirical_candidates(*values, stream="RP_Pos", query_group="query")


def test_strict_physical_join_preserves_inputs_and_all_original_coordinates():
    values = inputs()
    before = copy.deepcopy(values)
    proposals, targets, predictions, parent = call(values)
    assert proposals.feat_feature_id.tolist() == ["good"]
    assert proposals.iloc[0].feat_mz == values[1].iloc[0].mz
    assert proposals.iloc[0].feat_rt == values[1].iloc[0].rt
    assert (
        proposals.ppm_error.abs().lt(3).all()
        and (proposals.rt_error.abs() * 60).le(10).all()
    )
    evidence = empirical_retention_evidence(proposals, targets, predictions, parent)
    assert len(evidence.columns) == 8 and all(c.startswith(PREFIX) for c in evidence)
    assert evidence.iloc[0][PREFIX + "RT_error_seconds"] == pytest.approx(1)
    assert evidence.iloc[0][PREFIX + "exact_reference_available"] == 1
    assert all(pd.api.types.is_numeric_dtype(evidence[c]) for c in evidence)
    assert not {"name", "feat_feature_id"} & set(evidence)
    for a, b in zip(values[:5], before[:5]):
        pd.testing.assert_frame_equal(a, b)
    assert values[5:] == before[5:]


def test_actual_declared_mass_unsupported_ether_isotope_wrong_polarity_withheld():
    values = list(inputs())
    values[3] = pd.DataFrame({"ref_mz": [800.0]})
    assert call(values)[0].empty
    for edit in (
        {"class": "PE-O"},
        {"isotopologue": "M+1"},
        {"adduct": "[M-H]-"},
        {"carbon": 1000},
    ):
        values = list(inputs())
        values[0] = values[0].assign(**edit)
        assert call(values)[0].empty


def test_supervised_inputs_and_duplicate_original_feature_ids_rejected():
    values = list(inputs())
    values[4] = values[4].assign(training_positive=True)
    with pytest.raises(ValueError, match="supervised"):
        call(values)
    values = list(inputs())
    values[1] = pd.concat([values[1], values[1]], ignore_index=True)
    with pytest.raises(ValueError, match="Unique original"):
        call(values)


def test_unknown_target_evidence_is_missing_not_zero_error():
    proposals, targets, predictions, parent = call(inputs())
    proposals = proposals.assign(name="UNKNOWN")
    evidence = empirical_retention_evidence(proposals, targets, predictions, parent)
    assert evidence.iloc[0][PREFIX + "prediction_available"] == 0
    assert np.isnan(evidence.iloc[0][PREFIX + "RT_error_seconds"])


def test_empirical_predictors_require_explicit_query_excluded_reference_scopes():
    records, groups, _ = score_fixture()
    query = score_data("z", sorted(groups), ["z"])
    query["evidence"][PREFIX + "RT_error_seconds"] = 1.0
    with pytest.raises(ValueError, match="Explicit nested empirical"):
        validate_feature_scope(query, groups, {"z"}, query_is_training=False)
    query["empirical_reference_training_groups"] = sorted(groups)
    query["empirical_reference_excluded_groups"] = ["z"]
    validate_feature_scope(query, groups, {"z"}, query_is_training=False)
    query["empirical_reference_training_groups"] = [*groups, "z"]
    with pytest.raises(ValueError, match="excluded scoring"):
        validate_feature_scope(query, groups, {"z"}, query_is_training=False)
    training = copy.deepcopy(records["a"])
    training["evidence"][PREFIX + "RT_error_seconds"] = 1.0
    training["empirical_reference_training_groups"] = ["a", "b"]
    training["empirical_reference_excluded_groups"] = ["z"]
    with pytest.raises(ValueError, match="own project"):
        validate_feature_scope(training, groups, {"z"}, query_is_training=True)
