import copy

import numpy as np
import pytest

from matlab_annotation.family_evidence_experts import (
    chemical_family,
    fit_family_evidence_experts,
    predict_family_evidence_experts,
)
from matlab_annotation.mode_evidence_experts import (
    fit_mode_evidence_experts,
    predict_mode_evidence_experts,
)
from test_mode_evidence_experts import fixtures


@pytest.mark.parametrize(
    "cls,family",
    [
        ("PE-P", "glycerophospholipid"),
        ("1-acylLPE", "glycerophospholipid"),
        ("BMP", "glycerophospholipid"),
        ("SLBPA", "glycerophospholipid"),
        ("Hex2Cer-d", "sphingolipid"),
        ("GM3-d", "sphingolipid"),
        ("C1P-d", "sphingolipid"),
        ("S1P-d", "sphingolipid"),
        ("GA1-t", "sphingolipid"),
        ("TG-O", "glycerolipid"),
        ("DG-P", "glycerolipid"),
        ("FA-OH", "fatty_acid_ester"),
        ("AC-OH", "fatty_acid_ester"),
        ("CE", "fatty_acid_ester"),
        ("X4", "unsupported"),
    ],
)
def test_fixed_chemical_routes(cls, family):
    assert chemical_family(cls) == family


def setup():
    records, groups, params, query = fixtures()
    for data in [*records.values(), query]:
        data["evidence"]["class"] = "PE"
    fallback = fit_mode_evidence_experts(
        records, training_groups=groups, excluded_groups={"z"}, parameters=params
    )
    return records, groups, params, query, fallback


def test_sparse_family_fallback_is_bitexact_original_model():
    records, groups, params, query, fallback = setup()
    model = fit_family_evidence_experts(
        records,
        training_groups=groups,
        excluded_groups={"z"},
        parameters=params,
        fallback_model=fallback,
    )
    assert model["experts"] == {}
    assert model["training_pairs"] == fallback["training_pairs"]
    assert np.array_equal(
        predict_family_evidence_experts(query, model, fallback),
        predict_mode_evidence_experts(query, fallback),
    )


def test_all_modes_fit_without_added_labels_or_predictors():
    records, groups, params, query, fallback = setup()
    model = fit_family_evidence_experts(
        records,
        training_groups=groups,
        excluded_groups={"z"},
        parameters=params,
        fallback_model=fallback,
        minimum_rows=2,
        minimum_positive=1,
        minimum_negative=1,
        minimum_positive_projects=2,
    )
    assert len(model["experts"]) == 4
    assert model["training_pairs"] == 48 and model["positives"] == 24
    output = predict_family_evidence_experts(query, model, fallback)
    assert output.index.equals(query["pool"].index) and np.isfinite(output).all()
    bad = copy.deepcopy(model)
    next(iter(bad["experts"].values()))["excluded_groups"] = ["other"]
    with pytest.raises(ValueError, match="scopes"):
        predict_family_evidence_experts(query, bad, fallback)
    query["reference_training_groups"] = [*groups, "z"]
    with pytest.raises(ValueError, match="outside permitted"):
        predict_family_evidence_experts(query, model, fallback)


def test_actual_contributors_are_separate_from_reference_permission_scope():
    records, groups, params, query, fallback = setup()
    records["a"]["evidence"]["class"] = "TG"
    # Refit fallback against this unchanged-supervision alternative fixture.
    fallback = fit_mode_evidence_experts(
        records, training_groups=groups, excluded_groups={"z"}, parameters=params
    )
    model = fit_family_evidence_experts(
        records,
        training_groups=groups,
        excluded_groups={"z"},
        parameters=params,
        fallback_model=fallback,
        minimum_rows=2,
        minimum_positive=1,
        minimum_negative=1,
        minimum_positive_projects=2,
    )
    assert len(model["experts"]) == 4
    for expert in model["experts"].values():
        assert set(expert["training_cohorts"]) == groups
        assert set(expert["supervised_contributing_groups"]) == groups - {"a"}
    assert np.isfinite(predict_family_evidence_experts(query, model, fallback)).all()


def test_invalid_support_and_fallback_scope_rejected():
    records, groups, params, query, fallback = setup()
    with pytest.raises(ValueError, match="support"):
        fit_family_evidence_experts(
            records,
            training_groups=groups,
            excluded_groups={"z"},
            parameters=params,
            fallback_model=fallback,
            minimum_positive_projects=1,
        )
    fallback["excluded_groups"] = ["other"]
    with pytest.raises(ValueError, match="Fallback model scope"):
        fit_family_evidence_experts(
            records,
            training_groups=groups,
            excluded_groups={"z"},
            parameters=params,
            fallback_model=fallback,
        )
