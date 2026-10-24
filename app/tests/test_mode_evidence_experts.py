import copy

import numpy as np
import pandas as pd
import pytest

from matlab_annotation.mode_evidence_experts import (
    MODES,
    fit_mode_evidence_experts,
    predict_mode_evidence_experts,
)
from test_evidence_training import fixture, data


def fixtures():
    records, groups, params = fixture()
    for record in records.values():
        record["pool"] = pd.concat([record["pool"]] * 4, ignore_index=True)
        record["evidence"] = pd.concat(
            [record["evidence"].assign(stream=m) for m in MODES], ignore_index=True
        )
    query = data("z", sorted(groups), ["z"])
    query["pool"] = pd.concat([query["pool"]] * 4, ignore_index=True).drop(
        columns=["training_positive", "training_label_known"]
    )
    query["evidence"] = pd.concat(
        [query["evidence"].assign(stream=m) for m in MODES], ignore_index=True
    )
    return records, groups, params, query


def test_all_four_modes_fit_unchanged_supervision_and_predict_original_row_order():
    records, groups, params, query = fixtures()
    model = fit_mode_evidence_experts(
        records, training_groups=groups, excluded_groups={"z"}, parameters=params
    )
    assert set(model["experts"]) == set(MODES)
    assert model["training_pairs"] == 48 and model["positives"] == 24
    scores = predict_mode_evidence_experts(query, model)
    assert scores.index.equals(query["pool"].index)
    assert len(scores) == 16 and np.isfinite(scores).all()
    for expert in model["experts"].values():
        assert len(expert["categories"]["stream"]) == 1
        assert set(expert["training_cohorts"]) == groups


def test_expert_scope_mismatch_and_unknown_mode_rejected():
    records, groups, params, query = fixtures()
    model = fit_mode_evidence_experts(
        records, training_groups=groups, excluded_groups={"z"}, parameters=params
    )
    bad = copy.deepcopy(model)
    bad["experts"]["NP_Pos"]["excluded_groups"] = ["different"]
    with pytest.raises(ValueError, match="scopes"):
        predict_mode_evidence_experts(query, bad)
    query["evidence"]["stream"] = "UNKNOWN"
    with pytest.raises(ValueError, match="Unknown"):
        predict_mode_evidence_experts(query, model)


def test_a_project_may_not_be_silently_dropped_from_one_expert():
    records, groups, params, _ = fixtures()
    records["a"]["evidence"]["stream"] = "RP_Pos"
    with pytest.raises(ValueError, match="Every allowed training project"):
        fit_mode_evidence_experts(
            records, training_groups=groups, excluded_groups={"z"}, parameters=params
        )
