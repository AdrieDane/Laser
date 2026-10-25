"""Unused research: independently refitted experts for all four acquisition modes.

Fixed routing by measurement mode, not post-hoc selection of successful test
classes/cohorts. Labels and all original predictors stay the same. Each expert
retains proper query/reference/score exclusions and equal project weights.
"""

import numpy as np
import pandas as pd

from .evidence_training import (
    fit_grouped_evidence_model,
    predict_grouped_evidence,
    validate_feature_scope,
)

MODES = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")


def _subset(data, mode):
    if not {"stream"}.issubset(data["evidence"]):
        raise ValueError("Explicit measurement-mode predictor required")
    if not data["evidence"].stream.isin(MODES).all():
        raise ValueError("Unknown acquisition mode")
    rows = data["evidence"].stream.eq(mode)
    return dict(
        data,
        pool=data["pool"].loc[rows].copy(),
        evidence=data["evidence"].loc[rows].copy(),
    )


def fit_mode_evidence_experts(
    data_by_dataset, *, training_groups, excluded_groups, parameters, weight_policy=None
):
    """Fit ALL four experts with fixed180-round family, no class/mode cherry-pick."""
    for data in data_by_dataset.values():
        validate_feature_scope(
            data, training_groups, excluded_groups, query_is_training=True
        )
    experts = {}
    for mode in MODES:
        records = {
            dataset: _subset(data, mode) for dataset, data in data_by_dataset.items()
        }
        records = {
            dataset: data for dataset, data in records.items() if len(data["pool"])
        }
        experts[mode] = fit_grouped_evidence_model(
            records,
            training_groups=training_groups,
            excluded_groups=excluded_groups,
            parameters=parameters,
            weight_policy=weight_policy,
        )
    columns = next(iter(experts.values()))["columns"]
    assert all(m["columns"] == columns for m in experts.values())
    return dict(
        format="laser-fixed-four-mode-evidence-experts-v1",
        experts=experts,
        columns=columns,
        training_cohorts=sorted(training_groups),
        excluded_groups=sorted(excluded_groups),
        training_pairs=sum(m["training_pairs"] for m in experts.values()),
        positives=sum(m["positives"] for m in experts.values()),
        note="Fixed four-mode routing;cohort-balanced within mode;no new labels or own-query predictors. Research curator-agreement scores,not analytical identity probabilities/FDR.",
    )


def predict_mode_evidence_experts(data, model):
    if model.get("format") != "laser-fixed-four-mode-evidence-experts-v1" or set(
        model["experts"]
    ) != set(MODES):
        raise ValueError("Complete fixed four-mode expert model required")
    validate_feature_scope(
        data,
        model["training_cohorts"],
        model["excluded_groups"],
        query_is_training=False,
    )
    predictions = []
    for mode in MODES:
        expert = model["experts"][mode]
        if set(expert["training_cohorts"]) != set(model["training_cohorts"]) or set(
            expert["excluded_groups"]
        ) != set(model["excluded_groups"]):
            raise ValueError("Expert scoring scopes disagree with enclosing model")
        subset = _subset(data, mode)
        if len(subset["pool"]):
            predictions.append(predict_grouped_evidence(subset, expert))
    output = (
        pd.concat(predictions).reindex(data["pool"].index)
        if predictions
        else pd.Series(index=data["pool"].index, dtype=float)
    )
    if len(output) and not np.isfinite(output.to_numpy(float)).all():
        raise ValueError("Every query requires exactly one finite mode-expert score")
    return output
