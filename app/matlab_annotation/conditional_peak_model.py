"""Unused research: conditional within-identity peak preferences, NOT annotation.

This API only fits permitted TRAINING pairs or returns numerical preferences.
It never accepts/removes/moves a lipid, reads curation/files, or changes defaults.
Unsupported modes return unknown, not invented preference evidence.
"""

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class

from .conditional_peak_pairs import conditional_training_pairs, pair_predictors
from .evidence_training import evidence_matrix, validate_feature_scope
from .mode_evidence_experts import MODES, _subset

FORMAT = "laser-conditional-peak-comparison-v1"


def fit_conditional_peak_model(
    data_by_dataset,
    *,
    training_groups,
    excluded_groups,
    parameters,
    minimum_pair_rows=100,
    minimum_contributing_projects=3,
    minimum_class_projects=3,
):
    """All four fixed modes, explicit sparse fallback and actual contributors.

    Permitted reference groups need not all supply conditional negatives.
    This is recorded explicitly; fitting rows/projects are never invented.
    Pair construction balances orientations, identities and contributing projects.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    if parameters.get("early_stopping") is not False:
        raise ValueError("Implicit row-wise early stopping must be disabled")
    if (
        isinstance(minimum_pair_rows, bool)
        or minimum_pair_rows < 2
        or isinstance(minimum_contributing_projects, bool)
        or minimum_contributing_projects < 2
        or isinstance(minimum_class_projects, bool)
        or minimum_class_projects < 2
    ):
        raise ValueError("Explicit adequate pair/project support required")
    columns = None
    for data in data_by_dataset.values():
        validate_feature_scope(
            data, training_groups, excluded_groups, query_is_training=True
        )
        if columns is None:
            columns = list(data["evidence"])
        if list(data["evidence"]) != columns:
            raise ValueError("Original predictor schemas must match")
        if "stream" not in data["pool"] or not data["pool"].stream.equals(
            data["evidence"].stream
        ):
            raise ValueError("Physical/predictor acquisition modes disagree")
    if columns is None:
        raise ValueError("Training data required")
    experts = {}
    for mode in MODES:
        records = {name: _subset(data, mode) for name, data in data_by_dataset.items()}
        records = {name: data for name, data in records.items() if len(data["pool"])}
        try:
            pairs = conditional_training_pairs(
                records,
                training_groups=training_groups,
                excluded_groups=excluded_groups,
            )
        except ValueError as error:
            if str(error) != "No permitted conditional peak-ranking pairs":
                raise
            experts[mode] = dict(
                supported=False,
                pair_rows=0,
                supervised_contributing_projects=[],
                reason="no_conditional_pairs",
            )
            continue
        x, y = pairs["evidence"], pairs["positive"]
        contributors = pairs["supervised_contributing_projects"]
        if "left__class" not in x:
            raise ValueError(
                "Explicit chemical-class predictor required for support qualification"
            )
        class_projects = {}
        for cls, project in zip(
            x["left__class"].map(canonical_lipid_class), pairs["groups"]
        ):
            class_projects.setdefault(cls, set()).add(str(project))
        support = dict(
            pair_rows=len(x),
            supervised_contributing_projects=contributors,
            pair_rows_by_project=pairs["pair_rows_by_project"],
            identities_by_project=pairs["identities_by_project"],
            training_cohorts=sorted(training_groups),
            training_reference_permissions=sorted(training_groups),
            excluded_groups=sorted(excluded_groups),
            conditional_class_projects={
                c: sorted(g) for c, g in sorted(class_projects.items())
            },
        )
        if (
            len(x) < minimum_pair_rows
            or len(contributors) < minimum_contributing_projects
        ):
            experts[mode] = dict(
                supported=False,
                reason="insufficient_conditional_pair_support",
                **support,
            )
            continue
        categories = {
            c: {v: i for i, v in enumerate(sorted(x[c].astype(str).unique()))}
            for c in x
            if not pd.api.types.is_numeric_dtype(x[c])
        }
        if any(len(values) > 255 for values in categories.values()):
            raise ValueError("Conditional categorical cardinality exceeds HGB support")
        expert = dict(columns=list(x), categories=categories, supported=True, **support)
        estimator = HistGradientBoostingClassifier(
            **parameters, categorical_features=[c in categories for c in x]
        )
        estimator.fit(
            evidence_matrix(x, expert), y, sample_weight=pairs["sample_weight"]
        )
        assert estimator.classes_.tolist() == [0, 1]
        expert["estimator"] = estimator
        experts[mode] = expert
    return dict(
        format=FORMAT,
        experts=experts,
        original_columns=columns,
        training_cohorts=sorted(training_groups),
        training_reference_permissions=sorted(training_groups),
        excluded_groups=sorted(excluded_groups),
        minimum_pair_rows=minimum_pair_rows,
        minimum_contributing_projects=minimum_contributing_projects,
        minimum_class_projects=minimum_class_projects,
        note="Within-identity conditional preference ONLY,not annotation acceptance/analyticalidentityprobability/FDR. Full reference permissions andactualsupervisedcontributors separate. No names/FIDs/manualflags as predictors. No annotation decisions.",
    )


def predict_peak_preferences(data, model, left_indices, right_indices):
    """Antisymmetric preference for LEFT over RIGHT of the SAME lipid/adduct.

    Query labels, if present as separate pool metadata, are never inspected.
    No peak move is proposed here; unsupported modes return NaN.
    """
    if model.get("format") != FORMAT or set(model["experts"]) != set(MODES):
        raise ValueError("Complete conditional four-mode model required")
    validate_feature_scope(
        data,
        model["training_cohorts"],
        model["excluded_groups"],
        query_is_training=False,
    )
    if list(data["evidence"]) != model["original_columns"]:
        raise ValueError("Conditional original predictor schema differs")
    if "stream" not in data["pool"] or not data["pool"].stream.equals(
        data["evidence"].stream
    ):
        raise ValueError("Physical/predictor acquisition modes disagree")
    if not data["pool"].index.is_unique or len(left_indices) != len(right_indices):
        raise ValueError("Aligned original candidate row pairs required")
    left = data["pool"].loc[list(left_indices)]
    right = data["pool"].loc[list(right_indices)]
    keys = ["stream", "name", "adduct"]
    if not np.array_equal(
        left[keys].astype(str).to_numpy(), right[keys].astype(str).to_numpy()
    ):
        raise ValueError(
            "Preference pairs must have the same measurement mode/lipid/adduct"
        )
    if not left.stream.isin(MODES).all():
        raise ValueError("Unknown conditional acquisition mode")
    output = np.full(len(left), np.nan)
    a = data["evidence"].loc[list(left_indices)].reset_index(drop=True)
    b = data["evidence"].loc[list(right_indices)].reset_index(drop=True)
    for mode, expert in model["experts"].items():
        if not expert["supported"]:
            continue
        supported_classes = {
            c
            for c, g in expert["conditional_class_projects"].items()
            if len(g) >= model["minimum_class_projects"]
        }
        rows = np.flatnonzero(
            (
                left.stream.eq(mode)
                & left["class"].map(canonical_lipid_class).isin(supported_classes)
            ).to_numpy()
        )
        if not len(rows):
            continue
        if set(expert["training_cohorts"]) != set(model["training_cohorts"]) or set(
            expert["excluded_groups"]
        ) != set(model["excluded_groups"]):
            raise ValueError("Conditional expert scopes disagree")
        forward = pair_predictors(a.iloc[rows], b.iloc[rows])
        reverse = pair_predictors(b.iloc[rows], a.iloc[rows])
        pf = expert["estimator"].predict_proba(evidence_matrix(forward, expert))[:, 1]
        pr = expert["estimator"].predict_proba(evidence_matrix(reverse, expert))[:, 1]
        values = 0.5 * (pf + 1 - pr)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Invalid conditional preference")
        output[rows] = values
    return output
