"""Offline research scoring fits with explicit nested reference-scope checks.

No data files, manual reference, cutoff or annotation is read/written here.
Supervised flags may enter fitting only after label-free predictor preparation.
This module is unused by production and imports sklearn only during a fit.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

FORBIDDEN = {
    "dataset",
    "cohort",
    "name",
    "feat_feature_id",
    "feature_id",
    "query_group",
    "training_positive",
    "training_label_known",
    "manual_feature_id",
    "same_manual_peak",
}


def validate_feature_scope(
    data, training_groups, excluded_groups, *, query_is_training
):
    """No evaluated/calibration group can enter reference-derived predictors."""
    training, excluded = set(training_groups), set(excluded_groups)
    if not training or not excluded or training & excluded:
        raise ValueError("Explicit disjoint scoring training/excluded groups required")
    required = {
        "query_group",
        "reference_training_groups",
        "reference_excluded_groups",
        "pool",
        "evidence",
    }
    if not required.issubset(data):
        raise ValueError("Explicit query and reference-model provenance required")
    query = data["query_group"]
    references = set(data["reference_training_groups"])
    if not references or not references.issubset(training) or references & excluded:
        raise ValueError(
            "Reference models include groups outside permitted scoring training scope"
        )
    if query in references or query not in set(data["reference_excluded_groups"]):
        raise ValueError("A query's own project entered its reference models")
    if query_is_training:
        if query not in training:
            raise ValueError("Scoring fit query is not an allowed training project")
    elif query not in excluded:
        raise ValueError("Prediction query must be an explicitly excluded project")
    if not data["pool"].index.equals(data["evidence"].index):
        raise ValueError("Physical queries and predictor rows must align")
    if FORBIDDEN & set(data["evidence"]):
        raise ValueError(
            "Supervised labels or original identities entered predictor columns"
        )
    standard_keys = {
        "standard_reference_training_groups",
        "standard_reference_excluded_groups",
    }
    has_standard_context = any(
        str(c).startswith("q_declared_standard_") for c in data["evidence"]
    )
    if has_standard_context or standard_keys & set(data):
        if not standard_keys.issubset(data):
            raise ValueError(
                "Explicit nested standard-context reference scope required"
            )
        standard_groups = set(data["standard_reference_training_groups"])
        if (
            not standard_groups
            or not standard_groups.issubset(training)
            or standard_groups & excluded
        ):
            raise ValueError(
                "Standard-context references include excluded scoring groups"
            )
        if query in standard_groups or query not in set(
            data["standard_reference_excluded_groups"]
        ):
            raise ValueError(
                "A query's own project entered its standard-context references"
            )


def evidence_matrix(evidence, model):
    """Portable column/category mapping; unknown categories remain NaN."""
    if set(model["columns"]) != set(evidence):
        raise ValueError("Predictor schema differs from the fitted model")
    arrays = []
    for column in model["columns"]:
        values = evidence[column]
        if column in model["categories"]:
            arrays.append(
                values.astype(str).map(model["categories"][column]).to_numpy(float)
            )
        else:
            arrays.append(pd.to_numeric(values, errors="coerce").to_numpy(float))
    return np.column_stack(arrays)


def fit_grouped_evidence_model(
    data_by_dataset, *, training_groups, excluded_groups, parameters, weight_policy=None
):
    """Same fixed HGB/cohort-balanced weighting as007, genuinely excluded inputs."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    if weight_policy not in (None, "tempered_class_mode"):
        raise ValueError("Unknown research sample-weight policy")
    if parameters.get("early_stopping") is not False:
        raise ValueError(
            "Disable implicit row-wise validation/early stopping for grouped research fits"
        )
    xs, ys, groups = [], [], []
    schema = None
    for data in data_by_dataset.values():
        validate_feature_scope(
            data, training_groups, excluded_groups, query_is_training=True
        )
        pool, evidence = data["pool"], data["evidence"]
        if not {"training_positive", "training_label_known"}.issubset(pool):
            raise ValueError("Explicit separate supervised fitting flags required")
        if (
            not pool.training_positive.isin([True, False]).all()
            or not pool.training_label_known.isin([True, False]).all()
        ):
            raise ValueError("Supervised flags must be explicit booleans")
        if schema is None:
            schema = list(evidence)
        elif set(evidence) != set(schema):
            raise ValueError(
                "Predictor schemas must be identical across training queries"
            )
        known = pool.training_label_known.astype(bool)
        xs.append(evidence.loc[known, schema])
        ys.extend(pool.loc[known, "training_positive"].astype(int))
        groups.extend([data["query_group"]] * int(known.sum()))
    if set(groups) != set(training_groups):
        raise ValueError(
            "Every allowed training project must contribute known fitting rows"
        )
    x = pd.concat(xs, ignore_index=True)
    categories = {
        c: {v: i for i, v in enumerate(sorted(x[c].astype(str).unique()))}
        for c in x
        if not pd.api.types.is_numeric_dtype(x[c])
    }
    if any(len(values) > 255 for values in categories.values()):
        raise ValueError("Categorical cardinality exceeds fixed HGB support")
    model = dict(
        columns=list(x),
        categories=categories,
        training_cohorts=sorted(set(groups)),
        excluded_groups=sorted(set(excluded_groups)),
        training_pairs=len(x),
        positives=int(sum(ys)),
        note="Offline curator-agreement score, not an analytical identification probability/FDR.",
    )
    counts = Counter(groups)
    weights = np.asarray([len(groups) / len(counts) / counts[g] for g in groups])
    if weight_policy == "tempered_class_mode":
        from .class_balanced_weights import tempered_class_mode_weights

        if not {"stream", "class"}.issubset(x):
            raise ValueError("Explicit candidate stream/class required for balancing")
        weights = tempered_class_mode_weights(groups, x.stream, x["class"])
        totals = Counter()
        for cohort, weight in zip(groups, weights):
            totals[cohort] += float(weight)
        target_weight = len(groups) / len(counts)
        model["sample_weight_policy"] = dict(
            kind=weight_policy,
            exponent=0.5,
            minimum_factor=0.25,
            maximum_factor=4.0,
            minimum_weight=float(weights.min()),
            maximum_weight=float(weights.max()),
            cohort_total_target=float(target_weight),
            cohort_total_max_abs_error=float(
                max(abs(v - target_weight) for v in totals.values())
            ),
        )
    estimator = HistGradientBoostingClassifier(
        **parameters, categorical_features=[c in categories for c in x]
    )
    estimator.fit(evidence_matrix(x, model), np.asarray(ys), sample_weight=weights)
    model["estimator"] = estimator
    return model


def predict_grouped_evidence(data, model):
    """Never predicts on supervised flags; validates reference scope first."""
    validate_feature_scope(
        data,
        model["training_cohorts"],
        model["excluded_groups"],
        query_is_training=False,
    )
    return pd.Series(
        model["estimator"].predict_proba(evidence_matrix(data["evidence"], model))[
            :, 1
        ],
        index=data["pool"].index,
    )
