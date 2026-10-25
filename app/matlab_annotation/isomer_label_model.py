"""Unused numerical research model: compare ISOMER LABELS on one raw feature.

No annotation, acceptance, relabeling, file/curation access or default changes.
Fit only explicitly permitted TRAINING pairs. Query curation is never inspected.
"""

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class

from .conditional_peak_pairs import pair_predictors
from .evidence_training import evidence_matrix, validate_feature_scope
from .isomer_label_pairs import isomer_training_pairs
from .mode_evidence_experts import MODES, _subset

FORMAT = "laser-same-feature-isomer-label-comparison-v1"


def fit_isomer_label_model(
    data_by_dataset,
    *,
    training_groups,
    excluded_groups,
    parameters,
    minimum_pair_rows=100,
    minimum_contributing_projects=3,
    minimum_class_pair_projects=3,
):
    from sklearn.ensemble import HistGradientBoostingClassifier

    if parameters.get("early_stopping") is not False:
        raise ValueError("Implicit row-wise early stopping must be disabled")
    for v in (
        minimum_pair_rows,
        minimum_contributing_projects,
        minimum_class_pair_projects,
    ):
        if isinstance(v, bool) or not isinstance(v, int) or v < 2:
            raise ValueError("Explicit adequate pair/project support required")
    columns = None
    for data in data_by_dataset.values():
        validate_feature_scope(
            data, training_groups, excluded_groups, query_is_training=True
        )
        if columns is None:
            columns = list(data["evidence"])
        if columns != list(data["evidence"]):
            raise ValueError("Original predictor schemas must match")
        if not data["pool"].stream.equals(data["evidence"].stream):
            raise ValueError("Physical/predictor acquisition modes disagree")
    if columns is None:
        raise ValueError("Permitted training data required")
    permissions = dict(
        training_reference_permissions=sorted(training_groups),
        excluded_groups=sorted(excluded_groups),
    )
    experts = {}
    for mode in MODES:
        subsets = {d: _subset(v, mode) for d, v in data_by_dataset.items()}
        subsets = {d: v for d, v in subsets.items() if len(v["pool"])}
        try:
            pairs = isomer_training_pairs(
                subsets,
                training_groups=training_groups,
                excluded_groups=excluded_groups,
            )
        except ValueError as error:
            if str(error) != "No permitted same-feature isomer-label pairs":
                raise
            experts[mode] = dict(
                supported=False,
                pair_rows=0,
                supervised_contributing_projects=[],
                reason="no_conditional_isomer_labels",
                **permissions,
            )
            continue
        x = pairs["evidence"]
        metadata = dict(
            pair_rows=len(x),
            supervised_contributing_projects=pairs["supervised_contributing_projects"],
            pair_rows_by_project=pairs["pair_rows_by_project"],
            features_by_project=pairs["features_by_project"],
            class_pair_projects=pairs["class_pair_projects"],
            **permissions,
        )
        if (
            len(x) < minimum_pair_rows
            or len(metadata["supervised_contributing_projects"])
            < minimum_contributing_projects
        ):
            experts[mode] = dict(
                supported=False,
                reason="insufficient_actual_isomer_label_support",
                **metadata,
            )
            continue
        categories = {
            c: {v: i for i, v in enumerate(sorted(x[c].astype(str).unique()))}
            for c in x
            if not pd.api.types.is_numeric_dtype(x[c])
        }
        if any(len(v) > 255 for v in categories.values()):
            raise ValueError("Conditional categorical cardinality exceeds HGB support")
        expert = dict(
            columns=list(x), categories=categories, supported=True, **metadata
        )
        estimator = HistGradientBoostingClassifier(
            **parameters, categorical_features=[c in categories for c in x]
        )
        estimator.fit(
            evidence_matrix(x, expert),
            pairs["positive"],
            sample_weight=pairs["sample_weight"],
        )
        assert estimator.classes_.tolist() == [0, 1]
        expert["estimator"] = estimator
        experts[mode] = expert
    return dict(
        format=FORMAT,
        experts=experts,
        original_columns=columns,
        training_cohorts=sorted(training_groups),
        **permissions,
        minimum_pair_rows=minimum_pair_rows,
        minimum_contributing_projects=minimum_contributing_projects,
        minimum_class_pair_projects=minimum_class_pair_projects,
        note="Conditional isomer-label preference ONLY,not identity probability/FDR or annotation. Fullreferencepermissions separateactualcontributors. Querycuration neverread.",
    )


def predict_isomer_label_preferences(data, model, left_indices, right_indices):
    if model.get("format") != FORMAT or set(model["experts"]) != set(MODES):
        raise ValueError("Complete four-mode conditional isomer model required")
    training, excluded = (
        model["training_reference_permissions"],
        model["excluded_groups"],
    )
    validate_feature_scope(data, training, excluded, query_is_training=False)
    if list(data["evidence"]) != model["original_columns"]:
        raise ValueError("Conditional original predictor schema differs")
    pool, evidence = data["pool"], data["evidence"]
    if not pool.index.is_unique or len(left_indices) != len(right_indices):
        raise ValueError("Aligned unique physical row indices required")
    if not pool.stream.equals(evidence.stream):
        raise ValueError("Physical/predictor acquisition modes disagree")
    left, right = pool.loc[list(left_indices)], pool.loc[list(right_indices)]
    keys = ["stream", "feat_feature_id", "adduct"]
    if (
        not np.array_equal(
            left[keys].astype(str).to_numpy(), right[keys].astype(str).to_numpy()
        )
        or not left.stream.isin(MODES).all()
    ):
        raise ValueError("Same originalfeature/adduct/measurement mode required")
    a, b = left.ref_mz.to_numpy(float), right.ref_mz.to_numpy(float)
    if not (
        np.isfinite(a)
        & np.isfinite(b)
        & (a > 0)
        & (b > 0)
        & np.isclose(a, b, rtol=0, atol=1e-8)
    ).all():
        raise ValueError("Same finite positive theoretical isomer mass required")
    if (left.name.astype(str).to_numpy() == right.name.astype(str).to_numpy()).any():
        raise ValueError("Distinct isomer label hypotheses required")
    for side in (left, right):
        names = side.name.astype(str)
        if (
            not side.is_is.isin([True, False]).all()
            or side.is_is.any()
            or names.str.startswith("IS: ").any()
            or names.str.endswith(" (IS)").any()
        ):
            raise ValueError("Cannot compare standard label hypotheses")
    class_pairs = [
        tuple(sorted((canonical_lipid_class(a), canonical_lipid_class(b))))
        for a, b in zip(left["class"], right["class"])
    ]
    x, y = (
        evidence.loc[list(left_indices)].reset_index(drop=True),
        evidence.loc[list(right_indices)].reset_index(drop=True),
    )
    output = np.full(len(left), np.nan)
    for mode, expert in model["experts"].items():
        if set(expert["training_reference_permissions"]) != set(training) or set(
            expert["excluded_groups"]
        ) != set(excluded):
            raise ValueError("Conditional expert reference scopes disagree")
        if not expert["supported"]:
            continue
        supported = {
            pair
            for pair, projects in expert["class_pair_projects"].items()
            if len(projects) >= model["minimum_class_pair_projects"]
        }
        rows = np.flatnonzero(
            left.stream.eq(mode).to_numpy()
            & np.asarray([p in supported for p in class_pairs])
        )
        if not len(rows):
            continue
        forward, reverse = (
            pair_predictors(x.iloc[rows], y.iloc[rows]),
            pair_predictors(y.iloc[rows], x.iloc[rows]),
        )
        pf = expert["estimator"].predict_proba(evidence_matrix(forward, expert))[:, 1]
        pr = expert["estimator"].predict_proba(evidence_matrix(reverse, expert))[:, 1]
        values = 0.5 * (pf + 1 - pr)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Invalid conditional label preference")
        output[rows] = values
    return output
