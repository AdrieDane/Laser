"""Research class-specialist fits with explicit reference and supervision scopes.

Some allowed projects genuinely have no known rows for a particular class.
They may inform permitted reference features without inventing class labels.
All projects must remain represented and pass the original exclusion checks.
Never silently conflate allowed reference projects with contributing labels.
"""

from collections import Counter

import numpy as np
import pandas as pd

from .evidence_training import validate_feature_scope, evidence_matrix


def fit_class_specialist(
    data_by_dataset, *, training_groups, excluded_groups, parameters
):
    from sklearn.ensemble import HistGradientBoostingClassifier

    if parameters.get("early_stopping") is not False:
        raise ValueError("Implicit row-wise validation is forbidden")
    allowed = set(training_groups)
    if {d["query_group"] for d in data_by_dataset.values()} != allowed:
        raise ValueError("Every allowed project must be represented explicitly")
    xs, ys, row_groups, known_rows, schema = [], [], [], {}, None
    for dataset, data in data_by_dataset.items():
        validate_feature_scope(data, allowed, excluded_groups, query_is_training=True)
        pool, evidence = data["pool"], data["evidence"]
        for c in ("training_positive", "training_label_known"):
            if c not in pool or not pool[c].isin([True, False]).all():
                raise ValueError("Explicit Boolean training labels required")
        if schema is None:
            schema = list(evidence)
        elif set(schema) != set(evidence):
            raise ValueError("Identical specialist predictor schemas required")
        known = pool.training_label_known.astype(bool)
        if (pool.training_positive & ~known).any():
            raise ValueError("Positive specialist labels may not become unknown")
        known_rows[dataset] = int(known.sum())
        xs.append(evidence.loc[known, schema])
        ys.extend(pool.loc[known, "training_positive"].astype(int))
        row_groups.extend([data["query_group"]] * int(known.sum()))
    counts = Counter(row_groups)
    if len(counts) < 3 or set(ys) != {0, 1}:
        raise ValueError(
            "Specialist requires three contributing projects and both labels"
        )
    x = pd.concat(xs, ignore_index=True)
    categories = {
        c: {v: i for i, v in enumerate(sorted(x[c].astype(str).unique()))}
        for c in x
        if not pd.api.types.is_numeric_dtype(x[c])
    }
    if any(len(v) > 255 for v in categories.values()):
        raise ValueError("Specialist categorical support exceeds HGB limit")
    model = dict(
        columns=list(x),
        categories=categories,
        training_cohorts=sorted(allowed),
        excluded_groups=sorted(excluded_groups),
        supervised_training_cohorts=sorted(counts),
        allowed_without_supervision=sorted(allowed - set(counts)),
        training_query_known_rows=known_rows,
        supervision_scope_policy="explicit_allowed_and_contributing_v1",
        training_pairs=len(x),
        positives=int(sum(ys)),
        note="Allowed reference scope differs explicitly from actual label contributors. "
        "Equal weight per contributing project. No invented class labels or "
        "outer/query leakage; research agreement score,not identity probability/FDR.",
    )
    weights = np.asarray(
        [len(row_groups) / len(counts) / counts[g] for g in row_groups]
    )
    estimator = HistGradientBoostingClassifier(
        **parameters, categorical_features=[c in categories for c in x]
    )
    estimator.fit(evidence_matrix(x, model), np.asarray(ys), sample_weight=weights)
    model["estimator"] = estimator
    return model
