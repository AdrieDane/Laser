"""Unused research prototype: compare peaks WITHIN the same lipid identity.

No model fitting, scores, thresholds, annotations or files here. Pair labels
come only from explicitly permitted training projects after physical generation.
Manual-absent identities never become guessed peak-selection negatives. Names
and original feature IDs are used for grouping ONLY, never as predictors.
"""

from collections import Counter

import numpy as np
import pandas as pd

from .evidence_training import FORBIDDEN, validate_feature_scope


def pair_predictors(left, right):
    """Symmetric context and signed differences, including missingness.

    Mirroring a pair negates numeric differences/missingness, preserves numeric
    context, and swaps categorical sides. No category-order distance is invented.
    Both inputs must be original label-free predictor rows, not pool metadata.
    """
    if list(left) != list(right) or len(left) != len(right):
        raise ValueError("Aligned identical predictor schemas required")
    if FORBIDDEN & set(left):
        raise ValueError("Identity or supervision cannot enter pair predictors")
    output = {}
    for column in left:
        a, b = left[column], right[column]
        if pd.api.types.is_numeric_dtype(a) != pd.api.types.is_numeric_dtype(b):
            raise ValueError("Pair predictor dtypes disagree")
        if pd.api.types.is_numeric_dtype(a):
            av = pd.to_numeric(a, errors="coerce").to_numpy(float)
            bv = pd.to_numeric(b, errors="coerce").to_numpy(float)
            af, bf = np.isfinite(av), np.isfinite(bv)
            av, bv = np.where(af, av, np.nan), np.where(bf, bv, np.nan)
            output["difference__" + column] = av - bv
            output["midpoint__" + column] = av / 2 + bv / 2
            output["missing_difference__" + column] = (~af).astype(float) - (
                ~bf
            ).astype(float)
        else:
            output["left__" + column] = a.astype(str).to_numpy()
            output["right__" + column] = b.astype(str).to_numpy()
    return pd.DataFrame(output)


def conditional_training_pairs(
    data_by_dataset, *, training_groups, excluded_groups, vectorized=True
):
    """Return balanced two-orientation pairs and equal-project weights.

    A training identity contributes only when its one mapped positive feature
    and another KNOWN negative feature are both already physical candidates.
    Duplicate versions of one physical feature do not multiply votes. Each
    identity has total pre-normalization weight1, regardless of alternatives;
    every actually contributing project has equal final total weight. Allowed
    reference scope and actual supervised contributors are recorded separately.
    """
    frames, labels, projects, identity_weights = [], [], [], []
    contributions, identities = Counter(), Counter()
    for data in data_by_dataset.values():
        validate_feature_scope(
            data, training_groups, excluded_groups, query_is_training=True
        )
        pool, evidence = data["pool"], data["evidence"]
        if not pool.index.is_unique:
            raise ValueError("Original candidate row indices must be unique")
        required = {
            "stream",
            "name",
            "adduct",
            "feat_feature_id",
            "training_positive",
            "training_label_known",
        }
        if not required.issubset(pool):
            raise ValueError(
                "Explicit physical identity/feature and training flags required"
            )
        if (
            not pool[["training_positive", "training_label_known"]]
            .isin([True, False])
            .all()
            .all()
        ):
            raise ValueError("Explicit boolean supervision required")
        if (pool.training_positive & ~pool.training_label_known).any():
            raise ValueError("Positive training features cannot be unknown")
        keys = ["stream", "name", "adduct"]
        eligible_pool = pool
        if vectorized:
            flags = pool.groupby(keys + ["feat_feature_id"])[
                ["training_positive", "training_label_known"]
            ].nunique()
            if flags.gt(1).any().any():
                raise ValueError(
                    "Contradictory supervision for duplicate physical feature"
                )
            positive_features = (
                pool[pool.training_positive].groupby(keys).feat_feature_id.nunique()
            )
            if positive_features.gt(1).any():
                raise ValueError(
                    "Contradictory multiple positive original features for one identity"
                )
            eligible = pool[keys].copy()
            eligible["positive"] = pool.training_positive
            eligible["negative"] = pool.training_label_known & ~pool.training_positive
            grouped = eligible.groupby(keys, sort=False)
            usable = grouped.positive.transform("any") & grouped.negative.transform(
                "any"
            )
            eligible_pool = pool.loc[usable.fillna(False)]
        left_indices, right_indices = [], []
        for _, group in eligible_pool.groupby(keys, sort=False):
            flags = group.groupby("feat_feature_id")[
                ["training_positive", "training_label_known"]
            ].nunique()
            if flags.gt(1).any().any():
                raise ValueError(
                    "Contradictory supervision for duplicate physical feature"
                )
            # A physical feature may have alternative reference hypotheses.
            # Stable first-row selection is independent of labels and scores.
            group = group.drop_duplicates("feat_feature_id", keep="first")
            positive = group[group.training_positive]
            if len(positive) > 1:
                raise ValueError(
                    "Contradictory multiple positive original features for one identity"
                )
            negative = group[group.training_label_known & ~group.training_positive]
            if positive.empty or negative.empty:
                continue
            pi = positive.index[0]
            ni = negative.index
            if vectorized:
                left_indices.extend([pi] * len(negative) + list(ni))
                right_indices.extend(list(ni) + [pi] * len(negative))
            else:
                # Original scalar reference, retained for exact full-matrix QA.
                left = evidence.loc[[pi] * len(negative)].reset_index(drop=True)
                right = evidence.loc[ni].reset_index(drop=True)
                frames.extend(
                    (pair_predictors(left, right), pair_predictors(right, left))
                )
            labels.extend([1] * len(negative) + [0] * len(negative))
            projects.extend([data["query_group"]] * (2 * len(negative)))
            identity_weights.extend([1 / (2 * len(negative))] * (2 * len(negative)))
            identities[data["query_group"]] += 1
            contributions[data["query_group"]] += 2 * len(negative)
        if vectorized and left_indices:
            frames.append(
                pair_predictors(
                    evidence.loc[left_indices].reset_index(drop=True),
                    evidence.loc[right_indices].reset_index(drop=True),
                )
            )
    if not frames:
        raise ValueError("No permitted conditional peak-ranking pairs")
    matrix = pd.concat(frames, ignore_index=True)
    totals = Counter()
    for group, weight in zip(projects, identity_weights):
        totals[group] += weight
    target = len(projects) / len(totals)
    weights = np.asarray(
        [w * target / totals[g] for g, w in zip(projects, identity_weights)]
    )
    return dict(
        evidence=matrix,
        positive=np.asarray(labels, dtype=int),
        groups=np.asarray(projects),
        sample_weight=weights,
        training_reference_permissions=sorted(training_groups),
        excluded_groups=sorted(excluded_groups),
        supervised_contributing_projects=sorted(contributions),
        pair_rows_by_project=dict(contributions),
        identities_by_project=dict(identities),
        note="Synthetic pair orientation is not an independent biological observation or identification probability. Prototype pair construction ONLY,not fitted/used for annotation.",
    )
