"""Research-only pairwise physical evidence for fixed-identity peak choice.

This is NOT an annotation-presence classifier or a calibrated probability of
chemical identification. No accepted identities are added or removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .conditional_peak_choice import conditional_training_weights

# Constant occurrence descriptors cannot distinguish peaks within an identity.
# Remove prior choices/confidence so this model must compare physical evidence.
REMOVED_FEATURES = {
    "carbon",
    "db",
    "carbon_odd",
    "db_per_carbon",
    "composition_training_cohorts",
    "composition_joint_frequency",
    "composition_carbon_frequency",
    "composition_db_frequency",
    "q_context_carbon_median",
    "q_context_carbon_p90",
    "q_context_db_median",
    "q_context_db_p90",
    "q_carbon_minus_context_median",
    "q_db_minus_context_median",
    "q_previously_selected",
    "q_fitted_post_regression_confidence",
    "q_fitted_positional_isomer_selected",
    "q_fitted_source",
    "q_atlas_source",
}


def pair_evidence(left, right):
    """Relative/mean physical descriptors; row IDs never become model inputs."""
    if len(left) != len(right) or list(left) != list(right):
        raise ValueError("Paired evidence schemas/lengths must match")
    columns = {}
    for name in left:
        if name in REMOVED_FEATURES:
            continue
        a, b = left[name].reset_index(drop=True), right[name].reset_index(drop=True)
        if not pd.api.types.is_numeric_dtype(a):
            if not a.astype(str).equals(b.astype(str)):
                raise ValueError("Categorical identity descriptors must agree")
            columns[name] = a.astype(str)
        else:
            a = pd.to_numeric(a, errors="coerce").to_numpy(float)
            b = pd.to_numeric(b, errors="coerce").to_numpy(float)
            columns[name + "__difference"] = a - b
            columns[name + "__mean"] = (a + b) / 2
            columns[name + "__left_observed"] = np.isfinite(a).astype(float)
            columns[name + "__right_observed"] = np.isfinite(b).astype(float)
    return pd.DataFrame(columns)


def pair_training_data(pool, evidence, dataset, cohort):
    """One true-vs-alternative pair in BOTH orders; identity/group balanced."""
    if not pool.index.equals(evidence.index):
        raise ValueError("Candidate evidence indices must match")
    weights = conditional_training_weights(pool, dataset, cohort)
    keys = [
        dataset.astype(str),
        pool.stream.astype(str),
        pool.name.astype(str),
        pool.adduct.astype(str),
    ]
    eligible = pool[weights.gt(0)]
    positives = eligible[eligible.training_positive.astype(bool)]
    lookup = {
        tuple(k): i
        for i, k in zip(positives.index, zip(*(s.loc[positives.index] for s in keys)))
    }
    negatives = eligible[~eligible.training_positive.astype(bool)]
    wrong = negatives.index.to_numpy()
    true = np.asarray([lookup[tuple(k)] for k in zip(*(s.loc[wrong] for s in keys))])
    if not len(wrong):
        raise ValueError("No conditional training pairs")
    left = np.concatenate([true, wrong])
    right = np.concatenate([wrong, true])
    x = pair_evidence(evidence.loc[left], evidence.loc[right])
    y = np.concatenate([np.ones(len(wrong)), np.zeros(len(wrong))]).astype(int)
    w = np.tile(weights.loc[wrong].to_numpy(float) / 2, 2)
    metadata = pd.DataFrame(
        dict(
            cohort=cohort.loc[left].to_numpy(str),
            dataset=dataset.loc[left].to_numpy(str),
        )
    )
    return x, y, w, metadata, positives


def pairwise_peak_scores(pool, evidence, predict):
    """Minimum win over ALL alternatives, symmetrized in both pair orders.

    A score>=.75 means a candidate beats every alternative by the fixed pair
    criterion. Cyclic preferences yield no .75 winner, rather than a tie-break.
    Unique-peak identities cannot be reranked and receive missing scores.
    """
    if not pool.index.equals(evidence.index):
        raise ValueError("Candidate evidence indices must match")
    scores = pd.Series(np.nan, index=pool.index)
    left, right = [], []
    for _, part in pool.groupby(["stream", "name", "adduct"], sort=False):
        if part.feat_feature_id.astype(str).duplicated().any():
            raise ValueError("Pairwise candidates must have unique original features")
        ids = part.index.to_numpy()
        if len(ids) < 2:
            continue
        a, b = np.triu_indices(len(ids), k=1)
        left.extend(ids[a])
        right.extend(ids[b])
    if not left:
        return scores
    a, b = np.asarray(left), np.asarray(right)
    forward = np.asarray(
        predict(pair_evidence(evidence.loc[a], evidence.loc[b])), float
    )
    reverse = np.asarray(
        predict(pair_evidence(evidence.loc[b], evidence.loc[a])), float
    )
    if forward.shape != a.shape or reverse.shape != a.shape:
        raise ValueError("Pairwise prediction shape mismatch")
    p = (forward + 1 - reverse) / 2
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Pairwise predictions must be finite scores in[0,1]")
    wins = pd.DataFrame(
        dict(index=np.concatenate([a, b]), score=np.concatenate([p, 1 - p]))
    )
    minimum = wins.groupby("index").score.min()
    scores.loc[minimum.index] = minimum
    return scores
