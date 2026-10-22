"""Training-only support rules for experimental class/mode specialists.

No held-out labels, feature identities or dataset names enter prediction. Group
identities are used only to balance training weights and measure support.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def cohort_balanced_weights(groups: pd.Series) -> np.ndarray:
    """Give each training project group equal total weight, mean weight one."""
    values = groups.astype(str).tolist()
    counts = Counter(values)
    if not counts:
        return np.empty(0, dtype=float)
    return np.asarray([len(values) / len(counts) / counts[g] for g in values])


def specialist_training_plan(
    classes: pd.Series,
    positives: pd.Series,
    groups: pd.Series,
    *,
    min_positive: int = 100,
    min_negative: int = 100,
    min_groups: int = 6,
) -> dict[str, dict]:
    """Freeze actions using labelled TRAINING candidates only.

    Zero-positive classes are withheld only with at least ``min_groups`` groups
    of observed negative candidates. Unknown/new classes are not a whitelist:
    callers must fall back to the existing global model for unknown keys.
    """
    if not classes.index.equals(positives.index) or not classes.index.equals(
        groups.index
    ):
        raise ValueError(
            "Training classes, labels and groups must have identical indices"
        )
    if positives.isna().any() or not positives.isin([True, False, 0, 1]).all():
        raise ValueError("Only resolved binary TRAINING labels are supported")
    if classes.isna().any() or groups.isna().any():
        raise ValueError("Training classes and groups must be known")
    data = pd.DataFrame(
        {
            "class": classes.astype(str),
            "positive": positives.astype(bool),
            "group": groups.astype(str),
        }
    )
    result = {}
    for key, part in data.groupby("class", sort=True):
        positive = part[part.positive]
        npos, nneg = len(positive), int((~part.positive).sum())
        ngroups, positive_groups = part.group.nunique(), positive.group.nunique()
        if npos == 0 and ngroups >= min_groups:
            action = "withhold_zero_positive_support"
        elif (
            npos >= min_positive
            and nneg >= min_negative
            and ngroups >= min_groups
            and positive_groups >= min_groups
        ):
            action = "specialist"
        else:
            action = "global_fallback"
        result[key] = {
            "action": action,
            "positive_pairs": npos,
            "negative_pairs": nneg,
            "training_groups": int(ngroups),
            "positive_groups": int(positive_groups),
        }
    return result
