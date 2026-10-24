"""Unused research weights: temper large candidate-class dominance per cohort.

Uses only candidate classes/modes and split metadata, never curated class
presence, held-out labels or query identities as predictors. Not an FDR model.
"""

from collections import Counter

import numpy as np


def tempered_class_mode_weights(groups, streams, classes):
    """Preserve equal total weight per cohort; cap relative cell factors.

    Fixed square-root inverse cell-size factors, relative to median nonempty
    class/mode cell size in each cohort, clipped to[.25,4] BEFORE cohort
    normalization. No positive/negative reweighting or learned/test thresholds.
    """
    values = [np.asarray(v, dtype=str) for v in (groups, streams, classes)]
    if any(v.ndim != 1 for v in values) or not len(values[0]):
        raise ValueError("Nonempty one-dimensional cohort/mode/class vectors required")
    if len({len(v) for v in values}) != 1 or any(
        np.any(np.char.strip(v) == "") for v in values
    ):
        raise ValueError("Aligned explicit cohort/mode/class labels required")
    group, stream, cls = values
    cells = Counter(zip(group, stream, cls))
    medians = {
        g: np.median([n for (cohort, _, _), n in cells.items() if cohort == g])
        for g in np.unique(group)
    }
    factors = np.asarray(
        [
            np.clip(np.sqrt(medians[g] / cells[(g, s, c)]), 0.25, 4.0)
            for g, s, c in zip(group, stream, cls)
        ]
    )
    weights = np.empty(len(group), dtype=float)
    target = len(group) / len(medians)
    for g in medians:
        rows = group == g
        weights[rows] = factors[rows] * target / factors[rows].sum()
    assert np.isfinite(weights).all() and (weights > 0).all()
    return weights
