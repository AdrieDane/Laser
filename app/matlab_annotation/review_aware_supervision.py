"""Unused research: temper ambiguous negative TRAINING evidence, never labels.

Curated absence is not established chemical absence. A class missing in one
manual mode but present in another is less certain negative evidence, unless
the candidate contradicts a curated identity or occupies a curated peak.
No pseudo-positives, query-curation predictors, or benchmark relabeling.
"""

from collections import Counter

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class


def uncertain_review_negatives(
    pool,
    manual,
    *,
    query_group,
    training_groups,
    excluded_groups,
    normalize_species,
    normalize_adduct,
):
    """Return a fit-only flag AFTER scope authorization and physical generation.

    The caller must check these permissions before reading its manual file too.
    Manual must have the existing benchmark columns; standards do not establish
    endogenous class review. Ether/positional classes are NOT merged.
    """
    training, excluded = set(training_groups), set(excluded_groups)
    if (
        not training
        or not excluded
        or training & excluded
        or query_group not in training
        or query_group in excluded
    ):
        raise ValueError("Explicit authorized training-only curation scope required")
    if not {"training_positive", "training_label_known"}.issubset(pool):
        raise ValueError("Original positive/known flags required before review audit")
    if (
        not pool[["training_positive", "training_label_known"]]
        .isin([True, False])
        .all()
        .all()
    ):
        raise ValueError("Original supervision must be explicit boolean")
    if (pool.training_positive & ~pool.training_label_known).any():
        raise ValueError("A positive cannot be unknown")
    manual = manual[~manual.internal_standard]
    reviewed = {
        (s, canonical_lipid_class(c))
        for s, c in manual[["stream", "lipid_class"]].itertuples(index=False, name=None)
    }
    classes = {c for _, c in reviewed}
    identities = {
        (s, normalize_species(n), normalize_adduct(a))
        for s, n, a in manual[["stream", "species", "adduct"]].itertuples(
            index=False, name=None
        )
    }
    occupied = {
        (s, str(f))
        for s, f in manual.loc[manual.mapped, ["stream", "feature_id"]].itertuples(
            index=False, name=None
        )
    }
    flags = []
    for positive, known, s, c, n, a, f in pool[
        [
            "training_positive",
            "training_label_known",
            "stream",
            "class",
            "name",
            "adduct",
            "feat_feature_id",
        ]
    ].itertuples(index=False, name=None):
        c = canonical_lipid_class(c)
        flags.append(
            bool(
                known
                and not positive
                and c in classes
                and (s, c) not in reviewed
                and (s, normalize_species(n), normalize_adduct(a)) not in identities
                and (s, str(f)) not in occupied
            )
        )
    return pd.Series(flags, index=pool.index, dtype=bool)


def review_tempered_weights(groups, positive, uncertain):
    """Fixed quarter-weight uncertain negatives; equal totals per project.

    Labels and row counts remain identical. No learned/downstream tunable
    strength. Positives and established wrong-peak/label negatives retain
    their within-project relative weight of1.
    """
    groups = np.asarray(groups, dtype=str)
    positive, uncertain = np.asarray(positive), np.asarray(uncertain)
    if (
        not len(groups)
        or groups.ndim != 1
        or any(
            x.shape != groups.shape or not np.isin(x, [True, False]).all()
            for x in (positive, uncertain)
        )
        or np.any(groups == "")
    ):
        raise ValueError("Aligned nonempty project and boolean supervision required")
    if np.any(positive.astype(bool) & uncertain.astype(bool)):
        raise ValueError("Positive examples must not be downweighted as negatives")
    factors = np.where(uncertain.astype(bool), 0.25, 1.0)
    totals = Counter()
    for group, value in zip(groups, factors):
        totals[group] += float(value)
    target = len(groups) / len(totals)
    return np.asarray(
        [value * target / totals[group] for group, value in zip(groups, factors)]
    )
