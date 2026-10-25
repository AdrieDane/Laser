"""Unused research: separate label-free feature aggregation / TRAINING targets.

No files, models, scores, annotations or defaults. Physical identities only group
rows. Callers authorize query/outer/inner scopes before attaching supervision.
"""

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from standard_lipidomics.validation import _species
from .evidence_training import FORBIDDEN
from .standard_rt_library import ion_key

GROUP = ["stream", "feat_feature_id"]
DISALLOWED = FORBIDDEN | {
    "calibration_identity_in_manual",
    "manual_confirmed",
    "manually_confirmed",
    "manual_annotation",
    "internal_standard",
    "mapped",
    "positive",
    "known",
}


def physical_label_keys(pool):
    required = {*GROUP, "name", "adduct", "class", "is_is"}
    if not required.issubset(pool) or not pool.index.is_unique:
        raise ValueError("Unique explicit physical candidate rows required")
    if not pool.is_is.isin([True, False]).all():
        raise ValueError("Explicit standard booleans required")
    result = pool[GROUP].copy()
    result["feat_feature_id"] = result.feat_feature_id.astype(str)
    result["species"] = pool.name.map(_species)
    result["ion"] = pool.adduct.map(ion_key)
    result["lipid_class"] = pool["class"].map(canonical_lipid_class)
    result["standard"] = (
        pool.is_is
        | pool.name.astype(str).str.startswith("IS: ")
        | pool.name.astype(str).str.endswith(" (IS)")
    )
    # Two candidate versions of ONE reported label cannot disagree on class.
    if result.groupby([*GROUP, "species", "ion"]).lipid_class.nunique().gt(1).any():
        raise ValueError("A physical label has contradictory chemical classes")
    return result


def feature_evidence(pool, evidence):
    """Aggregate numeric original predictors without inspecting any labels."""
    if (
        not pool.index.equals(evidence.index)
        or not evidence.columns.is_unique
        or DISALLOWED & set(evidence)
    ):
        raise ValueError("Aligned label-free numerical/categorical evidence required")
    keys = physical_label_keys(pool)
    groups = pd.MultiIndex.from_frame(keys[GROUP])
    numeric = evidence.select_dtypes(include=["number", "bool"]).astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric.index = groups
    # Aggregation includes all physical versions, with grouping defined before
    # any supervision; no positive-row subset or fitted score is consulted.
    aggregates = numeric.groupby(level=[0, 1], sort=False).agg(["min", "median", "max"])
    aggregates.columns = [
        "feature_" + statistic + "__" + str(column)
        for column, statistic in aggregates.columns
    ]
    unique = keys.drop_duplicates([*GROUP, "species", "ion"])
    by = unique.groupby(GROUP, sort=False)
    aggregates["feature_distinct_label_hypotheses"] = by.size()
    aggregates["feature_distinct_classes"] = by.lipid_class.nunique()
    aggregates["feature_distinct_adducts"] = by.ion.nunique()
    return aggregates


def feature_training_targets(pool):
    """Explicit authorized flags only; unknown alternatives never become0."""
    keys = physical_label_keys(pool)
    required = {"training_positive", "training_label_known"}
    if (
        not required.issubset(pool)
        or not pool[list(required)].isin([True, False]).all().all()
    ):
        raise ValueError("Explicit separate authorized training labels required")
    if (pool.training_positive & ~pool.training_label_known).any():
        raise ValueError("Positive training labels must be known")
    version_keys = [*GROUP, "species", "ion"]
    data = keys.assign(positive=pool.training_positive, known=pool.training_label_known)
    if (
        data.groupby(version_keys)[["positive", "known", "standard"]]
        .nunique()
        .gt(1)
        .any()
        .any()
    ):
        raise ValueError("Duplicate physical versions have contradictory targets")
    distinct = data.drop_duplicates(version_keys)
    by = distinct.groupby(GROUP, sort=False)
    if by.positive.sum().gt(1).any():
        raise ValueError("Multiple curated lipid/adduct labels on one original feature")
    targets = by.agg(
        positive=("positive", "any"),
        all_labels_known=("known", "all"),
        standard=("standard", "any"),
        distinct_labels=("species", "size"),
    )
    targets["known"] = (targets.positive | targets.all_labels_known) & ~targets.standard
    targets["positive"] &= ~targets.standard
    group_index = pd.MultiIndex.from_frame(keys[GROUP])
    positive_feature = pd.Series(
        targets.positive.reindex(group_index).to_numpy(), index=pool.index
    )
    nonstandard_feature = pd.Series(
        (~targets.standard).reindex(group_index).to_numpy(), index=pool.index
    )
    ambiguous = pd.Series(
        targets.distinct_labels.reindex(group_index).gt(1).to_numpy(), index=pool.index
    )
    conditional = (
        positive_feature & nonstandard_feature & ambiguous & pool.training_label_known
    )
    return targets, conditional
