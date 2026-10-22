"""Research-only declared-standard RT transfer from independent class anchors.

The constant standard-specific residual is learned in OTHER acquisition groups.
All masses of declared spikes are excluded from automatic alignment anchors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .retention_atlas import predict_aligned_retention_atlas

FORMAT = "laser-declared-standard-class-transfer-v1"


def exclude_declared_standard_masses(automatic, targets):
    if automatic.empty or targets.empty or "feat_mz" not in automatic:
        return automatic.copy()
    masses = np.sort(
        pd.to_numeric(targets.ref_mz, errors="coerce").dropna().to_numpy(float)
    )
    observed = pd.to_numeric(automatic.feat_mz, errors="coerce").to_numpy(float)
    lower = np.searchsorted(masses, observed / (1 + 3e-6), side="left")
    upper = np.searchsorted(masses, observed / (1 - 3e-6), side="right")
    return automatic.loc[lower == upper].copy()


def standard_class_targets(targets, standard_library, *, stream):
    data = targets.copy()
    data["class"] = [
        standard_library["models"]
        .get(stream + "|" + r[0], {})
        .get("reporting_class", r[1])
        for r in data[["standard_compound_id", "class"]].itertuples(
            index=False, name=None
        )
    ]
    return data


def fit_standard_class_transfer(observations, *, minimum_cohorts=3):
    required = {"cohort", "stream", "standard_compound_id", "class", "residual_seconds"}
    if not required.issubset(observations) or minimum_cohorts < 3:
        raise ValueError("Standard transfer requires grouped residual observations")
    data = observations.copy()
    data["class"] = data["class"].map(canonical_lipid_class)
    data["residual_seconds"] = pd.to_numeric(data.residual_seconds, errors="coerce")
    data = data[np.isfinite(data.residual_seconds)]
    data = data.groupby(
        ["cohort", "stream", "standard_compound_id", "class"], as_index=False
    ).residual_seconds.median()
    models = {}
    for (stream, std), group in data.groupby(
        ["stream", "standard_compound_id"], sort=True
    ):
        n = int(group.cohort.nunique())
        if n < minimum_cohorts or group["class"].nunique() != 1:
            continue
        offset = float(group.residual_seconds.median())
        mad = float(np.median(np.abs(group.residual_seconds - offset)))
        if abs(offset) > 60 or mad > 5:
            continue
        models[stream + "|" + std] = dict(
            offset_seconds=offset,
            training_MAD_seconds=mad,
            training_cohorts=n,
            lipid_class=group["class"].iloc[0],
        )
    return dict(format=FORMAT, models=models, minimum_cohorts=minimum_cohorts)


def predict_standard_class_transfer(targets, automatic, atlas, transfer, *, stream):
    if transfer.get("format") != FORMAT:
        raise ValueError("Unsupported standard class-transfer model")
    output = pd.DataFrame(index=targets.index)
    output["standard_class_transfer_rt_seconds"] = np.nan
    output["standard_class_transfer_cohorts"] = 0
    output["standard_class_transfer_MAD_seconds"] = np.nan
    data = targets.copy()
    models = transfer["models"]
    supported = [
        i for i, r in data.iterrows() if stream + "|" + r.standard_compound_id in models
    ]
    if not supported:
        return output
    data = data.loc[supported].copy()
    data["class"] = [
        models[stream + "|" + s]["lipid_class"] for s in data.standard_compound_id
    ]
    clean = exclude_declared_standard_masses(automatic, targets)
    predicted = predict_aligned_retention_atlas(data, clean, atlas, stream=stream)
    for i, row in data.iterrows():
        model = models[stream + "|" + row.standard_compound_id]
        base = predicted.loc[i, "atlas_rt_seconds"]
        if not np.isfinite(base):
            continue
        output.loc[i, "standard_class_transfer_rt_seconds"] = (
            base + model["offset_seconds"]
        )
        output.loc[i, "standard_class_transfer_cohorts"] = model["training_cohorts"]
        output.loc[i, "standard_class_transfer_MAD_seconds"] = model[
            "training_MAD_seconds"
        ]
    return output


def combine_standard_rt_sources(primary, transferred):
    """Use class transfer only when standard calibration is unavailable.

    If both independent references exist but disagree by >10s, new completion
    remains unconfirmed. Existing accepted annotations are handled separately.
    """
    if not primary.index.equals(transferred.index):
        raise ValueError("Standard RT source indices must match")
    result = primary.join(transferred)
    original = result.standard_rt_seconds.copy()
    fallback = result.standard_class_transfer_rt_seconds
    result["standard_primary_rt_seconds"] = original
    conflict = original.notna() & fallback.notna() & (original - fallback).abs().gt(10)
    result["standard_rt_reference_source"] = np.where(
        original.notna(),
        "other_declared_standards",
        np.where(
            fallback.notna(), "independent_endogenous_class_transfer", "unavailable"
        ),
    )
    result["standard_rt_seconds"] = original.fillna(fallback).mask(conflict)
    result.loc[conflict, "standard_rt_reference_source"] = (
        "independent_RT_references_disagree"
    )
    return result
