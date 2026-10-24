"""Research prototype: independent spikes align exact-class RT predictions.

No files, query curation, new candidates or annotation decisions. A declared
spike detected independently of endogenous proposals can estimate an RT offset
relative to an OTHER-project isotope-specific transfer model. Transfer-only
spike detections cannot feed back as anchors for their own endogenous atlas.
"""

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .declared_standard_context import _qualified_measurements
from .retention_atlas import predict_aligned_retention_atlas
from .standard_rt_transfer import (
    exclude_declared_standard_masses,
    predict_standard_class_transfer,
    standard_class_targets,
)


def standard_anchored_retention(
    targets, automatic, declared, measurements, model, *, stream, query_group
):
    """Return numeric target predictions only, never accepted feature identities.

    Same class AND acquisition stream only; no ether/lyso positional merging.
    A single independent chemical needs >=10 transfer cohorts and MAD<=2s;
    multiple chemicals each need >=3 cohorts/MAD<=3s and offsets within3s.
    Maximum absolute offset30s; original observations/ranges remain unchanged.
    Caller must still enforce strict3ppm/10s candidate joins, protect measured
    spike feature IDs, and preserve own-query/outer/inner reference scopes.
    This unused prototype does not change production or experiment032.
    """
    if stream not in {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}:
        raise ValueError("Explicit acquisition stream required")
    if model.get("format") != "laser-nested-standard-context-models-v1":
        raise ValueError("Explicit nested standard reference required")
    training, excluded = set(model["training_groups"]), set(model["excluded_groups"])
    if (
        not training.isdisjoint(excluded)
        or query_group in training
        or query_group not in excluded
    ):
        raise ValueError("Query project must be excluded from reference training")
    if (
        not {"class", "carbon", "db", "ref_mz"}.issubset(targets)
        or not targets.index.is_unique
    ):
        raise ValueError("Unique chemical targets with C/DB/mass required")
    if not {"stream", "standard_compound_id", "ref_mz"}.issubset(declared):
        raise ValueError("Full declared chemical targets required")
    forbidden = {
        "training_positive",
        "training_label_known",
        "same_manual_peak",
        "manual_feature_id",
    }
    for frame in (targets, automatic, declared, measurements):
        if forbidden & set(frame):
            raise ValueError("Only label-free physical inputs permitted")
        if "stream" in frame and len(frame) and not frame.stream.eq(stream).all():
            raise ValueError("Cannot combine acquisition streams")
        for flag in ("manual_confirmed", "manually_confirmed", "manual_annotation"):
            if flag in frame and frame[flag].fillna(False).astype(bool).any():
                raise ValueError("Only label-free physical inputs permitted")
    clean = exclude_declared_standard_masses(automatic, declared)
    base = predict_aligned_retention_atlas(
        targets, clean, model["atlas"], stream=stream
    )
    out = pd.DataFrame(index=targets.index)
    out["standard_clean_atlas_rt_seconds"] = base.atlas_rt_seconds
    out["standard_aligned_rt_seconds"] = np.nan
    out["standard_alignment_shift_seconds"] = np.nan
    out["standard_alignment_anchor_compounds"] = 0.0
    out["standard_alignment_training_cohorts"] = np.nan
    out["standard_alignment_transfer_MAD_seconds"] = np.nan
    out["standard_alignment_offset_spread_seconds"] = np.nan
    out["standard_alignment_protected_mass"] = 0.0
    if targets.empty or measurements.empty or declared.empty:
        return out
    measured = _qualified_measurements(measurements)
    # A spike must be found from the independent standard-library RT route,
    # NOT the endogenous class-transfer fallback that we intend to align.
    measured = measured[
        measured.get(
            "standard_rt_reference_source", pd.Series("", index=measured.index)
        ).eq("other_declared_standards")
    ]
    if measured.empty:
        return out
    transfer = model["transfer"]
    if transfer.get("format") != "laser-declared-standard-class-transfer-v1":
        raise ValueError("Explicit isotope-specific standard transfer required")
    valid_models = {}
    for key, item in transfer["models"].items():
        n = item["training_cohorts"]
        mad, offset = item["training_MAD_seconds"], item["offset_seconds"]
        if (
            np.isfinite([n, mad, offset]).all()
            and 3 <= n <= len(training)
            and float(n).is_integer()
            and 0 <= mad <= 3
            and abs(offset) <= 60
        ):
            valid_models[key] = item
    if not valid_models:
        return out
    transfer = dict(transfer, models=valid_models)
    standardized = standard_class_targets(declared, model["library"], stream=stream)
    expected = predict_standard_class_transfer(
        standardized, clean, model["atlas"], transfer, stream=stream
    )
    standardized = standardized.join(expected)
    declared_by_compound = {
        k: g for k, g in standardized.groupby("standard_compound_id", sort=False)
    }
    anchors = []
    for compound, group in measured.groupby("standard_compound_id", sort=False):
        native = declared_by_compound.get(compound)
        if native is None or group._class.nunique() != 1:
            continue
        # Require actual declared masses, rather than a standard name alias.
        actual_masses = native.ref_mz.to_numpy(float)
        errors = np.abs(group.ref_mz.to_numpy(float)[:, None] - actual_masses[None, :])
        if not (errors <= np.maximum(1e-5, actual_masses * 0.05e-6)).any(axis=1).all():
            continue
        valid = native.dropna(subset=["standard_class_transfer_rt_seconds"])
        if valid.empty or valid["class"].map(canonical_lipid_class).nunique() != 1:
            continue
        cls = canonical_lipid_class(valid["class"].iloc[0])
        if cls != group._class.iloc[0]:
            continue
        predicted = valid.standard_class_transfer_rt_seconds.to_numpy(float)
        if np.ptp(predicted) > 1e-6:
            raise ValueError("Adducts of one standard disagree on class-transfer RT")
        cohorts = float(valid.standard_class_transfer_cohorts.min())
        mad = float(valid.standard_class_transfer_MAD_seconds.max())
        shift = float(group.feat_rt.median() * 60 - np.median(predicted))
        if (
            not np.isfinite([cohorts, mad, shift]).all()
            or cohorts < 3
            or not 0 <= mad <= 3
            or abs(shift) > 30
        ):
            continue
        anchors.append(dict(cls=cls, shift=shift, cohorts=cohorts, mad=mad))
    masses = pd.to_numeric(targets.ref_mz, errors="coerce").to_numpy(float)
    declared_masses = (
        pd.to_numeric(declared.ref_mz, errors="coerce").dropna().to_numpy(float)
    )
    protected = (
        np.abs(masses[:, None] - declared_masses[None, :])
        <= np.maximum(1e-5, declared_masses * 0.05e-6)
    ).any(axis=1)
    out["standard_alignment_protected_mass"] = protected.astype(float)
    if not anchors:
        return out
    anchors = pd.DataFrame(anchors)
    classes = targets["class"].map(canonical_lipid_class)
    for cls, group in anchors.groupby("cls", sort=False):
        if len(group) == 1 and (group.cohorts.iloc[0] < 10 or group["mad"].iloc[0] > 2):
            continue
        spread = float(group["shift"].max() - group["shift"].min())
        if spread > 3:
            continue
        shift = float(group["shift"].median())
        valid = classes.eq(cls) & ~pd.Series(protected, index=targets.index)
        valid &= base.atlas_in_domain & base.atlas_training_cohorts.ge(3)
        valid &= base.atlas_training_mad_seconds.le(5) & base.atlas_rt_seconds.notna()
        valid &= base.atlas_rt_seconds.add(shift).gt(0)
        ids = targets.index[valid]
        out.loc[ids, "standard_aligned_rt_seconds"] = (
            base.loc[ids, "atlas_rt_seconds"] + shift
        )
        out.loc[ids, "standard_alignment_shift_seconds"] = shift
        out.loc[ids, "standard_alignment_anchor_compounds"] = len(group)
        out.loc[ids, "standard_alignment_training_cohorts"] = group.cohorts.min()
        out.loc[ids, "standard_alignment_transfer_MAD_seconds"] = group["mad"].max()
        out.loc[ids, "standard_alignment_offset_spread_seconds"] = spread
    return out
