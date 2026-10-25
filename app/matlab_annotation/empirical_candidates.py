"""Unused research: strict empirical-RT proposals and numeric evidence ONLY.

No files, curation, scores, occupancy filtering or accepted annotations. Caller
preserves every old competitor and enforces matching nested reference scopes.
"""

import numpy as np
import pandas as pd

from annotator.composition_ranges import DEFAULT_COMPOSITION_RANGE_SETTINGS
from .automatic import matlab_annotation_adducts
from .class_candidates import _composition_allowed
from .empirical_retention import predict_aligned_empirical_retention
from .retention_atlas import predict_aligned_retention_atlas

PREFIX = "q_empirical_"
FIELDS = (
    "empirical_offset_seconds",
    "empirical_training_projects",
    "empirical_training_MAD_seconds",
)
KEYS = ["name", "adduct"]


def empirical_retention_evidence(pool, targets, predictions, parent):
    if (
        not targets.index.equals(predictions.index)
        or not targets.index.equals(parent.index)
        or targets.duplicated(KEYS).any()
    ):
        raise ValueError("Aligned unique chemical target predictions required")
    lookup = predictions[["atlas_rt_seconds", *FIELDS]].copy()
    lookup["parent_rt_seconds"] = parent.atlas_rt_seconds
    lookup.index = pd.MultiIndex.from_frame(targets[KEYS])
    matched = lookup.reindex(pd.MultiIndex.from_frame(pool[KEYS]))
    matched.index = pool.index
    out = pd.DataFrame(index=pool.index)
    error = pd.to_numeric(pool.feat_rt, errors="coerce") * 60 - matched.atlas_rt_seconds
    out[PREFIX + "RT_error_seconds"] = error
    out[PREFIX + "absolute_RT_error_seconds"] = error.abs()
    for field in FIELDS:
        out[PREFIX + field.removeprefix("empirical_")] = matched[field]
    out[PREFIX + "RT_change_from_parent_seconds"] = (
        matched.atlas_rt_seconds - matched.parent_rt_seconds
    )
    out[PREFIX + "exact_reference_available"] = (
        matched.empirical_offset_seconds.notna().astype(float)
    )
    out[PREFIX + "prediction_available"] = matched.atlas_rt_seconds.notna().astype(
        float
    )
    return out.replace([np.inf, -np.inf], np.nan)


def prepare_empirical_candidates(
    targets,
    features,
    automatic,
    declared,
    measurements,
    atlas,
    model,
    *,
    stream,
    query_group,
    ranges=DEFAULT_COMPOSITION_RANGE_SETTINGS,
):
    """Strict rawabsppm<3 / RT<=10s, same native targets/ranges and ORIGINAL IDs.

    Full physical alternatives before any acceptance/occupancy filtering. Actual
    spike masses AND independently measured spike FIDs protected. No replacement
    of old RT references or measured coordinates; old pool preserved upstream.
    """
    required = {"name", "class", "adduct", "carbon", "db", "ref_mz", "ref_rt"}
    if not required.issubset(targets) or not {
        "feature_id",
        "mz",
        "rt",
        "intensity",
    }.issubset(features):
        raise ValueError(
            "Full chemical targets and original physical features required"
        )
    if not features.feature_id.astype(str).is_unique:
        raise ValueError("Unique original feature IDs required")
    if stream not in {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}:
        raise ValueError("Known acquisition stream required")
    for frame in (targets, features, automatic, declared, measurements):
        if {
            "training_positive",
            "training_label_known",
            "same_manual_peak",
            "manual_feature_id",
        } & set(frame):
            raise ValueError("No supervised query inputs allowed")
    targets = targets.drop_duplicates(KEYS).reset_index(drop=True).copy()
    isotope = targets.get("isotopologue", pd.Series("", index=targets.index))
    targets = targets[isotope.fillna("").astype(str).eq("")].reset_index(drop=True)
    predictions = predict_aligned_empirical_retention(
        targets, automatic, atlas, model, stream=stream, query_group=query_group
    )
    parent = predict_aligned_retention_atlas(targets, automatic, atlas, stream=stream)
    standards = (
        automatic.get("is_is", pd.Series(False, index=automatic.index))
        .fillna(False)
        .astype(bool)
    )
    protected = (
        set(automatic.loc[standards, "feat_feature_id"].astype(str))
        if len(automatic)
        else set()
    )
    if len(measurements):
        protected.update(measurements.feat_feature_id.dropna().astype(str))
    spike_mz = (
        pd.to_numeric(declared.get("ref_mz", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .to_numpy(float)
    )
    mass = pd.to_numeric(features.mz, errors="coerce").to_numpy(float)
    rt = pd.to_numeric(features.rt, errors="coerce").to_numpy(float)
    intensity = pd.to_numeric(features.intensity, errors="coerce").to_numpy(float)
    ids = features.feature_id.astype(str).to_numpy()
    valid = np.flatnonzero(np.isfinite(mass) & np.isfinite(rt) & (mass > 0))
    order = valid[np.argsort(mass[valid], kind="stable")]
    accepted_ions = set(matlab_annotation_adducts(stream[:2]))
    rows = []
    # Provably empty original mass intervals cannot emit a physical pair.
    # Precompute ONLY this necessary condition, preserving every target and
    # predictor in the returned tables. The original exact per-hit mass/RT/
    # chemistry checks below remain unchanged and determine final proposals.
    _target_masses = targets.ref_mz.to_numpy(dtype=float)
    _target_lo = np.searchsorted(mass[order], _target_masses * (1 - 3e-6))
    _target_hi = np.searchsorted(mass[order], _target_masses * (1 + 3e-6), side="right")
    _mass_possible = _target_hi > _target_lo
    for i, target in targets.loc[_mass_possible].iterrows():
        predicted = float(predictions.loc[i, "atlas_rt_seconds"]) / 60
        mz = float(target.ref_mz)
        if (
            not np.isfinite([predicted, mz]).all()
            or predicted <= 0
            or mz <= 0
            or target.adduct not in accepted_ions
            or str(target.adduct)
            .rstrip(")")
            .endswith("+" if stream.endswith("Pos") else "-")
            is False
            or str(target["name"]).endswith(" (IS)")
        ):
            continue
        if (
            len(spike_mz)
            and (np.abs(spike_mz - mz) <= np.maximum(1e-5, spike_mz * 0.05e-6)).any()
        ):
            continue
        if not _composition_allowed(
            ranges, target["class"], target.carbon, target.db, target["name"]
        ):
            continue
        lo = np.searchsorted(mass[order], mz * (1 - 3e-6))
        hi = np.searchsorted(mass[order], mz * (1 + 3e-6), side="right")
        for p in order[lo:hi]:
            ppm = (mass[p] - mz) / mz * 1e6
            if abs(ppm) >= 3 or abs(rt[p] - predicted) * 60 > 10 or ids[p] in protected:
                continue
            row = target.to_dict()
            row.update(parent.loc[i].to_dict())
            row.update(
                stream=stream,
                feat_feature_id=ids[p],
                feat_mz=mass[p],
                feat_rt=rt[p],
                feat_intensity=intensity[p],
                baseline_ref_rt=target.ref_rt,
                ref_rt=predicted,
                rt_expected=predicted,
                rt_error=rt[p] - predicted,
                ppm_error=ppm,
                atlas_available=bool(pd.notna(parent.loc[i, "atlas_rt_seconds"])),
                isotopologue="",
                is_is=False,
                automatic_accepted=False,
                automatic_candidate=True,
                auto_annotation_status="empirical_RT_candidate",
                q_atlas_source=0.0,
                q_fitted_source=0.0,
                q_previously_selected=0.0,
                q_chemical_target_generated=float(
                    bool(target.get("chemical_target_generated", False))
                ),
            )
            row["q_atlas_RT_error_seconds"] = rt[p] * 60 - row["atlas_rt_seconds"]
            row["q_absolute_atlas_RT_error_seconds"] = abs(
                row["q_atlas_RT_error_seconds"]
            )
            rows.append(row)
    return pd.DataFrame(rows), targets, predictions, parent
