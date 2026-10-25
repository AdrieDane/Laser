"""Research-only strict discovery from independent-spike RT predictions.

No labels, files, occupancy filtering or annotation acceptance. The caller
must supply the matching nested standard/reference scope and protect spikes.
"""

import numpy as np
import pandas as pd

from annotator.composition_ranges import DEFAULT_COMPOSITION_RANGE_SETTINGS
from .automatic import matlab_annotation_adducts
from .class_candidates import _composition_allowed
from .retention_atlas import predict_aligned_retention_atlas
from .standard_anchored_retention import standard_anchored_retention
from .standard_rt_transfer import exclude_declared_standard_masses

PREFIX = "q_standard_aligned_"
FIELDS = (
    "standard_aligned_rt_seconds",
    "standard_alignment_shift_seconds",
    "standard_alignment_anchor_compounds",
    "standard_alignment_training_cohorts",
    "standard_alignment_transfer_MAD_seconds",
    "standard_alignment_offset_spread_seconds",
)
KEYS = ["name", "adduct"]


def aligned_retention_evidence(pool, targets, predictions):
    """Same numeric RT evidence for existing and newly discovered competitors."""
    if not targets.index.equals(predictions.index) or targets.duplicated(KEYS).any():
        raise ValueError("Unique chemical targets and aligned predictions required")
    lookup = predictions[list(FIELDS)].copy()
    lookup.index = pd.MultiIndex.from_frame(targets[KEYS])
    matched = lookup.reindex(pd.MultiIndex.from_frame(pool[KEYS]))
    matched.index = pool.index
    out = pd.DataFrame(index=pool.index)
    for c in FIELDS[1:]:
        out[PREFIX + c.removeprefix("standard_alignment_")] = matched[c]
    error = pd.to_numeric(pool.feat_rt, errors="coerce") * 60 - matched[FIELDS[0]]
    out[PREFIX + "RT_error_seconds"] = error
    out[PREFIX + "absolute_RT_error_seconds"] = error.abs()
    out[PREFIX + "available"] = matched[FIELDS[0]].notna().astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def prepare_standard_aligned_candidates(
    targets,
    features,
    automatic,
    declared,
    measurements,
    model,
    *,
    stream,
    query_group,
    ranges=DEFAULT_COMPOSITION_RANGE_SETTINGS,
):
    """Return new PROPOSALS and target predictions, never accepted annotations.

    Strict raw absppm<3 / RT<=10s, native accepted ions, original coordinates,
    configured C/DB ranges, no isotopes, actual spike masses AND measured FIDs
    protected. Existing physical competitors are preserved by the caller.
    """
    required = {"name", "class", "adduct", "carbon", "db", "ref_mz", "ref_rt"}
    if not required.issubset(targets):
        raise ValueError("Full chemical targets required")
    if features.feature_id.astype(str).duplicated().any():
        raise ValueError("Unique original feature IDs required")
    targets = targets.drop_duplicates(KEYS).reset_index(drop=True).copy()
    isotope = targets.get("isotopologue", pd.Series("", index=targets.index))
    targets = targets[isotope.fillna("").astype(str).eq("")].reset_index(drop=True)
    predictions = standard_anchored_retention(
        targets,
        automatic,
        declared,
        measurements,
        model,
        stream=stream,
        query_group=query_group,
    )
    clean = exclude_declared_standard_masses(automatic, declared)
    base = predict_aligned_retention_atlas(
        targets, clean, model["atlas"], stream=stream
    )
    standard = automatic.get("is_is", pd.Series(False, index=automatic.index))
    protected = set(
        automatic.loc[standard.fillna(False).astype(bool), "feat_feature_id"].astype(
            str
        )
    )
    if len(measurements):
        protected.update(measurements.feat_feature_id.dropna().astype(str))
    mass = pd.to_numeric(features.mz, errors="coerce").to_numpy(float)
    rt = pd.to_numeric(features.rt, errors="coerce").to_numpy(float)
    intensity = pd.to_numeric(features.intensity, errors="coerce").to_numpy(float)
    ids = features.feature_id.astype(str).to_numpy()
    valid = np.flatnonzero(np.isfinite(mass) & np.isfinite(rt) & (mass > 0))
    order = valid[np.argsort(mass[valid], kind="stable")]
    accepted_ions = set(matlab_annotation_adducts(stream[:2]))
    rows = []
    for i, target in targets.loc[
        predictions.standard_aligned_rt_seconds.notna()
    ].iterrows():
        predicted = predictions.loc[i, "standard_aligned_rt_seconds"] / 60
        mz = float(target.ref_mz)
        if (
            not np.isfinite(predicted)
            or not np.isfinite(mz)
            or mz <= 0
            or target.adduct not in accepted_ions
            or not str(target.adduct)
            .rstrip(")")
            .endswith("+" if stream.endswith("Pos") else "-")
            or str(target["name"]).endswith(" (IS)")
            or not _composition_allowed(
                ranges, target["class"], target.carbon, target.db, target["name"]
            )
        ):
            continue
        lo = np.searchsorted(mass[order], mz * (1 - 3e-6))
        hi = np.searchsorted(mass[order], mz * (1 + 3e-6), side="right")
        for p in order[lo:hi]:
            ppm = (mass[p] - mz) / mz * 1e6
            if abs(ppm) >= 3 or abs(rt[p] - predicted) * 60 > 10 or ids[p] in protected:
                continue
            row = target.to_dict()
            row.update(base.loc[i].to_dict())
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
                atlas_available=bool(pd.notna(base.loc[i, "atlas_rt_seconds"])),
                isotopologue="",
                is_is=False,
                automatic_accepted=False,
                automatic_candidate=True,
                auto_annotation_status="independent_standard_RT_candidate",
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
    pool = pd.DataFrame(rows)
    return pool, targets, predictions
