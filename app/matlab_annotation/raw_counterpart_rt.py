"""Unused numerical research: independent declared-spike polarity RT links.

No curation, files, endogenous anchors, annotations or default changes.
Alternate ions supply EVIDENCE ONLY, including native review-only channels.
"""

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from standard_lipidomics.validation import _boolean
from .adduct_family_evidence import native_adduct_families
from .standard_rt_library import ion_key
from .vendor.compound_model import PROTON_MASS

MODES = {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}
SUPERVISION = {
    "training_positive",
    "training_label_known",
    "same_manual_peak",
    "manual_feature_id",
    "manual_confirmed",
    "manually_confirmed",
    "calibration_identity_in_manual",
}


def opposite(stream):
    if stream not in MODES:
        raise ValueError("Explicit RP/NP polarity stream required")
    return stream[:3] + ("Neg" if stream.endswith("Pos") else "Pos")


def template_mass(mz, ion, family, *, positive):
    """Use existing native class/adduct offsets, not universal proton guesses."""
    ion = ion_key(ion)
    if ion not in family:
        return None
    offset, charge = family[ion]
    if not charge or (charge > 0) != positive:
        raise ValueError("Native source adduct disagrees with acquisition polarity")
    if not np.isfinite(mz) or mz <= 0:
        raise ValueError("Finite positive theoretical mass required")
    return float(charge * (np.sign(charge) * float(mz) - PROTON_MASS) - offset)


def _centers(measurements, stream, families):
    if SUPERVISION & set(measurements):
        raise ValueError("Declared-spike evidence refuses curator supervision")
    required = {
        "stream",
        "is_is",
        "name",
        "class",
        "adduct",
        "ref_mz",
        "feat_rt",
        "feat_intensity",
        "feat_feature_id",
        "standard_compound_id",
    }
    if not required.issubset(measurements):
        raise ValueError("Explicit declared-spike measurements required")
    std = measurements.is_is.map(_boolean).fillna(False).astype(bool)
    part = measurements.loc[measurements.stream.eq(stream) & std].copy()
    part = part.loc[
        part.standard_compound_id.notna() & part.standard_compound_id.astype(str).ne("")
    ]
    result = {}
    for cid, group in part.groupby("standard_compound_id", sort=False):
        classes = group["class"].map(canonical_lipid_class)
        if classes.nunique() != 1:
            continue
        cls = classes.iloc[0]
        rt = pd.to_numeric(group.feat_rt, errors="coerce").to_numpy(float) * 60
        intensity = pd.to_numeric(group.feat_intensity, errors="coerce").to_numpy(float)
        if (
            not np.isfinite(rt).all()
            or (rt <= 0).any()
            or not np.isfinite(intensity).all()
            or (intensity <= 0).any()
            or np.ptp(rt) > 3
        ):
            continue
        family = families.get((stream[:2], cls), {})
        masses = []
        for r in group.itertuples():
            m = template_mass(
                float(r.ref_mz), r.adduct, family, positive=stream.endswith("Pos")
            )
            if m is not None:
                masses.append(m)
        if not masses or np.ptp(masses) > 2e-5:
            continue
        result[str(cid)] = dict(
            standard_compound_id=str(cid),
            lipid_class=cls,
            rt_seconds=float(np.median(rt)),
            template_mass=float(np.median(masses)),
            feature_ids=tuple(group.feat_feature_id.astype(str)),
        )
    return result


def _independent_pairs(frame):
    result = frame.copy()
    for field in ("source_feature_ids", "target_feature_ids"):
        if field not in result:
            continue
        owners = {}
        for row in result.itertuples():
            fids = getattr(row, field)
            if not isinstance(fids, (tuple, list, set)):
                raise ValueError("Explicit original feature-ID collections required")
            for fid in fids:
                owners.setdefault(fid, set()).add(row.standard_compound_id)
        collided = {
            cid for values in owners.values() if len(values) > 1 for cid in values
        }
        result = result.loc[~result.standard_compound_id.isin(collided)]
    return result.reset_index(drop=True)


def independent_spike_pairs(
    measurements, target_features, reference_root, *, source_stream
):
    """Conservative alternate-ion augmentation; never returns annotations.

    A pre-existing shared declared spike seeds a bounded coarse30s search.
    Unique<1.5ppm opposite native-ion hits must agreeRTwithin3s; any ambiguous
    configured ion vetoes the compound, BEFORE any annotation occupancy filter.
    """
    target_stream = opposite(source_stream)
    families = native_adduct_families(str(reference_root))
    source = _centers(measurements, source_stream, families)
    target = _centers(measurements, target_stream, families)
    columns = [
        "standard_compound_id",
        "lipid_class",
        "source_rt_seconds",
        "target_rt_seconds",
        "source_feature_ids",
        "target_feature_ids",
        "origin",
        "maximum_abs_ppm",
    ]
    pairs = []
    for cid in sorted(set(source) & set(target)):
        a, b = source[cid], target[cid]
        if (
            a["lipid_class"] != b["lipid_class"]
            or abs(a["template_mass"] - b["template_mass"]) > 2e-5
        ):
            continue
        pairs.append(
            dict(
                standard_compound_id=cid,
                lipid_class=a["lipid_class"],
                source_rt_seconds=a["rt_seconds"],
                target_rt_seconds=b["rt_seconds"],
                source_feature_ids=a["feature_ids"],
                target_feature_ids=b["feature_ids"],
                origin="existing_independent_declared_pair",
                maximum_abs_ppm=np.nan,
            )
        )
    initial = _independent_pairs(pd.DataFrame(pairs, columns=columns))
    pairs = initial.to_dict("records")
    if initial.empty:
        return initial
    shift = float(
        np.median([r["target_rt_seconds"] - r["source_rt_seconds"] for r in pairs])
    )
    if abs(shift) > 30:
        return initial
    if (
        not {"feature_id", "mz", "rt", "intensity"}.issubset(target_features)
        or target_features.feature_id.astype(str).duplicated().any()
    ):
        raise ValueError("Unique original target-polarity features required")
    mz = pd.to_numeric(target_features.mz, errors="coerce").to_numpy(float)
    rt = pd.to_numeric(target_features.rt, errors="coerce").to_numpy(float) * 60
    intensity = pd.to_numeric(target_features.intensity, errors="coerce").to_numpy(
        float
    )
    valid = np.flatnonzero(
        np.isfinite(mz)
        & np.isfinite(rt)
        & np.isfinite(intensity)
        & (mz > 0)
        & (rt > 0)
        & (intensity > 0)
    )
    order = valid[np.argsort(mz[valid], kind="stable")]
    sorted_mass = mz[order]
    existing = {r["standard_compound_id"] for r in pairs}
    for cid, a in source.items():
        if cid in existing:
            continue
        family = families.get((source_stream[:2], a["lipid_class"]), {})
        hits = {}
        ambiguous = False
        for offset, charge in family.values():
            if not charge or (charge > 0) != target_stream.endswith("Pos"):
                continue
            expected = abs((a["template_mass"] + offset) / charge + PROTON_MASS)
            lo = np.searchsorted(sorted_mass, expected * (1 - 1.5e-6))
            hi = np.searchsorted(sorted_mass, expected * (1 + 1.5e-6), side="right")
            indices = order[lo:hi]
            indices = indices[
                (np.abs((mz[indices] / expected - 1) * 1e6) < 1.5)
                & (np.abs(rt[indices] - (a["rt_seconds"] + shift)) <= 30)
            ]
            if len(indices) > 1:
                ambiguous = True
                break
            for i in indices:
                hits[int(i)] = abs((mz[i] / expected - 1) * 1e6)
        if ambiguous or not hits:
            continue
        selected = list(hits)
        if np.ptp(rt[selected]) > 3:
            continue
        pairs.append(
            dict(
                standard_compound_id=cid,
                lipid_class=a["lipid_class"],
                source_rt_seconds=a["rt_seconds"],
                target_rt_seconds=float(np.median(rt[selected])),
                source_feature_ids=a["feature_ids"],
                target_feature_ids=tuple(
                    target_features.iloc[selected].feature_id.astype(str)
                ),
                origin="unique_native_alternate_ion_evidence_only",
                maximum_abs_ppm=float(max(hits.values())),
            )
        )
    result = _independent_pairs(pd.DataFrame(pairs, columns=columns))
    # Declared labels/ion aliases cannot manufacture independent calibrators
    # by reusing the same original physical peak for different compounds.
    if not result.origin.eq("existing_independent_declared_pair").any():
        return result.iloc[:0]
    return result.reset_index(drop=True)


def fit_independent_rt_links(pairs, *, source_stream):
    """No reliable relationship returns UNKNOWN, never a zero-residual claim."""
    opposite(source_stream)
    required = {
        "standard_compound_id",
        "lipid_class",
        "source_rt_seconds",
        "target_rt_seconds",
    }
    if not required.issubset(pairs) or pairs.standard_compound_id.duplicated().any():
        raise ValueError("Distinct explicit declared-spike RTpairs required")
    pairs = _independent_pairs(pairs)
    result = dict(
        source_stream=source_stream,
        target_stream=opposite(source_stream),
        global_map=None,
        class_maps={},
    )
    if pairs.empty:
        return result
    x, y = (
        pairs.source_rt_seconds.to_numpy(float),
        pairs.target_rt_seconds.to_numpy(float),
    )
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or (x <= 0).any()
        or (y <= 0).any()
    ):
        raise ValueError("Finite positive original RTs required")
    if source_stream.startswith("NP"):
        for cls, group in pairs.groupby("lipid_class", sort=False):
            delta = group.target_rt_seconds.to_numpy(
                float
            ) - group.source_rt_seconds.to_numpy(float)
            center = float(np.median(delta))
            mad = float(np.median(np.abs(delta - center)))
            if len(group) >= 2 and mad <= 2 and abs(center) <= 30:
                result["class_maps"][cls] = dict(
                    slope=1.0,
                    intercept_seconds=center,
                    anchor_compounds=len(group),
                    mad_seconds=mad,
                    minimum_source_seconds=float(group.source_rt_seconds.min()),
                    maximum_source_seconds=float(group.source_rt_seconds.max()),
                )
        return result
    if len(pairs) < 6 or np.ptp(x) < 120 or abs(np.median(y - x)) > 30:
        return result
    slope, intercept = np.polyfit(x, y, 1)
    if not 0.8 <= slope <= 1.2:
        return result
    errors = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        if np.ptp(x[keep]) < 120:
            return result
        s, b = np.polyfit(x[keep], y[keep], 1)
        if not 0.8 <= s <= 1.2:
            return result
        errors.append(abs(float(y[i] - (s * x[i] + b))))
    if np.median(errors) > 2 or np.quantile(errors, 0.9) > 4:
        return result
    result["global_map"] = dict(
        slope=float(slope),
        intercept_seconds=float(intercept),
        anchor_compounds=len(pairs),
        mad_seconds=float(np.median(errors)),
        minimum_source_seconds=float(x.min()),
        maximum_source_seconds=float(x.max()),
    )
    return result


def mapped_counterpart_rt(links, source_seconds, lipid_class):
    mapping = (
        links["global_map"]
        if links["source_stream"].startswith("RP")
        else links["class_maps"].get(canonical_lipid_class(lipid_class))
    )
    if (
        mapping is None
        or not np.isfinite(source_seconds)
        or not mapping["minimum_source_seconds"] - 30
        <= source_seconds
        <= mapping["maximum_source_seconds"] + 30
    ):
        return None
    return float(
        mapping["slope"] * source_seconds + mapping["intercept_seconds"]
    ), mapping
