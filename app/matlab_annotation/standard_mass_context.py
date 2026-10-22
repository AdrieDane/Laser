"""Research-only mass evidence from explicitly declared automatic standards.

Does not change measured m/z, candidate admission or annotation acceptance.
No curated workbook, fitted manual RT or dataset/feature-ID predictor is used.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .standard_rt_library import ion_key, nominal_standard_name


def _name(value):
    value = re.sub(r"^IS:\s*", "", str(value).strip()).removesuffix(" (IS)")
    return nominal_standard_name(value)


def validated_mass_anchors(automatic, declarations):
    """Match names/ions/actual labelled masses to positive-quantity declarations.

    Multiple XCMS features/adducts of ONE neutral chemical mass vote once.
    Matching the annotation's reference mass to the actual declaration prevents
    a labelled spike from being silently treated as its unlabelled analogue.
    Observed anchors must still be<3ppm from the declared theoretical mass.
    """
    columns = ["stream", "neutral_key", "feature_id", "reference_mz", "ppm"]
    needed = {
        "stream",
        "name",
        "adduct",
        "ref_mz",
        "feat_mz",
        "feat_feature_id",
        "is_is",
    }
    required = {
        "stream",
        "name",
        "adduct",
        "ref_mz",
        "internal_standard_neutral_mass",
        "internal_standard_quantity",
    }
    if not needed.issubset(automatic) or not required.issubset(declarations):
        raise ValueError(
            "Mass context requires explicit automatic rows and declared chemical masses"
        )
    targets = declarations[
        pd.to_numeric(declarations.internal_standard_quantity, errors="coerce").gt(0)
    ].copy()
    targets["_name"] = targets.name.map(_name)
    targets["_ion"] = targets.adduct.map(ion_key)
    groups = {k: g for k, g in targets.groupby(["stream", "_name", "_ion"], sort=False)}
    rows = []
    for row in automatic[automatic.is_is.fillna(False).astype(bool)].to_dict("records"):
        part = groups.get((row["stream"], _name(row["name"]), ion_key(row["adduct"])))
        if part is None:
            continue
        try:
            ref, observed = float(row["ref_mz"]), float(row["feat_mz"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite([ref, observed]).all() or ref <= 0 or observed <= 0:
            continue
        exact = part[np.abs(part.ref_mz - ref) / part.ref_mz * 1e6 <= 0.05]
        neutral = exact.internal_standard_neutral_mass.dropna().round(6).unique()
        if len(neutral) != 1:
            continue
        theory = float(exact.ref_mz.median())
        ppm = (observed / theory - 1) * 1e6
        if not np.isfinite(ppm) or abs(ppm) >= 3:
            continue
        rows.append(
            dict(
                stream=row["stream"],
                neutral_key=f"{neutral[0]:.6f}",
                feature_id=str(row["feat_feature_id"]),
                reference_mz=theory,
                ppm=ppm,
            )
        )
    frame = pd.DataFrame(rows, columns=columns).drop_duplicates()
    if len(frame):
        # One measured feature must not become two supposedly independent ions.
        conflicts = frame.groupby(["stream", "feature_id"]).neutral_key.nunique()
        bad = set(conflicts[conflicts.gt(1)].index)
        frame = frame.loc[
            [
                k not in bad
                for k in frame[["stream", "feature_id"]].itertuples(
                    index=False, name=None
                )
            ]
        ]
    return frame.reset_index(drop=True)


def _summary(anchors):
    residual = anchors.groupby("neutral_key").ppm.median()
    count = len(residual)
    center = float(residual.median()) if count else np.nan
    mad = float((residual - center).abs().median()) if count else np.nan
    p90 = float((residual - center).abs().quantile(0.9)) if count else np.nan
    enabled = (
        count >= 6
        and np.isfinite([center, mad, p90]).all()
        and abs(center) <= 2
        and mad <= 0.6
        and p90 <= 1.5
    )
    return dict(
        count=count,
        center=center if enabled else np.nan,
        mad=mad,
        p90=p90,
        enabled=bool(enabled),
    )


def standard_mass_evidence(pool, anchors):
    """Eight descriptors, preserving all old evidence/candidate coordinates.

    Each query excludes its own original feature AND all neutral standard
    groups whose admitted adduct mass is within3ppm of the query target. This
    prevents standard aliases from calibrating themselves. Unavailable mass
    centers remain NaN, not a fabricated0ppm calibration. Raw ppm remains an
    independent existing predictor; this function never widens a window.
    """
    if not {"stream", "ref_mz", "ppm_error", "feat_feature_id"}.issubset(pool):
        raise ValueError("Mass evidence requires original candidate coordinates")
    required = {"stream", "neutral_key", "feature_id", "reference_mz", "ppm"}
    if not required.issubset(anchors):
        raise ValueError("Mass evidence requires validated declared-standard anchors")
    output = pd.DataFrame(index=pool.index)
    descriptors = {
        "q_standard_mass_center_ppm": np.nan,
        "q_standard_mass_MAD_ppm": np.nan,
        "q_standard_mass_p90_ppm": np.nan,
        "q_standard_mass_anchor_count": 0.0,
        "q_standard_mass_enabled": 0.0,
        "q_centered_mass_error_ppm": np.nan,
        "q_absolute_centered_mass_error_ppm": np.nan,
        "q_squared_centered_mass_error_ppm": np.nan,
    }
    for col, value in descriptors.items():
        output[col] = value
    for stream, part in pool.groupby("stream", sort=False):
        local = anchors[anchors.stream.eq(stream)]
        cache = {}
        refs = local.reference_mz.to_numpy(float)
        neutral = local.neutral_key.to_numpy(str)
        ids = local.feature_id.to_numpy(str)
        for index, ref, error, fid in part[
            ["ref_mz", "ppm_error", "feat_feature_id"]
        ].itertuples(index=True, name=None):
            ref, error = float(ref), float(error)
            matching = ids == str(fid)
            if np.isfinite(ref) and ref > 0:
                matching |= np.abs(refs - ref) / refs * 1e6 < 3
            excluded = frozenset(neutral[matching])
            if excluded not in cache:
                cache[excluded] = _summary(local[~local.neutral_key.isin(excluded)])
            summary = cache[excluded]
            center = summary["center"]
            centered = (
                ((1 + error * 1e-6) / (1 + center * 1e-6) - 1) * 1e6
                if summary["enabled"] and np.isfinite(error)
                else np.nan
            )
            output.loc[index, list(descriptors)] = [
                center,
                summary["mad"],
                summary["p90"],
                summary["count"],
                float(summary["enabled"]),
                centered,
                abs(centered),
                centered**2,
            ]
    return output


def summarize_mass_anchors(anchors):
    return {str(s): _summary(g) for s, g in anchors.groupby("stream", sort=True)}
