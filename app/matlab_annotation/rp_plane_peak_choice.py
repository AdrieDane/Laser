"""Research-only original-peak replacement using independent RP RT planes.

Identity/adduct counts cannot grow. No model score or old feature-quality
evidence is falsely transferred to a newly selected original feature.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .standard_completion import _manual, _standards

_IDENTITY = {
    "stream",
    "name",
    "class",
    "base",
    "adduct",
    "carbon",
    "db",
    "ref_mz",
    "matlab_code",
    "matlab_compound",
    "is_is",
    "isotopologue",
    "chemical_target_generated",
    "chemical_target_origin",
}


def repeak_with_rp_plane(
    annotations, features, predictions, *, stream, declared_standard_masses=()
):
    """Old peak>=7s away; unique RAW<1.5ppm/<=3s alternative, no borrowing.

    Requires passed independent plane with>=12 OTHER anchors and median/p90
    leave-one-anchor-out errors<=1.5/3s. Occupied/manual/standard/spike features
    protected. Raw uniqueness BEFORE occupancy/spike filtering; conflicting
    new proposals ALL withheld, no iterative cascades. Replaced feature-level
    scores/quality fields are cleared, not inherited. Fixed identity multisets.
    """
    if stream not in {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}:
        raise ValueError("Unknown measurement stream")
    if not stream.startswith("RP_"):
        return annotations.copy(), pd.DataFrame(), pd.DataFrame()
    if (
        not annotations.index.equals(predictions.index)
        or not annotations.index.is_unique
    ):
        raise ValueError(
            "Independent predictions must align with unique annotation rows"
        )
    needed = {
        "rp_plane_rt_seconds",
        "rp_plane_status",
        "rp_plane_other_anchors",
        "rp_plane_loo_median_seconds",
        "rp_plane_loo_p90_seconds",
    }
    if not needed.issubset(predictions):
        raise ValueError("Explicit independent RP plane certificate required")
    keys = ["name", "adduct", "is_is"]
    protected = _manual(annotations) | _standards(annotations)
    if (annotations.duplicated(keys, keep=False) & ~protected).any():
        raise ValueError("Unprotected accepted identities must be unique")
    if (
        annotations.feat_feature_id.astype(str).duplicated().any()
        or features.feature_id.astype(str).duplicated().any()
    ):
        raise ValueError("Original features must be unique")
    mz = pd.to_numeric(features.mz, errors="coerce").to_numpy(float)
    rt = pd.to_numeric(features.rt, errors="coerce").to_numpy(float) * 60
    intensity = pd.to_numeric(features.intensity, errors="coerce").to_numpy(float)
    fid = features.feature_id.astype(str).to_numpy()
    valid = (
        np.isfinite(mz)
        & (mz > 0)
        & np.isfinite(rt)
        & np.isfinite(intensity)
        & (intensity > 0)
    )
    order = np.flatnonzero(valid)
    order = order[np.argsort(mz[order], kind="stable")]
    occupied = set(annotations.feat_feature_id.astype(str))
    spikes = np.asarray(list(declared_standard_masses), float)
    spikes = spikes[np.isfinite(spikes) & (spikes > 0)]
    proposed = []
    for index, row in annotations.iterrows():
        if protected.loc[index]:
            continue
        p = predictions.loc[index]
        expected = float(p.rp_plane_rt_seconds)
        certificate = [
            expected,
            float(p.rp_plane_other_anchors),
            float(p.rp_plane_loo_median_seconds),
            float(p.rp_plane_loo_p90_seconds),
        ]
        if (
            not np.isfinite(certificate).all()
            or expected <= 0
            or min(certificate[2:]) < 0
        ):
            continue
        if p.rp_plane_status != "independent_bounded_RP_plane" or not np.isfinite(
            expected
        ):
            continue
        if (
            p.rp_plane_other_anchors < 12
            or p.rp_plane_loo_median_seconds > 1.5
            or p.rp_plane_loo_p90_seconds > 3
        ):
            continue
        old_rt, mass = float(row.feat_rt) * 60, float(row.ref_mz)
        if (
            not np.isfinite([old_rt, mass]).all()
            or mass <= 0
            or abs(old_rt - expected) < 7
        ):
            continue
        lo, hi = np.searchsorted(mz[order], [mass * (1 - 1.5e-6), mass * (1 + 1.5e-6)])
        hits = order[lo:hi]
        hits = hits[
            (np.abs(mz[hits] / mass - 1) * 1e6 < 1.5)
            & (np.abs(rt[hits] - expected) <= 3)
        ]
        if len(hits) != 1:
            continue
        pos = int(hits[0])
        if fid[pos] in occupied:
            continue
        if len(spikes) and (
            (np.abs(spikes - mass) / spikes < 3e-6).any()
            or (np.abs(spikes - mz[pos]) / spikes < 3e-6).any()
        ):
            continue
        proposed.append((index, pos, expected))
    claims = Counter(fid[pos] for _, pos, _ in proposed)
    # Object storage accommodates NaN for obsolete bool/string quality fields
    # without altering values on any untouched or protected original row.
    selected = annotations.astype(object).copy()
    audit, new_candidates = [], []
    for index, pos, expected in proposed:
        if claims[fid[pos]] > 1:
            audit.append(
                dict(
                    action="withhold_conflicting_RP_plane_proposals",
                    feature_id=fid[pos],
                )
            )
            continue
        old = annotations.loc[index]
        replacement = {c: (old[c] if c in _IDENTITY else np.nan) for c in annotations}
        replacement.update(
            feat_feature_id=fid[pos],
            feat_mz=mz[pos],
            feat_rt=rt[pos] / 60,
            feat_intensity=intensity[pos],
            ref_rt=expected / 60,
            rt_expected=expected / 60,
            rt_error=(rt[pos] - expected) / 60,
            ppm_error=(mz[pos] / float(old.ref_mz) - 1) * 1e6,
            previous_feature_id=str(old.feat_feature_id),
            previous_ref_rt=old.get("ref_rt", np.nan),
            previous_class_selection_score=old.get("class_selection_score", np.nan),
            rp_plane_old_RT_error_seconds=float(old.feat_rt) * 60 - expected,
            rp_plane_new_RT_error_seconds=rt[pos] - expected,
            automatic_accepted=True,
            automatic_candidate=True,
            auto_annotation_status="bounded_RP_RT_original_peak_reselected",
            annotation_evidence="fixed accepted identity; >=12 OTHER automatic RT anchors; unique1.5ppm/3s original peak; old peak>=7s away",
        )
        replacement.update(predictions.loc[index].to_dict())
        for c, v in replacement.items():
            if c not in selected:
                selected[c] = pd.Series(None, index=selected.index, dtype=object)
            selected.loc[index, c] = v
        new_candidates.append(replacement)
        audit.append(
            dict(
                action="replace_original_peak_from_independent_RP_plane",
                name=old["name"],
                adduct=old.adduct,
                previous_feature_id=str(old.feat_feature_id),
                feature_id=fid[pos],
                old_RT_error_seconds=float(old.feat_rt) * 60 - expected,
                new_RT_error_seconds=rt[pos] - expected,
            )
        )
    assert len(selected) == len(annotations)
    assert Counter(selected[keys].itertuples(index=False, name=None)) == Counter(
        annotations[keys].itertuples(index=False, name=None)
    )
    assert not selected.feat_feature_id.astype(str).duplicated().any()
    changed = {i for i, pos, _ in proposed if claims[fid[pos]] == 1}
    unchanged = annotations.index[~annotations.index.isin(changed)]
    pd.testing.assert_frame_equal(
        annotations.loc[unchanged],
        selected.loc[unchanged, annotations.columns],
        check_dtype=False,
    )
    return selected, pd.DataFrame(new_candidates), pd.DataFrame(audit)
