"""Unused numerical RT prototype: other-project composition residual references.

No files, annotations, candidate joins or acceptance. Exact class identities
(including ether/positional distinctions) remain separate. A reproducible
other-project deviation can correct a smooth polynomial's local shape bias.
"""

import hashlib

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .retention_atlas import (
    predict_retention_atlas,
    _affine_alignment,
    _prepare_automatic_anchors,
    _alignment_from_prepared_anchors,
)

FORMAT = "laser-empirical-composition-RT-v1"
REQUIRED = {"cohort", "stream", "class", "carbon", "db", "rt_seconds"}


def _verify_atlas_scope(atlas, training):
    expected = hashlib.sha256("\n".join(sorted(training)).encode()).hexdigest()
    if atlas.get("training_scope_sha256") != expected:
        raise ValueError("Parent atlas and empirical reference training scopes differ")


def fit_empirical_retention(observations, atlas, *, training_groups, excluded_groups):
    """Fit residual references ONLY from explicitly authorized other projects.

    One value per project/class/CDB, no repeated-adduct weight multiplication.
    Same bounded class-balanced affine normalization as the parent RT atlas.
    Exact compositions need>=3projects,MAD<=5s,absmedian<=30s. Shrink toward
    polynomial by n/(n+3). Fixed prospective settings,not selected on test RTs.
    """
    training, excluded = set(training_groups), set(excluded_groups)
    if not training or not excluded or training & excluded:
        raise ValueError("Explicit disjoint training/excluded projects required")
    _verify_atlas_scope(atlas, training)
    if not REQUIRED.issubset(observations):
        raise ValueError("Explicit grouped RT reference observations required")
    data = observations[sorted(REQUIRED)].copy()
    if set(data.cohort) - training:
        raise ValueError("Excluded or unauthorized project entered RT fitting")
    if set(data.cohort) != training:
        raise ValueError("Every reference project must contribute RT observations")
    for c in ("carbon", "db", "rt_seconds"):
        data[c] = pd.to_numeric(data[c], errors="coerce")
    if (
        not np.isfinite(data[["carbon", "db", "rt_seconds"]]).all().all()
        or data.carbon.le(0).any()
        or data.db.lt(0).any()
        or data.rt_seconds.le(0).any()
    ):
        raise ValueError("Finite positive RT/carbon and nonnegative DB required")
    data["class"] = data["class"].map(canonical_lipid_class)
    keys = ["cohort", "stream", "class", "carbon", "db"]
    data = data.groupby(keys, as_index=False).rt_seconds.median()
    data["expected"] = predict_retention_atlas(data, atlas).atlas_rt_seconds
    data["normalized"] = data.rt_seconds
    for _, group in data.groupby(["cohort", "stream"], sort=False):
        valid = group[group.expected.notna()]
        if (
            len(valid) < 20
            or valid["class"].nunique() < 3
            or np.ptp(valid.expected) < 100
        ):
            continue
        weights = 1 / valid.groupby("class")["class"].transform("size").to_numpy(float)
        slope, intercept = _affine_alignment(valid.expected, valid.rt_seconds, weights)
        data.loc[group.index, "normalized"] = (group.rt_seconds - intercept) / slope
    data["residual"] = data.normalized - data.expected
    models = {}
    for (stream, cls, c, d), group in data.dropna(subset=["residual"]).groupby(
        ["stream", "class", "carbon", "db"], sort=True
    ):
        n = group.cohort.nunique()
        center = float(group.residual.median())
        mad = float(np.median(np.abs(group.residual - center)))
        if n < 3 or mad > 5 or abs(center) > 30:
            continue
        key = f"{stream}|{cls}|{float(c):g}|{float(d):g}"
        models[key] = dict(
            offset_seconds=center * n / (n + 3),
            unshrunk_offset_seconds=center,
            training_projects=int(n),
            training_MAD_seconds=mad,
        )
    return dict(
        format=FORMAT,
        training_groups=sorted(training),
        excluded_groups=sorted(excluded),
        models=models,
        note="Numerical RT reference ONLY. Project-normalized,exact-class/composition residuals;no query observations,annotation decisions or analytical certainty.",
    )


def _predict_empirical_reference_scalar(targets, atlas, model, *, query_group):
    if model.get("format") != FORMAT:
        raise ValueError("Explicit empirical RT reference required")
    training, excluded = set(model["training_groups"]), set(model["excluded_groups"])
    if training & excluded or query_group in training or query_group not in excluded:
        raise ValueError("Query project must be explicitly excluded")
    _verify_atlas_scope(atlas, training)
    out = predict_retention_atlas(targets, atlas)
    out["empirical_offset_seconds"] = np.nan
    out["empirical_training_projects"] = np.nan
    out["empirical_training_MAD_seconds"] = np.nan
    for row in targets[["stream", "class", "carbon", "db"]].itertuples():
        # itertuples renames the Python keyword class; positional access is
        # intentional and contains NO row/feature identifiers as predictors.
        index, stream, cls, c, d = row
        if not np.isfinite([c, d]).all() or pd.isna(out.loc[index, "atlas_rt_seconds"]):
            continue
        key = f"{stream}|{canonical_lipid_class(cls)}|{float(c):g}|{float(d):g}"
        item = model["models"].get(key)
        if item is None:
            continue
        corrected = out.loc[index, "atlas_rt_seconds"] + item["offset_seconds"]
        if not np.isfinite(corrected) or corrected <= 0:
            continue
        out.loc[index, "atlas_rt_seconds"] = corrected
        out.loc[index, "empirical_offset_seconds"] = item["offset_seconds"]
        out.loc[index, "empirical_training_projects"] = item["training_projects"]
        out.loc[index, "empirical_training_MAD_seconds"] = item["training_MAD_seconds"]
    return out


def predict_empirical_reference(targets, atlas, model, *, query_group):
    """Equivalent vector lookup; no cross-query/model prediction cache.

    Keep the scalar path for unusual dtypes, preserving public input behaviour.
    Keys, float additions and missing flags remain identical. Caller outputs
    and reference weights are never mutated.
    """
    if (
        not isinstance(targets, pd.DataFrame)
        or targets.empty
        or not {"carbon", "db"}.issubset(targets)
    ):
        return _predict_empirical_reference_scalar(
            targets, atlas, model, query_group=query_group
        )
    plain = all(
        isinstance(targets[c].dtype, np.dtype)
        and (
            np.issubdtype(targets[c].dtype, np.integer)
            or np.issubdtype(targets[c].dtype, np.floating)
        )
        for c in ("carbon", "db")
    )
    if not plain:
        return _predict_empirical_reference_scalar(
            targets, atlas, model, query_group=query_group
        )
    if model.get("format") != FORMAT:
        raise ValueError("Explicit empirical RT reference required")
    training, excluded = set(model["training_groups"]), set(model["excluded_groups"])
    if training & excluded or query_group in training or query_group not in excluded:
        raise ValueError("Query project must be explicitly excluded")
    _verify_atlas_scope(atlas, training)
    out = predict_retention_atlas(targets, atlas)
    for field in (
        "empirical_offset_seconds",
        "empirical_training_projects",
        "empirical_training_MAD_seconds",
    ):
        out[field] = np.nan
    keys = (
        targets.stream.astype(str)
        + "|"
        + targets["class"].map(canonical_lipid_class)
        + "|"
        + targets.carbon.map(lambda v: f"{float(v):g}")
        + "|"
        + targets.db.map(lambda v: f"{float(v):g}")
    )
    reference = model["models"]
    offsets = keys.map(
        {k: item["offset_seconds"] for k, item in reference.items()}
    ).to_numpy(float)
    corrected = out.atlas_rt_seconds.to_numpy(float) + offsets
    valid = (
        np.isfinite(targets.carbon.to_numpy(float))
        & np.isfinite(targets.db.to_numpy(float))
        & out.atlas_rt_seconds.notna().to_numpy()
        & np.isfinite(corrected)
        & (corrected > 0)
    )
    out.loc[valid, "atlas_rt_seconds"] = corrected[valid]
    out.loc[valid, "empirical_offset_seconds"] = offsets[valid]
    for field, key in (
        ("empirical_training_projects", "training_projects"),
        ("empirical_training_MAD_seconds", "training_MAD_seconds"),
    ):
        values = keys.map({k: item[key] for k, item in reference.items()}).to_numpy(
            float
        )
        out.loc[valid, field] = values[valid]
    return out


def predict_aligned_empirical_retention(
    targets, automatic, atlas, model, *, stream, query_group
):
    """Same label-free other-class warp and leave-composition-out local shift.

    Recompute anchor predictions under the corrected reference, so a local
    polynomial correction is NOT simply added twice to an old aligned value.
    Standard rows and weak automatic proposals cannot become anchors.
    """
    for frame in (targets, automatic):
        if {
            "training_positive",
            "training_label_known",
            "same_manual_peak",
            "manual_feature_id",
        } & set(frame):
            raise ValueError("No supervised query flags allowed in RT prediction")
        for flag in ("manual_confirmed", "manually_confirmed", "manual_annotation"):
            if flag in frame and frame[flag].fillna(False).astype(bool).any():
                raise ValueError("No curated query anchors allowed")
    out = predict_empirical_reference(
        targets.assign(stream=stream), atlas, model, query_group=query_group
    )
    anchors = _prepare_automatic_anchors(automatic, atlas, stream)
    if len(anchors):
        anchors["atlas_rt_seconds"] = predict_empirical_reference(
            anchors.assign(stream=stream), atlas, model, query_group=query_group
        ).atlas_rt_seconds
    out["atlas_global_status"] = "unavailable"
    out["atlas_local_support_count"] = 0
    out["atlas_local_shift_seconds"] = 0.0
    canonical = targets["class"].map(canonical_lipid_class)
    for cls, indices in canonical.groupby(canonical, sort=False).groups.items():
        warp = _alignment_from_prepared_anchors(
            anchors[anchors._canonical_class.ne(cls)]
        )
        out.loc[indices, "atlas_rt_seconds"] = (
            out.loc[indices, "atlas_rt_seconds"] * warp["slope"]
            + warp["intercept_seconds"]
        )
        out.loc[indices, "atlas_global_status"] = warp["status"]
        local = anchors[anchors._canonical_class.eq(cls)]
        if len(local) < 4:
            continue
        residual = local.feat_rt.to_numpy(float) * 60 - (
            local.atlas_rt_seconds.to_numpy(float) * warp["slope"]
            + warp["intercept_seconds"]
        )
        finite = np.isfinite(residual)
        local, residual = local.loc[finite], residual[finite]
        if len(local) < 4:
            continue
        c = targets.loc[indices, "carbon"].to_numpy(float)
        d = targets.loc[indices, "db"].to_numpy(float)
        same = (c[:, None] == local.carbon.to_numpy(float)[None, :]) & (
            d[:, None] == local.db.to_numpy(float)[None, :]
        )
        values = np.where(same, np.nan, residual[None, :])
        count = np.isfinite(values).sum(axis=1)
        shift = np.nanmedian(values, axis=1)
        mad = np.nanmedian(np.abs(values - shift[:, None]), axis=1)
        valid = (
            (count >= 4)
            & (np.abs(shift) <= 60)
            & (mad <= 5)
            & np.isfinite(out.loc[indices, "atlas_rt_seconds"].to_numpy(float))
        )
        good = pd.Index(indices)[valid]
        out.loc[good, "atlas_rt_seconds"] += shift[valid]
        out.loc[good, "atlas_local_support_count"] = count[valid]
        out.loc[good, "atlas_local_shift_seconds"] = shift[valid]
    return out
