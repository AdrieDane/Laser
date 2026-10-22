"""Portable class RT priors learned offline, with label-free run alignment.

The fitter is a development API: its observations must exclude the cohort being
evaluated. Annotation needs only the frozen numerical model and automatic
anchors. Neither feature IDs nor cohort names enter prediction features. This
module is not enabled by the production annotation workflow.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class


FORMAT = "laser-class-retention-atlas-v1"
STREAMS = frozenset({"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"})
REQUIRED = {"cohort", "stream", "class", "carbon", "db", "rt_seconds"}


def _design(carbon, db, center, degree):
    c = (np.asarray(carbon, dtype=float) - center[0]) / 10.0
    d = (np.asarray(db, dtype=float) - center[1]) / 4.0
    columns = [np.ones(len(c))]
    if degree >= 1:
        columns.extend([c, d])
    if degree >= 2:
        columns.extend([c * c, d * d, c * d])
    return np.column_stack(columns)


def _robust_fit(design, response, weights=None):
    response = np.asarray(response, dtype=float)
    initial = np.ones(len(response)) if weights is None else np.asarray(weights)
    current = initial.copy()
    ridge = np.eye(design.shape[1]) * 1e-4
    ridge[0, 0] = 0.0
    coefficients = np.zeros(design.shape[1])
    for _ in range(15):
        root = np.sqrt(current)
        x, y = design * root[:, None], response * root
        updated = np.linalg.lstsq(x.T @ x + ridge, x.T @ y, rcond=None)[0]
        residual = response - design @ updated
        scale = max(1.0, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        current = initial * np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-9))
        if np.max(np.abs(updated - coefficients)) < 1e-7:
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def _fit_class(group):
    center = [float(group.carbon.median()), float(group.db.median())]
    c, d = group.carbon.to_numpy(float), group.db.to_numpy(float)
    degree = 2 if len(group[["carbon", "db"]].drop_duplicates()) >= 8 else 1
    while degree and np.linalg.matrix_rank(_design(c, d, center, degree)) < (6 if degree == 2 else 3):
        degree -= 1
    weights = 1.0 / group.groupby("cohort").cohort.transform("size").to_numpy(float)
    design = _design(c, d, center, degree)
    coefficients = _robust_fit(design, group.aligned_rt_seconds.to_numpy(float), weights)
    residual = group.aligned_rt_seconds.to_numpy(float) - design @ coefficients
    return {
        "center": center,
        "degree": degree,
        "coefficients": coefficients.tolist(),
        "training_rows": len(group),
        "training_cohorts": int(group.cohort.nunique()),
        "training_compositions": len(group[["carbon", "db"]].drop_duplicates()),
        "median_absolute_training_residual_seconds": float(np.median(np.abs(residual))),
        "carbon_min": float(c.min()), "carbon_max": float(c.max()),
        "db_min": float(d.min()), "db_max": float(d.max()),
    }


def _predict_class(group, model):
    return _design(group.carbon, group.db, model["center"], model["degree"]) @ np.asarray(model["coefficients"])


def _affine_alignment(predicted, observed, weights=None):
    """Bound a run-level RT warp; coefficients act on seconds, not minutes."""
    predicted, observed = np.asarray(predicted, float), np.asarray(observed, float)
    # Center and scale to avoid a poorly conditioned intercept at long RTs.
    center = float(np.median(predicted))
    scale = max(1.0, float(np.ptp(predicted)))
    design = np.column_stack([np.ones(len(predicted)), (predicted - center) / scale])
    coefficients = _robust_fit(design, observed, weights)
    slope = float(np.clip(coefficients[1] / scale, 0.8, 1.2))
    intercept = float(np.clip(np.median(observed - slope * predicted), -90.0, 90.0))
    return slope, intercept


def fit_retention_atlas(observations: pd.DataFrame, *, minimum_cohorts: int = 3) -> dict:
    """Fit class/composition RT surfaces using TRAINING cohorts only.

    Related acquisitions must already share a cohort key. Repeated adducts or
    rows do not multiply a composition's influence. Training-cohort RT warps
    normalize acquisition offsets; only class coefficients/counts are exported.
    This API does not split data or choose an evaluation threshold for callers.
    """
    if not REQUIRED.issubset(observations):
        raise ValueError(f"RT observations missing columns: {sorted(REQUIRED - set(observations))}")
    if isinstance(minimum_cohorts, bool) or minimum_cohorts < 2:
        raise ValueError("At least two independent training cohorts are required")
    data = observations[list(sorted(REQUIRED))].copy()
    data["class"] = data["class"].map(canonical_lipid_class)
    for column in ("carbon", "db", "rt_seconds"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = np.isfinite(data[["carbon", "db", "rt_seconds"]]).all(axis=1)
    valid &= data.carbon.gt(0) & data.db.ge(0) & data.rt_seconds.gt(0)
    if not valid.all() or not data.stream.isin(STREAMS).all():
        raise ValueError("Invalid RT training values or measurement stream")
    if data.cohort.isna().any() or data.cohort.astype(str).eq("").any():
        raise ValueError("Training observations require explicit cohort groups")
    data["cohort"] = data.cohort.astype(str)
    data = data.groupby(["cohort", "stream", "class", "carbon", "db"], as_index=False).rt_seconds.median()
    data["aligned_rt_seconds"] = data.rt_seconds
    models = {}
    for _ in range(3):
        models = {
            stream + "|" + cls: _fit_class(group)
            for (stream, cls), group in data.groupby(["stream", "class"], sort=True)
            if group.cohort.nunique() >= minimum_cohorts
        }
        prediction = pd.Series(np.nan, index=data.index, dtype=float)
        for (stream, cls), group in data.groupby(["stream", "class"], sort=False):
            model = models.get(stream + "|" + cls)
            if model:
                prediction.loc[group.index] = _predict_class(group, model)
        for (_cohort, _stream), group in data.groupby(["cohort", "stream"], sort=False):
            usable = group[prediction.loc[group.index].notna()]
            if len(usable) < 20 or usable["class"].nunique() < 3:
                continue
            predicted = prediction.loc[usable.index].to_numpy(float)
            if np.ptp(predicted) < 100:
                continue
            weights = 1.0 / usable.groupby("class")["class"].transform("size").to_numpy(float)
            slope, intercept = _affine_alignment(predicted, usable.rt_seconds.to_numpy(float), weights)
            data.loc[group.index, "aligned_rt_seconds"] = (group.rt_seconds - intercept) / slope
    # Refit to the last aligned responses, not the preceding iteration.
    models = {
        stream + "|" + cls: _fit_class(group)
        for (stream, cls), group in data.groupby(["stream", "class"], sort=True)
        if group.cohort.nunique() >= minimum_cohorts
    }
    scope = "\n".join(sorted(data.cohort.unique()))
    return {
        "format": FORMAT,
        "rt_unit": "seconds",
        "training_cohorts": int(data.cohort.nunique()),
        "training_scope_sha256": hashlib.sha256(scope.encode()).hexdigest(),
        "minimum_cohorts": minimum_cohorts,
        "models": models,
        "note": "Training residuals are not held-out accuracy. RT reference only, not a lipid identification or analytical FDR estimate.",
    }


def predict_retention_atlas(targets: pd.DataFrame, atlas: Mapping) -> pd.DataFrame:
    """Predict seconds for known classes near the training composition domain.

    Unknown classes and distant compositions remain unavailable, not zero.
    Range padding is for RT interpolation only; user composition filters still
    belong at the theoretical-target boundary and are not overridden here.
    """
    if atlas.get("format") != FORMAT or atlas.get("rt_unit") != "seconds":
        raise ValueError("Unsupported retention-atlas format or RT unit")
    if not {"stream", "class", "carbon", "db"}.issubset(targets):
        raise ValueError("RT targets require stream, class, carbon and db")
    if not targets.index.is_unique:
        raise ValueError("RT target index must be unique")
    data = targets[["stream", "class", "carbon", "db"]].copy()
    data["class"] = data["class"].map(canonical_lipid_class)
    output = pd.DataFrame(index=targets.index)
    output["atlas_rt_seconds"] = np.nan
    output["atlas_training_mad_seconds"] = np.nan
    output["atlas_training_cohorts"] = 0
    output["atlas_in_domain"] = False
    for (stream, cls), group in data.groupby(["stream", "class"], sort=False):
        model = atlas["models"].get(stream + "|" + cls)
        if not model:
            continue
        c, d = pd.to_numeric(group.carbon, errors="coerce"), pd.to_numeric(group.db, errors="coerce")
        valid = np.isfinite(c) & np.isfinite(d)
        valid &= c.between(model["carbon_min"] - 2, model["carbon_max"] + 2)
        valid &= d.between(max(0, model["db_min"] - 1), model["db_max"] + 1)
        usable = group.loc[valid].assign(carbon=c[valid], db=d[valid])
        predicted = _predict_class(usable, model)
        finite = np.isfinite(predicted) & (predicted > 0)
        ids = usable.index[finite]
        output.loc[ids, "atlas_rt_seconds"] = predicted[finite]
        output.loc[ids, "atlas_training_mad_seconds"] = model["median_absolute_training_residual_seconds"]
        output.loc[ids, "atlas_training_cohorts"] = model["training_cohorts"]
        output.loc[ids, "atlas_in_domain"] = True
    return output


def _prepare_automatic_anchors(automatic_annotations, atlas, stream):
    required = {"class", "carbon", "db", "feat_rt", "ppm_error", "post_regression_confidence"}
    if automatic_annotations.empty or not required.issubset(automatic_annotations):
        return pd.DataFrame(columns=["_canonical_class", "carbon", "db", "feat_rt", "atlas_rt_seconds"])
    anchors = automatic_annotations.copy().assign(stream=stream)
    anchors["_canonical_class"] = anchors["class"].map(canonical_lipid_class)
    confidence = pd.to_numeric(anchors.post_regression_confidence, errors="coerce")
    mass = pd.to_numeric(anchors.ppm_error, errors="coerce")
    standard = anchors.get("is_is", pd.Series(False, index=anchors.index)).fillna(False).astype(bool)
    anchors = anchors[confidence.ge(0.9) & mass.abs().lt(1.5) & ~standard]
    anchors = anchors.drop_duplicates(["_canonical_class", "carbon", "db"])
    atlas_rt = predict_retention_atlas(anchors, atlas)
    anchors = anchors.join(atlas_rt).dropna(subset=["atlas_rt_seconds", "feat_rt"])
    return anchors


def _alignment_from_prepared_anchors(anchors):
    identity = dict(slope=1.0, intercept_seconds=0.0, anchor_count=0, anchor_classes=0,
                    status="insufficient_automatic_anchors")
    if "feat_intensity" in anchors:
        anchors = anchors.sort_values("feat_intensity", ascending=False, kind="stable")
    anchors = anchors.groupby("_canonical_class", sort=False).head(8)
    n, classes = len(anchors), anchors._canonical_class.nunique()
    identity.update(anchor_count=n, anchor_classes=int(classes))
    if n < 20 or classes < 3 or np.ptp(anchors.atlas_rt_seconds) < 100:
        return identity
    predicted = anchors.atlas_rt_seconds.to_numpy(float)
    observed = pd.to_numeric(anchors.feat_rt, errors="coerce").to_numpy(float) * 60
    weights = 1.0 / anchors.groupby("_canonical_class")._canonical_class.transform("size").to_numpy(float)
    slope, intercept = _affine_alignment(predicted, observed, weights)
    before = float(np.median(np.abs(observed - predicted)))
    after = float(np.median(np.abs(observed - (predicted * slope + intercept))))
    if after >= before * 0.95:
        return {**identity, "status": "identity_warp_no_reliable_improvement",
                "median_absolute_before_seconds": before,
                "median_absolute_after_seconds": before}
    return dict(slope=slope, intercept_seconds=intercept, anchor_count=n,
                anchor_classes=int(classes), status="automatic_anchor_alignment",
                median_absolute_before_seconds=before, median_absolute_after_seconds=after)


def align_retention_atlas(
    automatic_annotations: pd.DataFrame, atlas: Mapping, *, stream: str,
    exclude_class: str | None = None,
) -> dict:
    """Align with strong automatic anchors, never with held-out manual RTs.

    Excluding the evaluated class avoids a class calibrating its own candidates.
    Unknown confidence does not count as strong evidence. Missing/weak evidence
    returns an explicitly unaligned identity warp rather than a fabricated fit.
    """
    anchors = _prepare_automatic_anchors(automatic_annotations, atlas, stream)
    if exclude_class is not None:
        anchors = anchors[anchors._canonical_class.ne(canonical_lipid_class(exclude_class))]
    return _alignment_from_prepared_anchors(anchors)


def predict_aligned_retention_atlas(
    targets: pd.DataFrame, automatic_annotations: pd.DataFrame, atlas: Mapping,
    *, stream: str,
) -> pd.DataFrame:
    """Apply other-class run alignment and leave-composition-out local anchors.

    A class can have a reproducible acquisition offset not explained by a
    single whole-run warp. At least four OTHER distinct compositions must agree
    before applying such a local offset. A target never supplies its own local
    support, repeated adducts do not multiply support, and manual RTs are neither
    required nor consulted. This is alignment evidence, not an identification.
    """
    prediction = predict_retention_atlas(targets.assign(stream=stream), atlas)
    prediction["atlas_global_status"] = "unavailable"
    prediction["atlas_local_support_count"] = 0
    prediction["atlas_local_shift_seconds"] = 0.0
    canonical = targets["class"].map(canonical_lipid_class)
    anchors = _prepare_automatic_anchors(automatic_annotations, atlas, stream)
    for cls, indices in canonical.groupby(canonical, sort=False).groups.items():
        warp = _alignment_from_prepared_anchors(anchors[anchors._canonical_class.ne(cls)])
        prediction.loc[indices, "atlas_rt_seconds"] = (
            prediction.loc[indices, "atlas_rt_seconds"] * warp["slope"] + warp["intercept_seconds"]
        )
        prediction.loc[indices, "atlas_global_status"] = warp["status"]
        local = anchors[anchors._canonical_class.eq(cls)]
        if len(local) < 4:
            continue
        residual = local.feat_rt.to_numpy(float) * 60 - (
            local.atlas_rt_seconds.to_numpy(float) * warp["slope"] + warp["intercept_seconds"]
        )
        finite = np.isfinite(residual)
        local, residual = local.loc[finite], residual[finite]
        if len(local) < 4:
            continue
        c = pd.to_numeric(targets.loc[indices, "carbon"], errors="coerce").to_numpy(float)
        d = pd.to_numeric(targets.loc[indices, "db"], errors="coerce").to_numpy(float)
        same = ((c[:, None] == local.carbon.to_numpy(float)[None, :]) &
                (d[:, None] == local.db.to_numpy(float)[None, :]))
        values = np.where(same, np.nan, residual[None, :])
        count = np.isfinite(values).sum(axis=1)
        shift = np.nanmedian(values, axis=1)
        mad = np.nanmedian(np.abs(values - shift[:, None]), axis=1)
        valid = ((count >= 4) & (np.abs(shift) <= 60) & (mad <= 5) &
                 np.isfinite(prediction.loc[indices, "atlas_rt_seconds"].to_numpy(float)))
        good = pd.Index(indices)[valid]
        prediction.loc[good, "atlas_rt_seconds"] += shift[valid]
        prediction.loc[good, "atlas_local_support_count"] = count[valid]
        prediction.loc[good, "atlas_local_shift_seconds"] = shift[valid]
    return prediction
