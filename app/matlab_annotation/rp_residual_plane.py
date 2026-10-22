"""Unused research: bounded RP RT shape adjustment from automatic anchors.

Produces RT hypotheses only, NEVER annotations or chemical confidence scores.
The caller must exclude each evaluated project from the supplied frozen atlas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .retention_atlas import (
    _prepare_automatic_anchors,
    _alignment_from_prepared_anchors,
    _robust_fit,
    _predict_class,
    predict_retention_atlas,
)
from .standard_completion import _manual, _standards


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _inside_hull(points, query):
    points = sorted(set(map(tuple, np.asarray(points, float))))
    if len(points) < 3:
        return False
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return len(hull) >= 3 and all(
        _cross(a, b, query) >= -1e-9 for a, b in zip(hull, hull[1:] + hull[:1])
    )


def _plane_fit(points, residual, query):
    """Query has ALREADY been excluded; each anchor also predicts leave-one-out."""
    if len(points) < 12 or np.ptp(points[:, 0]) < 6 or np.ptp(points[:, 1]) < 2:
        return None, "insufficient_other_composition_span"
    if not _inside_hull(points, query):
        return None, "query_outside_other_anchor_hull"
    center = np.median(points, axis=0)
    scaled = (points - center) / np.array([10.0, 4.0])
    design = np.column_stack([np.ones(len(points)), scaled])
    if np.linalg.matrix_rank(design) < 3 or np.linalg.cond(design) > 100:
        return None, "ill_conditioned_other_anchor_plane"
    errors = []
    for i in range(len(points)):
        keep = np.arange(len(points)) != i
        if (
            np.linalg.matrix_rank(design[keep]) < 3
            or np.linalg.cond(design[keep]) > 100
        ):
            return None, "ill_conditioned_leave_one_anchor_out"
        coefficients = _robust_fit(design[keep], residual[keep])
        errors.append(float(design[i] @ coefficients - residual[i]))
    median, p90 = np.quantile(np.abs(errors), [0.5, 0.9])
    if median > 1.5 or p90 > 3:
        return None, "other_anchor_leave_one_out_disagreement"
    coefficients = _robust_fit(design, residual)
    if abs(coefficients[1] / 10) > 3 or abs(coefficients[2] / 4) > 5:
        return None, "residual_slope_bound_exceeded"
    query_design = np.r_[1.0, (query - center) / np.array([10.0, 4.0])]
    shift = float(query_design @ coefficients)
    if max(np.max(np.abs(design @ coefficients)), abs(shift)) > 60:
        return None, "residual_shift_bound_exceeded"
    return dict(
        coefficients=coefficients,
        shift=shift,
        loo_median=float(median),
        loo_p90=float(p90),
    ), "independent_bounded_RP_plane"


def predict_rp_residual_plane(targets, automatic, atlas, *, stream):
    """At least12 OTHER strong compositions; no RT extrapolation or self-votes.

    Frozen automatic class score>=.95 and abs mass error<1.5ppm. Repeated adducts
    vote once; standards/manual rows do not calibrate. Median/p90 leave-one-
    anchor-out RT errors<=1.5/3s. Residual C/DB slopes bounded3/5sec per unit,
    residual shift<=60s, target inside OTHER-anchor CDB convex hull. Final RP
    surface must retain later RT with carbon and earlier RT with unsaturation.
    Thresholds are fixed physical safeguards, NOT trained identification scores.
    """
    output = pd.DataFrame(index=targets.index)
    output["rp_plane_rt_seconds"] = np.nan
    output["rp_plane_other_anchors"] = 0
    output["rp_plane_loo_median_seconds"] = np.nan
    output["rp_plane_loo_p90_seconds"] = np.nan
    output["rp_plane_status"] = (
        "not_RP" if not stream.startswith("RP_") else "unavailable"
    )
    if not stream.startswith("RP_"):
        return output
    if not {
        "name",
        "class_selection_score",
        "class",
        "carbon",
        "db",
        "feat_rt",
        "ppm_error",
    }.issubset(automatic):
        return output
    source = automatic.copy()
    source["post_regression_confidence"] = pd.to_numeric(
        source.class_selection_score, errors="coerce"
    )
    source = source[
        source.post_regression_confidence.ge(0.95)
        & ~_manual(source)
        & ~_standards(source)
    ]
    source = source.sort_values(
        "post_regression_confidence", ascending=False, kind="stable"
    )
    source = source.drop(
        columns=[
            "atlas_rt_seconds",
            "atlas_training_mad_seconds",
            "atlas_training_cohorts",
            "atlas_in_domain",
        ],
        errors="ignore",
    )
    anchors = _prepare_automatic_anchors(source, atlas, stream)
    reference = predict_retention_atlas(targets.assign(stream=stream), atlas)
    canonical = targets["class"].map(canonical_lipid_class)
    for cls, indices in canonical.groupby(canonical, sort=False).groups.items():
        local = anchors[anchors._canonical_class.eq(cls)]
        model = atlas["models"].get(stream + "|" + cls)
        if len(local) < 12 or model is None:
            continue
        warp = _alignment_from_prepared_anchors(
            anchors[anchors._canonical_class.ne(cls)]
        )
        residual = local.feat_rt.to_numpy(float) * 60 - (
            local.atlas_rt_seconds.to_numpy(float) * warp["slope"]
            + warp["intercept_seconds"]
        )
        coordinates = local[["carbon", "db"]].to_numpy(float)
        cache = {}
        for index in indices:
            query = targets.loc[index, ["carbon", "db"]].to_numpy(float)
            expected = float(reference.loc[index, "atlas_rt_seconds"])
            if not np.isfinite(query).all() or not np.isfinite(expected):
                continue
            key = tuple(query)
            other = ~np.all(coordinates == query[None, :], axis=1)
            output.loc[index, "rp_plane_other_anchors"] = int(other.sum())
            if key not in cache:
                cache[key] = _plane_fit(coordinates[other], residual[other], query)
            fitted, status = cache[key]
            output.loc[index, "rp_plane_status"] = status
            if fitted is None:
                continue
            # Check total surface derivatives, not merely residual slope signs.
            check = np.vstack([coordinates[other], query[None, :]])
            c, d = check[:, 0], check[:, 1]
            dc = (
                _predict_class(pd.DataFrame(dict(carbon=c + 0.5, db=d)), model)
                - _predict_class(pd.DataFrame(dict(carbon=c - 0.5, db=d)), model)
            ) * warp["slope"] + fitted["coefficients"][1] / 10
            dd = (
                _predict_class(pd.DataFrame(dict(carbon=c, db=d + 0.5)), model)
                - _predict_class(pd.DataFrame(dict(carbon=c, db=d - 0.5)), model)
            ) * warp["slope"] + fitted["coefficients"][2] / 4
            if not ((dc > 0).all() and (dd < 0).all()):
                output.loc[index, "rp_plane_status"] = "non_monotone_RP_surface"
                continue
            predicted = (
                expected * warp["slope"] + warp["intercept_seconds"] + fitted["shift"]
            )
            if not np.isfinite(predicted) or predicted <= 0:
                continue
            output.loc[
                index,
                [
                    "rp_plane_rt_seconds",
                    "rp_plane_loo_median_seconds",
                    "rp_plane_loo_p90_seconds",
                ],
            ] = [predicted, fitted["loo_median"], fitted["loo_p90"]]
    return output
