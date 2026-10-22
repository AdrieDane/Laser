"""Research-only, leave-composition-out RT surfaces from automatic anchors.

No manual annotation or held-out RT is read. This produces an optional RT
proposal, not accepted annotations or chemical identification probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class


def _fit_local_plane(carbon, db, rt):
    center = np.asarray([np.median(carbon), np.median(db)], float)
    design = np.column_stack(
        [np.ones(len(carbon)), (carbon - center[0]) / 2, db - center[1]]
    )
    keep = np.ones(len(rt), bool)
    for _ in range(3):
        x, y = design[keep], rt[keep]
        if len(y) < 8 or np.linalg.matrix_rank(x) < 3:
            return None
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = rt - design @ coefficients
        scale = max(
            0.5, 1.4826 * np.median(np.abs(residual[keep] - np.median(residual[keep])))
        )
        updated = np.abs(residual) <= max(4.0, 3 * scale)
        if np.array_equal(updated, keep):
            break
        keep = updated
    x, y = design[keep], rt[keep]
    if len(y) < 8 or np.linalg.matrix_rank(x) < 3:
        return None
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ coefficients
    leverage = np.sum(x * (x @ np.linalg.pinv(x.T @ x)), axis=1)
    if np.max(leverage) >= 0.9:
        return None
    loo = np.abs(residual / np.maximum(1 - leverage, 1e-6))
    mad = float(np.median(np.abs(residual)))
    if np.median(loo) > 3 or np.quantile(loo, 0.9) > 6 or mad > 2.5:
        return None
    return dict(
        center=center,
        coefficients=coefficients,
        support=int(keep.sum()),
        analytic_loo_median_seconds=float(np.median(loo)),
        analytic_loo_p90_seconds=float(np.quantile(loo, 0.9)),
        training_median_absolute_seconds=mad,
        minimum_rt=float(y.min()),
        maximum_rt=float(y.max()),
    )


def predict_local_anchor_surfaces(
    targets, automatic, *, minimum_anchors=8, neighbors=12
):
    """Fit a local C:DB RT plane using OTHER strong automatic compositions.

    At least eight distinct compositions are required. Repeated adducts do not
    multiply support; the target composition is excluded completely. Fixed
    analytical leave-one-out checks apply to the final trimmed support set,
    not independent validation data, so these diagnostics are not probability
    calibration. Unsupported/sparse/unstable fits remain NaN, never zero.
    """
    if minimum_anchors < 8 or neighbors < minimum_anchors:
        raise ValueError("Local RT surfaces need >=8 distinct anchor compositions")
    if not {"class", "carbon", "db"}.issubset(targets):
        raise ValueError("RT targets require class/C/DB")
    if not targets.index.is_unique:
        raise ValueError("RT target indices must be unique")
    output = pd.DataFrame(index=targets.index)
    fields = [
        "local_surface_rt_seconds",
        "local_surface_analytic_LOO_seconds",
        "local_surface_training_MAD_seconds",
        "local_surface_nearest_distance",
    ]
    for c in fields:
        output[c] = np.nan
    output["local_surface_support"] = 0
    required = {
        "class",
        "carbon",
        "db",
        "feat_rt",
        "ppm_error",
        "post_regression_confidence",
    }
    if automatic.empty or not required.issubset(automatic):
        return output
    anchors = automatic.copy()
    standard = (
        anchors.get("is_is", pd.Series(False, index=anchors.index))
        .fillna(False)
        .astype(bool)
    )
    if "name" in anchors:
        standard |= anchors.name.astype(str).str.endswith(
            " (IS)"
        ) | anchors.name.astype(str).str.startswith("IS: ")
    valid = (
        ~standard
        & pd.to_numeric(anchors.post_regression_confidence, errors="coerce").ge(0.9)
        & pd.to_numeric(anchors.ppm_error, errors="coerce").abs().lt(1.5)
    )
    anchors = anchors.loc[valid].copy()
    anchors["canonical_class"] = anchors["class"].map(canonical_lipid_class)
    for c in ("carbon", "db", "feat_rt"):
        anchors[c] = pd.to_numeric(anchors[c], errors="coerce")
    anchors = anchors[
        np.isfinite(anchors[["carbon", "db", "feat_rt"]]).all(axis=1)
        & anchors.feat_rt.gt(0)
    ]
    if "feat_intensity" in anchors:
        anchors = anchors.sort_values("feat_intensity", ascending=False, kind="stable")
    anchors = anchors.drop_duplicates(["canonical_class", "carbon", "db"]).reset_index(
        drop=True
    )
    canonical = targets["class"].map(canonical_lipid_class)
    for cls, indices in canonical.groupby(canonical, sort=False).groups.items():
        own = anchors[anchors.canonical_class.eq(cls)]
        if len(own) < minimum_anchors:
            continue
        c, d = own.carbon.to_numpy(float), own.db.to_numpy(float)
        rt = own.feat_rt.to_numpy(float) * 60
        cache = {}
        unique = targets.loc[indices, ["carbon", "db"]].drop_duplicates()
        for row in unique.itertuples():
            tc, td = float(row.carbon), float(row.db)
            if not np.isfinite([tc, td]).all():
                continue
            distance = np.hypot((c - tc) / 2, d - td)
            distance[(c == tc) & (d == td)] = np.inf
            selected = np.argsort(distance, kind="stable")[:neighbors]
            selected = selected[distance[selected] <= 8]
            if len(selected) < minimum_anchors or distance[selected[0]] > 2:
                continue
            if (
                not c[selected].min() - 2 <= tc <= c[selected].max() + 2
                or not d[selected].min() - 1 <= td <= d[selected].max() + 1
            ):
                continue
            key = tuple(sorted(selected))
            if key not in cache:
                cache[key] = _fit_local_plane(c[list(key)], d[list(key)], rt[list(key)])
            fit = cache[key]
            if fit is None or fit["support"] < minimum_anchors:
                continue
            prediction = float(
                np.asarray([1.0, (tc - fit["center"][0]) / 2, td - fit["center"][1]])
                @ fit["coefficients"]
            )
            if (
                prediction <= 0
                or not fit["minimum_rt"] - 10 <= prediction <= fit["maximum_rt"] + 10
            ):
                continue
            matching = pd.Index(indices)[
                targets.loc[indices, "carbon"].eq(tc)
                & targets.loc[indices, "db"].eq(td)
            ]
            output.loc[matching, "local_surface_rt_seconds"] = prediction
            output.loc[matching, "local_surface_support"] = fit["support"]
            output.loc[matching, "local_surface_analytic_LOO_seconds"] = fit[
                "analytic_loo_median_seconds"
            ]
            output.loc[matching, "local_surface_training_MAD_seconds"] = fit[
                "training_median_absolute_seconds"
            ]
            output.loc[matching, "local_surface_nearest_distance"] = distance[
                selected[0]
            ]
    return output
