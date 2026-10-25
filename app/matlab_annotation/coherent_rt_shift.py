"""Unused numerical prototype: a shared RT offset from unlabeled mass hits.

No references are fitted/read here, no annotations/IDs are emitted, and no
measurements/defaults change. A future adapter MUST supply query-excluded RT
templates, phase-specific chemistry families and independently protected spikes.
Identifiers only limit repeated votes; their text never predicts an offset.
"""

import numpy as np
import pandas as pd

FIELDS = {
    "feature_id",
    "composition_id",
    "series_id",
    "carbon",
    "db",
    "mass_error_ppm",
    "rt_residual_seconds",
    "intensity",
}


def estimate_coherent_rt_shift(
    matches,
    *,
    maximum_shift_seconds=90.0,
    grid_step_seconds=0.5,
    bandwidth_seconds=2.0,
    anchor_mass_tolerance_ppm=1.5,
    minimum_anchors=6,
    minimum_carbon_counts=3,
    minimum_doublebond_counts=2,
    minimum_series_links=3,
    minimum_peak_ratio=1.25,
):
    """Numerical shift or explicit UNKNOWN, never a class/feature assignment.

    Positive residual means measuredRT-templateRT, in seconds. Each composition
    contributes its strongest edge; each physical feature's total hypothesis
    weight is <=1. Duplicate references/adducts cannot invent extra votes.
    Competing modes or ambiguous peak hypotheses withhold the shift.
    Returned support counts are unlabeled features, NOT independent projects.
    """
    if set(matches) != FIELDS:
        raise ValueError(
            "Only the explicit label-free physical match schema is allowed"
        )
    if not all(
        np.isfinite(v) and v > 0
        for v in (
            maximum_shift_seconds,
            grid_step_seconds,
            bandwidth_seconds,
            anchor_mass_tolerance_ppm,
        )
    ):
        raise ValueError("Finite positive numerical search parameters required")
    for n in (
        minimum_anchors,
        minimum_carbon_counts,
        minimum_doublebond_counts,
        minimum_series_links,
    ):
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise ValueError("Explicit adequate anchor/series support required")
    if not np.isfinite(minimum_peak_ratio) or minimum_peak_ratio <= 1:
        raise ValueError("A distinct competing-mode margin is required")
    if (
        grid_step_seconds > bandwidth_seconds
        or maximum_shift_seconds / grid_step_seconds > 4000
    ):
        raise ValueError("Bounded resolved numerical search required")
    result = dict(
        supported=False,
        shift_seconds=None,
        reason="no_eligible_physical_matches",
        anchors=0,
        features=0,
        compositions=0,
        carbon_counts=0,
        doublebond_counts=0,
        series_links=0,
        peak_ratio=None,
        residual_MAD_seconds=None,
        note="Unlabeled query RT-coherence ONLY,not identity/FDR/biological replication. Reference exclusion,spike/phase/family/domain guards are caller obligations;no candidates/annotations here.",
    )
    frame = matches.copy()
    numeric = ["carbon", "db", "mass_error_ppm", "rt_residual_seconds", "intensity"]
    for c in numeric:
        frame[c] = pd.to_numeric(frame[c], errors="raise")
    valid = np.isfinite(frame[numeric].to_numpy(float)).all(axis=1)
    valid &= (
        frame.feature_id.notna()
        & frame.composition_id.notna()
        & frame.series_id.notna()
    )
    valid &= frame.intensity.gt(0) & frame.carbon.gt(0) & frame.db.ge(0)
    valid &= frame.carbon.eq(frame.carbon.round()) & frame.db.eq(frame.db.round())
    valid &= frame.mass_error_ppm.abs().lt(anchor_mass_tolerance_ppm)
    valid &= frame.rt_residual_seconds.abs().le(maximum_shift_seconds)
    frame = frame.loc[valid].copy()
    if frame.empty:
        return result
    for c in ("feature_id", "composition_id", "series_id"):
        frame[c] = frame[c].astype(str)
    if (
        frame.groupby("composition_id")[["series_id", "carbon", "db"]]
        .nunique()
        .gt(1)
        .any()
        .any()
    ):
        raise ValueError(
            "One composition must have consistent series/carbon/DB metadata"
        )
    # Adduct/reference aliases of the same chemical composition get ONE vote.
    frame["composition_id"] = pd.MultiIndex.from_frame(
        frame[["series_id", "carbon", "db"]]
    ).factorize(sort=True)[0]
    if frame.groupby("feature_id").intensity.nunique().gt(1).any():
        return dict(result, reason="inconsistent_physical_feature_intensity")
    duplicate_values = frame.groupby(["feature_id", "composition_id"])[numeric].nunique(
        dropna=False
    )
    if duplicate_values.gt(1).any().any():
        return dict(result, reason="conflicting_duplicate_physical_hypotheses")
    frame = frame.drop_duplicates(
        ["feature_id", "composition_id"], keep="first"
    ).reset_index(drop=True)
    if (
        frame.feature_id.nunique() < minimum_anchors
        or frame.composition_id.nunique() < minimum_anchors
    ):
        return dict(result, reason="insufficient_distinct_physical_support")
    groups, unique = pd.factorize(frame.composition_id, sort=True)
    hypotheses = (
        frame.groupby("feature_id").composition_id.transform("nunique").to_numpy(float)
    )
    mass_weight = np.exp(
        -0.5
        * (frame.mass_error_ppm.to_numpy(float) / (anchor_mass_tolerance_ppm / 2)) ** 2
    )
    abundance = np.minimum(
        1.0,
        np.sqrt(
            frame.intensity.to_numpy(float)
            / float(frame.drop_duplicates("feature_id").intensity.quantile(0.9))
        ),
    )
    weights = mass_weight * abundance / hypotheses
    residuals = frame.rt_residual_seconds.to_numpy(float)
    grid = np.arange(
        -maximum_shift_seconds,
        maximum_shift_seconds + grid_step_seconds / 2,
        grid_step_seconds,
    )
    evidence = weights[:, None] * np.exp(
        -0.5 * ((residuals[:, None] - grid[None, :]) / bandwidth_seconds) ** 2
    )
    per_composition = np.zeros((len(unique), len(grid)))
    np.maximum.at(per_composition, groups, evidence)
    density = per_composition.sum(axis=0)
    tied = np.flatnonzero(np.isclose(density, density.max(), rtol=1e-12, atol=0))
    best = min(tied, key=lambda i: (abs(grid[i]), grid[i]))
    rivals = density[np.abs(grid - grid[best]) > 3 * bandwidth_seconds]
    second = float(rivals.max()) if len(rivals) else 0.0
    ratio = float(density[best] / second) if second > 0 else float("inf")
    result["peak_ratio"] = ratio if np.isfinite(ratio) else None
    if ratio < minimum_peak_ratio:
        return dict(result, reason="competing_RT_shift_modes")
    frame["_contribution"] = evidence[:, best]
    frame["_physical_weight"] = weights
    chosen = []
    for _, group in frame.groupby("feature_id", sort=False):
        order = group.sort_values(
            ["_contribution", "composition_id"], ascending=[False, True], kind="stable"
        )
        top = order.iloc[0]
        if (
            top._contribution < 0.15
            or abs(float(top.rt_residual_seconds) - float(grid[best]))
            > 2 * bandwidth_seconds
        ):
            continue
        if len(order) > 1 and float(
            top._contribution - order.iloc[1]._contribution
        ) < 0.1 * float(top._contribution):
            continue
        chosen.append(order.index[0])
    unique_anchors = []
    for _, group in frame.loc[chosen].groupby("composition_id", sort=True):
        order = group.sort_values(
            ["_contribution", "rt_residual_seconds"],
            ascending=[False, True],
            kind="stable",
        )
        top = order.iloc[0]
        # Do not break two plausible, separated original peaks by row/ID order.
        rivals = order.iloc[1:]
        ambiguous = (
            rivals._physical_weight.ge(0.9 * float(top._physical_weight))
            & (rivals.rt_residual_seconds - float(grid[best]))
            .abs()
            .le(2 * bandwidth_seconds)
            & (rivals.rt_residual_seconds - float(top.rt_residual_seconds))
            .abs()
            .gt(bandwidth_seconds / 2)
        )
        if not ambiguous.any():
            unique_anchors.append(order.index[0])
    anchors = frame.loc[unique_anchors]
    result.update(
        anchors=len(anchors),
        features=anchors.feature_id.nunique(),
        compositions=anchors.composition_id.nunique(),
        carbon_counts=anchors.carbon.nunique(),
        doublebond_counts=anchors.db.nunique(),
    )
    links = 0
    for _, group in anchors.groupby("series_id", sort=False):
        coordinates = set(zip(group.carbon.astype(int), group.db.astype(int)))
        links += sum(
            ((c + 2, d) in coordinates) + ((c, d + 1) in coordinates)
            for c, d in coordinates
        )
    result["series_links"] = int(links)
    if (
        len(anchors) < minimum_anchors
        or result["carbon_counts"] < minimum_carbon_counts
        or result["doublebond_counts"] < minimum_doublebond_counts
        or links < minimum_series_links
    ):
        return dict(result, reason="insufficient_unambiguous_coherent_series")
    order = np.argsort(anchors.rt_residual_seconds.to_numpy(float), kind="stable")
    values = anchors.rt_residual_seconds.to_numpy(float)[order]
    anchor_weights = anchors._contribution.to_numpy(float)[order]
    offset = float(
        values[np.searchsorted(np.cumsum(anchor_weights), anchor_weights.sum() / 2)]
    )
    result.update(
        supported=True,
        shift_seconds=offset,
        reason="unlabeled_coherent_RT_shift",
        residual_MAD_seconds=float(np.median(np.abs(values - offset))),
        carbon_min=float(anchors.carbon.min()),
        carbon_max=float(anchors.carbon.max()),
        db_min=float(anchors.db.min()),
        db_max=float(anchors.db.max()),
    )
    return result
