"""Grid coherence V2, copied from LASER-4D a188f72 (2026-11-04).

The regular precursor m/z, RT, and CCS criteria judge every feature-reference
pair independently.  This module adds an optional second check: a candidate
should fit the local cluster formed by other carbon/double-bond nodes of its
class and adduct. A systematic dataset-wide RT or CCS shift therefore cancels,
while an isolated feature inside the broad hard windows can be flagged.

The filter is deliberately conservative:

* duplicate features assigned to the same lipid node never inflate support;
* at least two distinct neighbouring nodes are required;
* insufficiently supported edge/sparse nodes are unassessed and pass;
* missing CCS falls back to m/z + RT when CCS filtering is enabled.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from scientific_defaults import SCIENTIFIC_DEFAULTS


GRID_COHERENCE_COLUMNS = [
    "grid_coherence_score",
    "grid_coherence_neighbors",
    "grid_coherence_dimensions",
    "grid_coherence_method",
    "grid_coherence_worst_axis",
    "grid_coherence_local_distance",
    "grid_coherence_assessed",
    "grid_coherence_outlier",
    "grid_coherence_pass",
]

# Annotation windows are eligibility limits, not estimates of within-cluster
# variation. These small native-unit floors prevent identical neighbours from
# amplifying measurement noise. The usual spread of this class/adduct supplies
# the more important, data-driven scale when it is larger.
MIN_LOCAL_SPREAD_NATIVE = (1.2, 3.0 / 60.0, 1.0)  # ppm, minutes, CCS percent
TYPICAL_SPREAD_MULTIPLIER = 2.5
GRID_COHERENCE_METHOD = "local_median_mad_v3"


def _grid_class(data: pd.DataFrame) -> pd.Series:
    if "grid_class" in data:
        return data["grid_class"].fillna("").astype(str)
    lipid_class = data.get(
        "class", pd.Series("", index=data.index, dtype=object)
    ).fillna("").astype(str)
    base = data.get(
        "base", pd.Series("", index=data.index, dtype=object)
    ).fillna("").astype(str)
    return pd.Series(
        [
            f"{lipid_class_value}-{base_value}"
            if base_value else lipid_class_value
            for lipid_class_value, base_value in zip(lipid_class, base)
        ],
        index=data.index,
    )


def _empty_scores(data: pd.DataFrame) -> pd.DataFrame:
    scored = data.copy()
    scored["grid_coherence_score"] = np.nan
    scored["grid_coherence_neighbors"] = 0
    scored["grid_coherence_dimensions"] = 0
    scored["grid_coherence_method"] = GRID_COHERENCE_METHOD
    scored["grid_coherence_worst_axis"] = ""
    scored["grid_coherence_local_distance"] = np.nan
    scored["grid_coherence_assessed"] = False
    scored["grid_coherence_outlier"] = False
    scored["grid_coherence_pass"] = True
    return scored


def score_grid_coherence(
    annotations: pd.DataFrame,
    ppm_tolerance: float,
    rt_tolerance: float,
    ccs_tolerance_percent: float,
    *,
    use_ccs: bool = True,
    threshold: float = SCIENTIFIC_DEFAULTS.grid_coherence_threshold,
    min_neighbors: int = 2,
    max_neighbors: int = 6,
    max_grid_distance: float = 2.5,
) -> pd.DataFrame:
    """Add a conservative local grid-coherence score to every candidate.

    The score ranges from zero to one. It measures the largest standardized
    deviation from the local median of nearby, distinct carbon/double-bond
    nodes. The scale reflects observed within-cluster spread, with a small
    native-unit floor; broad annotation windows do not inflate the score.
    Values below ``threshold`` are marked as outliers.

    Carbon differences are divided by two before calculating grid distance,
    reflecting the common two-carbon spacing of lipid homologues.  The residual
    dimensions are normalized for computation, but the observed spread and
    native-unit floor cancel that normalization in the final score.
    """
    if annotations.empty:
        return _empty_scores(annotations)
    if ppm_tolerance <= 0 or rt_tolerance <= 0:
        raise ValueError("m/z and RT tolerances must be positive")
    if use_ccs and ccs_tolerance_percent <= 0:
        raise ValueError("CCS tolerance must be positive when CCS is enabled")
    if not 0 <= threshold <= 1:
        raise ValueError("Grid-coherence threshold must be between 0 and 1")
    if min_neighbors < 1 or max_neighbors < min_neighbors:
        raise ValueError("Invalid grid-coherence neighbor limits")

    required = {
        "adduct", "carbon", "db", "ppm_error", "rt_error",
    }
    if not required.issubset(annotations.columns):
        return _empty_scores(annotations)

    scored = _empty_scores(annotations)
    scored["_grid_class"] = _grid_class(scored)
    scored["_carbon"] = pd.to_numeric(scored["carbon"], errors="coerce")
    scored["_db"] = pd.to_numeric(scored["db"], errors="coerce")
    scored["_ppm_norm"] = (
        pd.to_numeric(scored["ppm_error"], errors="coerce") / ppm_tolerance
    )
    scored["_rt_norm"] = (
        pd.to_numeric(scored["rt_error"], errors="coerce") / rt_tolerance
    )
    if use_ccs and "ccs_error_percent" in scored:
        scored["_ccs_norm"] = (
            pd.to_numeric(scored["ccs_error_percent"], errors="coerce")
            / ccs_tolerance_percent
        )
    else:
        scored["_ccs_norm"] = np.nan

    # A median per node prevents split peaks from inflating support. Choosing
    # the feature nearest the reference instead can favour a wrong candidate
    # when the whole observed class is shifted relative to that reference.
    group_columns = ["_grid_class", "adduct"]
    for _, group in scored.groupby(group_columns, sort=False, dropna=False):
        valid_nodes = group[
            group["_carbon"].notna() & group["_db"].notna()
        ]
        if valid_nodes.empty:
            continue
        residual_columns = ["_ppm_norm", "_rt_norm", "_ccs_norm"]
        representatives = (
            valid_nodes.groupby(["_carbon", "_db"], sort=False)[residual_columns]
            .median()
            .reset_index()
        )
        if len(representatives) <= min_neighbors:
            continue

        # Every candidate in the group is scored against the same small set of
        # node representatives, so the comparison is one dense array operation
        # rather than a per-candidate pandas filter/sort.
        candidates = group[group["_carbon"].notna() & group["_db"].notna()]
        if candidates.empty:
            continue

        candidate_carbon = candidates["_carbon"].to_numpy(float)
        candidate_db = candidates["_db"].to_numpy(float)
        candidate_residuals = candidates[residual_columns].to_numpy(float)
        node_carbon = representatives["_carbon"].to_numpy(float)
        node_db = representatives["_db"].to_numpy(float)
        node_residuals = representatives[residual_columns].to_numpy(float)

        grid_distance = np.sqrt(
            ((node_carbon[None, :] - candidate_carbon[:, None]) / 2.0) ** 2
            + (node_db[None, :] - candidate_db[:, None]) ** 2
        )
        # A candidate never supports itself, and distant nodes are dropped.
        # Sorting these to infinity reproduces the filter-then-sort-then-head
        # selection exactly, including its stable tie order.
        excluded = (
            (node_carbon[None, :] == candidate_carbon[:, None])
            & (node_db[None, :] == candidate_db[:, None])
        ) | (grid_distance > max_grid_distance)
        ranked_distance = np.where(excluded, np.inf, grid_distance)
        order = np.argsort(ranked_distance, axis=1, kind="stable")
        order = order[:, :max_neighbors]

        rows = np.arange(len(candidates))[:, None]
        neighbour_distance = ranked_distance[rows, order]
        within_range = np.isfinite(neighbour_distance)

        # Exclude the target node before estimating the observed local cluster.
        # Each axis has its own robust centre and spread, so a single deviant
        # RT or CCS value cannot disappear inside an average similarity.
        comparison = np.where(
            within_range[:, :, None], node_residuals[order], np.nan
        )
        dimension_valid = np.isfinite(
            candidate_residuals[:, None, :] - comparison
        )
        dimensions = dimension_valid.sum(axis=2)
        contributes = within_range & (dimensions >= 2)
        comparison = np.where(
            contributes[:, :, None]
            & np.isfinite(candidate_residuals[:, None, :]),
            comparison, np.nan,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            centre = np.nanmedian(comparison, axis=1)
            local_spread = 1.4826 * np.nanmedian(
                np.abs(comparison - centre[:, None, :]), axis=1
            )
        axis_support = np.isfinite(comparison).sum(axis=1)
        neighbour_count = contributes.sum(axis=1)
        adequately_supported = (
            (neighbour_count >= min_neighbors)
            & ((axis_support >= min_neighbors).sum(axis=1) >= 2)
        )
        has_support = adequately_supported[:, None] & (axis_support >= min_neighbors)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            typical_spread = np.nanmedian(
                np.where(has_support, local_spread, np.nan), axis=0
            )
        typical_spread = np.where(np.isfinite(typical_spread), typical_spread, 0.0)
        native_floor = np.asarray(MIN_LOCAL_SPREAD_NATIVE) / np.asarray((
            ppm_tolerance, rt_tolerance,
            ccs_tolerance_percent if use_ccs else 1.0,
        ))
        spread = np.maximum(
            local_spread,
            np.maximum(TYPICAL_SPREAD_MULTIPLIER * typical_spread, native_floor),
        )
        standardized = (candidate_residuals - centre) / spread
        usable_axes = np.isfinite(standardized) & (axis_support >= min_neighbors)
        standardized = np.where(usable_axes, standardized, 0.0)
        worst_dimension = np.max(np.abs(standardized), axis=1)
        usable_dimensions = usable_axes.sum(axis=1)
        assessed = (neighbour_count >= min_neighbors) & (usable_dimensions >= 2)
        if not assessed.any():
            continue

        score = np.exp(-0.5 * worst_dimension ** 2)
        mean_dimensions = np.divide(
            (dimensions * contributes).sum(axis=1), neighbour_count,
            out=np.zeros(len(candidates)), where=neighbour_count > 0,
        )

        index = candidates.index[assessed]
        scored.loc[index, "grid_coherence_score"] = score[assessed]
        scored.loc[index, "grid_coherence_neighbors"] = neighbour_count[assessed]
        scored.loc[index, "grid_coherence_dimensions"] = (
            np.round(mean_dimensions[assessed]).astype(int)
        )
        scored.loc[index, "grid_coherence_worst_axis"] = np.asarray(
            ("m/z", "RT", "CCS"), dtype=object
        )[np.argmax(np.abs(standardized[assessed]), axis=1)]
        scored.loc[index, "grid_coherence_local_distance"] = worst_dimension[assessed]
        scored.loc[index, "grid_coherence_assessed"] = True
        scored.loc[index, "grid_coherence_outlier"] = score[assessed] < threshold
        scored.loc[index, "grid_coherence_pass"] = score[assessed] >= threshold

    return scored.drop(
        columns=[
            "_grid_class", "_carbon", "_db", "_ppm_norm", "_rt_norm",
            "_ccs_norm",
        ]
    )
