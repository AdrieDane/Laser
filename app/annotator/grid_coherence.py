"""Experimental local grid-coherence scoring for lipid candidates.

The regular precursor m/z, RT, and CCS criteria judge every feature-reference
pair independently.  This module adds an optional second check: candidates at
nearby carbon/double-bond nodes in the same class and adduct should have similar
residuals.  A systematic dataset-wide RT or CCS shift therefore cancels, while
an isolated feature that merely falls inside the broad hard windows can be
flagged as a local grid outlier.

The filter is deliberately conservative:

* duplicate features assigned to the same lipid node never inflate support;
* at least two distinct neighbouring nodes are required;
* insufficiently supported edge/sparse nodes are unassessed and pass;
* missing CCS falls back to m/z + RT when CCS filtering is enabled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scientific_defaults import SCIENTIFIC_DEFAULTS


GRID_COHERENCE_COLUMNS = [
    "grid_coherence_score",
    "grid_coherence_neighbors",
    "grid_coherence_dimensions",
    "grid_coherence_assessed",
    "grid_coherence_outlier",
    "grid_coherence_pass",
]


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

    The score ranges from zero to one.  It is the grid-distance-weighted mean
    similarity between a candidate's normalized residual vector and the best
    representative at each nearby, distinct carbon/double-bond node.  Values
    below ``threshold`` are marked as outliers.

    Carbon differences are divided by two before calculating grid distance,
    reflecting the common two-carbon spacing of lipid homologues.  The residual
    dimensions are normalized by the same tolerances used for annotation.
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

    # One representative per node prevents split peaks or repeated candidates
    # for one species from making a grid look more strongly supported.
    scored["_origin_distance"] = np.sqrt(
        scored["_ppm_norm"].fillna(0.0) ** 2
        + scored["_rt_norm"].fillna(0.0) ** 2
        + scored["_ccs_norm"].fillna(0.0) ** 2
    )
    group_columns = ["_grid_class", "adduct"]
    for _, group in scored.groupby(group_columns, sort=False, dropna=False):
        valid_nodes = group[
            group["_carbon"].notna() & group["_db"].notna()
        ]
        if valid_nodes.empty:
            continue
        representatives = (
            valid_nodes.sort_values("_origin_distance", kind="stable")
            .drop_duplicates(["_carbon", "_db"], keep="first")
        )
        if len(representatives) <= min_neighbors:
            continue

        # Every candidate in the group is scored against the same small set of
        # node representatives, so the comparison is one dense array operation
        # rather than a per-candidate pandas filter/sort.
        candidates = group[group["_carbon"].notna() & group["_db"].notna()]
        if candidates.empty:
            continue

        residual_columns = ["_ppm_norm", "_rt_norm", "_ccs_norm"]
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

        # Residual differences per neighbour and dimension (candidates x
        # neighbours x 3). A dimension counts only where both sides are finite.
        difference = candidate_residuals[:, None, :] - node_residuals[order]
        dimension_valid = np.isfinite(difference)
        dimensions = dimension_valid.sum(axis=2)
        squared_distance = np.where(dimension_valid, difference, 0.0)
        squared_distance = (squared_distance ** 2).sum(axis=2)

        contributes = within_range & (dimensions >= 2)
        similarity = np.where(contributes, np.exp(-0.5 * squared_distance), 0.0)
        weight = np.where(
            contributes,
            np.exp(-0.5 * np.where(within_range, neighbour_distance, 0.0) ** 2),
            0.0,
        )

        neighbour_count = contributes.sum(axis=1)
        weight_total = weight.sum(axis=1)
        assessed = (neighbour_count >= min_neighbors) & (weight_total > 0)
        if not assessed.any():
            continue

        score = np.divide(
            (similarity * weight).sum(axis=1), weight_total,
            out=np.zeros(len(candidates)), where=assessed,
        )
        mean_dimensions = np.divide(
            (dimensions * weight).sum(axis=1), weight_total,
            out=np.zeros(len(candidates)), where=assessed,
        )

        index = candidates.index[assessed]
        scored.loc[index, "grid_coherence_score"] = score[assessed]
        scored.loc[index, "grid_coherence_neighbors"] = neighbour_count[assessed]
        scored.loc[index, "grid_coherence_dimensions"] = (
            np.round(mean_dimensions[assessed]).astype(int)
        )
        scored.loc[index, "grid_coherence_assessed"] = True
        scored.loc[index, "grid_coherence_outlier"] = score[assessed] < threshold
        scored.loc[index, "grid_coherence_pass"] = score[assessed] >= threshold

    return scored.drop(
        columns=[
            "_grid_class", "_carbon", "_db", "_ppm_norm", "_rt_norm",
            "_ccs_norm", "_origin_distance",
        ]
    )
