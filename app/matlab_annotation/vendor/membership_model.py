"""Resolve stored MATLAB inlier/outlier coordinates onto current viewer rows."""

from __future__ import annotations

import numpy as np

from .project import LegacyProject


def resolve_project_memberships(
    project: LegacyProject,
    phase: str,
    compounds: tuple[int, ...] | list[int] | np.ndarray,
    *,
    observed_mz: np.ndarray,
    observed_rt_seconds: np.ndarray,
    observed_viewer_rows: np.ndarray,
) -> np.ndarray:
    """Port ``w_tot`` construction and its 0.5 m/z/RT coordinate mapping."""
    mz = np.asarray(observed_mz, dtype=float).reshape(-1)
    rt = np.asarray(observed_rt_seconds, dtype=float).reshape(-1)
    viewers = np.asarray(observed_viewer_rows, dtype=np.int64).reshape(-1)
    if not (mz.shape == rt.shape == viewers.shape):
        raise ValueError("observed m/z, RT, and viewer rows must align")
    stored: list[tuple[int, int, float, float, float]] = []
    for sibling, compound in enumerate(
        np.asarray(compounds, dtype=np.int64).reshape(-1), start=1
    ):
        for row in project.compounds.inliers[int(compound) - 1]:
            if str(row[0]) != phase:
                continue
            raw_code = float(row[1])
            weight = 1.0 - 0.5 * (int(round(raw_code * 2)) % 2)
            stored.append((
                sibling, int(np.floor(raw_code)), float(row[2]), float(row[3]), weight
            ))
        for row in project.compounds.outliers[int(compound) - 1]:
            if str(row[0]) != phase:
                continue
            stored.append((
                sibling, int(np.floor(float(row[1]))), float(row[2]), float(row[3]), -1.0
            ))
    resolved: list[tuple[int, int, int, float]] = []
    for sibling, code, target_mz, target_rt, weight in stored:
        candidates = np.flatnonzero(
            (np.abs(mz - target_mz) <= 0.5)
            & (np.abs(rt - target_rt) <= 0.5)
        )
        if not len(candidates):
            continue
        distance = (
            np.abs(mz[candidates] - target_mz) * 1000.0
            + np.abs(rt[candidates] - target_rt)
        )
        order = np.lexsort((viewers[candidates], distance))
        observation = candidates[order[0]]
        resolved.append((sibling, code, int(viewers[observation]), weight))
    if not resolved:
        return np.empty((0, 4), dtype=float)
    result = np.asarray(resolved, dtype=float)
    order = np.lexsort(tuple(result[:, column] for column in reversed(range(4))))
    result = result[order]
    exact_duplicate = np.r_[False, np.all(np.diff(result, axis=0) == 0, axis=1)]
    result = result[~exact_duplicate]
    key_duplicate = np.r_[
        False, np.all(np.diff(result[:, :3], axis=0) == 0, axis=1)
    ]
    result = result[~key_duplicate]
    return result
