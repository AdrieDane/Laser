"""Smooth class-specific carbon/double-bond evidence inside hard limits."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


COMPOSITION_FEATURES = (
    "composition_distance", "odd_carbon", "unsaturation_fraction",
)


def fit_composition_priors(
    curated: pd.DataFrame, *, minimum_observations: int = 12,
) -> dict:
    """Fit broad ellipses from training-only, curated class compositions.

    Adding variances of four carbon units squared and one double bond squared
    prevents a narrow training distribution from acting as a hard whitelist.
    Classes with insufficient observations receive no learned distance penalty.
    """
    priors = {}
    for lipid_class, group in curated.groupby("lipid_class", sort=True):
        points = group[["carbon", "db"]].apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(points) < minimum_observations:
            continue
        covariance = np.cov(points.to_numpy().T) + np.diag([4.0, 1.0])
        priors[str(lipid_class)] = {
            "center": points.median().astype(float).tolist(),
            "inverse_covariance": np.linalg.inv(covariance).tolist(),
            "observations": int(len(points)),
        }
    return priors


def composition_features(
    candidates: pd.DataFrame, priors: Mapping,
) -> pd.DataFrame:
    """Return graded evidence, without rejecting unusual/odd-chain species."""
    result = pd.DataFrame(index=candidates.index)
    carbon = pd.to_numeric(candidates["carbon"], errors="coerce")
    double_bonds = pd.to_numeric(candidates["db"], errors="coerce")
    result["composition_distance"] = 0.0
    result["odd_carbon"] = carbon.mod(2)
    result["unsaturation_fraction"] = double_bonds / carbon.where(carbon.gt(0))
    for lipid_class, group in candidates.groupby("lipid_class", sort=False):
        prior = priors.get(str(lipid_class))
        if prior is None:
            continue
        points = np.column_stack([
            carbon.loc[group.index], double_bonds.loc[group.index],
        ])
        delta = points - np.asarray(prior["center"], dtype=float)
        squared_distance = np.einsum(
            "ij,jk,ik->i", delta,
            np.asarray(prior["inverse_covariance"], dtype=float), delta,
        )
        result.loc[group.index, "composition_distance"] = np.sqrt(
            np.maximum(squared_distance, 0.0)
        )
    return result.replace([np.inf, -np.inf], np.nan)


__all__ = [
    "COMPOSITION_FEATURES", "composition_features", "fit_composition_priors",
]
