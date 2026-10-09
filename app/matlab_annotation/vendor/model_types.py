"""Shared immutable value types for the native MATLAB annotation model.

This module intentionally depends only on the spline value objects.  Keeping
the model contract here prevents the RT builders and annotation orchestrator
from importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spline import OneDimensionalSplineTerm, TwoDimensionalSplineTerm


@dataclass(frozen=True)
class AnnotationModel:
    """Scientific RT-model inputs prepared before candidate mapping."""

    ranges: np.ndarray
    thresholds_seconds: np.ndarray
    rt_preprocessing: np.ndarray
    term: OneDimensionalSplineTerm | TwoDimensionalSplineTerm | None
