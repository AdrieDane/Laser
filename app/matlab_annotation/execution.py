"""Execution API around mechanically preserved MATLAB-equivalent science."""

from matlab_annotation.engine import (
    CompoundOutcome,
    MatlabAnnotationError,
    run_legacy_matlab_annotation,
    run_matlab_annotation,
)

__all__ = [
    "CompoundOutcome",
    "MatlabAnnotationError",
    "run_legacy_matlab_annotation",
    "run_matlab_annotation",
]
