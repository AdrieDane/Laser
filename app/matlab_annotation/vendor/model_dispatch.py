"""Dispatch ProjectData annotation definitions to verified native RT models."""

from __future__ import annotations

import numpy as np

from .model_types import AnnotationModel
from .compound_model import phase_composition
from .project import LegacyProject
from .rtdb_model import ProjectFaCalibration
from .rt_model import (
    build_empty_r_annotation_model,
    build_fa_annotation_model,
    build_isoprene_annotation_model,
)


class UnsupportedPhaseDefinition(ValueError):
    """The selected catalog definition cannot generate targets in this phase."""


def _r_group(project: LegacyProject, compound: int) -> str:
    value = np.asarray(
        project.compounds.legacy_fields["R_group"], dtype=object
    ).reshape(-1)[compound - 1]
    array = np.asarray(value, dtype=object).reshape(-1)
    return "" if array.size == 0 else str(array[0])


def build_calibrated_annotation_model(
    project: LegacyProject,
    calibration: ProjectFaCalibration,
    *,
    phase: str,
    compounds: tuple[int, ...] | list[int] | np.ndarray,
    integration_steps_per_unit: int = 1000,
) -> AnnotationModel:
    """Choose the MATLAB model branch for one resolved annotation group."""
    active = tuple(int(value) for value in np.asarray(compounds).reshape(-1))
    if not active:
        raise ValueError("at least one compound is required")
    for compound in active:
        model = phase_composition(
            project.compounds.composition_attr[compound - 1], phase
        )
        if model.bases[0] == 0:
            raise UnsupportedPhaseDefinition(
                f"compound {compound} has no charged adduct in {phase}"
            )
    groups = tuple(_r_group(project, compound) for compound in active)
    if len(set(groups)) != 1:
        raise ValueError(f"active siblings mix R-groups {groups!r}")
    r_group = groups[0]
    indices = np.asarray(active, dtype=np.int64) - 1
    if r_group in {"FA", "FAfull"}:
        beta = calibration.preprocessing_beta[:, indices]
        if len(active) == 1:
            beta = beta[:, 0]
        if not np.all(np.isfinite(beta)):
            raise ValueError("FA model has no finite project calibration")
        return build_fa_annotation_model(
            project,
            phase=phase,
            compounds=active,
            preprocessing_beta=beta,
            integration_steps_per_unit=integration_steps_per_unit,
        )
    if r_group == "isoprene":
        if len(active) != 1:
            raise ValueError("isoprene annotation groups must be singletons")
        beta = calibration.isoprene_preprocessing_beta[:, indices[0]]
        if not np.all(np.isfinite(beta)):
            raise ValueError("isoprene model has no finite project calibration")
        return build_isoprene_annotation_model(
            project,
            phase=phase,
            compound=active[0],
            preprocessing_beta=beta,
            integration_steps_per_unit=integration_steps_per_unit,
        )
    if r_group == "":
        return build_empty_r_annotation_model(
            project, phase=phase, compounds=active
        )
    raise UnsupportedPhaseDefinition(
        f"R-group {r_group!r} has no verified annotation model"
    )
