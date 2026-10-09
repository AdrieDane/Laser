"""End-to-end native scientific service for semi-targeted annotation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .annotation_runner import AnnotationRunResult, run_annotation
from .model_types import AnnotationModel
from .engine import resolve_annotation_siblings
from .model_dispatch import build_calibrated_annotation_model
from .project import LegacyProject
from .reference import ReferenceData
from .rtdb_model import ProjectFaCalibration, calibrate_project_fa_preprocessing
from .target_model import CompoundSetTargetTable


@dataclass
class AnnotationEngine:
    """Join project state, RTDB calibration and the native mapper.

    This is the fixture-free production boundary the GUI can call in the next
    phase.  Calibrations are lazy and cached independently for NP and RP.
    """

    project: LegacyProject
    reference: ReferenceData
    _calibrations: dict[str, ProjectFaCalibration] = field(
        default_factory=dict, init=False, repr=False
    )

    def calibration(self, phase: str) -> ProjectFaCalibration:
        if phase not in self.project.phases:
            raise ValueError(f"project has no phase {phase!r}")
        if phase not in self._calibrations:
            self._calibrations[phase] = calibrate_project_fa_preprocessing(
                self.project, self.reference, phase=phase
            )
        return self._calibrations[phase]

    def siblings(self, phase: str, compound: int) -> tuple[int, ...]:
        if phase not in self.project.phases:
            raise ValueError(f"project has no phase {phase!r}")
        active = tuple(resolve_annotation_siblings(
            int(compound),
            self.project.compounds.combined,
            self.project.compounds.priority[phase],
        ))
        if not active:
            raise ValueError(
                f"compound {compound} is not selectable in phase {phase}"
            )
        return active

    def run(
        self,
        *,
        phase: str,
        compound: int,
        model: AnnotationModel | None = None,
        targets: CompoundSetTargetTable | None = None,
        database_rt_seconds: np.ndarray | None = None,
    ) -> AnnotationRunResult:
        active = self.siblings(phase, compound)
        if model is None:
            model = build_calibrated_annotation_model(
                self.project,
                self.calibration(phase),
                phase=phase,
                compounds=active,
            )
        return run_annotation(
            self.project,
            self.reference,
            phase=phase,
            compounds=active,
            model=model,
            targets=targets,
            database_rt_seconds=database_rt_seconds,
        )

    @property
    def calibrated_phases(self) -> tuple[str, ...]:
        return tuple(self._calibrations)
