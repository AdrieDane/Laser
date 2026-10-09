"""Canonical scientific defaults shared by every LASER entry point.

The GUI, headless workflows, CLI templates, reports, and annotation engines
must not carry independent fallback values.  This dependency-free module is
the single source for those defaults.  It deliberately returns fresh mappings
so callers can customize a run without mutating process-wide state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScientificDefaults:
    """Versioned default policy for LASER annotation and evidence gates."""

    policy_version: int = 2

    # Candidate generation and orthogonal evidence.
    four_d_ppm_tolerance: float = 5.0
    standard_ppm_tolerance: float = 4.0
    processing_qc_ppm_tolerance: float = 4.0
    rt_tolerance_min: float = 0.5
    ccs_tolerance_percent: float = 5.0

    # LASER-owned post-MZmine selection gates.
    detection_min_percent: float = 50.0
    grid_coherence_threshold: float = 0.45
    minimum_mzmine_shape_score: float = 0.60
    maximum_boundary_apex_ratio: float = 0.075

    # Standard-lipidomics untargeted follow-up. These gates are applied only
    # after accepted annotations have been removed from the XCMS feature set.
    untargeted_p_value_threshold: float = 0.001
    untargeted_min_abs_log2_fold_change: float = 1.0
    untargeted_min_detection_percent: float = 50.0
    untargeted_cluster_min_neighbors: int = 3
    untargeted_cluster_min_dominance: float = 0.60
    untargeted_ch2_mass_tolerance_da: float = 0.01
    untargeted_ch2_rt_tolerance_min: float = 0.50
    untargeted_opposite_polarity_rt_tolerance_min: float = 0.20
    untargeted_isotope_rt_tolerance_min: float = 0.05

    # Evidence trace loading windows.
    trace_ppm_tolerance: float = 7.0
    trace_rt_tolerance_min: float = 1.0
    trace_ccs_tolerance_percent: float = 10.0

    # Internal-standard calibration candidate windows.
    calibration_ppm_tolerance: float = 5.0
    calibration_rt_tolerance_min: float = 0.5
    calibration_ccs_tolerance_percent: float = 5.0

    def __post_init__(self) -> None:
        positive = (
            "four_d_ppm_tolerance",
            "standard_ppm_tolerance",
            "processing_qc_ppm_tolerance",
            "rt_tolerance_min",
            "ccs_tolerance_percent",
            "trace_ppm_tolerance",
            "trace_rt_tolerance_min",
            "trace_ccs_tolerance_percent",
            "calibration_ppm_tolerance",
            "calibration_rt_tolerance_min",
            "calibration_ccs_tolerance_percent",
            "untargeted_opposite_polarity_rt_tolerance_min",
            "untargeted_isotope_rt_tolerance_min",
            "untargeted_ch2_mass_tolerance_da",
            "untargeted_ch2_rt_tolerance_min",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        fractions = (
            "grid_coherence_threshold",
            "minimum_mzmine_shape_score",
            "maximum_boundary_apex_ratio",
            "untargeted_p_value_threshold",
            "untargeted_cluster_min_dominance",
        )
        for name in fractions:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one.")
        if not 0.0 < float(self.detection_min_percent) <= 100.0:
            raise ValueError(
                "detection_min_percent must be greater than zero and at most 100."
            )
        if not 0.0 < float(self.untargeted_min_detection_percent) <= 100.0:
            raise ValueError(
                "untargeted_min_detection_percent must be greater than zero "
                "and at most 100."
            )
        if float(self.untargeted_min_abs_log2_fold_change) < 0.0:
            raise ValueError(
                "untargeted_min_abs_log2_fold_change cannot be negative."
            )
        if int(self.untargeted_cluster_min_neighbors) < 1:
            raise ValueError(
                "untargeted_cluster_min_neighbors must be at least one."
            )

    def four_d_annotation_block(self) -> dict[str, object]:
        """Return a new mutable 4D annotation configuration mapping."""
        return {
            "engine": "reference",
            "ppm": self.four_d_ppm_tolerance,
            "use_rt": True,
            "rt_tolerance_min": self.rt_tolerance_min,
            "align_rt": False,
            "use_ccs": True,
            "ccs_tolerance_percent": self.ccs_tolerance_percent,
            "use_detection_filter": True,
            "detection_scope_any_group": True,
            "detection_min_percent": self.detection_min_percent,
            "use_grid_coherence": True,
            "grid_coherence_threshold": self.grid_coherence_threshold,
            "use_mzmine_shape_filter": True,
            "minimum_mzmine_shape_score": self.minimum_mzmine_shape_score,
            "maximum_boundary_apex_ratio": self.maximum_boundary_apex_ratio,
            "composition_ranges": {
                "enabled": True,
                "use_group_ranges": True,
            },
            "trace_ppm": self.trace_ppm_tolerance,
            "trace_rt_tolerance_min": self.trace_rt_tolerance_min,
            "trace_ccs_tolerance_percent": self.trace_ccs_tolerance_percent,
            "trace_require_msms": False,
        }

    def calibration_block(self) -> dict[str, object]:
        """Return a new mutable calibration configuration mapping."""
        return {
            "detection_ppm": self.calibration_ppm_tolerance,
            "detection_rt_min": self.calibration_rt_tolerance_min,
            "detection_ccs_percent": self.calibration_ccs_tolerance_percent,
            "use_raw_ccs_for_detection": False,
            "include_agilent_tune_mix": False,
            "tune_mix_rt_max": 3.0,
            "ccs": {"mode": "auto", "on_failure": "error"},
            "rt": {"mode": "none", "on_failure": "disable"},
            "mass": {"mode": "none", "on_failure": "disable"},
        }


SCIENTIFIC_DEFAULTS = ScientificDefaults()
