"""Concrete engines for the engine-neutral Standard Lipidomics use case."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from annotator.np_rt import DEFAULT_NP_RT_REFERENCE_PATH, load_np_rt_reference
from annotator.reference import load_reference
from standard_lipidomics.workflow import annotate_standard_batch
from standard_lipidomics.policy import (
    DEFAULT_MATLAB_POLICY_PATH,
    discover_matlab_fa_ranges,
    discover_matlab_policy_file,
    load_matlab_lipid_policy,
)
from application.standard_workflow import (
    LASER_STANDARD_ENGINE,
    MATLAB_STANDARD_ENGINE,
    StandardAnnotationRequest,
    StandardAnnotationResult,
    StandardEngine,
)
from extension_registry import ComponentRegistry
from matlab_annotation.configuration import (
    AutomaticAnchorSettings,
    DEFAULT_MATLAB_PIPELINE_REFERENCE,
)
from matlab_annotation.execution import run_matlab_annotation
from scientific_defaults import SCIENTIFIC_DEFAULTS
from errors import WorkflowConfigurationError


def _float_setting(value: object) -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise WorkflowConfigurationError(
            f"Expected a numeric standard-workflow setting, got {value!r}."
        ) from exc


def _int_setting(value: object) -> int:
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise WorkflowConfigurationError(
            f"Expected a whole-number standard-workflow setting, got {value!r}."
        ) from exc


def _int_pair_setting(
    value: object,
    *,
    name: str,
) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
    ):
        raise WorkflowConfigurationError(
            f"{name} must contain exactly two integers"
        )
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise WorkflowConfigurationError(
            f"{name} must contain exactly two integers"
        ) from exc


def _phase_setting(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise WorkflowConfigurationError("phases must be a list or tuple")
    return tuple(str(item) for item in value)


def build_automatic_anchor_settings(
    values: Mapping[str, object],
) -> AutomaticAnchorSettings:
    """Translate public workflow settings into the MATLAB adapter contract."""
    defaults = AutomaticAnchorSettings()
    return AutomaticAnchorSettings(
        anchor_selection_mode=str(values.get(
            "anchor_selection_mode", defaults.anchor_selection_mode,
        )),
        positional_isomer_workflow_mode=str(values.get(
            "positional_isomer_workflow_mode",
            defaults.positional_isomer_workflow_mode,
        )),
        ppm_tolerance=_float_setting(values.get(
            "anchor_ppm", defaults.ppm_tolerance,
        )),
        rt_tolerance_seconds=_float_setting(values.get(
            "anchor_rt_seconds", defaults.rt_tolerance_seconds,
        )),
        maximum_score=_float_setting(values.get(
            "anchor_maximum_score", defaults.maximum_score,
        )),
        minimum_score_margin=_float_setting(values.get(
            "anchor_score_margin", defaults.minimum_score_margin,
        )),
        minimum_anchors=_int_setting(values.get(
            "minimum_anchors", defaults.minimum_anchors,
        )),
        maximum_anchors=_int_setting(values.get(
            "maximum_anchors", defaults.maximum_anchors,
        )),
        minimum_carbon_span=_int_setting(values.get(
            "minimum_anchor_carbon_span", defaults.minimum_carbon_span,
        )),
        minimum_double_bond_span=_int_setting(values.get(
            "minimum_anchor_db_span", defaults.minimum_double_bond_span,
        )),
        positional_isomer_rt_tolerance_seconds=_float_setting(values.get(
            "positional_isomer_rt_seconds",
            defaults.positional_isomer_rt_tolerance_seconds,
        )),
        positional_isomer_gap_tolerance_seconds=_float_setting(values.get(
            "positional_isomer_gap_seconds",
            defaults.positional_isomer_gap_tolerance_seconds,
        )),
        positional_isomer_minimum_separation_seconds=_float_setting(values.get(
            "positional_isomer_minimum_separation_seconds",
            defaults.positional_isomer_minimum_separation_seconds,
        )),
        positional_isomer_score_margin=_float_setting(values.get(
            "positional_isomer_score_margin",
            defaults.positional_isomer_score_margin,
        )),
        positional_isomer_local_rt_adjustment_seconds=_float_setting(values.get(
            "positional_isomer_local_rt_adjustment_seconds",
            defaults.positional_isomer_local_rt_adjustment_seconds,
        )),
        positional_isomer_local_support_groups=_int_setting(values.get(
            "positional_isomer_local_support_groups",
            defaults.positional_isomer_local_support_groups,
        )),
        positional_isomer_local_composition_distance=_int_setting(values.get(
            "positional_isomer_local_composition_distance",
            defaults.positional_isomer_local_composition_distance,
        )),
        use_grid_coherence=bool(values.get(
            "use_grid_coherence", defaults.use_grid_coherence,
        )),
        grid_coherence_threshold=_float_setting(values.get(
            "grid_coherence_threshold",
            SCIENTIFIC_DEFAULTS.grid_coherence_threshold,
        )),
        use_post_regression_completion=bool(values.get(
            "use_post_regression_completion",
            defaults.use_post_regression_completion,
        )),
        post_regression_confidence_threshold=_float_setting(values.get(
            "post_regression_confidence_threshold",
            defaults.post_regression_confidence_threshold,
        )),
        post_regression_ambiguity_margin=_float_setting(values.get(
            "post_regression_ambiguity_margin",
            defaults.post_regression_ambiguity_margin,
        )),
        use_fitted_grid_completion=bool(values.get(
            "use_fitted_grid_completion", defaults.use_fitted_grid_completion,
        )),
        fitted_grid_feature_ambiguity_margin=_float_setting(values.get(
            "fitted_grid_feature_ambiguity_margin",
            defaults.fitted_grid_feature_ambiguity_margin,
        )),
    )


class MatlabXcmsEngine:
    name = MATLAB_STANDARD_ENGINE

    def annotate(self, request: StandardAnnotationRequest) -> StandardAnnotationResult:
        reference = request.matlab_reference_path or Path(
            DEFAULT_MATLAB_PIPELINE_REFERENCE
        )
        payload = run_matlab_annotation(
            request.project,
            reference_dir=reference,
            phases=_phase_setting(request.settings.get("phases", ("RP", "NP"))),
            only_accepted=bool(request.settings.get("only_accepted", True)),
            anchor_settings=build_automatic_anchor_settings(request.settings),
            composition_ranges=request.composition_ranges,
            progress=request.progress,
            mass_calibrations=cast(
                Mapping[str, object] | None,
                request.settings.get("mass_calibrations"),
            ),
        )
        return StandardAnnotationResult.from_mapping(self.name, payload)


class LaserReferenceEngine:
    name = LASER_STANDARD_ENGINE

    def annotate(self, request: StandardAnnotationRequest) -> StandardAnnotationResult:
        values = request.settings
        reference_path = request.reference_path
        if reference_path is None:
            raise WorkflowConfigurationError(
                "LASER reference engine requires reference_path"
            )
        discovered_carbon, discovered_db, discovered_source = (
            discover_matlab_fa_ranges(request.project)
        )
        carbon_range = _int_pair_setting(
            values.get("fa_carbon_range", discovered_carbon),
            name="fa_carbon_range",
        )
        db_range = _int_pair_setting(
            values.get("fa_db_range", discovered_db),
            name="fa_db_range",
        )
        range_source = values.get("range_source", discovered_source)
        if range_source is not None and not isinstance(range_source, (str, Path)):
            raise WorkflowConfigurationError(
                "range_source must be a path or string"
            )
        policy_path = request.matlab_policy_path
        if policy_path is None:
            policy_path = (
                discover_matlab_policy_file(request.project)
                or Path(DEFAULT_MATLAB_POLICY_PATH)
            )
        policy = (
            None
            if bool(values.get("disable_matlab_policy", False))
            else load_matlab_lipid_policy(
                policy_path,
                fa_carbon_range=carbon_range,
                fa_db_range=db_range,
                range_source=range_source,
            )
        )
        np_reference_path = (
            request.np_rt_reference_path or Path(DEFAULT_NP_RT_REFERENCE_PATH)
        )
        np_reference = (
            None
            if bool(values.get("disable_np_rt_reference", False))
            else load_np_rt_reference(np_reference_path)
        )
        payload = annotate_standard_batch(
            request.resolved_sources(),
            load_reference(reference_path),
            ppm_tolerance=_float_setting(values.get(
                "ppm", SCIENTIFIC_DEFAULTS.standard_ppm_tolerance,
            )),
            rp_rt_tolerance_min=_float_setting(values.get(
                "rp_rt_tolerance_min", SCIENTIFIC_DEFAULTS.rt_tolerance_min,
            )),
            align_rp_rt=bool(values.get("align_rp_rt", False)),
            adapt_class_rt=bool(values.get("adapt_class_rt", True)),
            matlab_policy=policy,
            composition_ranges=request.composition_ranges,
            np_rt_reference=np_reference,
            np_rt_tolerance_min=_float_setting(values.get(
                "np_rt_tolerance_min", 0.30,
            )),
            use_grid_coherence=bool(values.get("use_grid_coherence", True)),
            grid_coherence_threshold=_float_setting(values.get(
                "grid_coherence_threshold",
                SCIENTIFIC_DEFAULTS.grid_coherence_threshold,
            )),
        )
        return StandardAnnotationResult.from_mapping(self.name, payload)


standard_engine_registry: ComponentRegistry[StandardEngine] = ComponentRegistry({
    LASER_STANDARD_ENGINE: LaserReferenceEngine(),
    MATLAB_STANDARD_ENGINE: MatlabXcmsEngine(),
}, expected_type=StandardEngine, name_attribute="name")
