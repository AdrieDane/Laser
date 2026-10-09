"""Separate stage selection and XCMS mass calibration for standard lipidomics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from annotator.mass_calibrants import (
    build_mass_reference_panel,
    detect_mass_calibration_candidates,
)
from annotator.matching import (
    NEGATIVE_CCS_INTERNAL_STANDARDS,
    POSITIVE_CCS_INTERNAL_STANDARDS,
)
from annotator.reference import load_reference
from presentation.mass_calibration_ui import (
    MassCalibrationViewInput,
    render_mass_calibration,
)
from resource_paths import REFERENCE_DATA_DIRECTORY
from scientific_defaults import SCIENTIFIC_DEFAULTS
from standard_lipidomics.mass_calibration import (
    freeze_mass_calibration,
    write_calibrated_xcms_files,
)
from standard_lipidomics.workflow import (
    discover_standard_streams,
    load_standard_feature_list,
)


STANDARD_PIPELINE_PAGE_KEY = "standard_pipeline_page"
STANDARD_PIPELINE_SELECTION_KEY = "standard_pipeline_stage_selection"
STANDARD_ANNOTATION_PROJECT_KEY = "standard_annotation_project_override"

_INPUT_KEY = "standard_calibration_input_directory"
_OUTPUT_KEY = "standard_calibration_output_directory"
_STREAMS_KEY = "standard_calibration_streams"
_INDEX_KEY = "standard_calibration_index"
_MODELS_KEY = "standard_calibration_models"
_FINISHED_KEY = "standard_calibration_finished"
_REFERENCE = REFERENCE_DATA_DIRECTORY / "4d_lipid_reference.csv"


@dataclass(frozen=True)
class StandardPipelineStageSelection:
    """Independent, composable standard-lipidomics stages."""

    run_calibration: bool = True
    run_annotation: bool = True

    @property
    def any_selected(self) -> bool:
        return self.run_calibration or self.run_annotation


def selected_standard_stages() -> StandardPipelineStageSelection:
    value = st.session_state.get(STANDARD_PIPELINE_SELECTION_KEY)
    if isinstance(value, StandardPipelineStageSelection):
        return value
    return StandardPipelineStageSelection()


def _clear_calibration_run(*, keep_paths: bool = True) -> None:
    for key in (_STREAMS_KEY, _INDEX_KEY, _MODELS_KEY, _FINISHED_KEY):
        st.session_state.pop(key, None)
    if not keep_paths:
        st.session_state.pop(_INPUT_KEY, None)
        st.session_state.pop(_OUTPUT_KEY, None)


def reset_standard_pipeline() -> None:
    """Return the standard workflow to module selection."""
    _clear_calibration_run(keep_paths=True)
    st.session_state.pop(STANDARD_PIPELINE_SELECTION_KEY, None)
    st.session_state.pop(STANDARD_ANNOTATION_PROJECT_KEY, None)
    st.session_state[STANDARD_PIPELINE_PAGE_KEY] = "select"


def render_standard_stage_selection(*, back_action: Callable[[], object]) -> None:
    """Render the same explicit stage-choice pattern used by the 4D workflow."""
    st.title("Standard lipidomics")
    st.caption(
        "Choose independent modules. Calibration writes reusable XCMS files; "
        "annotation reads XCMS files and never launches calibration itself."
    )
    previous = selected_standard_stages()
    calibration = st.toggle(
        "XCMS m/z calibration",
        value=previous.run_calibration,
        key="standard-pipeline-run-calibration",
        help=(
            "Review one precursor-mass model per available RP/NP and POS/NEG "
            "stream, then export annotation-ready XCMS peak-group lists."
        ),
    )
    annotation = st.toggle(
        "Annotation",
        value=previous.run_annotation,
        key="standard-pipeline-run-annotation",
        help=(
            "Run either standard annotation engine from original or previously "
            "calibrated XCMS peak-group lists."
        ),
    )
    selection = StandardPipelineStageSelection(
        run_calibration=bool(calibration), run_annotation=bool(annotation)
    )
    st.session_state[STANDARD_PIPELINE_SELECTION_KEY] = selection
    back_col, continue_col = st.columns(2, gap="medium")
    if back_col.button("Back to start", width="stretch", key="standard-stage-back"):
        reset_standard_pipeline()
        back_action()
        st.rerun()
    if continue_col.button(
        "Continue",
        type="primary",
        width="stretch",
        key="standard-stage-continue",
        disabled=not selection.any_selected,
        help=None if selection.any_selected else "Select at least one module.",
    ):
        _clear_calibration_run(keep_paths=True)
        st.session_state[STANDARD_PIPELINE_PAGE_KEY] = (
            "calibration" if selection.run_calibration else "annotation"
        )
        st.rerun()


def render_standard_pipeline_entry(
    *,
    default_project: str,
    back_action: Callable[[], object],
) -> str | None:
    """Route module selection/calibration or prepare the annotation header."""
    page = st.session_state.setdefault(STANDARD_PIPELINE_PAGE_KEY, "select")
    if page == "select":
        render_standard_stage_selection(back_action=back_action)
        return None
    if page == "calibration":
        render_standard_calibration(default_project=default_project)
        return None
    title_col, back_col = st.columns([5, 1])
    title_col.title("Standard lipidomics · Annotation")
    if back_col.button("Back to modules", width="stretch"):
        st.session_state[STANDARD_PIPELINE_PAGE_KEY] = "select"
        st.rerun()
    st.caption(
        "Annotation reads existing XCMS peak-group lists. To calibrate first, "
        "return to modules and run XCMS m/z calibration."
    )
    return str(
        st.session_state.get(STANDARD_ANNOTATION_PROJECT_KEY, default_project)
    )


def _prefix(stream: str) -> str:
    return f"standard_export_{stream.casefold()}_mass"


def _initialize_controls(prefix: str) -> None:
    defaults = {
        f"{prefix}_reference_panel": "Unlabelled IS + Agilent tune mix",
        f"{prefix}_search_ppm_tol": SCIENTIFIC_DEFAULTS.calibration_ppm_tolerance,
        f"{prefix}_search_rt_tol": SCIENTIFIC_DEFAULTS.calibration_rt_tolerance_min,
        f"{prefix}_tune_mix_rt_max": 3.0,
        f"{prefix}_fit_model": "Linear",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _search_signature(stream: str, source: Path, prefix: str) -> str:
    stat = source.stat()
    payload = (
        stream,
        str(source.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        st.session_state[f"{prefix}_reference_panel"],
        float(st.session_state[f"{prefix}_search_ppm_tol"]),
        float(st.session_state[f"{prefix}_search_rt_tol"]),
        float(st.session_state[f"{prefix}_tune_mix_rt_max"]),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _search_calibrants(features: pd.DataFrame, *, stream: str, prefix: str) -> dict:
    polarity = "pos" if stream.endswith("Pos") else "neg"
    standards = (
        POSITIVE_CCS_INTERNAL_STANDARDS
        if polarity == "pos"
        else NEGATIVE_CCS_INTERNAL_STANDARDS
    )
    references = build_mass_reference_panel(
        load_reference(_REFERENCE),
        polarity=polarity,
        panel_name=str(st.session_state[f"{prefix}_reference_panel"]),
        unlabelled_standards=standards,
        chromatography=stream.split("_", 1)[0],
    )
    return detect_mass_calibration_candidates(
        features,
        references,
        ppm_tol=float(st.session_state[f"{prefix}_search_ppm_tol"]),
        rt_tol=float(st.session_state[f"{prefix}_search_rt_tol"]),
        tune_mix_rt_max=float(st.session_state[f"{prefix}_tune_mix_rt_max"]),
        use_lipid_rt=True,
    )


def _render_calibration_setup(default_project: str) -> None:
    default_input = str(st.session_state.get(_INPUT_KEY, default_project))
    default_output = str(st.session_state.get(_OUTPUT_KEY, ""))
    with st.form("standard-xcms-calibration-setup", border=True):
        input_directory = st.text_input(
            "XCMS project folder",
            value=default_input,
            help=(
                "Project root, Processing/Annotation folder, or any folder "
                "containing peakgrouplist_<phase>_<polarity>.txt files."
            ),
        )
        output_directory = st.text_input(
            "Calibrated XCMS output folder",
            value=default_output,
            help=(
                "LASER writes standalone peakgrouplist files and a JSON audit "
                "manifest here. Existing calibration outputs are not overwritten."
            ),
        )
        phase_choice = st.radio(
            "Chromatography", ("Both", "RP", "NP"), horizontal=True
        )
        start = st.form_submit_button(
            "Review XCMS calibration", type="primary", width="stretch"
        )
    if not start:
        return
    if not input_directory:
        st.error("Choose the XCMS project folder.")
        return
    try:
        sources = discover_standard_streams(input_directory)
    except ValueError as error:
        st.error(str(error))
        return
    phases = ("RP", "NP") if phase_choice == "Both" else (phase_choice,)
    review_order = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")
    streams = tuple(
        stream for stream in review_order
        if stream in sources and stream.split("_", 1)[0] in phases
    )
    if not streams:
        st.error("No selected XCMS peak-group lists were found.")
        return
    output = Path(output_directory).expanduser() if output_directory else (
        Path(input_directory).expanduser().resolve() / "LASER_calibrated_XCMS"
    )
    st.session_state[_INPUT_KEY] = str(Path(input_directory).expanduser().resolve())
    st.session_state[_OUTPUT_KEY] = str(output.resolve())
    st.session_state[_STREAMS_KEY] = streams
    st.session_state[_INDEX_KEY] = 0
    st.session_state[_MODELS_KEY] = {}
    st.session_state.pop(_FINISHED_KEY, None)
    st.rerun()


def _finish_calibration() -> None:
    input_directory = Path(st.session_state[_INPUT_KEY])
    output_directory = Path(st.session_state[_OUTPUT_KEY])
    streams = tuple(st.session_state[_STREAMS_KEY])
    all_sources = discover_standard_streams(input_directory)
    sources = {stream: all_sources[stream] for stream in streams}
    calibrations = dict(st.session_state.get(_MODELS_KEY, {}))
    paths = write_calibrated_xcms_files(
        sources, calibrations, output_directory, overwrite=False
    )
    st.session_state[_FINISHED_KEY] = {
        "output_directory": str(output_directory.resolve()),
        "paths": {stream: str(path.resolve()) for stream, path in paths.items()},
        "applied": tuple(stream for stream in streams if stream in calibrations),
    }
    st.session_state[STANDARD_ANNOTATION_PROJECT_KEY] = str(
        output_directory.resolve()
    )
    for key in (_STREAMS_KEY, _INDEX_KEY, _MODELS_KEY):
        st.session_state.pop(key, None)
    st.rerun()


def _render_finished() -> None:
    finished = st.session_state[_FINISHED_KEY]
    applied = set(finished["applied"])
    rows = [
        {
            "stream": stream,
            "calibration": "applied" if stream in applied else "not applied",
            "XCMS file": path,
        }
        for stream, path in finished["paths"].items()
    ]
    st.success(
        "Calibration module complete. Annotation-ready XCMS files were exported "
        f"to {finished['output_directory']}."
    )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "LASER_standard_mass_calibration.json records every source, fitted model, "
        "and skipped stream. Annotation can use this folder directly."
    )
    selection = selected_standard_stages()
    left, right = st.columns(2, gap="medium")
    if left.button(
        "Calibrate another project", width="stretch", key="standard-calibrate-again"
    ):
        _clear_calibration_run(keep_paths=False)
        st.rerun()
    next_label = "Continue to annotation" if selection.run_annotation else "Choose modules"
    if right.button(
        next_label,
        type="primary",
        width="stretch",
        key="standard-calibration-finished-next",
    ):
        st.session_state.pop(_FINISHED_KEY, None)
        st.session_state[STANDARD_PIPELINE_PAGE_KEY] = (
            "annotation" if selection.run_annotation else "select"
        )
        st.rerun()


def render_standard_calibration(*, default_project: str) -> None:
    """Render a calibration-only workflow that has no annotation side effects."""
    title_col, back_col = st.columns([5, 1])
    title_col.title("Standard lipidomics · XCMS m/z calibration")
    if back_col.button(
        "Back to modules", width="stretch", key="standard-calibration-back"
    ):
        _clear_calibration_run(keep_paths=True)
        st.session_state[STANDARD_PIPELINE_PAGE_KEY] = "select"
        st.rerun()
    st.caption(
        "This module only calibrates and exports XCMS peak-group lists. It does "
        "not start annotation."
    )
    if st.session_state.get(_FINISHED_KEY):
        _render_finished()
        return
    streams = tuple(st.session_state.get(_STREAMS_KEY, ()))
    if not streams:
        _render_calibration_setup(default_project)
        return
    index = int(st.session_state.get(_INDEX_KEY, 0))
    if index >= len(streams):
        try:
            _finish_calibration()
        except (OSError, ValueError) as error:
            st.error(str(error))
            if st.button("Choose another output folder", width="stretch"):
                _clear_calibration_run(keep_paths=True)
                st.session_state.pop(_OUTPUT_KEY, None)
                st.rerun()
        return

    stream = streams[index]
    source = discover_standard_streams(st.session_state[_INPUT_KEY])[stream]
    features = load_standard_feature_list(source)
    prefix = _prefix(stream)
    _initialize_controls(prefix)
    phase, polarity = stream.split("_", 1)
    st.caption(
        f"Calibration {index + 1} of {len(streams)} · {phase} "
        f"{polarity.lower()} mode. Each XCMS stream receives an independent model."
    )
    if phase == "NP":
        st.info(
            "NP internal standards are searched with NP-specific retention times. "
            "Standards without an NP library RT are reported as unavailable."
        )
    models = dict(st.session_state.get(_MODELS_KEY, {}))
    decision = render_mass_calibration(MassCalibrationViewInput(
        search_signature=_search_signature(stream, source, prefix),
        search_candidates=lambda: _search_calibrants(
            features, stream=stream, prefix=prefix
        ),
        return_stage="standard_calibration",
        confirmed_calibration=models.get(stream),
        features=features,
        polarity="pos" if stream.endswith("Pos") else "neg",
        key_prefix=prefix,
        title=f"{phase} {polarity} m/z calibration",
        validate_lipids=phase == "RP",
        use_lipid_rt=True,
    ))
    if decision.action == "none":
        return
    if decision.action == "back":
        if index == 0:
            _clear_calibration_run(keep_paths=True)
        else:
            st.session_state[_INDEX_KEY] = index - 1
        st.rerun()
    if decision.action == "skip":
        models.pop(stream, None)
    elif decision.action == "apply" and decision.calibration is not None:
        calibration = dict(decision.calibration)
        calibration["source_mass_search_signature"] = decision.search_signature
        models[stream] = freeze_mass_calibration(calibration)
    st.session_state[_MODELS_KEY] = models
    st.session_state[_INDEX_KEY] = index + 1
    st.rerun()


__all__ = [
    "STANDARD_ANNOTATION_PROJECT_KEY",
    "STANDARD_PIPELINE_PAGE_KEY",
    "STANDARD_PIPELINE_SELECTION_KEY",
    "StandardPipelineStageSelection",
    "render_standard_calibration",
    "render_standard_pipeline_entry",
    "render_standard_stage_selection",
    "reset_standard_pipeline",
    "selected_standard_stages",
]
