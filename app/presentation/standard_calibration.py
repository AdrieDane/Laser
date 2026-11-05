"""Separate stage selection and XCMS mass calibration for standard lipidomics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from presentation.app_theme import page_header
from presentation.standard_rt_calibration import render_rt_calibration
from standard_lipidomics.internal_standards import (
    PROJECT_IS_PANELS, project_is_definitions, project_is_reference, project_is_signature,
)
from annotator.mass_calibrants import (
    MASS_REFERENCE_PANELS,
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
    apply_standard_mass_calibration,
)
from standard_lipidomics.project_store import save_calibration_workflow
from standard_lipidomics.project_layout import workflow_folder, xcms_project_root
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
_RT_MODELS_KEY = "standard_calibration_rt_models"
_STAGE_KEY = "standard_calibration_stage"
_KINDS_KEY = "standard_calibration_kinds"
_FINISHED_KEY = "standard_calibration_finished"
_REFERENCE = REFERENCE_DATA_DIRECTORY / "4d_lipid_reference.csv"


@dataclass(frozen=True)
class StandardPipelineStageSelection:
    """Independent, composable standard-lipidomics stages."""

    run_calibration: bool = True
    run_annotation: bool = True
    run_untargeted: bool = False

    @property
    def any_selected(self) -> bool:
        return self.run_calibration or self.run_annotation or self.run_untargeted


def selected_standard_stages() -> StandardPipelineStageSelection:
    value = st.session_state.get(STANDARD_PIPELINE_SELECTION_KEY)
    if isinstance(value, StandardPipelineStageSelection):
        return value
    return StandardPipelineStageSelection()


def _clear_calibration_run(*, keep_paths: bool = True) -> None:
    for key in (_STREAMS_KEY, _INDEX_KEY, _MODELS_KEY, _RT_MODELS_KEY, _STAGE_KEY, _KINDS_KEY, _FINISHED_KEY):
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
    """Five independent modules on the LASER start page."""
    st.title("LASER Standard Lipidomics")
    st.caption("Choose a module, then enter your project folder. LASER finds the input files and saved results.")
    modules = (
        ("Calibration", "Review mass calibration and save reusable XCMS files.", "calibration"),
        ("Annotation", "Annotate raw XCMS data or choose a saved calibration.", "annotation"),
        ("Untargeted analysis", "Compare XCMS features between biological groups.", "untargeted"),
        ("Open completed annotation", "Reopen a saved annotation workspace.", "open_annotation"),
        ("Open completed untargeted analysis", "Reopen saved untargeted results and review decisions.", "open_untargeted"),
    )
    with st.container(key="home-module-buttons"):
        for label, description, page in modules:
            if st.button(label, key=f"standard-module-{page}", width="stretch", help=description):
                _clear_calibration_run(keep_paths=False)
                st.session_state.pop(STANDARD_ANNOTATION_PROJECT_KEY, None)
                st.session_state[STANDARD_PIPELINE_SELECTION_KEY] = StandardPipelineStageSelection(
                    run_calibration=page == "calibration", run_annotation=page == "annotation",
                    run_untargeted=page == "untargeted")
                st.session_state[STANDARD_PIPELINE_PAGE_KEY] = page
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
    titles = {"annotation": "Annotation", "untargeted": "Untargeted analysis",
              "open_annotation": "Open completed annotation",
              "open_untargeted": "Open completed untargeted analysis"}
    if page_header(titles.get(page, "LASER Standard Lipidomics"), key="standard-module-back"):
        st.session_state[STANDARD_PIPELINE_PAGE_KEY] = "select"
        st.rerun()
    return str(
        st.session_state.get(STANDARD_ANNOTATION_PROJECT_KEY, default_project)
    )


def _prefix(stream: str) -> str:
    return f"standard_export_{stream.casefold()}_mass"


def _initialize_controls(prefix: str, panels: tuple[str, ...] = MASS_REFERENCE_PANELS) -> None:
    defaults = {
        f"{prefix}_reference_panel": "Unlabelled IS + Agilent tune mix",
        f"{prefix}_search_ppm_tol": SCIENTIFIC_DEFAULTS.calibration_ppm_tolerance,
        f"{prefix}_search_rt_tol": SCIENTIFIC_DEFAULTS.calibration_rt_tolerance_min,
        f"{prefix}_tune_mix_rt_max": 3.0,
        f"{prefix}_fit_model": "Linear",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state[f"{prefix}_reference_panel"] not in panels:
        st.session_state[f"{prefix}_reference_panel"] = panels[0]


def _search_signature(stream: str, source: Path, prefix: str) -> str:
    stat = source.stat()
    payload = (
        stream,
        project_is_signature(st.session_state.get(_INPUT_KEY, source.parent)),
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
    project = st.session_state.get(_INPUT_KEY)
    path, _ = project_is_definitions(project) if project else (None, ())
    panel = str(st.session_state[f"{prefix}_reference_panel"])
    if path is not None:
        references = project_is_reference(project, load_reference(_REFERENCE), stream=stream)
        if panel == "Agilent tune mix":
            references = references.iloc[:0]
        if "Agilent tune mix" in panel:
            tune = build_mass_reference_panel(load_reference(_REFERENCE), polarity=polarity,
                panel_name="Agilent tune mix", unlabelled_standards={}, chromatography=stream.split("_")[0])
            references = pd.concat([references, tune], ignore_index=True)
    else:
        references = build_mass_reference_panel(
            load_reference(_REFERENCE), polarity=polarity, panel_name=panel,
            unlabelled_standards=standards, chromatography=stream.split("_", 1)[0])
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
    with st.form("standard-xcms-calibration-setup", border=True):
        input_directory = st.text_input(
            "Project folder",
            value=default_input,
            help=(
                "Project root, Processing/Annotation folder, or any folder "
                "containing peakgrouplist_<phase>_<polarity>.txt files."
            ),
        )
        st.caption("Calibration exports are saved in the project's Calibration folder.")
        mass_enabled = st.checkbox("m/z calibration", value=True)
        rt_enabled = st.checkbox("RT calibration", value=False)
        phase_choice = st.radio(
            "Chromatography", ("Both", "RP", "NP"), horizontal=True
        )
        start = st.form_submit_button(
            "Review XCMS calibration", type="primary", width="stretch"
        )
    if not start:
        return
    if not (mass_enabled or rt_enabled):
        st.error("Select m/z calibration, RT calibration, or both.")
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
    root = xcms_project_root(input_directory)
    st.session_state[_INPUT_KEY] = str(root)
    st.session_state[_OUTPUT_KEY] = str(workflow_folder(root, "Calibration"))
    st.session_state[_STREAMS_KEY] = streams
    st.session_state[_INDEX_KEY] = 0
    st.session_state[_MODELS_KEY] = {}
    st.session_state[_RT_MODELS_KEY] = {}
    st.session_state[_KINDS_KEY] = (mass_enabled, rt_enabled)
    st.session_state[_STAGE_KEY] = "mass" if mass_enabled else "rt"
    st.session_state.pop(_FINISHED_KEY, None)
    st.rerun()


def _finish_calibration() -> None:
    input_directory = Path(st.session_state[_INPUT_KEY])
    output_directory = Path(st.session_state[_OUTPUT_KEY])
    streams = tuple(st.session_state[_STREAMS_KEY])
    all_sources = discover_standard_streams(input_directory)
    sources = {stream: all_sources[stream] for stream in streams}
    calibrations = dict(st.session_state.get(_MODELS_KEY, {}))
    st.session_state[_FINISHED_KEY] = {
        "output_directory": str(output_directory.resolve()),
        "sources": sources, "models": calibrations,
        "rt_models": dict(st.session_state.get(_RT_MODELS_KEY, {})), "paths": {}, "saved": False,
        "applied": tuple(stream for stream in streams if stream in calibrations),
    }
    for key in (_STREAMS_KEY, _INDEX_KEY, _MODELS_KEY):
        st.session_state.pop(key, None)
    st.rerun()


def _render_finished() -> None:
    finished = st.session_state[_FINISHED_KEY]
    applied = set(finished["applied"])
    rows = [
        {
            "stream": stream,
            "m/z calibration": "applied" if stream in applied else "not applied",
            "RT calibration": "applied" if stream in finished.get("rt_models", {}) else "not applied",
            "XCMS file": path,
        }
        for stream, path in finished["paths"].items()
    ]
    if finished["saved"]:
        st.success(f"Calibration saved: {finished['output_directory']}")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.success("Calibration review complete. Save this calibration to use it in annotation.")
    st.caption(f"Save location: {finished['output_directory']}")
    existing = st.radio(
        "When an old project exists", ("Archive old project", "Overwrite old project"),
        horizontal=True, key="standard-calibration-existing")
    if st.button("Save calibration", type="primary", width="stretch"):
        try:
            paths, archive = save_calibration_workflow(
                st.session_state[_INPUT_KEY], finished["sources"], finished["models"],
                existing="archive" if existing.startswith("Archive") else "overwrite",
                rt_calibrations=finished.get("rt_models", {}))
            finished["paths"] = {name: str(path) for name, path in paths.items()}
            finished["saved"] = True
            st.session_state[STANDARD_ANNOTATION_PROJECT_KEY] = st.session_state[_INPUT_KEY]
            st.session_state["standard-annotation-input-kind"] = "Calibrated XCMS data"
            st.success("Calibration saved.")
            if archive:
                st.caption(f"Previous calibration archived at {archive}")
        except (OSError, ValueError) as exc:
            st.error(f"Calibration could not be saved: {exc}")
    st.caption("The calibration audit contains the reviewed models and skipped streams. Archived calibrations remain available in Annotation.")
    selection = selected_standard_stages()
    left, right = st.columns(2, gap="medium")
    if left.button(
        "Calibrate another project", width="stretch", key="standard-calibrate-again"
    ):
        _clear_calibration_run(keep_paths=False)
        st.rerun()
    next_label = (
        "Continue to annotation" if selection.run_annotation else
        "Continue to untargeted analysis" if selection.run_untargeted else
        "Choose modules"
    )
    if right.button(
        next_label,
        type="primary",
        width="stretch",
        key="standard-calibration-finished-next",
    ):
        st.session_state.pop(_FINISHED_KEY, None)
        st.session_state[STANDARD_PIPELINE_PAGE_KEY] = (
            "annotation" if selection.run_annotation else
            "untargeted" if selection.run_untargeted else "select"
        )
        st.rerun()


def render_standard_calibration(*, default_project: str) -> None:
    """Render a calibration-only workflow that has no annotation side effects."""
    if page_header("Calibration", key="standard-calibration-back"):
        _clear_calibration_run(keep_paths=True)
        st.session_state[STANDARD_PIPELINE_PAGE_KEY] = "select"
        st.rerun()
    st.caption("Review independent m/z and RT calibrations for each XCMS stream.")
    st.info("RT calibration uses reviewed standards. Saved XCMS files use the reference RT scale and retain the original RTs in audit columns.")
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
    try:
        is_path, _ = project_is_definitions(st.session_state[_INPUT_KEY])
    except ValueError as exc:
        st.error(str(exc))
        return
    panels = PROJECT_IS_PANELS if is_path is not None else MASS_REFERENCE_PANELS
    mass_enabled, rt_enabled = st.session_state.get(_KINDS_KEY, (True, False))
    if st.session_state.get(_STAGE_KEY) == "rt":
        features = apply_standard_mass_calibration(features, st.session_state.get(_MODELS_KEY, {}).get(stream))
        rt_decision = render_rt_calibration(features, load_reference(_REFERENCE), stream=stream, prefix=prefix,
            project=st.session_state[_INPUT_KEY])
        if rt_decision.action == "none":
            return
        if rt_decision.action == "back":
            if mass_enabled:
                st.session_state[_STAGE_KEY] = "mass"
            elif index > 0:
                st.session_state[_INDEX_KEY] = index - 1
            else:
                _clear_calibration_run(keep_paths=True)
        else:
            rt_models = dict(st.session_state.get(_RT_MODELS_KEY, {}))
            if rt_decision.action == "apply":
                rt_models[stream] = rt_decision.calibration
            else:
                rt_models.pop(stream, None)
            st.session_state[_RT_MODELS_KEY] = rt_models
            st.session_state[_INDEX_KEY] = index + 1
            st.session_state[_STAGE_KEY] = "mass" if mass_enabled else "rt"
        st.rerun()
    _initialize_controls(prefix, panels)
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
        reference_panels=panels,
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
    if rt_enabled:
        st.session_state[_STAGE_KEY] = "rt"
    else:
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
