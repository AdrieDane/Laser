"""Standard-lipidomics setup presenters outside the workflow facade."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from annotator.np_rt import DEFAULT_NP_RT_REFERENCE_PATH
from standard_lipidomics.workflow import (
    DEFAULT_STANDARD_PPM_TOLERANCE,
    MAX_STANDARD_PPM_TOLERANCE,
    STREAM_ORDER,
    discover_standard_streams,
)
from standard_lipidomics.policy import (
    DEFAULT_MATLAB_POLICY_PATH,
    discover_matlab_fa_ranges,
    discover_matlab_policy_file,
)
from matlab_annotation.configuration import (
    AUTOMATIC_REGRESSION_INLIERS,
    BIOLOGICAL_ABUNDANCE_PRIOR,
    ITERATIVE_POSITIONAL_REFIT,
    POST_REGRESSION_POSITIONAL_RESOLUTION,
    UNIFIED_EVIDENCE_ANCHORS,
)
from matlab_annotation.execution import MatlabAnnotationError
from scientific_defaults import SCIENTIFIC_DEFAULTS


@dataclass(frozen=True)
class MatlabFormState:
    phases: tuple[str, ...]
    input_directory: str
    reference_directory: str
    anchor_selection_mode: str
    positional_workflow_mode: str
    anchor_ppm: float
    anchor_rt_seconds: float
    anchor_score_margin: float
    minimum_anchors: int
    use_grid_coherence: bool
    grid_coherence_threshold: float
    composition_ranges: Any
    submitted: bool


@dataclass(frozen=True)
class LaserFormState:
    input_directory: str
    use_np_rt: bool
    reference_path: str
    policy_path: str
    np_rt_reference_path: str
    uploads: dict[str, Any]
    composition_ranges: Any
    ppm: float
    rt_tolerance: float
    adapt_class_rt: bool
    np_rt_tolerance: float
    use_grid_coherence: bool
    grid_coherence_threshold: float
    detect: bool
    run: bool


def _matlab_form(
    *, default_project: str, default_reference: Path,
    render_composition_ranges: Callable[[str], Any],
) -> MatlabFormState:
    with st.form("matlab-engine-inputs", border=True):
        phase_choice = st.radio(
            "Chromatography", ("Both", "RP", "NP"), horizontal=True,
            help=("Both runs RP and NP automatically but keeps their RT models and "
                  "grids isolated. Choose RP or NP to run only one phase. Positional "
                  "LPC/LPE pairing applies only to RP; NP uses the combined one-peak "
                  "phase identity."),
        )
        phases = ("RP", "NP") if phase_choice == "Both" else (phase_choice,)
        input_directory = st.text_input(
            "XCMS project folder", value=default_project,
            help=("The project root or its Processing/Annotation folder, the folder "
                  "containing peakgrouplist_<phase>_Pos.txt and/or "
                  "peakgrouplist_<phase>_Neg.txt. LASER detects them automatically."),
        )
        with st.expander("Annotation settings", expanded=False):
            reference_directory = st.text_input(
                "MATLAB reference bundle (Code / DataBase / RTDB)",
                value=str(default_reference),
                help=("LASER ships the lipid database, parameter file and RT database "
                      "this engine needs. Override only deliberately."),
            )
            anchor_label = st.selectbox(
                "Automatic RT-anchor selection",
                ("Unified validated evidence (default)",
                 "Biological abundance prior (legacy diagnostic)",
                 "Automatic regression inliers (legacy diagnostic)"),
                index=0,
                help=("The default builds one RT model from validated evidence. The "
                      "legacy choices remain only for controlled comparisons."),
            )
            anchor_mode = (UNIFIED_EVIDENCE_ANCHORS if anchor_label.startswith("Unified")
                           else BIOLOGICAL_ABUNDANCE_PRIOR if anchor_label.startswith("Biological")
                           else AUTOMATIC_REGRESSION_INLIERS)
            positional_label = st.selectbox(
                "RP positional-isomer workflow",
                ("Post-regression resolver (current default)",
                 "Two-pass resolver and RT refit (experimental)"), index=0,
                help=("The current workflow resolves LPC/LPE after RT regression. The "
                      "experimental workflow resolves provisional pairs and refits the "
                      "RT model. NP is unchanged because positional forms coelute."),
            )
            positional_mode = (ITERATIVE_POSITIONAL_REFIT
                               if positional_label.startswith("Two-pass")
                               else POST_REGRESSION_POSITIONAL_RESOLUTION)
            anchor_ppm = st.number_input(
                "Automatic-anchor m/z tolerance (ppm)", min_value=0.1,
                max_value=3.0, value=3.0, step=0.1,
                help="Maximum precursor-mass error for an automatic RT anchor.")
            anchor_rt = st.number_input(
                "Automatic-anchor RT tolerance (seconds)", min_value=5.0,
                max_value=120.0, value=30.0, step=5.0,
                help="Maximum reference-RT difference for an automatic anchor.")
            margin = st.number_input(
                "Minimum ambiguity-score margin", min_value=0.0, max_value=1.0,
                value=0.20, step=0.05,
                help="Required score advantage over the next-best anchor assignment.")
            minimum = st.number_input(
                "Minimum anchors per lipid grid", min_value=3, max_value=20,
                value=5, step=1,
                help="Minimum trusted anchors required to calibrate a class RT grid.")
            use_coherence = st.checkbox(
                "Use grid coherence as an annotation filter", value=True,
                key="standard_matlab_use_grid_coherence",
                help="Exclude assessed local grid outliers from the final annotation table.")
            coherence = st.number_input(
                "Minimum grid coherence", min_value=0.0, max_value=1.0,
                value=SCIENTIFIC_DEFAULTS.grid_coherence_threshold, step=0.05,
                key="standard_matlab_grid_coherence_threshold",
                help="Scores below this value are excluded while the filter is enabled.")
        with st.expander("Carbon and double-bond annotation limits", expanded=False):
            composition_ranges = render_composition_ranges("standard_matlab")
        submitted = st.form_submit_button(
            "Run MATLAB annotation", type="primary", width="stretch")
    return MatlabFormState(
        phases=phases, input_directory=input_directory,
        reference_directory=reference_directory, anchor_selection_mode=str(anchor_mode),
        positional_workflow_mode=str(positional_mode), anchor_ppm=float(anchor_ppm),
        anchor_rt_seconds=float(anchor_rt), anchor_score_margin=float(margin),
        minimum_anchors=int(minimum), use_grid_coherence=bool(use_coherence),
        grid_coherence_threshold=float(coherence),
        composition_ranges=composition_ranges, submitted=submitted,
    )


def _execute_matlab_form(
    state: MatlabFormState, *, cache_version: int,
    project_signature: Callable[[Path, tuple[str, ...]], tuple],
    run_cached: Callable[..., dict],
) -> None:
    if not state.submitted:
        return
    if not state.input_directory:
        st.error("Choose the XCMS project folder.")
        return
    project = Path(state.input_directory).expanduser()
    sources = discover_standard_streams(project)
    available = tuple(phase for phase in state.phases
                      if any(name.startswith(f"{phase}_") for name in sources))
    if not available:
        expected = " or ".join(f"peakgrouplist_{phase}_Pos/Neg"
                               for phase in state.phases)
        st.error(f"No selected XCMS peak-group lists were found. Expected {expected}.")
        return
    missing = tuple(phase for phase in state.phases if phase not in available)
    if missing:
        st.warning(f"No {'/'.join(missing)} peak-group lists were found; "
                   f"running {'/'.join(available)} only.")
    _run_matlab_form(
        state,
        cache_version=cache_version,
        project_signature=project_signature,
        run_cached=run_cached,
    )


def _run_matlab_form(
    state: MatlabFormState,
    *,
    cache_version: int,
    project_signature: Callable[[Path, tuple[str, ...]], tuple],
    run_cached: Callable[..., dict],
) -> None:
    """Run annotation directly from original or pre-calibrated XCMS files."""
    project = Path(state.input_directory).expanduser()
    sources = discover_standard_streams(project)
    status, bar = st.empty(), st.progress(0.0)

    def report(fraction: float, message: str) -> None:
        bar.progress(min(max(fraction, 0.0), 1.0))
        status.caption(message)

    try:
        result = run_cached(
            str(project.resolve()), project_signature(project, state.phases),
            str(Path(state.reference_directory).expanduser()), cache_version,
            state.phases, state.anchor_selection_mode,
            state.positional_workflow_mode, state.anchor_ppm,
            state.anchor_rt_seconds, state.anchor_score_margin,
            state.minimum_anchors, state.use_grid_coherence,
            state.grid_coherence_threshold, state.composition_ranges,
            report,
        )
    except MatlabAnnotationError as error:
        bar.empty()
        status.empty()
        st.error(str(error))
        return
    bar.empty()
    status.empty()
    st.session_state.standard_annotation_result = result
    st.session_state.standard_results_page = "menu"
    st.session_state.standard_project_directory = str(project.resolve())
    st.session_state.standard_input_sources = dict(sources)
    for key in ("standard_report_html", "standard_report_signature",
                "standard_untargeted_review_basket", "standard_untargeted_analysis",
                "standard_untargeted_signature"):
        st.session_state.pop(key, None)
    st.session_state.standard_headless_config = {
        "workflow": "standard", "engine": "matlab",
        "inputs": {"project": str(project.resolve()),
                   "matlab_reference": str(Path(state.reference_directory).expanduser().resolve())},
        "annotation": {
            "phases": list(state.phases), "only_accepted": True,
            "anchor_selection_mode": state.anchor_selection_mode,
            "positional_isomer_workflow_mode": state.positional_workflow_mode,
            "anchor_ppm": state.anchor_ppm,
            "anchor_rt_seconds": state.anchor_rt_seconds,
            "anchor_score_margin": state.anchor_score_margin,
            "minimum_anchors": state.minimum_anchors,
            "use_grid_coherence": state.use_grid_coherence,
            "grid_coherence_threshold": state.grid_coherence_threshold,
            "use_post_regression_completion": True,
            "post_regression_confidence_threshold": 0.50,
            "post_regression_ambiguity_margin": 0.10,
            "use_fitted_grid_completion": True,
            "fitted_grid_feature_ambiguity_margin": 0.10,
            "composition_ranges": state.composition_ranges.to_dict(),
        },
        "output": {"directory": "results/standard_matlab", "overwrite": False},
    }


def render_matlab_engine(
    *, default_project: str, default_reference: Path, cache_version: int,
    render_composition_ranges: Callable[[str], Any],
    project_signature: Callable[[Path, tuple[str, ...]], tuple],
    run_cached: Callable[..., dict],
) -> None:
    """Render and execute the automatic XCMS-only MATLAB annotation setup."""
    state = _matlab_form(
        default_project=default_project, default_reference=default_reference,
        render_composition_ranges=render_composition_ranges,
    )
    _execute_matlab_form(
        state, cache_version=cache_version, project_signature=project_signature,
        run_cached=run_cached,
    )


def _laser_reference_inputs(
    default_reference: Path, use_np_rt: bool,
) -> tuple[str, str, str, dict[str, Any]]:
    with st.expander("Annotation settings", expanded=False):
        st.caption(
            "LASER ships with the chemical reference, standard MATLAB lipid "
            "policy, and NP RT reference. Change these paths only deliberately."
        )
        reference = st.text_input(
            "Chemical reference CSV", value=str(default_reference),
            help="Chemical identity and exact-mass reference used by the matcher.",
        )
        policy = st.text_input(
            "MATLAB lipid policy override (lipids.tsv)",
            value=str(DEFAULT_MATLAB_POLICY_PATH),
            help="Leave the bundled standard policy selected for routine analysis.",
        )
        np_reference = st.text_input(
            "NP retention-time reference CSV", value=str(DEFAULT_NP_RT_REFERENCE_PATH),
            disabled=not use_np_rt,
            help="Identity-level NP RT reference used after guarded calibration.",
        )
        st.caption("Optional uploads override detected files for the same stream.")
        uploads = {
            stream: column.file_uploader(
                stream, type=["txt", "tsv", "csv"], key=f"standard_{stream}",
                help=f"Optional manual {stream} peak-group list.",
            )
            for column, stream in zip(st.columns(4), STREAM_ORDER)
        }
    return reference, policy, np_reference, uploads


def _laser_annotation_criteria(
    use_np_rt: bool,
) -> tuple[float, float, bool, float, bool, float]:
    first, second, third = st.columns(3)
    ppm = first.number_input(
        "m/z tolerance (ppm)", min_value=0.1,
        max_value=MAX_STANDARD_PPM_TOLERANCE,
        value=DEFAULT_STANDARD_PPM_TOLERANCE, step=0.5,
        help="Maximum precursor-mass deviation allowed for a reference match.")
    rt_tolerance = second.number_input(
        "RP RT tolerance (min)", min_value=0.01,
        value=SCIENTIFIC_DEFAULTS.rt_tolerance_min, step=0.05,
        help="Maximum RP RT deviation after validated class-grid adjustment.")
    adapt = third.checkbox(
        "Automatically adapt class RT grids", value=True,
        help="Apply only class corrections that pass leave-one-out validation.")
    np_tolerance = st.number_input(
        "NP RT tolerance (min)", min_value=0.01, value=0.30, step=0.05,
        disabled=not use_np_rt,
        help="Applied after the NP reference is shifted by validated anchors.")
    coherence_col, threshold_col = st.columns(2)
    use_coherence = coherence_col.checkbox(
        "Use grid coherence as an annotation filter", value=True,
        key="standard_laser_use_grid_coherence",
        help="Exclude assessed local grid outliers from final annotations.")
    threshold = threshold_col.number_input(
        "Minimum grid coherence", min_value=0.0, max_value=1.0,
        value=SCIENTIFIC_DEFAULTS.grid_coherence_threshold, step=0.05,
        key="standard_laser_grid_coherence_threshold",
        help="Scores below this value are filtered while the option is enabled.")
    return (float(ppm), float(rt_tolerance), bool(adapt), float(np_tolerance),
            bool(use_coherence), float(threshold))


def _laser_form(
    *, default_project: str, default_reference: Path,
    render_composition_ranges: Callable[[str], Any],
) -> LaserFormState:
    with st.form("standard-lipidomics-inputs", border=True):
        project = st.text_input(
            "MATLAB/XCMS project folder", value=default_project,
            help="Choose the project root or Processing/Annotation folder.")
        use_np_rt = st.checkbox(
            "Use anchored NP retention-time grid", value=True,
            help="Apply the NP reference only after guarded dataset calibration.")
        reference, policy, np_reference, uploads = _laser_reference_inputs(
            default_reference, use_np_rt)
        with st.expander("Carbon and double-bond annotation limits", expanded=False):
            composition = render_composition_ranges("standard_laser")
        criteria = _laser_annotation_criteria(use_np_rt)
        detect_col, run_col = st.columns(2)
        detect = detect_col.form_submit_button("Detect project inputs", width="stretch")
        run = run_col.form_submit_button(
            "Run standard annotation", type="primary", width="stretch")
    return LaserFormState(
        input_directory=project, use_np_rt=use_np_rt, reference_path=reference,
        policy_path=policy, np_rt_reference_path=np_reference, uploads=uploads,
        composition_ranges=composition, ppm=criteria[0], rt_tolerance=criteria[1],
        adapt_class_rt=criteria[2], np_rt_tolerance=criteria[3],
        use_grid_coherence=criteria[4], grid_coherence_threshold=criteria[5],
        detect=detect, run=run,
    )


def _resolve_laser_inputs(
    state: LaserFormState,
) -> tuple[Path, Path, Path | None, dict[str, bytes], str | None]:
    reference = Path(state.reference_path).expanduser()
    policy = Path(state.policy_path).expanduser() if state.policy_path else None
    np_reference = (Path(state.np_rt_reference_path).expanduser()
                    if state.use_np_rt else None)
    sources: dict[str, bytes] = {}
    error = None
    if state.input_directory:
        try:
            sources.update({name: path.read_bytes() for name, path in
                            discover_standard_streams(state.input_directory).items()})
        except ValueError as exc:
            error = str(exc)
    if policy is None and state.input_directory:
        policy = discover_matlab_policy_file(state.input_directory)
    policy = policy or DEFAULT_MATLAB_POLICY_PATH
    for stream, upload in state.uploads.items():
        if upload is not None:
            sources[stream] = upload.getvalue()
    if not reference.is_file():
        error = f"Reference file not found: {reference}"
    elif not policy.is_file():
        error = f"MATLAB lipid policy not found: {policy}"
    elif np_reference is not None and not np_reference.is_file():
        error = f"NP retention-time reference not found: {np_reference}"
    elif not sources and error is None:
        error = "Choose an XCMS folder or upload at least one stream."
    return reference, policy, np_reference, sources, error


def _store_laser_result(
    result: dict, state: LaserFormState, reference: Path, policy: Path,
    np_reference: Path | None, sources: dict[str, bytes],
) -> None:
    project = (str(Path(state.input_directory).expanduser().resolve())
               if state.input_directory else None)
    st.session_state.standard_annotation_result = result
    st.session_state.standard_results_page = "menu"
    st.session_state.standard_project_directory = project
    st.session_state.standard_input_sources = dict(sources)
    for key in ("standard_report_html", "standard_report_signature",
                "standard_untargeted_review_basket", "standard_untargeted_analysis",
                "standard_untargeted_signature"):
        st.session_state.pop(key, None)
    st.session_state.standard_headless_config = {
        "workflow": "standard", "engine": "laser",
        "inputs": {
            "project": project or "REPLACE_WITH_XCMS_PROJECT",
            "reference": str(reference.resolve()),
            "matlab_policy": str(policy.resolve()),
            "np_rt_reference": str(np_reference.resolve()) if np_reference else None,
        },
        "annotation": {
            "ppm": state.ppm, "rp_rt_tolerance_min": state.rt_tolerance,
            "align_rp_rt": False, "adapt_class_rt": state.adapt_class_rt,
            "disable_matlab_policy": False,
            "disable_np_rt_reference": not state.use_np_rt,
            "np_rt_tolerance_min": state.np_rt_tolerance,
            "use_grid_coherence": state.use_grid_coherence,
            "grid_coherence_threshold": state.grid_coherence_threshold,
            "composition_ranges": state.composition_ranges.to_dict(),
        },
        "output": {"directory": "results/standard", "overwrite": False},
    }


def _execute_laser_form(state: LaserFormState, run_cached: Callable[..., dict]) -> None:
    if not (state.detect or state.run):
        return
    reference, policy, np_reference, sources, error = _resolve_laser_inputs(state)
    if error:
        st.error(error)
        return
    if state.detect:
        source = ("project policy" if state.input_directory and
                  discover_matlab_policy_file(state.input_directory) == policy
                  else "bundled standard policy")
        st.success(f"Detected {len(sources)} stream(s): {', '.join(sources)}. "
                   f"Using {source}: {policy.name}.")
        return
    search_root = state.input_directory or policy.parent
    carbon_range, db_range, range_path = discover_matlab_fa_ranges(search_root)
    with st.spinner("Annotating conventional lipidomics streams…"):
        result = run_cached(
            str(reference), reference.stat().st_size, reference.stat().st_mtime_ns,
            state.ppm, state.rt_tolerance, state.adapt_class_rt, str(policy),
            policy.stat().st_size, policy.stat().st_mtime_ns,
            str(np_reference) if np_reference else "",
            np_reference.stat().st_size if np_reference else 0,
            np_reference.stat().st_mtime_ns if np_reference else 0,
            state.np_rt_tolerance, state.use_grid_coherence,
            state.grid_coherence_threshold, carbon_range, db_range,
            str(range_path) if range_path else "", state.composition_ranges, sources,
        )
    _store_laser_result(result, state, reference, policy, np_reference, sources)


def render_laser_engine(
    *, default_project: str, default_reference: Path,
    render_composition_ranges: Callable[[str], Any],
    run_cached: Callable[..., dict],
) -> None:
    """Render and execute the LASER reference-matcher setup."""
    state = _laser_form(
        default_project=default_project, default_reference=default_reference,
        render_composition_ranges=render_composition_ranges,
    )
    st.info(
        "RP applies only validated class-specific corrections. Global RT alignment "
        "is disabled. NP uses the identity-level reference only when guarded "
        "dataset calibration passes. MATLAB lipids.tsv policies are enforced."
    )
    _execute_laser_form(state, run_cached)
