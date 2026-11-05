"""Streamlit presentation layer for conventional XCMS lipidomics."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st

from annotator.composition_ranges import CompositionRangeSettings
from annotator.standard_annotated_inputs import append_standard_annotations, remove_standard_annotation_columns
from saved_project import resolve_artifact_path
from standard_lipidomics.workflow import (
    discover_standard_streams,
)
from matlab_annotation.configuration import (
    DEFAULT_MATLAB_PIPELINE_REFERENCE,
)
from standard_lipidomics.grid_browser import render_standard_grid_browser
from standard_lipidomics.report import render_standard_lipidomics_report
from scientific_defaults import SCIENTIFIC_DEFAULTS
from standard_lipidomics.preferences import default_project
from standard_lipidomics.project_store import (
    load_annotation_workflow,
    load_untargeted_workflow,
    save_annotation_workflow,
    save_untargeted_workflow,
    stage_project_metadata,
    workflow_folder,
    xcms_project_root,
)
from standard_lipidomics.project_layout import available_calibrations
from standard_lipidomics.internal_standards import project_is_definitions, project_is_signature
from application.standard_workflow import (
    LASER_STANDARD_ENGINE,
    MATLAB_STANDARD_ENGINE,
    StandardAnnotationRequest,
)
from composition import standard_annotation_service
from presentation.standard_workspace import (
    streamlit_safe_report_frame as _streamlit_safe_report_frame,  # noqa: F401
)
from presentation.standard_setup import render_laser_engine, render_matlab_engine
from application.annotation_jobs import active_annotation_job, read_annotation_job
from application.untargeted_jobs import active_untargeted_job, load_untargeted_inputs, start_untargeted_job
from presentation.job_terminal import render_job_terminal
from presentation.app_theme import page_header
from presentation.standard_calibration import render_standard_pipeline_entry
from presentation.standard_calibration import (
    STANDARD_PIPELINE_PAGE_KEY,
)
from presentation.standard_composition import (
    render_composition_range_settings as _render_composition_range_settings_presenter,
)
from presentation.standard_exports import (
    render_standard_exports as _render_standard_exports_presenter,
)
from presentation.standard_audit import render_matlab_audit
from presentation.standard_untargeted import (
    _mirrored_untargeted_table,  # noqa: F401
    _p_value_marker_colours,  # noqa: F401
    _selected_untargeted_point,  # noqa: F401
    _untargeted_abundance_table,  # noqa: F401
    _untargeted_group_summary,  # noqa: F401
    _untargeted_map_families,  # noqa: F401
    render_untargeted_workspace,
    _restore_group_metadata_from_inputs,
    _untargeted_criteria,
)

LASER_ENGINE = "LASER reference matcher"
MATLAB_ENGINE = "Automatic MATLAB engine (XCMS)"
MATLAB_ENGINE_CACHE_VERSION = 28
DEFAULT_MATLAB_XCMS_PROJECT = default_project()


def _render_composition_range_settings(
    key_prefix: str,
) -> CompositionRangeSettings:
    """Compatibility wrapper around the focused composition presenter."""
    return _render_composition_range_settings_presenter(key_prefix)


@st.cache_data(show_spinner=False)
def _run_standard_cached(
    reference_path: str,
    reference_size: int,
    reference_mtime_ns: int,
    ppm_tolerance: float,
    rp_rt_tolerance_min: float,
    adapt_class_rt: bool,
    policy_path: str,
    policy_size: int,
    policy_mtime_ns: int,
    np_rt_reference_path: str,
    np_rt_reference_size: int,
    np_rt_reference_mtime_ns: int,
    np_rt_tolerance_min: float,
    use_grid_coherence: bool,
    grid_coherence_threshold: float,
    fa_carbon_range: tuple[int, int],
    fa_db_range: tuple[int, int],
    range_source: str,
    composition_ranges: CompositionRangeSettings,
    _sources: dict[str, bytes],
    source_signature: tuple = (),
    grid_coherence_version: str = "v1",
    input_project_path: str = "",
    internal_standards_signature: tuple = (),
) -> dict:
    return standard_annotation_service.run(
        StandardAnnotationRequest(
            engine=LASER_STANDARD_ENGINE,
            project=Path(input_project_path) if input_project_path else Path(policy_path).expanduser().parent,
            sources=_sources,
            reference_path=Path(reference_path),
            matlab_policy_path=Path(policy_path),
            np_rt_reference_path=(
                Path(np_rt_reference_path) if np_rt_reference_path else None
            ),
            settings={
                "ppm": ppm_tolerance,
                "rp_rt_tolerance_min": rp_rt_tolerance_min,
                "align_rp_rt": False,
                "adapt_class_rt": adapt_class_rt,
                "fa_carbon_range": fa_carbon_range,
                "fa_db_range": fa_db_range,
                "range_source": range_source or None,
                "composition_ranges": composition_ranges.to_dict(),
                "np_rt_tolerance_min": np_rt_tolerance_min,
                "use_grid_coherence": use_grid_coherence,
                "grid_coherence_threshold": grid_coherence_threshold,
                "grid_coherence_version": grid_coherence_version,
            },
        )
    ).to_mapping()


@st.cache_data(show_spinner=False)
def _run_matlab_cached(
    project_path: str,
    project_signature: tuple,
    reference_path: str,
    cache_schema_version: int,
    phases: tuple[str, ...],
    anchor_selection_mode: str,
    positional_isomer_workflow_mode: str,
    anchor_ppm: float,
    anchor_rt_seconds: float,
    anchor_score_margin: float,
    minimum_anchors: int,
    use_grid_coherence: bool,
    grid_coherence_threshold: float,
    composition_ranges: CompositionRangeSettings,
    _progress,
    use_pe_rebuild: bool = True,
    pe_rebuild_score_threshold: float = 0.10,
    use_experimental_pe_selector: bool = False,
    grid_coherence_version: str = "v1",
) -> dict:
    return standard_annotation_service.run(
        StandardAnnotationRequest(
            engine=MATLAB_STANDARD_ENGINE,
            project=Path(project_path),
            matlab_reference_path=Path(reference_path),
            settings={
                "phases": phases,
                "only_accepted": True,
                "anchor_selection_mode": anchor_selection_mode,
                "positional_isomer_workflow_mode": positional_isomer_workflow_mode,
                "anchor_ppm": anchor_ppm,
                "anchor_rt_seconds": anchor_rt_seconds,
                "anchor_score_margin": anchor_score_margin,
                "minimum_anchors": minimum_anchors,
                "use_grid_coherence": use_grid_coherence,
                "grid_coherence_threshold": grid_coherence_threshold,
                "grid_coherence_version": grid_coherence_version,
                "composition_ranges": composition_ranges.to_dict(),
                "use_pe_rebuild": use_pe_rebuild,
                "pe_rebuild_score_threshold": pe_rebuild_score_threshold,
                "use_experimental_pe_selector": use_experimental_pe_selector,
            },
            progress=_progress,
        )
    ).to_mapping()


def _project_signature(
    input_directory: Path,
    phases: tuple[str, ...],
) -> tuple:
    """Size and mtime of selected-phase XCMS inputs for cache invalidation."""
    sources = discover_standard_streams(input_directory)
    signature = [
        (name, path.stat().st_size, path.stat().st_mtime_ns)
        for name, path in sorted(sources.items())
        if any(name.startswith(f"{phase}_") for phase in phases)
    ]
    csv_signature = project_is_signature(input_directory)
    if csv_signature[0]:
        signature.append(("internal_standards.csv", *csv_signature))
    standard_candidates = [input_directory / "internal_standards.tsv"]
    standard_candidates.extend(
        path.parent / "internal_standards.tsv" for path in sources.values()
    )
    for path in standard_candidates:
        if path.is_file():
            signature.append(
                (
                    "internal_standards.tsv",
                    path.stat().st_size,
                    path.stat().st_mtime_ns,
                )
            )
            break
    return tuple(signature)


def _render_matlab_engine(default_project: str) -> None:
    """Inputs and run control for automatic XCMS-only MATLAB annotation."""
    render_matlab_engine(
        default_project=default_project,
        default_reference=Path(DEFAULT_MATLAB_PIPELINE_REFERENCE),
        cache_version=MATLAB_ENGINE_CACHE_VERSION,
        render_composition_ranges=_render_composition_range_settings,
        project_signature=_project_signature,
        run_cached=_run_matlab_cached,
    )


def render_standard_lipidomics_mode(default_reference: Path) -> None:
    """Render the standalone standard-lipidomics workflow."""
    if st.session_state.get("standard_pipeline_page") in {"annotation_job", "untargeted_job"}:
        _render_annotation_job()
        return
    if st.session_state.get("standard_annotation_result"):
        _render_standard_results()
        return
    annotation_project = render_standard_pipeline_entry(
        default_project=default_project(),
        back_action=lambda: None,
    )
    if annotation_project is None:
        return
    page = st.session_state.get(STANDARD_PIPELINE_PAGE_KEY)
    if page in {"open_annotation", "open_untargeted"}:
        _render_saved_workflow_open("annotation" if page == "open_annotation" else "untargeted")
        return
    if st.session_state.get(STANDARD_PIPELINE_PAGE_KEY) == "untargeted":
        _render_untargeted_input(annotation_project)
        return
    annotation_project = _render_annotation_input(annotation_project)
    if annotation_project is None:
        return
    st.radio("When an old project exists", ("Archive old project", "Overwrite old project"),
             horizontal=True, key="standard-annotation-existing")
    engine = st.radio(
        "Annotation engine",
        [MATLAB_ENGINE, LASER_ENGINE],
        horizontal=True,
        key="standard_annotation_engine",
        help=(
            "Automatic MATLAB engine (XCMS): reproduces the Amsterdam UMC MATLAB "
            "annotation logic using theoretical targets, calibrated RTDB splines and "
            "staged regression. LASER reference matcher: directly matches XCMS "
            "features using configured mass, RT, policy and optional anchored NP RT "
            "rules."
        ),
    )
    if engine == MATLAB_ENGINE:
        _render_matlab_engine(annotation_project)
    else:
        render_laser_engine(
            default_project=annotation_project,
            default_reference=default_reference,
            render_composition_ranges=_render_composition_range_settings,
            run_cached=_run_standard_cached,
        )
    _render_standard_results()


def _render_annotation_input(default_project: str) -> str | None:
    project = st.text_input("Project folder", value=default_project, key="standard-annotation-project")
    kind = st.radio("Annotation input", ("Raw XCMS data", "Calibrated XCMS data"),
                    horizontal=True, key="standard-annotation-input-kind")
    if not project.strip():
        st.info("Enter the project folder to detect its XCMS inputs.")
        return None
    root = xcms_project_root(project)
    running = active_annotation_job(root)
    if running is not None:
        st.info("Annotation is running for this project.")
        if st.button("View annotation progress", type="primary"):
            st.session_state.standard_annotation_job_path = str(running)
            st.session_state[STANDARD_PIPELINE_PAGE_KEY] = "annotation_job"
            st.rerun()
        return None
    try:
        if kind == "Raw XCMS data":
            sources = discover_standard_streams(root)
            if not sources:
                raise ValueError("No raw XCMS peak-group lists were found in the project input folder.")
            source = root
        else:
            choices = available_calibrations(root)
            if not choices:
                raise ValueError("No saved calibration was found. Run Calibration and save it first.")
            def label(name: str) -> str:
                path = choices[name]
                try:
                    manifest = json.loads((path / "LASER_standard_mass_calibration.json").read_text())
                    date = str(manifest.get("saved_at", ""))
                    streams = ", ".join(manifest.get("streams", {}))
                    return " · ".join(value for value in (name, date, streams) if value)
                except (OSError, ValueError):
                    return name
            selected = st.selectbox("Calibration version", list(choices), format_func=label,
                                    key=f"standard-calibration-choice-{root}")
            source = choices[selected]
            sources = discover_standard_streams(source)
            if not sources:
                raise ValueError("The selected calibration contains no XCMS peak-group lists.")
        project_is_definitions(source)
        st.session_state.standard_selected_input_directory = str(source)
        return str(source)
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return None


def _render_saved_workflow_open(workflow: str) -> None:
    project = st.text_input(
        "Project folder", value=default_project(),
        key="standard-open-project-folder",
    )
    open_annotation = workflow == "annotation"
    requested = st.button(
        "Open completed annotation" if open_annotation else "Open completed untargeted analysis",
        disabled=not bool(project.strip()), type="primary", width="stretch")
    if not requested:
        return
    try:
        root = xcms_project_root(project)
        if open_annotation:
            result, saved = load_annotation_workflow(project)
            page = "annotation"
            workflow = "annotation"
        else:
            result, saved = load_untargeted_workflow(project)
            page = "untargeted"
            workflow = "untargeted"
        st.session_state.standard_annotation_result = result
        st.session_state.standard_results_page = page
        st.session_state.standard_loaded_workflow = workflow
        st.session_state["standard-results-enable-untargeted"] = False
        st.session_state.standard_auto_save_annotation_pending = False
        st.session_state.standard_auto_save_untargeted_pending = False
        st.session_state.standard_project_directory = str(root)
        st.session_state.standard_headless_config = {
            "workflow": "standard", "engine": result.get("engine", "untargeted"),
            "inputs": dict(saved.get("inputs") or {"project": str(root)}),
            "annotation": dict(saved.get("settings") or {}),
        }
        try:
            source_folder = saved.get("inputs", {}).get("xcms_source") or saved.get(
                "inputs", {}
            ).get("project") or root
            st.session_state.standard_input_sources = discover_standard_streams(
                source_folder
            ) or discover_standard_streams(root)
        except ValueError:
            st.session_state.standard_input_sources = discover_standard_streams(root)
        if workflow == "annotation":
            # Reopen the exact selected input even if the current calibration
            # was subsequently overwritten or moved into history.
            snapshots = {}
            for stream in result.get("streams", {}):
                path = resolve_artifact_path(saved, f"annotated_input_{stream}")
                if path is not None and path.is_file():
                    snapshots[stream] = remove_standard_annotation_columns(path)
            if snapshots:
                st.session_state.standard_input_sources = snapshots
        if workflow == "untargeted":
            analysis = result["untargeted_analysis"]
            criteria = analysis.get("criteria", {})
            for key in (
                "standard_untargeted_p_value",
                "standard_untargeted_fold_change",
                "standard_untargeted_detection",
            ):
                st.session_state.pop(key, None)
            for field, key in (
                ("group_a", "standard_untargeted_group_a"),
                ("group_b", "standard_untargeted_group_b"),
            ):
                if field in criteria:
                    st.session_state[key] = criteria[field]
            st.session_state.standard_untargeted_analysis = analysis
        st.rerun()
    except (OSError, ValueError, KeyError) as exc:
        st.error(str(exc))


def _render_untargeted_input(default_project: str) -> None:
    project = st.text_input(
        "Project folder", value=default_project,
        key="standard-untargeted-input-project",
    )
    if not project.strip() or not Path(project).expanduser().is_dir():
        return
    try:
        if active_annotation_job(project):
            st.info("Wait for this project's annotation to finish before starting untargeted analysis.")
            return
        running = active_untargeted_job(project)
        if running:
            if st.button("View untargeted progress", type="primary"):
                st.session_state.standard_untargeted_job_path = str(running)
                st.session_state[STANDARD_PIPELINE_PAGE_KEY] = "untargeted_job"
                st.rerun()
            return
        root = xcms_project_root(project)
        manifest = root / "LASER" / "LASER_PROJECT.json"
        watched = [manifest, root / "LASER" / "Annotation" / "LASER_PROJECT.json"]
        watched.extend(discover_standard_streams(root).values())
        watched.extend((root / "metadata").glob("*.xls*"))
        signature = (str(root), tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns)
                                      for path in watched if path.is_file()))
        cached = st.session_state.get("standard_untargeted_launch_inputs")
        if cached is None or cached[0] != signature:
            cached = (signature, *load_untargeted_inputs(project))
            st.session_state.standard_untargeted_launch_inputs = cached
        _, result, provenance = cached
        if provenance.startswith("raw XCMS"):
            st.info("No completed annotation found. This run will include all XCMS features.")
        configured = _untargeted_criteria(result)
        if configured is None:
            return
        criteria, _ = configured
        existing = st.radio("When an old project exists", ("Archive old project", "Overwrite old project"),
            horizontal=True, key="standard-untargeted-launch-existing")
        if not st.button("Run untargeted analysis", type="primary", width="stretch"):
            return
        job = start_untargeted_job(project, criteria, result=result, provenance=provenance,
            existing="archive" if existing.startswith("Archive") else "overwrite")
        st.session_state.standard_untargeted_job_path = str(job)
        st.session_state[STANDARD_PIPELINE_PAGE_KEY] = "untargeted_job"
        st.rerun()
    except (OSError, ValueError) as exc:
        st.error(str(exc))


def _stream_summary(result: dict) -> pd.DataFrame:
    """One row per stream, describing what that engine decided."""
    matlab = result.get("engine") == "matlab"
    rows = []
    for name, stream_result in result["streams"].items():
        annotations = stream_result["annotations"]
        accepted = (
            int(annotations["automatic_accepted"].sum()) if not annotations.empty else 0
        )
        row = {
            "stream": name,
            "input_features": len(stream_result["features"]),
            "candidate_pairs_passing": len(annotations),
            "automatic_accepted": accepted,
        }
        if matlab:
            # MATLAB's mapper decides acceptance itself; a row that survives the
            # final regression stage is annotated, so there is no withheld tier
            # to report. What varies instead is how many peaks were reviewed.
            row["reviewed_candidates"] = len(stream_result["all_pairs"])
            row["RT evidence"] = "MATLAB RTDB spline"
            row["reference grid nodes"] = len(stream_result["reference"])
        else:
            row["ambiguous_or_withheld"] = (
                int(
                    annotations["rank_for_feature"].eq(0).sum()
                    - annotations["automatic_accepted"].sum()
                )
                if not annotations.empty
                else 0
            )
            row["RT evidence"] = (
                "guarded class RT"
                if name.startswith("RP_")
                else (
                    "anchored NP RT grid"
                    if not annotations.empty
                    and annotations["rt_evidence_mode"]
                    .astype(str)
                    .str.contains("anchored_NP")
                    .any()
                    else "m/z only; abstained"
                )
            )
            policy_audit = stream_result["policy_audit"]
            row["policy reference rows allowed"] = (
                int(policy_audit["policy_result"].eq("allowed").sum())
                if not policy_audit.empty
                else 0
            )
            row["policy reference rows rejected"] = (
                int(policy_audit["policy_result"].ne("allowed").sum())
                if not policy_audit.empty
                else 0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _render_matlab_audit(result: dict) -> None:
    """Per-definition outcome of the MATLAB sweep."""
    render_matlab_audit(result)


def _render_standard_grid_coherence_audit(result: dict) -> None:
    """Show the shared grid-coherence audit for the LASER matcher."""
    coherence = result.get("grid_coherence_audit", pd.DataFrame())
    if coherence is None or coherence.empty:
        return
    enabled = bool(result.get("grid_coherence_enabled", False))
    mode_label = "annotation filter enabled" if enabled else "diagnostic only"
    with st.expander(f"Grid coherence ({mode_label})", expanded=False):
        threshold = float(
            result.get(
                "grid_coherence_threshold",
                SCIENTIFIC_DEFAULTS.grid_coherence_threshold,
            )
        )
        filtered = int(
            pd.to_numeric(
                coherence.get("filtered_annotations", pd.Series(dtype=float)),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        st.caption(
            f"Minimum score {threshold:.2f}; assessed "
            f"{int(pd.to_numeric(coherence.get('assessed', pd.Series(dtype=float)), errors='coerce').fillna(0).sum())} "
            f"candidate(s). "
            + (
                f"The filter excluded {filtered} assessed outlier(s)."
                if enabled
                else "The score was calculated but did not exclude annotations."
            )
        )
        st.dataframe(coherence, hide_index=True, width="stretch")
        st.download_button(
            "Download grid-coherence audit CSV",
            coherence.to_csv(index=False).encode(),
            file_name="LASER_standard_grid_coherence_audit.csv",
            mime="text/csv",
            width="stretch",
        )


def _render_standard_exports(
    result: dict, *, show_annotation_table: bool = True
) -> None:
    """Compatibility wrapper around the focused export presenter."""
    _render_standard_exports_presenter(
        result,
        report_renderer=render_standard_lipidomics_report,
        append_annotations=append_standard_annotations,
        show_annotation_table=show_annotation_table,
    )


def _render_standard_results_menu(result: dict) -> None:
    """Landing page separating annotation review from untargeted follow-up."""
    if st.session_state.get("standard_loaded_workflow") == "untargeted":
        _standard_results_page_header("Untargeted results")
        if st.button("Open untargeted analysis", type="primary", width="stretch"):
            st.session_state.standard_results_page = "untargeted"
            st.rerun()
        if st.button("Start another workflow", width="stretch"):
            st.session_state.pop("standard_annotation_result", None)
            st.session_state.pop("standard_results_page", None)
            st.session_state.pop("standard_loaded_workflow", None)
            st.session_state[STANDARD_PIPELINE_PAGE_KEY] = "select"
            st.rerun()
        return
    _standard_results_page_header("Standard lipidomics results")
    identified = result.get("identified_lipids", pd.DataFrame())
    equivalents = result.get("equivalent_lipids", pd.DataFrame())
    candidates = result.get("all_candidates", pd.DataFrame())
    metrics = st.columns(4)
    metrics[0].metric("Input streams", len(result.get("streams", {})))
    metrics[1].metric("Primary lipids", len(identified))
    metrics[2].metric("Equivalent features", len(equivalents))
    metrics[3].metric("Candidate identities", len(candidates))
    st.caption("The annotation result is complete and can be saved or reviewed without rerunning it.")
    annotation_col, save_col = st.columns(2)
    if annotation_col.button(
        "Open annotation review",
        type="primary",
        width="stretch",
        key="standard-open-annotation-review",
    ):
        st.session_state.standard_results_page = "annotation"
        st.rerun()
    if save_col.button(
        "Save project",
        width="stretch",
        key="standard-open-save-project",
    ):
        st.session_state.standard_results_page = "save"
        st.rerun()
    if st.button(
        "Start a new annotation",
        width="stretch",
        key="standard-start-new-annotation",
    ):
        st.session_state.pop("standard_annotation_result", None)
        st.session_state.pop("standard_results_page", None)
        st.rerun()


def _render_standard_project_save(result: dict) -> None:
    """Save the completed in-memory result without hiding it in an expander."""
    _standard_results_page_header("Save LASER project")
    st.caption("Save the current annotation and remake its exports without rerunning annotation.")
    config = st.session_state.get("standard_headless_config", {})
    source_project = (
        st.session_state.get("standard_project_directory")
        or result.get("xcms_input_dir")
        or config.get("inputs", {}).get("project")
    )
    if source_project == "REPLACE_WITH_XCMS_PROJECT":
        source_project = None
    if not source_project:
        source_project = st.text_input(
            "XCMS project folder for LASER results",
            key="standard-project-save-folder",
        )
        if not source_project:
            st.info("Enter a project folder to save this annotation.")
            return
        st.session_state.standard_project_directory = source_project
    destination = workflow_folder(source_project, "Annotation")
    st.code(str(destination))
    existing = st.radio(
        "When an old project exists",
        ("Archive old project", "Overwrite old project"),
        horizontal=True, key="standard-annotation-existing",
    )
    include_html = st.checkbox(
        "Also create an HTML export",
        value=False,
        key="standard-project-include-html",
        help=(
            "Optional convenience export. The native LASER workspace already "
            "contains the same analytical sections."
        ),
    )
    if not st.button(
        "Save LASER project",
        type="primary",
        width="stretch",
        key="save-standard-laser-project",
    ):
        return
    try:
        paths, archive = _save_current_annotation(
            result, include_html=include_html,
            existing="archive" if existing.startswith("Archive") else "overwrite",
        )
        st.session_state.standard_saved_project = str(destination)
        st.success(f"Project saved: {destination}")
        if archive:
            st.caption(f"Previous export archived at {archive}")
        st.caption(f"Manifest: {paths['project_manifest']}")
    except Exception as exc:
        st.error(f"Project could not be saved: {exc}")


def _save_current_annotation(
    result: dict, *, include_html: bool, existing: str,
) -> tuple[dict[str, Path], Path | None]:
    config = st.session_state.get("standard_headless_config", {})
    source_project = st.session_state.get("standard_project_directory")
    if not source_project:
        raise ValueError("Choose an XCMS project folder before saving.")
    input_sources = st.session_state.get("standard_input_sources", {})
    if not input_sources:
        raise ValueError(
            "The original XCMS peak-group lists are unavailable; annotated "
            "peak lists cannot be regenerated. Restore them in the project folder."
        )
    # Keep original XCMS lists intact when annotation used calibrated input.
    raw_sources = discover_standard_streams(xcms_project_root(source_project))
    stage_project_metadata(source_project, raw_sources)
    input_settings = dict(config.get("inputs", {}))
    if input_settings.get("project") in (None, "REPLACE_WITH_XCMS_PROJECT"):
        input_settings["project"] = str(xcms_project_root(source_project))
    annotated_inputs = {
        stream: append_standard_annotations(
            source, result.get("automatic_annotations", pd.DataFrame()),
            stream=stream,
        )
        for stream, source in input_sources.items()
    }
    html_report = (
        render_standard_lipidomics_report(
            result, settings=config.get("annotation", {}),
            project=input_settings.get("project"),
        ) if include_html else None
    )
    return save_annotation_workflow(
        source_project, result,
        settings=config.get("annotation", {}),
        inputs=input_settings,
        annotated_inputs=annotated_inputs, html_report=html_report,
        existing=existing,
    )


def _render_untargeted_save(result: dict, analysis: dict) -> None:
    source_project = st.session_state.get("standard_project_directory")
    if not source_project:
        st.error("Choose an XCMS project folder before saving.")
        return
    destination = workflow_folder(source_project, "Untargeted")
    with st.container(border=True):
        st.subheader("Save untargeted analysis")
        st.code(str(destination))
        existing = st.radio(
            "When an old project exists",
            ("Archive old project", "Overwrite old project"),
            horizontal=True, key="standard-untargeted-existing",
        )
        if not st.button("Save untargeted analysis", type="primary", width="stretch"):
            return
        try:
            _save_current_untargeted(
                result, analysis,
                existing="archive" if existing.startswith("Archive") else "overwrite",
            )
            st.success(f"Untargeted analysis saved: {destination}")
        except Exception as exc:
            st.error(f"Untargeted analysis could not be saved: {exc}")


def _save_current_untargeted(
    result: dict, analysis: dict, *, existing: str,
) -> tuple[dict[str, Path], Path | None]:
    source_project = st.session_state.get("standard_project_directory")
    if not source_project:
        raise ValueError("Choose an XCMS project folder before saving.")
    stage_project_metadata(
        source_project, discover_standard_streams(xcms_project_root(source_project))
    )
    return save_untargeted_workflow(
        source_project, result, analysis,
        inputs={
            "project": str(xcms_project_root(source_project)),
            "xcms_source": st.session_state.get(
                "standard_headless_config", {}
            ).get("inputs", {}).get("xcms_source")
            or st.session_state.get("standard_headless_config", {}).get(
                "inputs", {}
            ).get("project"),
        },
        existing=existing,
    )


def _standard_results_page_header(title: str) -> None:
    if page_header(title, key=f"standard-back-to-start-{title}"):
        st.session_state.pop("standard_annotation_result", None)
        st.session_state.pop("standard_results_page", None)
        st.session_state[STANDARD_PIPELINE_PAGE_KEY] = "select"
        st.rerun()


def _render_standard_annotation_review_body(result: dict) -> None:
    """Render only the live MS/grid viewer."""
    render_standard_grid_browser(result)


def _render_standard_annotation_audit_body(result: dict) -> None:
    """Render the run summary and collapsible scientific audit tables."""
    with st.expander("Annotation run summary", expanded=False):
        identified = result["identified_lipids"]
        equivalents = result["equivalent_lipids"]
        candidates = result["all_candidates"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Input streams", len(result["streams"]))
        c2.metric("Automatic primary lipids", len(identified))
        c3.metric("Automatic equivalents", len(equivalents))
        c4.metric("All candidate identities", len(candidates))
        st.dataframe(_stream_summary(result), hide_index=True, width="stretch")
    if result.get("engine") == "matlab":
        _render_matlab_audit(result)
    else:
        _render_standard_grid_coherence_audit(result)


def _render_standard_annotation_workspace(result: dict) -> None:
    """Show only the annotation reviewer; no report tabs."""
    _restore_group_metadata_from_inputs(result)
    _render_standard_annotation_review_body(result)


def _render_standard_results() -> None:
    """Render whichever engine's result is in session state."""
    result = st.session_state.get("standard_annotation_result")
    if not result:
        return
    if st.session_state.get("standard_auto_save_annotation_pending"):
        st.session_state.standard_auto_save_annotation_pending = False
        try:
            paths, archive = _save_current_annotation(
                result, include_html=False, existing="archive"
            )
            st.session_state.standard_saved_project = str(paths["project_manifest"].parent)
            if archive:
                st.session_state.standard_auto_save_notice = (
                    f"Annotation saved; previous export archived at {archive}."
                )
            else:
                st.session_state.standard_auto_save_notice = (
                    f"Annotation saved in {paths['project_manifest'].parent}."
                )
        except Exception as exc:
            st.session_state.standard_auto_save_notice = (
                f"Automatic annotation save failed: {exc}. Use Save annotation to retry."
            )
    notice = st.session_state.pop("standard_auto_save_notice", None)
    if notice:
        if notice.startswith("Automatic"):
            st.warning(notice)
        else:
            st.success(notice)
    standalone = st.session_state.get("standard_loaded_workflow") == "untargeted"
    default_page = "untargeted" if standalone else "annotation" if result.get("benchmark_context") else "menu"
    page = st.session_state.setdefault("standard_results_page", default_page)
    if page == "menu":
        _render_standard_results_menu(result)
        return
    if page == "untargeted":
        _standard_results_page_header("Untargeted analysis")
        untargeted = render_untargeted_workspace(result)
        if untargeted:
            if st.session_state.get("standard_auto_save_untargeted_pending"):
                st.session_state.standard_auto_save_untargeted_pending = False
                try:
                    _, archive = _save_current_untargeted(
                        result, untargeted, existing="archive",
                    )
                    st.success(
                        "Untargeted analysis saved in "
                        f"{workflow_folder(st.session_state.standard_project_directory, 'Untargeted')}."
                    )
                    if archive:
                        st.caption(f"Previous export archived at {archive}")
                except Exception as exc:
                    st.warning(f"Automatic untargeted save failed: {exc}. Use Save untargeted analysis to retry.")
            _render_untargeted_save(result, untargeted)
            if not standalone:
                with st.expander("Reports and exports", expanded=False):
                    _render_standard_exports(result, show_annotation_table=False)
        return
    if page == "save":
        _render_standard_project_save(result)
        return
    _standard_results_page_header("Annotation review")
    if st.button("Save annotation", key="standard-save-from-annotation"):
        st.session_state.standard_results_page = "save"
        st.rerun()
    _render_standard_annotation_workspace(result)


@st.fragment(run_every="2s")
def _render_annotation_job() -> None:
    untargeted = st.session_state.get(STANDARD_PIPELINE_PAGE_KEY) == "untargeted_job"
    title = "Untargeted analysis" if untargeted else "Annotation"
    path_key = "standard_untargeted_job_path" if untargeted else "standard_annotation_job_path"
    if page_header(title, key="standard-job-back-to-start"):
        st.session_state[STANDARD_PIPELINE_PAGE_KEY] = "select"
        st.session_state.pop(path_key, None)
        st.rerun()
    directory = st.session_state.get(path_key)
    if not directory:
        st.error("No run was selected.")
        return
    try:
        state = read_annotation_job(directory)
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return
    if state["status"] == "complete":
        st.success(f"{title} complete")
    elif state["status"] == "failed":
        st.error(f"{title} failed: {state['message']}")
    else:
        st.progress(float(state.get("progress", 0.0)), text=state.get("message", "Running annotation"))
    render_job_terminal(directory, state, "untargeted.log" if untargeted else "annotation.log")
    if state["status"] in {"queued", "running"}:
        st.info(f"{title} continues if you close the browser.")
