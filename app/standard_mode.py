"""Streamlit presentation layer for conventional XCMS lipidomics."""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import streamlit as st

from annotator.composition_ranges import CompositionRangeSettings
from annotator.standard_annotated_inputs import append_standard_annotations
from standard_lipidomics.workflow import (
    discover_standard_streams,
)
from matlab_annotation.configuration import (
    DEFAULT_MATLAB_PIPELINE_REFERENCE,
)
from standard_lipidomics.grid_browser import render_standard_grid_browser
from standard_lipidomics.report import render_standard_lipidomics_report
from scientific_defaults import SCIENTIFIC_DEFAULTS
from standard_lipidomics.project_store import (
    load_annotation_workflow,
    load_untargeted_workflow,
    save_annotation_workflow,
    save_untargeted_workflow,
    stage_project_metadata,
    workflow_folder,
    xcms_project_root,
)
from standard_lipidomics.workflow import load_standard_feature_list
from application.standard_workflow import (
    LASER_STANDARD_ENGINE,
    MATLAB_STANDARD_ENGINE,
    StandardAnnotationRequest,
)
from composition import standard_annotation_service
from presentation.standard_workspace import (
    render_standard_annotation_workspace,
    streamlit_safe_report_frame as _streamlit_safe_report_frame,  # noqa: F401
)
from presentation.standard_setup import render_laser_engine, render_matlab_engine
from presentation.standard_calibration import render_standard_pipeline_entry
from presentation.standard_calibration import (
    STANDARD_PIPELINE_PAGE_KEY,
    selected_standard_stages,
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
)

LASER_ENGINE = "LASER reference matcher"
MATLAB_ENGINE = "Automatic MATLAB engine (XCMS)"
MATLAB_ENGINE_CACHE_VERSION = 28
DEFAULT_MATLAB_XCMS_PROJECT = os.environ.get(
    "LASER_STANDARD_PROJECT",
    "",
)


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
) -> dict:
    return standard_annotation_service.run(
        StandardAnnotationRequest(
            engine=LASER_STANDARD_ENGINE,
            project=Path(policy_path).expanduser().parent,
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
    if st.session_state.get("standard_annotation_result"):
        _render_standard_results()
        return
    annotation_project = render_standard_pipeline_entry(
        default_project=DEFAULT_MATLAB_XCMS_PROJECT,
        back_action=lambda: None,
    )
    if annotation_project is None:
        if st.session_state.get(STANDARD_PIPELINE_PAGE_KEY) == "select":
            _render_saved_workflow_open()
        return
    if st.session_state.get(STANDARD_PIPELINE_PAGE_KEY) == "untargeted":
        _render_untargeted_input(annotation_project)
        return
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


def _render_saved_workflow_open() -> None:
    st.divider()
    st.subheader("Open saved LASER results")
    project = st.text_input(
        "XCMS project folder",
        value=DEFAULT_MATLAB_XCMS_PROJECT,
        key="standard-open-project-folder",
    )
    annotation_col, untargeted_col = st.columns(2)
    open_annotation = annotation_col.button(
        "Open saved annotation", disabled=not bool(project), width="stretch"
    )
    open_untargeted = untargeted_col.button(
        "Open saved untargeted analysis", disabled=not bool(project), width="stretch"
    )
    if not (open_annotation or open_untargeted):
        return
    try:
        root = xcms_project_root(project)
        if open_annotation:
            result, saved = load_annotation_workflow(root)
            page = "menu"
            workflow = "annotation"
        else:
            result, saved = load_untargeted_workflow(root)
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
        "XCMS project folder", value=default_project,
        key="standard-untargeted-input-project",
    )
    if not st.button("Load XCMS features for untargeted analysis", type="primary"):
        return
    try:
        sources = discover_standard_streams(project)
        if not sources:
            raise ValueError("No XCMS peak-group lists were found in this project.")
        streams = {
            name: {"features": load_standard_feature_list(path)}
            for name, path in sources.items()
        }
        result = {
            "engine": "untargeted", "streams": streams,
            "automatic_annotations": pd.DataFrame(),
            "xcms_input_dir": str(xcms_project_root(project)),
        }
        st.session_state.standard_annotation_result = result
        st.session_state.standard_results_page = "untargeted"
        st.session_state.standard_loaded_workflow = "untargeted"
        st.session_state.standard_auto_save_annotation_pending = False
        st.session_state.standard_auto_save_untargeted_pending = True
        st.session_state.standard_project_directory = str(xcms_project_root(project))
        st.session_state.standard_input_sources = sources
        st.session_state.standard_headless_config = {
            "workflow": "standard", "engine": "untargeted",
            "inputs": {"project": str(xcms_project_root(project)),
                       "xcms_source": str(Path(project).expanduser().resolve())},
            "annotation": {},
        }
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
        st.header("Untargeted results")
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
    st.header("Standard lipidomics results")
    identified = result.get("identified_lipids", pd.DataFrame())
    equivalents = result.get("equivalent_lipids", pd.DataFrame())
    candidates = result.get("all_candidates", pd.DataFrame())
    metrics = st.columns(4)
    metrics[0].metric("Input streams", len(result.get("streams", {})))
    metrics[1].metric("Primary lipids", len(identified))
    metrics[2].metric("Equivalent features", len(equivalents))
    metrics[3].metric("Candidate identities", len(candidates))
    st.caption("The annotation result is complete and can be saved or reviewed without rerunning it.")
    annotation_col, save_col, untargeted_col = st.columns(3)
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
    enable_untargeted = st.checkbox(
        "Enable untargeted analysis",
        value=selected_standard_stages().run_untargeted,
        key="standard-results-enable-untargeted",
        help="Off by default. Uses the XCMS features already in this result.",
    )
    if enable_untargeted and not result.get("untargeted_analysis"):
        st.session_state.standard_auto_save_untargeted_pending = True
    if untargeted_col.button(
        "Open untargeted analysis",
        width="stretch",
        key="standard-open-untargeted-analysis",
        disabled=not enable_untargeted,
    ):
        st.session_state.standard_results_page = "untargeted"
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
        "When an export already exists",
        ("Archive previous export", "Overwrite previous export"),
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
    stage_project_metadata(source_project, input_sources)
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
    with st.expander("Save untargeted analysis", expanded=True):
        st.caption("Remake the untargeted exports from the current result and review decisions.")
        st.code(str(destination))
        existing = st.radio(
            "When an export already exists",
            ("Archive previous export", "Overwrite previous export"),
            horizontal=True, key="standard-untargeted-existing",
        )
        if not st.button("Save untargeted analysis", type="primary", width="stretch"):
            return
        try:
            paths, archive = _save_current_untargeted(
                result, analysis,
                existing="archive" if existing.startswith("Archive") else "overwrite",
            )
            st.success(f"Untargeted analysis saved: {destination}")
            if archive:
                st.caption(f"Previous export archived at {archive}")
            st.caption(f"Manifest: {paths['project_manifest']}")
        except Exception as exc:
            st.error(f"Untargeted analysis could not be saved: {exc}")


def _save_current_untargeted(
    result: dict, analysis: dict, *, existing: str,
) -> tuple[dict[str, Path], Path | None]:
    source_project = st.session_state.get("standard_project_directory")
    if not source_project:
        raise ValueError("Choose an XCMS project folder before saving.")
    stage_project_metadata(
        source_project, st.session_state.get("standard_input_sources", {})
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
    heading_col, back_col = st.columns([5, 1])
    heading_col.header(title)
    if back_col.button(
        "Back to results",
        width="stretch",
        key=f"standard-back-to-results-{title}",
    ):
        st.session_state.standard_results_page = "menu"
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
    """Native, immediately available counterpart of the standard HTML report."""
    config = st.session_state.get("standard_headless_config", {})
    render_standard_annotation_workspace(
        result,
        config=config,
        render_annotation=lambda: _render_standard_annotation_review_body(result),
        render_audit=lambda: _render_standard_annotation_audit_body(result),
        render_exports=lambda: _render_standard_exports(
            result,
            show_annotation_table=False,
        ),
    )


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
