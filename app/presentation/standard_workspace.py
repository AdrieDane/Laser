"""Native standard-lipidomics report workspace presentation.

The standard-mode page owns navigation and user input. This presenter owns a
context-driven registry of completed-project tabs, so an extension can insert
a section without changing the workflow controller or this renderer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from presentation.workspace import (
    WorkspaceSection,
    WorkspaceSectionRegistry,
    render_workspace_sections,
)
from standard_lipidomics.report import build_standard_report_data


FrameDefinitions = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class StandardWorkspaceContext:
    """All data and page callbacks required by registered standard tabs."""

    result: Mapping[str, Any]
    report: Mapping[str, Any]
    config: Mapping[str, Any]
    render_annotation: Callable[[], None]
    render_audit: Callable[[], None]
    render_exports: Callable[[], None]


def streamlit_safe_report_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Serialize nested report cells for Arrow without changing report data."""
    safe = frame.copy()

    def display_value(value: object) -> str:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(value, default=str, ensure_ascii=False)
        if value is None:
            return ""
        return str(value)

    for column in safe.columns:
        if (
            pd.api.types.is_object_dtype(safe[column].dtype)
            or pd.api.types.is_string_dtype(safe[column].dtype)
        ):
            safe[column] = safe[column].map(display_value)
    return safe


def render_named_report_frames(
    source: Mapping[str, Any], definitions: FrameDefinitions,
) -> None:
    rendered = False
    for key, title in definitions:
        frame = source.get(key, pd.DataFrame())
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            rendered = True
            st.markdown(f"**{title}**")
            st.dataframe(
                streamlit_safe_report_frame(frame),
                hide_index=True,
                width="stretch",
                height=360,
            )
    if not rendered:
        st.caption("No data were produced for this section.")


def _annotation(context: StandardWorkspaceContext) -> None:
    context.render_annotation()


def _audit(context: StandardWorkspaceContext) -> None:
    context.render_audit()


def _exports(context: StandardWorkspaceContext) -> None:
    context.render_exports()


def _summary(context: StandardWorkspaceContext) -> None:
    report = context.report
    metrics = st.columns(4)
    metrics[0].metric("Input XCMS features", report["input_features"])
    metrics[1].metric("Primary lipid identities", len(report["selected"]))
    metrics[2].metric("Equivalent features", len(report["equivalents"]))
    metrics[3].metric("Review candidates", len(report["review"]))
    class_summary = report["class_summary"]
    if not class_summary.empty:
        st.bar_chart(
            class_summary.set_index("lipid_class")[[
                "unique_primary_identities", "equivalent_features",
            ]],
            x_label="Lipid class",
            y_label="Unique identities / features",
        )
    st.markdown("**Counts by chromatography/polarity stream**")
    st.dataframe(report["stream_summary"], hide_index=True, width="stretch")
    st.info(
        "Standard lipidomics provides XCMS coordinates, intensities and "
        "annotation/grid evidence. It does not invent CCS, mobility, raw "
        "chromatograms or feature-linked MS/MS that are absent from the inputs."
    )


def _annotation_table(
    key: str, caption: str,
) -> Callable[[StandardWorkspaceContext], None]:
    def render(context: StandardWorkspaceContext) -> None:
        st.caption(caption)
        st.dataframe(
            streamlit_safe_report_frame(context.report[key]),
            hide_index=True,
            width="stretch",
            height=720,
        )

    return render


def _untargeted(context: StandardWorkspaceContext) -> None:
    st.caption(
        "Untargeted results appear here after they have been calculated; "
        "accepted annotation features are excluded before testing."
    )
    render_named_report_frames(context.report, (
        ("untargeted_criteria", "Screening criteria"),
        ("untargeted_differences", "Significant unannotated features"),
        ("untargeted_review", "Feature review basket"),
        ("untargeted_reacquisition", "Targeted reacquisition list"),
        ("untargeted_groups", "Comparison groups"),
        ("untargeted_metadata_audit", "Sample metadata join audit"),
    ))


def _distributions(context: StandardWorkspaceContext) -> None:
    st.caption(
        "Distributions describe retained annotation evidence and XCMS group "
        "metadata, overall and per lipid class."
    )
    st.dataframe(
        context.report["metric_summary"],
        hide_index=True,
        width="stretch",
        height=720,
    )


def _rt_grid(context: StandardWorkspaceContext) -> None:
    render_named_report_frames(context.result, (
        ("matlab_seed_audit", "Automatic anchor selection"),
        ("matlab_reference_seed_audit", "Independent anchor evidence"),
        ("matlab_automatic_anchors", "Accepted automatic anchors"),
        ("matlab_internal_standard_audit", "Internal-standard recovery"),
        ("matlab_grid_coherence_audit", "Grid coherence"),
        ("matlab_post_regression_audit", "Post-regression confidence completion"),
        ("matlab_fitted_grid_audit", "Fitted-grid completion and feature selection"),
        ("grid_coherence_audit", "Grid-coherence audit"),
        ("matlab_positional_isomer_audit", "RP positional-isomer resolution"),
        ("matlab_positional_refit_audit", "Experimental positional RT refit"),
    ))
    for stream_name, stream_result in context.result.get("streams", {}).items():
        for key, title in (
            ("rt_models", "Class RT models"),
            ("np_rt_alignment", "NP anchor alignment"),
        ):
            frame = stream_result.get(key, pd.DataFrame())
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                st.markdown(f"**{stream_name} · {title}**")
                st.dataframe(frame, hide_index=True, width="stretch")


def _policy(context: StandardWorkspaceContext) -> None:
    render_named_report_frames(context.result, (
        ("policy_audit", "MATLAB lipid-policy decisions"),
        ("np_rt_audit", "NP reference identity coverage"),
        ("matlab_compound_audit", "MATLAB lipid-definition audit"),
        ("composition_range_audit", "Carbon/double-bond range decisions"),
    ))
    composition = context.result.get("composition_range_settings")
    if composition:
        st.markdown("**Carbon and double-bond limits**")
        st.json(composition, expanded=False)


def _inputs(context: StandardWorkspaceContext) -> None:
    st.metric("Annotation engine", context.report["engine"])
    st.caption(f"Project: {context.report['project']}")
    st.dataframe(context.report["settings"], hide_index=True, width="stretch")
    with st.expander("Complete reproducible configuration", expanded=False):
        st.json(dict(context.config), expanded=True)


def _availability(context: StandardWorkspaceContext) -> None:
    st.dataframe(
        context.report["availability"], hide_index=True, width="stretch"
    )


STANDARD_WORKSPACE_SECTIONS = WorkspaceSectionRegistry((
    WorkspaceSection("annotation", "Annotation review", _annotation),
    WorkspaceSection("audit", "Annotation audit", _audit),
    WorkspaceSection("summary", "Summary", _summary),
    WorkspaceSection(
        "selected",
        "Selected annotations",
        _annotation_table(
            "selected", "One final primary feature per lipid identity."
        ),
    ),
    WorkspaceSection(
        "equivalents",
        "Equivalent features",
        _annotation_table(
            "equivalents",
            "Additional accepted feature matches remain explicit and do not "
            "inflate the unique primary-identity count.",
        ),
    ),
    WorkspaceSection(
        "review",
        "Review candidates",
        _annotation_table(
            "review",
            "Candidates not selected as a primary or equivalent annotation.",
        ),
    ),
    WorkspaceSection("untargeted", "Untargeted", _untargeted),
    WorkspaceSection(
        "distributions", "Evidence distributions", _distributions,
    ),
    WorkspaceSection("rt-grid", "RT, anchors & grid", _rt_grid),
    WorkspaceSection("policy", "Policy & limits", _policy),
    WorkspaceSection("inputs", "Inputs & settings", _inputs),
    WorkspaceSection(
        "availability", "Evidence availability", _availability,
    ),
    WorkspaceSection("exports", "Exports", _exports),
))


def render_standard_annotation_workspace(
    result: dict[str, Any],
    *,
    config: Mapping[str, Any],
    render_annotation: Callable[[], None],
    render_audit: Callable[[], None],
    render_exports: Callable[[], None],
    sections: WorkspaceSectionRegistry[StandardWorkspaceContext] = (
        STANDARD_WORKSPACE_SECTIONS
    ),
) -> None:
    """Render the native counterpart of the optional standard HTML report."""
    report = build_standard_report_data(
        result,
        settings=config.get("annotation", {}),
        project=config.get("inputs", {}).get("project"),
    )
    context = StandardWorkspaceContext(
        result=result,
        report=report,
        config=config,
        render_annotation=render_annotation,
        render_audit=render_audit,
        render_exports=render_exports,
    )
    render_workspace_sections(sections, context)


__all__ = [
    "STANDARD_WORKSPACE_SECTIONS",
    "StandardWorkspaceContext",
    "render_named_report_frames",
    "render_standard_annotation_workspace",
    "streamlit_safe_report_frame",
]
