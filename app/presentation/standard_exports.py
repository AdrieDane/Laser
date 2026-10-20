"""Download controls for completed standard-lipidomics runs."""

from __future__ import annotations

import json
from typing import Callable

import pandas as pd
import streamlit as st


def _report_signature(result: dict, config: dict) -> tuple:
    identified = result["identified_lipids"]
    candidates = result["all_candidates"]
    return (
        str(result.get("engine", "laser")), len(identified),
        len(result["equivalent_lipids"]), len(candidates),
        str(identified.iloc[0].get("feature_identity", identified.iloc[0].get("name", ""))) if not identified.empty else "",
        str(identified.iloc[-1].get("feature_identity", identified.iloc[-1].get("name", ""))) if not identified.empty else "",
        json.dumps(config.get("annotation", {}), sort_keys=True),
        len(result.get("untargeted_analysis", {}).get("significant_features", pd.DataFrame())),
        json.dumps(result.get("untargeted_review_basket", pd.DataFrame()).to_dict(orient="records"), sort_keys=True, default=str),
    )


def _render_input_copies(result: dict, append_annotations: Callable) -> None:
    sources = st.session_state.get("standard_input_sources", {})
    if not sources:
        return
    with st.expander("Annotated copies of the input peak-group lists", expanded=False):
        st.caption("Original rows, columns, ordering and numeric text are retained. Only laser_annotation and adduct are inserted as columns 2 and 3. Multiple accepted identities for one feature are semicolon-separated.")
        columns = st.columns(min(len(sources), 4))
        for index, (stream, source) in enumerate(sources.items()):
            columns[index % len(columns)].download_button(
                f"Download {stream}",
                append_annotations(source, result.get("automatic_annotations", pd.DataFrame()), stream=stream),
                file_name=f"peakgrouplist_{stream}_annotated.txt",
                mime="text/tab-separated-values", width="stretch",
            )


def render_standard_exports(
    result: dict, *, report_renderer: Callable, append_annotations: Callable,
    show_annotation_table: bool = True,
) -> None:
    """Render standard workflow tables, reports, and reproducibility files."""
    identified, equivalents = result["identified_lipids"], result["equivalent_lipids"]
    automatic, candidates = result["automatic_annotations"], result["all_candidates"]
    columns = st.columns(3)
    downloads = (
        ("Download identified lipids CSV", identified, "LASER_standard_identified_lipids.csv"),
        ("Download equivalent lipids CSV", equivalents, "LASER_standard_equivalent_lipids.csv"),
        ("Download all candidate identities CSV", candidates, "LASER_standard_all_candidates.csv"),
    )
    for column, (label, frame, filename) in zip(columns, downloads):
        column.download_button(label, frame.to_csv(index=False).encode(), file_name=filename, mime="text/csv", width="stretch")
    config = st.session_state.get("standard_headless_config", {})
    signature = _report_signature(result, config)
    if st.session_state.get("standard_report_signature") != signature:
        inputs = config.get("inputs", {})
        st.session_state.standard_report_html = report_renderer(result, settings=config.get("annotation", {}), project=inputs.get("project") or result.get("xcms_input_dir"))
        st.session_state.standard_report_signature = signature
    st.download_button("Download optional HTML export", st.session_state.standard_report_html.encode("utf-8"), file_name="LASER_standard_lipidomics_report.html", mime="text/html", width="stretch", help="Convenience export only. Use Save LASER project on the results screen for a result that can be reopened inside LASER.")
    _render_input_copies(result, append_annotations)
    if config:
        st.download_button("Download reproducible headless configuration (JSON)", json.dumps(config, indent=2).encode(), file_name="laser_standard_workflow.json", mime="application/json", help="Run with windows_tools\\LASER CLI.bat run laser_standard_workflow.json.", width="stretch")
    if not show_annotation_table:
        return
    if automatic.empty:
        st.warning("No candidates met the automatic acceptance safeguards. Review the all-candidates export rather than treating mass-only matches as identifications.")
    else:
        st.dataframe(automatic.head(1000), hide_index=True, width="stretch")
