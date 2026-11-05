"""Standalone Streamlit entry point for LASER Standard Lipidomics."""

from pathlib import Path

import streamlit as st

from resource_paths import REFERENCE_DATA_DIRECTORY
from standard_mode import render_standard_lipidomics_mode
from presentation.app_theme import apply_app_theme


st.set_page_config(
    page_title="LASER Standard Lipidomics",
    page_icon="🧪",
    layout="wide",
)


def main() -> None:
    """Render the standard-lipidomics workflow directly."""
    apply_app_theme(
        show_selector=st.session_state.get("standard_pipeline_page", "select") == "select",
        viewer=bool(st.session_state.get("standard_annotation_result")),
    )
    render_standard_lipidomics_mode(
        Path(REFERENCE_DATA_DIRECTORY) / "4d_lipid_reference.csv"
    )


main()
