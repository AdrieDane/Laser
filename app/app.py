"""Standalone Streamlit entry point for LASER Standard Lipidomics."""

from pathlib import Path

import streamlit as st

from resource_paths import REFERENCE_DATA_DIRECTORY
from standard_mode import render_standard_lipidomics_mode


st.set_page_config(
    page_title="LASER Standard Lipidomics",
    page_icon="🧪",
    layout="wide",
)


def main() -> None:
    """Render the standard-lipidomics workflow directly."""
    render_standard_lipidomics_mode(
        Path(REFERENCE_DATA_DIRECTORY) / "4d_lipid_reference.csv"
    )


main()
