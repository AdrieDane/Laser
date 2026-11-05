"""Streamlit controls for the standard-lipidomics grid reviewer."""

from __future__ import annotations

import numpy as np
import streamlit as st
from presentation.standard_grid.dimensions import DIMENSIONS


CONTROLS_STATE = "standard_review_controls_visible"
BENCHMARK_SHOW_UNBENCHMARKED_STATE = (
    "standard_benchmark_show_unbenchmarked_candidates"
)


def _set_visible(visible: bool) -> None:
    st.session_state[CONTROLS_STATE] = bool(visible)


def review_criteria_signature(
    annotation_run_key: tuple,
    run_ppm_tolerance: float,
    run_rt_tolerance: float,
    run_use_grid_coherence: bool,
    run_grid_coherence_threshold: float,
) -> tuple:
    return (
        *annotation_run_key,
        float(run_ppm_tolerance),
        float(run_rt_tolerance),
        bool(run_use_grid_coherence),
        float(run_grid_coherence_threshold),
    )


def _reset(
    run_ppm_tolerance: float,
    run_rt_tolerance: float,
    run_use_grid_coherence: bool,
    run_grid_coherence_threshold: float,
) -> None:
    st.session_state.standard_viewer_ppm_tol = float(run_ppm_tolerance)
    st.session_state.standard_viewer_rt_tol = float(run_rt_tolerance)
    st.session_state.standard_viewer_use_grid_coherence = bool(
        run_use_grid_coherence
    )
    st.session_state.standard_viewer_grid_coherence_threshold = float(
        run_grid_coherence_threshold
    )


def _initialise_run_values(
    annotation_run_key: tuple,
    run_ppm_tolerance: float,
    run_rt_tolerance: float,
    run_use_grid_coherence: bool,
    run_grid_coherence_threshold: float,
) -> tuple[float, float]:
    ppm_maximum = max(10.0, float(run_ppm_tolerance) * 4.0)
    rt_maximum = max(5.0, float(run_rt_tolerance) * 4)
    signature = review_criteria_signature(
        annotation_run_key,
        run_ppm_tolerance,
        run_rt_tolerance,
        run_use_grid_coherence,
        run_grid_coherence_threshold,
    )
    if st.session_state.get("standard_viewer_run_criteria_signature") != signature:
        _reset(
            run_ppm_tolerance,
            run_rt_tolerance,
            run_use_grid_coherence,
            run_grid_coherence_threshold,
        )
        st.session_state.standard_viewer_run_criteria_signature = signature
    st.session_state.standard_viewer_ppm_tol = float(np.clip(
        st.session_state.get("standard_viewer_ppm_tol", ppm_maximum),
        0.1,
        ppm_maximum,
    ))
    st.session_state.standard_viewer_rt_tol = float(np.clip(
        st.session_state.get("standard_viewer_rt_tol", run_rt_tolerance),
        0.01,
        rt_maximum,
    ))
    st.session_state.standard_viewer_grid_coherence_threshold = float(np.clip(
        st.session_state.get(
            "standard_viewer_grid_coherence_threshold",
            run_grid_coherence_threshold,
        ),
        0.0,
        1.0,
    ))
    return ppm_maximum, rt_maximum


def _render_criteria(
    *,
    annotation_run_key: tuple,
    run_ppm_tolerance: float,
    run_rt_tolerance: float,
    uses_rt_grid: bool,
    run_use_grid_coherence: bool,
    run_grid_coherence_threshold: float,
) -> tuple[float, float, bool, float]:
    ppm_maximum, rt_maximum = _initialise_run_values(
        annotation_run_key,
        run_ppm_tolerance,
        run_rt_tolerance,
        run_use_grid_coherence,
        run_grid_coherence_threshold,
    )
    ppm = st.slider(
        "m/z tolerance (ppm)", 0.1, ppm_maximum, step=0.1,
        key="standard_viewer_ppm_tol",
        help=(
            f"The run opened at ±{run_ppm_tolerance:g} ppm. Tightening hides "
            "weaker matches; widening may reveal light-blue review candidates."
            " Accepted PE from the expanded method uses its independently corrected "
            "mass error; the raw error remains in the details."
        ),
    )
    rt = st.slider(
        "RT tolerance (min)", 0.01, rt_maximum, step=0.01,
        key="standard_viewer_rt_tol", disabled=not uses_rt_grid,
        help=(
            "Deviation from the anchored grid (the independent local RT model for "
            "accepted PE from the expanded method). This is a view-only window."
            if uses_rt_grid else
            "This stream was annotated on mass alone, so no RT window applies."
        ),
    )
    use_grid = st.checkbox(
        "Use grid coherence", key="standard_viewer_use_grid_coherence",
        help=(
            "Assessed candidates must meet the displayed minimum. Unassessed "
            "edge nodes remain visible; changing this never changes exports."
        ),
    )
    threshold = st.number_input(
        "Minimum grid coherence", 0.0, 1.0, step=0.05, format="%.2f",
        key="standard_viewer_grid_coherence_threshold", disabled=not use_grid,
        help=(
            f"The annotation run used {run_grid_coherence_threshold:.2f}; this "
            "control only changes the reviewer."
        ),
    )
    st.button(
        "Reset to run criteria", width="stretch",
        key="standard-reset-review-tolerances", on_click=_reset,
        args=(
            run_ppm_tolerance,
            run_rt_tolerance,
            run_use_grid_coherence,
            run_grid_coherence_threshold,
        ),
    )
    return float(ppm), float(rt), bool(use_grid), float(threshold)


def render_review_controls(
    *,
    annotation_run_key: tuple,
    run_ppm_tolerance: float,
    run_rt_tolerance: float,
    uses_rt_grid: bool,
    run_use_grid_coherence: bool,
    run_grid_coherence_threshold: float,
    benchmark_mode: bool = False,
) -> dict:
    """Render view, criteria, and grid controls and return their values."""
    visible = bool(st.session_state.setdefault(CONTROLS_STATE, True))
    if not visible:
        st.button("›", help="Show review controls", on_click=_set_visible, args=(True,))
    panel_key = "standard-review-controls-panel" + ("" if visible else "-hidden")
    with st.container(border=True, key=panel_key):
        if visible:
            st.button("‹", help="Hide review controls", on_click=_set_visible, args=(False,))
        st.markdown("**View**")
        x_dimension = st.selectbox(
            "X dimension", DIMENSIONS, index=0,
            key="standard_viewer_x_dimension",
        )
        y_dimension = st.selectbox("Y dimension", DIMENSIONS, index=1,
            key="standard_viewer_y_dimension")
        st.divider()
        st.markdown("**Live review criteria**")
        st.caption(
            "These open at the annotation-run values. Changes affect only this view."
        )
        ppm, rt, use_grid, threshold = _render_criteria(
            annotation_run_key=annotation_run_key,
            run_ppm_tolerance=run_ppm_tolerance,
            run_rt_tolerance=run_rt_tolerance,
            uses_rt_grid=uses_rt_grid,
            run_use_grid_coherence=run_use_grid_coherence,
            run_grid_coherence_threshold=run_grid_coherence_threshold,
        )
        st.divider()
        st.markdown("**Grid**")
        show_mesh = st.checkbox("show predicted grid", value=True, key="standard_show_mesh")
        show_labels = st.checkbox("show C:DB labels", value=True, key="standard_show_labels")
        show_cloud = st.checkbox("show grey points", value=False, key="standard_show_cloud")
        show_unbenchmarked = True
        if benchmark_mode:
            show_unbenchmarked = st.checkbox(
                "show unbenchmarked blue candidates", value=True,
                key=BENCHMARK_SHOW_UNBENCHMARKED_STATE,
            )
    return {
        "x_dimension": x_dimension,
        "y_dimension": y_dimension,
        "ppm_tolerance": ppm,
        "rt_tolerance": rt,
        "use_grid_coherence": use_grid,
        "grid_coherence_threshold": threshold,
        "show_mesh": bool(show_mesh),
        "show_labels": bool(show_labels),
        "show_cloud": bool(show_cloud),
        "show_unbenchmarked": bool(show_unbenchmarked),
    }
