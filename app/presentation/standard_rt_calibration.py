"""RT-standard review adapted from LASER-4D a188f72 for XCMS streams."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from annotator.mass_calibrants import build_mass_reference_panel, detect_mass_calibration_candidates
from annotator.matching import (highest_abundance_internal_standard_indices,
    POSITIVE_CCS_INTERNAL_STANDARDS, NEGATIVE_CCS_INTERNAL_STANDARDS)
from annotator.rt_calibration import (RT_CALIBRATION_MODELS, apply_internal_standard_rt_calibration,
    fit_confirmed_internal_standard_rt_calibration, is_rt_calibration_model)
from presentation.viewer_theme import viewer_plotly_chart
from standard_lipidomics.internal_standards import project_is_definitions, project_is_reference, project_is_signature


@dataclass(frozen=True)
class RtCalibrationDecision:
    action: str = "none"
    calibration: dict | None = None


def _rt_figure(calibration: dict) -> go.Figure:
    anchors = calibration["anchors"]
    low, high = float(anchors["ref_rt"].min()), float(anchors["ref_rt"].max())
    pad = max((high - low) * 0.12, 0.25)
    line = np.linspace(low - pad, high + pad, 100)
    figure = go.Figure()
    figure.add_scatter(x=line, y=line, mode="lines", name="Unadjusted database RT", line=dict(color="#9a9a9a", dash="dot", width=1.5))
    figure.add_scatter(x=line, y=apply_internal_standard_rt_calibration(line, calibration), mode="lines", name="RT calibration line", line=dict(color="#2677c9", width=3))
    figure.add_scatter(
        x=anchors["ref_rt"], y=anchors["feat_rt"], mode="markers+text",
        text=anchors["standard"], textposition="top center", name="Confirmed IS",
        marker=dict(color="#ffd400", size=11, line=dict(color="#8a5a00", width=1.5)),
    )
    figure.update_layout(
        height=360, template="plotly_white", xaxis_title="Reference-database RT (min)",
        yaxis_title="Observed / adjusted reference RT (min)",
        margin=dict(l=60, r=20, t=25, b=55), legend=dict(orientation="h", y=1.08, x=0),
        dragmode=False,
    )
    figure.update_xaxes(fixedrange=True)
    figure.update_yaxes(fixedrange=True)
    return figure


def _render_rt_result(calibration: dict) -> None:
    if not is_rt_calibration_model(calibration):
        st.warning(calibration["label"])
        return
    st.success(calibration["label"])
    for warning in calibration.get("quality_warnings", ()):
        st.warning(warning)
    metrics = st.columns(4, gap="medium")
    metrics[0].metric("IS RT RMSE before", f"{calibration['rmse_before_min']:.3f} min")
    metrics[1].metric("IS RT RMSE after", f"{calibration['rmse_after_min']:.3f} min")
    loo_rmse = float(calibration.get("loo_rmse_min", np.nan))
    metrics[2].metric(
        "Leave-one-out RMSE",
        f"{loo_rmse:.3f} min" if np.isfinite(loo_rmse) else "Unavailable",
    )
    metrics[3].metric("Confirmed RT standards", calibration["n_used"])
    viewer_plotly_chart(
        _rt_figure(calibration), width="stretch", key="is-rt-calibration-line",
        config={"displayModeBar": False, "scrollZoom": False},
    )
    anchors = calibration["anchors"]
    st.dataframe(
        anchors[["standard", "feat_feature_id", "ref_rt", "feat_rt", "raw_rt_error", "model_weight", "adjusted_ref_rt", "adjusted_rt_error"]],
        hide_index=True, width="stretch",
    )


def render_rt_calibration(features: pd.DataFrame, reference: pd.DataFrame, *, stream: str,
                          prefix: str, project: str | None = None) -> RtCalibrationDecision:
    st.subheader(f"{stream} · RT calibration")
    st.caption("Confirm internal standards, review the fitted model and its validation, then apply or skip.")
    is_path, _ = project_is_definitions(project) if project else (None, ())
    panels = ("Project internal standards",) if is_path else ("Unlabelled IS", "Labelled IS")
    if st.session_state.get(f"{prefix}-rt-panel") not in panels:
        st.session_state[f"{prefix}-rt-panel"] = panels[0]
    with st.form(f"{prefix}-rt-search", border=True):
        panel = st.selectbox("Reference RT-calibration list", panels,
                             key=f"{prefix}-rt-panel")
        columns = st.columns(2)
        ppm = columns[0].number_input("RT-search m/z tolerance (ppm)", min_value=0.1,
                                     max_value=100.0, value=5.0, step=0.5)
        rt = columns[1].number_input("RT-search deviation (min)", min_value=0.01,
                                    max_value=10.0, value=1.0, step=0.05)
        submitted = st.form_submit_button("Search RT calibration masses", type="primary", width="stretch")
    payload = pd.util.hash_pandas_object(features[["feature_id", "mz", "rt"]], index=False).values.tobytes()
    signature = hashlib.sha256(payload + repr((stream, panel, ppm, rt, project_is_signature(project) if project else ())).encode()).hexdigest()
    cache_key = f"{prefix}-rt-search-result"
    if submitted:
        polarity = "pos" if stream.endswith("Pos") else "neg"
        standards = POSITIVE_CCS_INTERNAL_STANDARDS if polarity == "pos" else NEGATIVE_CCS_INTERNAL_STANDARDS
        references = (project_is_reference(project, reference, stream=stream) if is_path else
            build_mass_reference_panel(reference, polarity=polarity, panel_name=panel,
                unlabelled_standards=standards, chromatography=stream.split("_")[0]))
        # Like the 4D search, only standards with measured library RTs are eligible.
        references = references.loc[pd.to_numeric(references["ref_rt"], errors="coerce").notna()]
        result = detect_mass_calibration_candidates(features, references, ppm_tol=ppm, rt_tol=rt)
        st.session_state[cache_key] = (signature, result)
    stored = st.session_state.get(cache_key)
    selected = pd.DataFrame()
    if stored is None:
        st.info("Choose a reference list and search for RT calibration masses.")
    elif stored[0] != signature:
        st.warning("RT search settings or calibrated masses changed. Search again.")
    else:
        result = stored[1]
        candidates = result["candidates"]
        if candidates.empty:
            st.warning("No RT standards matched. Widen the search windows or change the reference list.")
        else:
            st.success(f"Found {candidates['standard'].nunique()} of {result['reference_count']} RT standards.")
            if result["missing"]:
                with st.expander("RT standards not found"):
                    st.write(", ".join(result["missing"]))
            editor = candidates.copy()
            editor.insert(0, "Confirm", editor.index.isin(highest_abundance_internal_standard_indices(editor)))
            edited = st.data_editor(editor, key=f"{prefix}-rt-candidates-{signature}",
                width="stretch", hide_index=True,
                disabled=[column for column in editor.columns if column != "Confirm"],
                column_order=["Confirm", "standard", "source", "internal_standard_concentration", "internal_standard_unit", "feat_feature_id", "feat_intensity",
                              "feat_mz", "ref_mz", "ppm_error", "feat_rt", "ref_rt", "rt_error"],
                column_config={"Confirm": st.column_config.CheckboxColumn("Confirm", default=False)})
            selected = edited.loc[edited["Confirm"]].copy()
    model_name = st.selectbox("RT calibration model", tuple(RT_CALIBRATION_MODELS), index=1,
        key=f"{prefix}-rt-model",
        help="Piecewise fitting needs six distinct, well-distributed standards in matching elution order. Outside that range it uses the nearest boundary shift.")
    calibration = fit_confirmed_internal_standard_rt_calibration(selected, model=RT_CALIBRATION_MODELS[model_name])
    if stored is not None and stored[0] == signature:
        _render_rt_result(calibration)
    columns = st.columns(3)
    if columns[0].button("Back", key=f"{prefix}-rt-back", width="stretch"):
        return RtCalibrationDecision("back")
    if columns[1].button("Skip RT calibration", key=f"{prefix}-rt-skip", width="stretch"):
        return RtCalibrationDecision("skip")
    if columns[2].button("Apply RT calibration", key=f"{prefix}-rt-apply", width="stretch", type="primary",
                         disabled=not is_rt_calibration_model(calibration)):
        return RtCalibrationDecision("apply", calibration)
    return RtCalibrationDecision()
