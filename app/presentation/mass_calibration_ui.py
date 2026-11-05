"""Focused Streamlit presenter for reviewed precursor-mass calibration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from presentation.viewer_theme import viewer_plotly_chart

from annotator.mass_calibrants import MASS_REFERENCE_PANELS
from annotator.mass_calibrants import evaluate_lipid_validation_panel
from annotator.matching import (
    MASS_CALIBRATION_MODELS,
    apply_mass_calibration,
    fit_confirmed_mass_calibration,
    highest_abundance_internal_standard_indices,
    is_mass_calibration_model,
)


@dataclass(frozen=True)
class MassCalibrationViewInput:
    search_signature: str
    search_candidates: Callable[[], dict]
    return_stage: str
    confirmed_calibration: dict | None
    features: pd.DataFrame
    polarity: str
    key_prefix: str = "mass"
    title: str = "Mass calibration"
    validate_lipids: bool = True
    use_lipid_rt: bool = True


@dataclass(frozen=True)
class MassCalibrationDecision:
    action: str = "none"
    calibration: dict | None = None
    search_signature: str = ""


def _key(view: MassCalibrationViewInput, suffix: str) -> str:
    return f"{view.key_prefix}_{suffix}"


def _selection_signature(
    search_signature: str, model: str, selected: pd.DataFrame
) -> str:
    identities = sorted(zip(
        selected.get("standard", pd.Series(dtype=str)).astype(str),
        selected.get("feat_feature_id", pd.Series(dtype=str)).astype(str),
    ))
    payload = json.dumps([search_signature, model, identities], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_search_form(view: MassCalibrationViewInput) -> dict | None:
    with st.form(_key(view, "reference_search"), border=True):
        st.selectbox(
            "Reference m/z calibration list",
            MASS_REFERENCE_PANELS,
            key=_key(view, "reference_panel"),
            help=(
                "Labelled panels use exact isotope-labelled masses. Unlabelled "
                "panels use short/odd-chain lipid standards. Tune-mix panels use "
                "polarity-specific Agilent ESI-L exact masses."
            ),
        )
        ppm_col, rt_col, tune_col = st.columns(3, gap="medium")
        ppm_col.number_input(
            "Search m/z tolerance (ppm)", min_value=0.1, max_value=100.0,
            step=0.5, key=_key(view, "search_ppm_tol"),
            help="Maximum absolute precursor-mass error used to find candidates.",
        )
        rt_col.number_input(
            "IS RT deviation (min)", min_value=0.01, max_value=10.0,
            step=0.05, key=_key(view, "search_rt_tol"),
            disabled=not view.use_lipid_rt,
            help=(
                "Maximum difference from the chromatography-specific library RT "
                "for lipid internal standards."
            ),
        )
        tune_col.number_input(
            "Tune-mix maximum RT (min)", min_value=0.05, max_value=10.0,
            step=0.25, key=_key(view, "tune_mix_rt_max"),
            help="Tune-mix ions have no library RT and must occur before this RT.",
        )
        search_clicked = st.form_submit_button(
            "Search reference masses", type="primary", width="stretch"
        )
    if search_clicked:
        with st.spinner("Searching the selected mass-reference panel…"):
            st.session_state[_key(view, "search_result")] = view.search_candidates()
            st.session_state[_key(view, "search_signature")] = view.search_signature
            st.session_state.pop(_key(view, "calibration_preview"), None)
            st.session_state.pop(_key(view, "calibration_preview_signature"), None)
    if st.session_state.get(_key(view, "search_signature")) != view.search_signature:
        return None
    return st.session_state.get(_key(view, "search_result"))


def _render_candidate_editor(
    search_result: dict | None, view: MassCalibrationViewInput
) -> pd.DataFrame:
    candidates = (
        search_result.get("candidates", pd.DataFrame())
        if search_result else pd.DataFrame()
    )
    if search_result is None:
        st.info("Choose a reference list and click **Search reference masses**.")
        return pd.DataFrame()
    if candidates.empty:
        st.warning("No candidates matched. Widen the ppm/RT window or change panel.")
        return pd.DataFrame()
    st.success(
        f"Found {candidates['standard'].nunique()} of "
        f"{search_result['reference_count']} standards "
        f"({len(candidates)} candidate features)."
    )
    if search_result.get("missing"):
        with st.expander("Reference masses not found", expanded=False):
            st.write(", ".join(search_result["missing"]))
    confirmed_ids: set[str] = set()
    current = view.confirmed_calibration
    if (
        current is not None
        and is_mass_calibration_model(current)
        and current.get("source_mass_search_signature") == view.search_signature
    ):
        confirmed_ids = set(current["anchors"]["feat_feature_id"].astype(str))
    default_indices = highest_abundance_internal_standard_indices(candidates)
    editor = candidates.copy()
    editor.insert(0, "Confirm", (
        editor["feat_feature_id"].astype(str).isin(confirmed_ids)
        if confirmed_ids else editor.index.isin(default_indices)
    ))
    return st.data_editor(
        editor, key=f"{view.key_prefix}-candidates::{view.search_signature}",
        width="stretch", hide_index=True,
        disabled=[column for column in editor.columns if column != "Confirm"],
        column_order=[
            "Confirm", "standard", "source", "feat_feature_id",
            "feat_intensity", "feat_mz", "ref_mz", "ppm_error",
            "feat_rt", "ref_rt", "rt_error",
        ],
        column_config={
            "Confirm": st.column_config.CheckboxColumn("Confirm", default=False),
            "standard": "Calibration standard", "source": "Reference source",
            "feat_feature_id": "Feature",
            "feat_intensity": st.column_config.NumberColumn("Intensity", format="%.3g"),
            "feat_mz": st.column_config.NumberColumn("Observed m/z", format="%.6f"),
            "ref_mz": st.column_config.NumberColumn("Reference m/z", format="%.6f"),
            "ppm_error": st.column_config.NumberColumn("Raw Δm/z (ppm)", format="%+.3f"),
            "feat_rt": st.column_config.NumberColumn("RT (min)", format="%.3f"),
            "ref_rt": st.column_config.NumberColumn("Library RT", format="%.3f"),
            "rt_error": st.column_config.NumberColumn("ΔRT (min)", format="%+.3f"),
        },
    )


def _fit_preview(
    selected: pd.DataFrame, search_signature: str, view: MassCalibrationViewInput
) -> tuple[dict | None, str]:
    model_col, fit_col = st.columns([2.0, 1.0], gap="medium")
    model_label = model_col.selectbox(
        "Calibration model", tuple(MASS_CALIBRATION_MODELS),
        key=_key(view, "fit_model"),
        help=(
            "Constant ppm is most conservative; linear is the historical "
            "default; robust linear downweights an outlier; quadratic requires "
            "at least five standards across 100 m/z and must remain monotonic."
        ),
    )
    fit_col.markdown(
        '<div style="height: 1.75rem"></div>', unsafe_allow_html=True
    )
    fit_clicked = fit_col.button(
        "Calibrate and view results", type="primary", width="stretch",
        disabled=selected.empty, key=_key(view, "fit_mass_calibration"),
    )
    signature = _selection_signature(
        search_signature, MASS_CALIBRATION_MODELS[model_label], selected
    )
    if fit_clicked:
        st.session_state[_key(view, "calibration_preview")] = fit_confirmed_mass_calibration(
            selected, model=MASS_CALIBRATION_MODELS[model_label]
        )
        st.session_state[_key(view, "calibration_preview_signature")] = signature
    preview = (
        st.session_state.get(_key(view, "calibration_preview"))
        if st.session_state.get(_key(view, "calibration_preview_signature")) == signature
        else None
    )
    return preview, signature


def _render_fit_result(
    calibration: dict | None, view: MassCalibrationViewInput
) -> bool:
    if calibration is None:
        return False
    if not is_mass_calibration_model(calibration):
        st.warning(calibration["label"])
        return False
    st.success(calibration["label"])
    for warning in calibration.get("quality_warnings", ()):
        st.warning(warning)
    anchors = calibration["anchors"]
    before, after, count = st.columns(3, gap="medium")
    before.metric("RMS mass error before", f"{calibration['rmse_before_ppm']:.3f} ppm")
    after.metric("RMS mass error after", f"{calibration['rmse_after_ppm']:.3f} ppm")
    count.metric("Confirmed mass standards", calibration["n_used"])
    x_min, x_max = float(anchors["feat_mz"].min()), float(anchors["feat_mz"].max())
    padding = max((x_max - x_min) * 0.08, 1.0)
    x_line = np.linspace(x_min - padding, x_max + padding, 100)
    figure = go.Figure()
    figure.add_scatter(
        x=x_line, y=x_line, mode="lines", name="No correction",
        line=dict(color="#9a9a9a", dash="dot", width=1.5),
    )
    figure.add_scatter(
        x=x_line, y=apply_mass_calibration(x_line, calibration), mode="lines",
        name="Mass calibration model", line=dict(color="#007aff", width=3),
    )
    figure.add_scatter(
        x=anchors["feat_mz"], y=anchors["ref_mz"], mode="markers+text",
        text=anchors["standard"], textposition="top center",
        name="Confirmed standards",
        marker=dict(color="#ffd400", size=10, line=dict(color="#8a5a00", width=1.3)),
    )
    figure.update_layout(
        height=360, template="plotly_white", xaxis_title="Observed m/z",
        yaxis_title="Reference / calibrated m/z",
        margin=dict(l=60, r=20, t=25, b=55),
        legend=dict(orientation="h", y=1.08, x=0), dragmode=False,
    )
    figure.update_xaxes(fixedrange=True)
    figure.update_yaxes(fixedrange=True)
    viewer_plotly_chart(
        figure, width="stretch", key=_key(view, "calibration_line"),
        config={"displayModeBar": False, "scrollZoom": False},
    )
    st.dataframe(anchors[[
        "standard", "feat_feature_id", "feat_mz", "ref_mz", "raw_ppm_error",
        "model_weight", "calibrated_mz", "calibrated_ppm_error",
    ]], hide_index=True, width="stretch")
    return True


def _metric_ppm(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return f"{float(numeric):.2f} ppm" if np.isfinite(numeric) else "Unavailable"


def _render_lipid_validation(evaluation: dict) -> None:
    st.subheader("Independent lipid validation effect")
    details = evaluation.get("details", pd.DataFrame())
    if details.empty:
        st.warning(
            "No independent lipid targets were matched. Widen the mass-reference "
            "search window or confirm that the selected polarity and RP RTs match."
        )
        return
    detected, median_after, p95_after, within = st.columns(4, gap="medium")
    detected.metric(
        "Validation targets detected",
        f"{len(details)} / {evaluation.get('reference_count', len(details))}",
    )
    median_after.metric(
        "Median |error| after",
        _metric_ppm(evaluation.get("median_absolute_ppm_after")),
    )
    p95_after.metric(
        "95th percentile |error|",
        _metric_ppm(evaluation.get("p95_absolute_ppm_after")),
    )
    percentage = pd.to_numeric(
        evaluation.get("within_5ppm_percent"), errors="coerce"
    )
    within.metric(
        "Targets within ±5 ppm",
        f"{float(percentage):.1f}%" if np.isfinite(percentage) else "Unavailable",
    )
    status = evaluation.get("status")
    message = (
        "PASS: median absolute residual ≤2 ppm and 95th percentile ≤5 ppm."
        if status == "PASS"
        else "REVIEW: the independent residual panel exceeds a review threshold."
    )
    (st.success if status == "PASS" else st.warning)(message)
    before = details[["target", "theoretical_mz", "ppm_before", "intensity"]].rename(
        columns={"ppm_before": "PPM error", "theoretical_mz": "Theoretical m/z"}
    )
    before["Stage"] = "Before calibration"
    after = details[["target", "theoretical_mz", "ppm_after", "intensity"]].rename(
        columns={"ppm_after": "PPM error", "theoretical_mz": "Theoretical m/z"}
    )
    after["Stage"] = "After calibration"
    plot = pd.concat((before, after), ignore_index=True).dropna(
        subset=["PPM error", "Theoretical m/z"]
    )
    st.scatter_chart(
        plot,
        x="Theoretical m/z",
        y="PPM error",
        color="Stage",
        size="intensity",
    )
    st.dataframe(details, hide_index=True, width="stretch")
    st.download_button(
        "Download independent validation CSV",
        data=details.to_csv(index=False).encode("utf-8"),
        file_name=f"laser_mass_calibration_validation_{evaluation.get('polarity', '')}.csv",
        mime="text/csv",
        width="stretch",
    )
    st.caption(
        "These lipids are independent and never fit the LASER correction. The "
        "evaluation uses one highest-abundance aligned feature per target; the "
        "MZmine calibration test remains the per-sample raw-scan evaluation."
    )
    missing = evaluation.get("missing", [])
    if missing:
        with st.expander("Independent targets not detected", expanded=False):
            st.write(", ".join(missing))


def render_mass_calibration(
    view: MassCalibrationViewInput,
) -> MassCalibrationDecision:
    st.markdown(
        f'<div class="matlab-title">{view.title}</div>', unsafe_allow_html=True
    )
    result = _render_search_form(view)
    edited = _render_candidate_editor(result, view)
    selected = edited[edited["Confirm"]].copy() if not edited.empty else pd.DataFrame()
    preview, _ = _fit_preview(selected, view.search_signature, view)
    valid = _render_fit_result(preview, view)
    if valid and view.validate_lipids:
        _render_lipid_validation(evaluate_lipid_validation_panel(
            view.features,
            preview,
            polarity=view.polarity,
            ppm_tol=max(
                float(st.session_state[_key(view, "search_ppm_tol")]), 30.0
            ),
            rt_tol=0.20,
        ))
    back_col, skip_col, apply_col = st.columns(3, gap="medium")
    if back_col.button(
        "Back without changes", width="stretch", key=_key(view, "back_to_setup")
    ):
        return MassCalibrationDecision(action="back")
    if skip_col.button(
        "Skip mass calibration", width="stretch",
        key=_key(view, "skip_mass_calibration")
    ):
        return MassCalibrationDecision(action="skip")
    if apply_col.button(
        "Apply mass calibration", type="primary", width="stretch",
        key=_key(view, "apply_mass_calibration"), disabled=not valid,
    ):
        return MassCalibrationDecision(
            action="apply", calibration=preview,
            search_signature=view.search_signature,
        )
    return MassCalibrationDecision()
