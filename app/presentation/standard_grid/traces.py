"""Chromatogram trace presentation for the standard grid browser."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from standard_traces import StandardSampleTrace, load_standard_feature_traces
from viewer_common import apply_matlab_plot_style, stable_color_map

FEATURE_TRACE_COLOURS = (
    "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#6A3D9A",
)
TRACE_DIAGNOSTIC_PPM = 3.0
TRACE_DIAGNOSTIC_RT_SECONDS = 10.0

def _trace_number(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(row.get(column), errors="coerce")
    return float(value) if pd.notna(value) else float("nan")


def _feature_trace_range(row: pd.Series) -> tuple[float, float] | None:
    """Return a compact RT window around an XCMS feature for the viewer."""
    centre = _trace_number(row, "feat_rt")
    if not np.isfinite(centre):
        return None
    lower = _trace_number(row, "feat_rtmin")
    upper = _trace_number(row, "feat_rtmax")
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower, upper = centre - 0.20, centre + 0.20
    # Keep a little context around the XCMS integration window without letting
    # very broad grouped windows flatten the selected chromatogram.
    width = max(upper - lower, 0.08)
    padding = max(width * 0.35, 0.08)
    lower -= padding
    upper += padding
    # Grouped XCMS bounds occasionally span a large part of the run (for
    # example after a split/merged peak).  They are useful metadata, but make
    # a selected EIC look flat.  Keep the diagnostic view local to the feature
    # centre while retaining the original window whenever it is reasonable.
    maximum_view_width = 0.35  # minutes
    if upper - lower > maximum_view_width:
        half = maximum_view_width / 2.0
        lower, upper = centre - half, centre + half
    return lower, upper


def _apply_trace_y_range(
    figure: go.Figure,
    traces: list[StandardSampleTrace],
    x_range: tuple[float, float] | list[float] | None,
) -> None:
    """Scale intensity to the RT segment that is actually shown.

    Plotly otherwise considers the full extracted run when autoscaling y.  A
    large unrelated chromatographic peak outside the selected RT window can
    therefore compress the feature of interest to a line at zero.
    """
    values: list[np.ndarray] = []
    for trace in traces:
        rt = np.asarray(trace.rt_values, dtype=float)
        intensity = np.asarray(trace.rt_intensities, dtype=float)
        valid = np.isfinite(intensity)
        if x_range is not None and len(rt) == len(intensity):
            valid &= np.isfinite(rt)
            valid &= rt >= float(x_range[0])
            valid &= rt <= float(x_range[1])
        if np.any(valid):
            values.append(intensity[valid])
    if not values:
        return
    maximum = float(np.nanmax(np.concatenate(values)))
    if not np.isfinite(maximum) or maximum <= 0:
        return
    figure.update_yaxes(range=[0.0, maximum * 1.08])


def _feature_trace_colour_map(feature_ids) -> dict[str, str]:
    """Assign stable, reusable colours to every competing feature ID."""
    return stable_color_map(feature_ids, FEATURE_TRACE_COLOURS)


def _same_id_trace_candidates(
    candidates: pd.DataFrame,
    primary_row: pd.Series,
) -> tuple[pd.DataFrame, int]:
    """Keep chemically plausible same-ID rows for the trace diagnostics.

    ``all_pairs`` deliberately retains MATLAB's broad early-stage proposals,
    but those rows are not equivalent annotations.  Showing every 6-ppm/
    60-second proposal when an anchor is clicked creates a misleading wall of
    chromatograms.  The trace view uses the final LASER contract (strictly
    below 3 ppm and 10 seconds from the MATLAB RT prediction), while retaining
    any row the automatic mapper accepted and always retaining the selected
    primary feature.  The full broad pool remains available in the grid and
    exports for audit.
    """
    if candidates.empty:
        return candidates.copy(), 0
    rows = candidates.copy()
    ppm = pd.to_numeric(
        rows["ppm_error"] if "ppm_error" in rows else
        pd.Series(np.nan, index=rows.index),
        errors="coerce",
    ).abs()
    rt_seconds = (
        pd.to_numeric(
            rows["rt_error"] if "rt_error" in rows else
            pd.Series(np.nan, index=rows.index),
            errors="coerce",
        ).abs() * 60.0
    )
    strict = ppm.lt(TRACE_DIAGNOSTIC_PPM) & rt_seconds.lt(
        TRACE_DIAGNOSTIC_RT_SECONDS
    )
    accepted = rows.get(
        "automatic_candidate", pd.Series(False, index=rows.index)
    ).fillna(False).astype(bool)
    keep = (strict | accepted).fillna(False)
    primary_id = str(primary_row.get("feat_feature_id", ""))
    feature_ids = rows.get(
        "feat_feature_id", pd.Series("", index=rows.index)
    ).astype(str)
    keep |= feature_ids.eq(primary_id)
    selected = rows.loc[keep].drop_duplicates(
        "feat_feature_id", keep="first"
    ).copy()
    omitted = int(rows[~keep].get("feat_feature_id", pd.Series(
        dtype=object
    )).astype(str).nunique())
    return selected, omitted


def _standard_trace_figure(
    traces: list[StandardSampleTrace],
    row: pd.Series,
    *,
    title: str,
    key: str,
    colour_by_sample: dict[str, str] | None = None,
    show_legend: bool = False,
    line_dash: str = "solid",
) -> None:
    """Render one selected XCMS feature as an all-sample RT overlay."""
    if not traces:
        return
    representative = max(traces, key=lambda trace: trace.peak_height)
    colours = colour_by_sample or {
        trace.sample_name: colour
        for trace, colour in zip(
            traces,
            ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"),
            strict=False,
        )
    }
    figure = go.Figure()
    for trace in traces:
        is_representative = trace.sample_name == representative.sample_name
        label = (
            f"{trace.sample_name} · highest peak height"
            if is_representative else trace.sample_name
        )
        figure.add_trace(go.Scatter(
            x=trace.rt_values,
            y=trace.rt_intensities,
            mode="lines",
            name=label,
            line=dict(
                color=colours.get(trace.sample_name, "#0072B2"),
                width=3.0 if is_representative else 1.3,
                dash=line_dash,
            ),
            opacity=1.0 if is_representative else 0.68,
            hovertemplate=(
                f"{trace.sample_name}<br>"
                "RT %{x:.4f} min<br>intensity %{y:.4g}<extra></extra>"
            ),
            showlegend=show_legend,
        ))
    apply_matlab_plot_style(figure)
    figure.update_layout(
        height=320,
        title=dict(text=title, x=0.5, xanchor="center"),
        margin=dict(l=68, r=20, t=52, b=58),
        xaxis_title="retention time (min)",
        yaxis_title="intensity",
        showlegend=show_legend,
        uirevision=key,
    )
    x_range = _feature_trace_range(row)
    if x_range is not None:
        figure.update_xaxes(range=list(x_range))
        _apply_trace_y_range(figure, traces, x_range)
        centre = _trace_number(row, "feat_rt")
        figure.add_vline(
            x=centre,
            line=dict(color="#8e8e93", width=1, dash="dot"),
        )
        lower = _trace_number(row, "feat_rtmin")
        upper = _trace_number(row, "feat_rtmax")
        if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
            figure.add_vrect(
                x0=lower, x1=upper,
                fillcolor="rgba(0,114,178,0.08)", line_width=0,
            )
    st.plotly_chart(
        figure,
        width="stretch",
        key=key,
        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
    )


def _candidate_trace_specs(
    primary_row: pd.Series,
    same_id_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, str, list[tuple[str, float, float]]]:
    rows = same_id_candidates.copy()
    primary_id = str(primary_row.get("feat_feature_id", ""))
    known_ids = set(rows.get("feat_feature_id", pd.Series(dtype=object)).astype(str))
    if primary_id and primary_id not in known_ids:
        rows = pd.concat([rows, pd.DataFrame([primary_row.to_dict()])], ignore_index=True)
    rows["feat_feature_id"] = rows["feat_feature_id"].astype(str)
    rows = rows.drop_duplicates("feat_feature_id", keep="first")
    specs = []
    for row in rows.itertuples(index=False):
        values = row._asdict()
        mz = pd.to_numeric(values.get("feat_mz"), errors="coerce")
        rt = pd.to_numeric(values.get("feat_rt"), errors="coerce")
        if pd.notna(mz) and pd.notna(rt):
            specs.append((str(values["feat_feature_id"]), float(mz), float(rt)))
    return rows, primary_id, specs


def _render_primary_trace(
    primary_traces: list[StandardSampleTrace],
    primary_row: pd.Series,
    *,
    stream_name: str,
    species_id: str,
    primary_id: str,
) -> None:
    if not primary_traces:
        st.info(f"No raw scan trace was found for primary feature {primary_id}.")
        return
    sample_traces = {trace.sample_name: trace for trace in primary_traces}
    options = ["All samples (overlay)", *sample_traces]
    selected = st.selectbox(
        "Trace display",
        options,
        key=f"standard-trace-display::{stream_name}::{species_id}::{primary_id}",
        label_visibility="collapsed",
    )
    displayed = primary_traces if selected == options[0] else [sample_traces[selected]]
    _standard_trace_figure(
        displayed,
        primary_row,
        title=f"Chromatogram · feature {primary_id}",
        key=f"standard-feature-chromatogram::{stream_name}::{species_id}::{primary_id}",
    )
    integration = []
    for label in ("feat_rtmin", "feat_rtmax", "feat_into", "feat_intb", "feat_maxo", "feat_sn"):
        value = _trace_number(primary_row, label)
        if np.isfinite(value):
            integration.append(f"{label.removeprefix('feat_')}={value:.4g}")
    metadata = (
        " · ".join(integration)
        if integration else "XCMS integration metadata not present in this export."
    )
    st.caption(
        "Dashed line: XCMS feature RT; blue band: XCMS grouped RT bounds. "
        f"{metadata} Traces are diagnostic only and do not alter annotation."
    )


def _add_alternative_feature(
    figure: go.Figure,
    feature_id: str,
    candidate_row: pd.Series,
    traces: list[StandardSampleTrace],
    colour: str,
) -> tuple[tuple[float, float] | None, list[StandardSampleTrace]]:
    dash = "dot" if not bool(candidate_row.get("automatic_candidate", False)) else "solid"
    representative = max(traces, key=lambda trace: trace.peak_height)
    for trace in traces:
        is_representative = trace.sample_name == representative.sample_name
        figure.add_trace(go.Scatter(
            x=trace.rt_values,
            y=trace.rt_intensities,
            mode="lines",
            name=feature_id,
            legendgroup=feature_id,
            showlegend=False,
            line=dict(color=colour, width=2.8 if is_representative else 1.0, dash=dash),
            opacity=1.0 if is_representative else 0.32,
            hovertemplate=(
                f"feature {feature_id}<br>{trace.sample_name}<br>"
                "RT %{x:.4f} min<br>intensity %{y:.4g}<extra></extra>"
            ),
        ))
    figure.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", name=feature_id,
        legendgroup=feature_id, showlegend=True,
        line=dict(color=colour, width=2.8, dash=dash), hoverinfo="skip",
    ))
    candidate_rt = _trace_number(candidate_row, "feat_rt")
    if np.isfinite(candidate_rt):
        figure.add_vline(x=candidate_rt, line=dict(color=colour, width=1, dash="dot"), opacity=0.7)
    lower = _trace_number(candidate_row, "feat_rtmin")
    upper = _trace_number(candidate_row, "feat_rtmax")
    if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
        figure.add_vrect(x0=lower, x1=upper, fillcolor=colour, opacity=0.045, line_width=0)
    return _feature_trace_range(candidate_row), traces


def _render_alternative_traces(
    rows: pd.DataFrame,
    traces_by_feature: dict[str, list[StandardSampleTrace]],
    *,
    primary_id: str,
    stream_name: str,
    species_id: str,
) -> None:
    alternative_rows = rows[rows["feat_feature_id"].ne(primary_id)]
    alternative_ids = list(dict.fromkeys(
        feature_id for feature_id in alternative_rows["feat_feature_id"].astype(str)
        if feature_id in traces_by_feature
    ))
    if not alternative_ids:
        if len(rows["feat_feature_id"].unique()) > 1:
            st.caption("Additional same-ID features had no available raw trace.")
        return
    st.markdown(
        f'<div class="matlab-title">Same lipid ID candidate traces · '
        f'{len(alternative_ids)} feature(s)</div>',
        unsafe_allow_html=True,
    )
    colours = _feature_trace_colour_map(alternative_ids)
    figure = go.Figure()
    row_by_feature = alternative_rows.drop_duplicates("feat_feature_id").set_index("feat_feature_id")
    ranges, all_traces = [], []
    for feature_id in alternative_ids:
        trace_range, traces = _add_alternative_feature(
            figure,
            feature_id,
            row_by_feature.loc[feature_id],
            traces_by_feature[feature_id],
            colours[feature_id],
        )
        if trace_range is not None:
            ranges.append(trace_range)
        all_traces.extend(traces)
    apply_matlab_plot_style(figure)
    figure.update_layout(
        height=360,
        title=dict(text="Chromatograms of other matching features", x=0.5, xanchor="center"),
        margin=dict(l=68, r=20, t=52, b=96),
        xaxis_title="retention time (min)", yaxis_title="intensity",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.28, yanchor="top"),
        uirevision=f"standard-same-id-chromatograms::{stream_name}::{species_id}",
    )
    if ranges:
        x_range = [min(item[0] for item in ranges), max(item[1] for item in ranges)]
        figure.update_xaxes(range=x_range)
        _apply_trace_y_range(figure, all_traces, x_range)
    st.plotly_chart(
        figure,
        width="stretch",
        key=f"standard-same-id-chromatograms::{stream_name}::{species_id}",
        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
    )
    st.caption(
        "Each colour is one alternative XCMS feature; thick lines are the "
        "highest-height sample. Dotted lines were not automatically accepted."
    )


def _render_standard_feature_traces(
    result: dict,
    stream_name: str,
    primary_row: pd.Series,
    same_id_candidates: pd.DataFrame,
    species_id: str,
) -> None:
    """Render primary and competing same-identity chromatograms."""
    project_dir = result.get("xcms_input_dir")
    if not project_dir:
        st.caption("Chromatograms require an XCMS project folder with converted MS1 scans.")
        return
    rows, primary_id, specs = _candidate_trace_specs(primary_row, same_id_candidates)
    if not specs:
        st.caption("The selected annotation has no finite m/z/RT pair to trace.")
        return
    traces_by_feature = load_standard_feature_traces(
        project_dir, stream_name, tuple(specs), ppm_tolerance=5.0,
    )
    if not traces_by_feature:
        st.info("No matching converted .mzXML/.mzML MS1 files were found.")
        return
    st.markdown(
        f'<div class="matlab-title">Feature trace · {species_id}</div>',
        unsafe_allow_html=True,
    )
    _render_primary_trace(
        traces_by_feature.get(primary_id, []),
        primary_row,
        stream_name=stream_name,
        species_id=species_id,
        primary_id=primary_id,
    )
    _render_alternative_traces(
        rows,
        traces_by_feature,
        primary_id=primary_id,
        stream_name=stream_name,
        species_id=species_id,
    )
