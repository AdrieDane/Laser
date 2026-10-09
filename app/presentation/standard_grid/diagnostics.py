"""Diagnostic-strip figures for the standard-lipidomics reviewer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viewer_common import MATLAB_FONT, matlab_axis_config, row_sample_values


COLUMN_WIDTHS = [0.52, 0.16, 0.16, 0.16]
HORIZONTAL_SPACING = 0.032


def _span(values, minimum):
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    return max(minimum, float(finite.abs().max()) * 1.15) if len(finite) else minimum


def _subplot_specs(kind: str) -> list[list[dict[str, str]]]:
    return [[
        {"type": "heatmap" if kind == "body" else "scatter"},
        {"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"},
    ]]


def _build_body_traces(
    detail: pd.DataFrame,
    xlabels: list[str],
    row_ids: list[str],
    marker: dict,
    point_custom: np.ndarray,
    heat_custom: np.ndarray,
    heat_hovertemplate: str,
    point_hover_prefix: str,
) -> go.Figure:
    matrix = np.vstack([
        row_sample_values(row, xlabels) for _, row in detail.iterrows()
    ]) if len(detail) else np.zeros((0, len(xlabels)))
    figure = make_subplots(
        rows=1, cols=4, shared_yaxes=True, column_widths=COLUMN_WIDTHS,
        horizontal_spacing=HORIZONTAL_SPACING, specs=_subplot_specs("body"),
    )
    figure.add_trace(go.Heatmap(
        z=np.log10(np.clip(matrix, 1, None)), x=xlabels, y=row_ids,
        customdata=heat_custom, colorscale="Turbo", showscale=False,
        hovertemplate=heat_hovertemplate,
    ), row=1, col=1)
    coherence = pd.to_numeric(detail.get("grid_coherence_score"), errors="coerce")
    missing = ~np.isfinite(coherence)
    figure.add_trace(go.Scatter(
        x=np.where(missing, 0.5, coherence), y=row_ids, mode="markers+text",
        text=np.where(missing, "n/a", ""), textposition="middle center",
        marker=marker, customdata=point_custom,
        hovertemplate=f"{point_hover_prefix}<br>grid coherence %{{x:.3f}}<extra></extra>",
    ), row=1, col=2)
    for column, values, hover in (
        (3, detail["rt_error"] * 60, "ΔRT vs grid %{x:.1f} s"),
        (4, detail["ppm_error"], "Δmass %{x:.2f} ppm"),
    ):
        figure.add_trace(go.Scatter(
            x=values, y=row_ids, mode="markers", marker=marker,
            customdata=point_custom,
            hovertemplate=f"{point_hover_prefix}<br>{hover}<extra></extra>",
        ), row=1, col=column)
    return figure


def _build_header(xlabels: list[str], spans: tuple[float, float], left_margin: int) -> go.Figure:
    titles = ("intensity/sample", "grid coherence", "Δ RT (s)", "Δ mass (ppm)")
    figure = make_subplots(
        rows=1, cols=4, shared_yaxes=True, column_widths=COLUMN_WIDTHS,
        horizontal_spacing=HORIZONTAL_SPACING, specs=_subplot_specs("header"),
    )
    figure.add_trace(go.Scatter(
        x=xlabels, y=np.zeros(len(xlabels)), mode="markers",
        marker=dict(opacity=0), hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    figure.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0], mode="markers", marker=dict(opacity=0),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=2)
    for column, span in enumerate(spans, start=3):
        figure.add_trace(go.Scatter(
            x=[-span, span], y=[0, 0], mode="markers", marker=dict(opacity=0),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=column)
    axes = {"xaxis": matlab_axis_config(
        False, side="top", fixedrange=True, showgrid=False,
        title=dict(text=titles[0], standoff=18), showticklabels=False,
        ticks="", showline=False, mirror=False,
        categoryorder="array", categoryarray=xlabels,
    )}
    axes["xaxis2"] = matlab_axis_config(
        False, side="top", fixedrange=True, showgrid=False, range=[0, 1],
        title=dict(text=titles[1], standoff=18), nticks=3, ticklen=3,
    )
    for index, (span, title) in enumerate(zip(spans, titles[2:]), start=3):
        axes[f"xaxis{index}"] = matlab_axis_config(
            False, side="top", fixedrange=True, showgrid=False,
            range=[-span, span], zeroline=False,
            title=dict(text=title, standoff=18), nticks=3, ticklen=3,
        )
    for index in range(1, 5):
        axes["yaxis" if index == 1 else f"yaxis{index}"] = matlab_axis_config(
            False, visible=False, fixedrange=True, range=[-1, 1],
        )
    figure.update_layout(
        height=62, title="", margin=dict(l=left_margin, r=20, t=46, b=0),
        showlegend=False, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        template="plotly_white", font=MATLAB_FONT, dragmode=False, **axes,
    )
    return figure


def _configure_body(
    figure: go.Figure,
    detail: pd.DataFrame,
    xlabels: list[str],
    row_ids: list[str],
    row_labels: list[str],
    spans: tuple[float, float],
    left_margin: int,
    body_height: int,
) -> None:
    axes = {
        "xaxis": matlab_axis_config(
            showticklabels=False, categoryorder="array",
            categoryarray=xlabels, fixedrange=True,
        ),
        "yaxis": matlab_axis_config(
            categoryorder="array", categoryarray=row_ids, tickmode="array",
            tickvals=row_ids, ticktext=row_labels,
            tickfont=dict(size=10, color="#3a3a3c"), fixedrange=True,
            range=[-0.5, len(row_ids) - 0.5],
        ),
    }
    threshold_values = pd.to_numeric(
        detail.get("grid_coherence_threshold"), errors="coerce"
    ).dropna()
    threshold = float(threshold_values.iloc[0]) if len(threshold_values) else 0.35
    axes["xaxis2"] = matlab_axis_config(
        range=[0, 1], zeroline=False, title=dict(text=""),
        showticklabels=False, fixedrange=True,
    )
    axes["yaxis2"] = matlab_axis_config(
        showticklabels=False, fixedrange=True, range=[-0.5, len(row_ids) - 0.5],
    )
    shapes = [
        dict(type="rect", x0=0, x1=threshold, y0=0, y1=1, xref="x2", yref="paper",
             line=dict(width=0), fillcolor="rgba(214, 55, 55, 0.08)", layer="below"),
        dict(type="line", x0=threshold, x1=threshold, y0=0, y1=1,
             xref="x2", yref="paper", line=dict(color="rgba(214,55,55,0.55)", width=1)),
    ]
    for index, span in enumerate(spans, start=3):
        axes[f"xaxis{index}"] = matlab_axis_config(
            range=[-span, span], zeroline=False, title=dict(text=""),
            showticklabels=False, fixedrange=True,
        )
        axes[f"yaxis{index}"] = matlab_axis_config(
            showticklabels=False, fixedrange=True, range=[-0.5, len(row_ids) - 0.5],
        )
        shapes.append(dict(
            type="line", x0=0, x1=0, y0=0, y1=1, xref=f"x{index}", yref="paper",
            line=dict(color="rgba(0,0,0,0.32)", width=1),
        ))
    figure.update_layout(
        height=body_height, title="", margin=dict(l=left_margin, r=20, t=4, b=8),
        showlegend=False, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        template="plotly_white", font=MATLAB_FONT, dragmode=False,
        shapes=shapes, **axes,
    )


def build_standard_diagnostic_figures(
    detail: pd.DataFrame,
    sample_cols: list[str],
    *,
    row_ids: list[str],
    row_labels: list[str],
    marker_colors: list[str],
    marker_sizes: list[int],
    point_custom: np.ndarray,
    heat_custom: np.ndarray,
    heat_hovertemplate: str,
    point_hover_prefix: str,
    left_margin: int,
    body_height: int,
) -> tuple[go.Figure, go.Figure]:
    """Build the fixed header and scrollable diagnostic body."""
    xlabels = list(sample_cols)
    marker = dict(
        color=marker_colors, size=marker_sizes,
        line=dict(width=0.6, color="white"),
    )
    spans = (_span(detail["rt_error"] * 60, 10), _span(detail["ppm_error"], 5))
    body = _build_body_traces(
        detail, xlabels, row_ids, marker, point_custom, heat_custom,
        heat_hovertemplate, point_hover_prefix,
    )
    header = _build_header(xlabels, spans, left_margin)
    _configure_body(
        body, detail, xlabels, row_ids, row_labels, spans, left_margin, body_height,
    )
    return header, body
