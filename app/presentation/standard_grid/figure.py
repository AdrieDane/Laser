"""Layered construction of the standard RT–m/z grid figure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from viewer_common import apply_matlab_plot_style


@dataclass(frozen=True)
class StandardGridFigureServices:
    """Narrow services required to draw a conventional grid figure."""

    cloud_dimension: Callable
    observed_dimension: Callable
    node_dimension: Callable
    marker_groups: Callable
    safe_species_label: Callable
    benchmark_point_colours: Callable
    benchmark_visible_plot_candidates: Callable
    benchmark_status_colours: Mapping[str, str]
    candidate_columns: tuple[str, ...]
    hover_template: str
    dimension_titles: Mapping[str, str]


def _empty_differential() -> pd.DataFrame:
    return pd.DataFrame({
        "feature_id": [], "rt": [], "mz": [],
        "log2_fold_change_b_vs_a": [], "p_value": [],
        "detection_a_percent": [], "detection_b_percent": [],
        "cluster_hypothesis": [], "homologous_series_hypothesis": [],
        "group_a": [], "group_b": [],
    })


def _visible_differential(
    differential_features: pd.DataFrame | None,
    axis_ranges: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    differential = (
        pd.DataFrame() if differential_features is None
        else differential_features.copy()
    )
    if not differential.empty:
        differential = differential[
            differential.get("significant", pd.Series(False, index=differential.index))
            .fillna(False).astype(bool)
        ].copy()
        differential["rt"] = pd.to_numeric(differential.get("rt"), errors="coerce")
        differential["mz"] = pd.to_numeric(differential.get("mz"), errors="coerce")
        differential = differential.dropna(subset=["rt", "mz"])
        if axis_ranges:
            xlo, xhi, ylo, yhi = axis_ranges
            differential = differential[
                differential["rt"].between(xlo, xhi)
                & differential["mz"].between(ylo, yhi)
            ]
    if differential.empty:
        return _empty_differential()
    if "homologous_series_hypothesis" not in differential:
        differential["homologous_series_hypothesis"] = (
            "CH2-series evidence not available for this saved run"
        )
    return differential


def _add_feature_layers(
    figure: go.Figure,
    features: pd.DataFrame,
    differential_features: pd.DataFrame | None,
    *,
    show_cloud: bool,
    x_dimension: str,
    y_dimension: str,
    axis_ranges: tuple[float, float, float, float] | None,
    services: StandardGridFigureServices,
) -> None:
    cloud = features.copy() if show_cloud and features is not None else pd.DataFrame()
    if not cloud.empty:
        cloud["rt"] = pd.to_numeric(cloud["rt"], errors="coerce")
        cloud["mz"] = pd.to_numeric(cloud["mz"], errors="coerce")
        cloud = cloud.dropna(subset=["rt", "mz"])
        if axis_ranges:
            xlo, xhi, ylo, yhi = axis_ranges
            cloud = cloud[cloud["rt"].between(xlo, xhi) & cloud["mz"].between(ylo, yhi)]
    if cloud.empty:
        cloud = pd.DataFrame({"rt": [], "mz": [], "feature_id": []})
    figure.add_trace(go.Scattergl(
        x=services.cloud_dimension(cloud, x_dimension),
        y=services.cloud_dimension(cloud, y_dimension), mode="markers",
        marker=dict(color="rgba(165,165,165,0.42)", size=4),
        customdata=cloud[["feature_id", "rt", "mz"]].to_numpy(dtype=object),
        hovertemplate=(
            "feature %{customdata[0]}<br>RT %{customdata[1]:.3f} min<br>"
            "m/z %{customdata[2]:.4f}<extra></extra>"
        ),
        name="features", showlegend=False,
    ))
    differential = _visible_differential(differential_features, axis_ranges)
    fold_change = pd.to_numeric(differential["log2_fold_change_b_vs_a"], errors="coerce")
    p_values = pd.to_numeric(differential["p_value"], errors="coerce")
    sizes = np.clip(-np.log10(p_values.clip(lower=1e-12)) + 8.0, 9.0, 17.0)
    figure.add_trace(go.Scattergl(
        x=services.cloud_dimension(differential, x_dimension),
        y=services.cloud_dimension(differential, y_dimension), mode="markers",
        marker=dict(
            color=np.where(fold_change.ge(0), "#e23b3b", "#1484d6"),
            size=sizes, symbol="diamond", line=dict(width=1.2, color="white"),
        ),
        customdata=differential[[
            "feature_id", "rt", "mz", "group_b", "group_a",
            "log2_fold_change_b_vs_a", "p_value", "detection_a_percent",
            "detection_b_percent", "cluster_hypothesis",
            "homologous_series_hypothesis",
        ]].to_numpy(dtype=object),
        hovertemplate=(
            "unannotated feature %{customdata[0]}<br>"
            "RT %{customdata[1]:.3f} min · m/z %{customdata[2]:.4f}<br>"
            "%{customdata[3]} vs %{customdata[4]} log2FC %{customdata[5]:+.2f}<br>"
            "raw p %{customdata[6]:.3g}<br>detection A/B %{customdata[7]:.0f}% / "
            "%{customdata[8]:.0f}%<br>%{customdata[9]}<br>"
            "%{customdata[10]}<extra></extra>"
        ),
        name="differential unannotated", showlegend=not differential.empty,
    ))


def _add_grid_layers(
    figure: go.Figure, grid_state: dict, plot_df: pd.DataFrame,
    show_mesh: bool, services: StandardGridFigureServices,
) -> None:
    for family in ("db_lines", "c_lines"):
        line = grid_state["mesh_lines"].get(family) if show_mesh else None
        figure.add_trace(go.Scattergl(
            x=line[0] if line else [], y=line[1] if line else [], mode="lines",
            line=dict(color="rgba(150,110,220,0.55)", width=1),
            hoverinfo="skip", showlegend=False,
        ))
    conn = plot_df.dropna(subset=["feat_rt", "feat_mz", "node_rt", "node_mz"])
    inc_xs: list[float | None] = []
    inc_ys: list[float | None] = []
    dec_xs: list[float | None] = []
    dec_ys: list[float | None] = []
    for _, row in conn.iterrows():
        xs = [services.observed_dimension(row, "RT"), services.node_dimension(row, "RT"), None]
        ys = [services.observed_dimension(row, "m/z"), services.node_dimension(row, "m/z"), None]
        target = (inc_xs, inc_ys) if row["feat_rt"] >= row["node_rt"] else (dec_xs, dec_ys)
        target[0].extend(xs)
        target[1].extend(ys)
    for xs, ys, colour, name in (
        (inc_xs, inc_ys, "rgba(255, 202, 40, 0.62)", "increased RT"),
        (dec_xs, dec_ys, "rgba(220, 55, 55, 0.58)", "decreased RT"),
    ):
        figure.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(color=colour, width=1.4),
            hoverinfo="skip", showlegend=False, name=name,
        ))
    figure.add_trace(go.Scatter(
        x=services.node_dimension(conn, "RT"), y=services.node_dimension(conn, "m/z"),
        mode="markers", marker=dict(color="rgba(150,110,220,0.8)", size=4, symbol="x"),
        hoverinfo="skip", showlegend=False, name="grid nodes",
    ))


def _add_species_layers(
    figure: go.Figure,
    plot_df: pd.DataFrame,
    *,
    x_dimension: str,
    y_dimension: str,
    show_labels: bool,
    selected_keys: set[str],
    projected_best_key: str | None,
    services: StandardGridFigureServices,
) -> None:
    benchmark_active = bool(
        "benchmark_status" in plot_df
        and plot_df["benchmark_status"].astype(str).isin(services.benchmark_status_colours).any()
    )
    for mask, marker, name in services.marker_groups(plot_df):
        sub = plot_df[mask]
        if projected_best_key is not None:
            sub = sub[~sub["key"].astype(str).eq(projected_best_key)]
        figure.add_trace(go.Scatter(
            x=services.observed_dimension(sub, x_dimension),
            y=services.observed_dimension(sub, y_dimension),
            mode="markers+text" if show_labels else "markers",
            marker=dict(marker),
            selectedpoints=np.flatnonzero(sub["key"].isin(selected_keys).to_numpy()).tolist(),
            selected=(dict(marker=dict(opacity=1.0)) if benchmark_active
                      else dict(marker=dict(color="#d62728"))),
            unselected=dict(marker=dict(opacity=1.0)),
            text=services.safe_species_label(sub) if show_labels and not sub.empty else None,
            textposition="top center", textfont=dict(size=9, color="rgba(30,30,30,0.9)"),
            customdata=sub.reindex(columns=services.candidate_columns).to_numpy(dtype=object),
            hovertemplate=services.hover_template, name=name, showlegend=False,
        ))


def _add_projection_layer(
    figure: go.Figure,
    rows: pd.DataFrame,
    *,
    best: bool,
    x_dimension: str,
    y_dimension: str,
    show_labels: bool,
    selected_keys: set[str],
    services: StandardGridFigureServices,
) -> None:
    fallback = "#2e9f64" if best else "#af52de"
    figure.add_trace(go.Scatter(
        x=services.observed_dimension(rows, x_dimension),
        y=services.observed_dimension(rows, y_dimension),
        mode="markers+text" if show_labels else "markers",
        marker=dict(
            color=services.benchmark_point_colours(rows, fallback=fallback),
            size=12 if best else 10,
            line=dict(width=3.0 if best else 0.8, color="#af52de" if best else "#5e2a84"),
        ),
        selectedpoints=(np.flatnonzero(rows["key"].isin(selected_keys).to_numpy()).tolist()
                        if not rows.empty and not best else None),
        selected=dict(marker=dict(color="#af52de", size=12)),
        unselected=dict(marker=dict(opacity=1.0)),
        text=rows["feat_feature_id"].astype(str) if show_labels and not rows.empty else None,
        textposition="top center", textfont=dict(size=9, color="#1d1d1f" if best else "#6d2f8f"),
        customdata=rows.reindex(columns=services.candidate_columns).to_numpy(dtype=object),
        hovertemplate=services.hover_template,
        name="best same-ID feature" if best else "same lipid ID feature",
        showlegend=False,
    ))


def build_standard_grid_figure(
    grid_state: dict,
    features: pd.DataFrame,
    *,
    x_dimension: str = "RT", y_dimension: str = "m/z",
    show_mesh: bool = True, show_labels: bool = True, show_cloud: bool = False,
    selected_keys: set[str] | None = None,
    projected_alternatives: pd.DataFrame | None = None,
    projected_best: pd.DataFrame | None = None,
    axis_ranges: tuple[float, float, float, float] | None = None,
    saved_view: dict | None = None, uirevision: str | None = None,
    show_unbenchmarked: bool = True,
    differential_features: pd.DataFrame | None = None,
    services: StandardGridFigureServices,
) -> go.Figure:
    """Compose stable feature, grid, species, and projection layers."""
    selected_keys = selected_keys or set()
    plot_df = services.benchmark_visible_plot_candidates(
        grid_state["plot_df"], show_unbenchmarked=show_unbenchmarked,
    )
    empty = plot_df.iloc[0:0]
    alternatives = empty if projected_alternatives is None else projected_alternatives
    best = empty if projected_best is None else projected_best
    best_key = str(best.iloc[0]["key"]) if not best.empty else None
    figure = go.Figure()
    _add_feature_layers(
        figure, features, differential_features, show_cloud=show_cloud,
        x_dimension=x_dimension, y_dimension=y_dimension, axis_ranges=axis_ranges,
        services=services,
    )
    _add_grid_layers(figure, grid_state, plot_df, show_mesh, services)
    _add_species_layers(
        figure, plot_df, x_dimension=x_dimension, y_dimension=y_dimension,
        show_labels=show_labels, selected_keys=selected_keys,
        projected_best_key=best_key,
        services=services,
    )
    _add_projection_layer(
        figure, alternatives, best=False, x_dimension=x_dimension,
        y_dimension=y_dimension, show_labels=show_labels, selected_keys=selected_keys,
        services=services,
    )
    _add_projection_layer(
        figure, best, best=True, x_dimension=x_dimension,
        y_dimension=y_dimension, show_labels=show_labels, selected_keys=selected_keys,
        services=services,
    )
    apply_matlab_plot_style(figure)
    figure.update_xaxes(title=services.dimension_titles[x_dimension])
    figure.update_yaxes(title=services.dimension_titles[y_dimension])
    saved_view = saved_view or {}
    if saved_view.get("xaxis") and saved_view.get("yaxis"):
        figure.update_xaxes(range=saved_view["xaxis"])
        figure.update_yaxes(range=saved_view["yaxis"])
    elif axis_ranges and x_dimension == "RT" and y_dimension == "m/z":
        xlo, xhi, ylo, yhi = axis_ranges
        figure.update_xaxes(range=[xlo, xhi])
        figure.update_yaxes(range=[ylo, yhi])
    figure.update_layout(
        height=790, clickmode="event", dragmode="pan", uirevision=uirevision,
        title="", showlegend=False, margin=dict(l=12, r=12, t=18, b=12),
    )
    return figure
