"""Candidate-quality diagnostic strip for the standard grid browser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import streamlit as st

from presentation.standard_grid.diagnostics import build_standard_diagnostic_figures


@dataclass(frozen=True)
class StandardGridDiagnosticServices:
    """Narrow interaction and styling dependencies for diagnostics."""

    benchmark_status_colours: Mapping[str, str]
    left_margin: int
    plotly_click_bridge: Callable
    sync_component_selection: Callable



def _display_labels(detail: pd.DataFrame) -> pd.Series:
    labels = detail["name"].astype(str)
    if detail["adduct"].nunique() > 1:
        labels += " " + detail["adduct"].astype(str)
    automatic = detail["automatic_candidate"].fillna(False).astype(bool)
    run_candidate = detail.get(
        "run_candidate", pd.Series(True, index=detail.index)
    ).fillna(False).astype(bool)
    labels = labels.mask(~automatic & run_candidate, labels + " [candidate]")
    return labels.mask(
        ~automatic & ~run_candidate, labels + " [widened window]"
    )


def _marker_colours(
    detail: pd.DataFrame, active_keys: set[str],
    services: StandardGridDiagnosticServices,
) -> list[str]:
    normal = [
        "#d62728" if key in active_keys else
        "#ffd400" if anchor else
        "#af52de" if confirmed else
        "#2e9f64" if automatic else
        "#2677c9" if run_candidate else "#83bbed"
        for key, anchor, confirmed, automatic, run_candidate in zip(
            detail["key"],
            detail["anchor"].fillna(False),
            detail["confirmed"].fillna(False),
            detail["automatic_candidate"].fillna(False),
            detail.get(
                "run_candidate", pd.Series(True, index=detail.index)
            ).fillna(False),
        )
    ]
    benchmark = detail.get(
        "benchmark_status", pd.Series("", index=detail.index)
    ).astype(str)
    return [
        services.benchmark_status_colours.get(status, fallback)
        for status, fallback in zip(benchmark, normal)
    ]


def _diagnostic_figures(
    detail: pd.DataFrame, sample_cols: list[str], active_keys: set[str],
    services: StandardGridDiagnosticServices,
):
    detail["row_id"] = [
        f"{index:04d}|{name}|{adduct}"
        for index, (name, adduct) in enumerate(zip(detail["name"], detail["adduct"]))
    ]
    heat_custom: np.ndarray = np.empty((len(detail), len(sample_cols), 3), dtype=object)
    heat_custom[:, :, 0] = detail["name"].astype(str).to_numpy()[:, None]
    heat_custom[:, :, 1] = detail["adduct"].astype(str).to_numpy()[:, None]
    heat_custom[:, :, 2] = detail["key"].astype(str).to_numpy()[:, None]
    point_custom = np.column_stack([
        detail["name"].astype(str), detail["adduct"].astype(str),
        detail["key"].astype(str),
    ])
    return build_standard_diagnostic_figures(
        detail, sample_cols, row_ids=detail["row_id"].tolist(),
        row_labels=_display_labels(detail).tolist(),
        marker_colors=_marker_colours(detail, active_keys, services),
        marker_sizes=[10 if key in active_keys else 7 for key in detail["key"]],
        point_custom=point_custom, heat_custom=heat_custom,
        heat_hovertemplate=(
            "%{customdata[0]} %{customdata[1]}<br>"
            "%{x}<br>log10 %{z:.2f}<extra></extra>"
        ),
        point_hover_prefix="%{customdata[0]} %{customdata[1]}",
        left_margin=services.left_margin,
        body_height=max(350, 22 * len(detail) + 30),
    )


def _render_figures(
    figures, detail: pd.DataFrame, chart_key: str,
    services: StandardGridDiagnosticServices,
) -> None:
    header, body = figures
    component_key = "standard_diagnostic_click::viewer"
    config = {
        "displayModeBar": False, "scrollZoom": False,
        "doubleClick": False, "responsive": True,
    }
    with st.container(border=True, key="standard-annotation-diagnostics-shell"):
        st.plotly_chart(
            header, width="stretch", key="standard-detail-header::viewer",
            config=config,
        )
        with st.container(height=470, border=False):
            st.plotly_chart(
                body, width="stretch", key="standard-detail::viewer",
                config=config,
            )
            services.plotly_click_bridge()(
                key=component_key, data={"chartKey": chart_key}, default={},
                height=0,
                on_point_change=lambda: services.sync_component_selection(
                    component_key, chart_key, set(detail["key"])
                ),
            )


def _render_annotation_diagnostics(
    plot_df: pd.DataFrame,
    sample_cols: list[str],
    *,
    chart_key: str,
    active_keys: set[str],
    services: StandardGridDiagnosticServices,
):
    """Render intensity and errors for each species on the active grid."""
    detail = (
        plot_df.sort_values(["carbon", "db", "name", "adduct", "absresid"])
        .drop_duplicates(subset=["name", "adduct"], keep="first")
        .reset_index(drop=True)
    )
    if not sample_cols or detail.empty:
        st.caption("Sample-level intensity columns were not found for this stream.")
        return
    figures = _diagnostic_figures(detail, sample_cols, active_keys, services)
    _render_figures(figures, detail, chart_key, services)
