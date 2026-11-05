"""Appearance-only adaptation at the interactive viewer's chart boundary."""

from __future__ import annotations

from typing import Any
import re

import plotly.graph_objects as go
import streamlit as st

from presentation.app_theme import current_theme

SURFACE = "#213e50"
TEXT = "#e9f6fa"
MUTED = "#bdd3dd"
GRID = "#38586b"
BORDER = "#64899b"
_NEUTRAL_INK = {
    "black",
    "#000",
    "#000000",
    "#1d1d1f",
    "#303030",
    "#333333",
    "#3a3a3c",
    "#4d4d4d",
    "#6e6e73",
    "#8e8e93",
    "rgba(30,30,30,0.9)",
}


def _readable_ink(colour: Any) -> Any:
    if isinstance(colour, str):
        return TEXT if colour.lower().replace(" ", "") in _NEUTRAL_INK else colour
    if isinstance(colour, (tuple, list)) or hasattr(colour, "tolist"):
        return [_readable_ink(item) for item in colour]
    return colour


def _theme_fonts(node: Any) -> None:
    """Change neutral UI ink, never group/status colours or colour scales."""
    if isinstance(node, list):
        for item in node:
            _theme_fonts(item)
    elif isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("font") and isinstance(value, dict) and "color" in value:
                value["color"] = _readable_ink(value["color"])
            _theme_fonts(value)


def _theme_axes(layout: dict) -> None:
    for name, axis in layout.items():
        if not name.startswith(("xaxis", "yaxis", "scene")) or not isinstance(
            axis, dict
        ):
            continue
        if name.startswith("scene"):
            axis["bgcolor"] = SURFACE
            for dimension in ("xaxis", "yaxis", "zaxis"):
                panel = axis.setdefault(dimension, {})
                panel.update(
                    backgroundcolor=SURFACE,
                    gridcolor=GRID,
                    linecolor=BORDER,
                    zerolinecolor=BORDER,
                    color=MUTED,
                )
        else:
            axis.update(
                gridcolor=GRID, linecolor=BORDER, zerolinecolor=BORDER, color=MUTED
            )
            if axis.get("ticktext") is not None:
                axis["ticktext"] = [
                    re.sub(r'color:[^;"\']+', f"color:{MUTED}", text)
                    if isinstance(text, str)
                    else text
                    for text in axis["ticktext"]
                ]


def themed_viewer_figure(figure: go.Figure, theme: str) -> go.Figure:
    """Keep coordinates, identities, click data, ranges and camera state intact."""
    if theme != "Midnight":
        return figure
    # Never mutate a cached figure shared by another session or the light theme.
    themed = go.Figure(figure)
    layout = themed.layout.to_plotly_json()
    layout.update(template="plotly_dark", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE)
    layout.setdefault("font", {})["color"] = TEXT
    layout.setdefault("title", {}).setdefault("font", {})["color"] = TEXT
    layout.setdefault("legend", {}).update(bgcolor=SURFACE, bordercolor=GRID)
    layout["legend"].setdefault("font", {})["color"] = TEXT
    layout.setdefault("hoverlabel", {}).update(bgcolor=SURFACE, bordercolor=BORDER)
    layout["hoverlabel"].setdefault("font", {})["color"] = TEXT
    _theme_axes(layout)
    _theme_fonts(layout)
    themed.layout = layout
    for trace in themed.data:
        for name in ("textfont", "line", "marker"):
            part = getattr(trace, name, None)
            if part is not None and hasattr(part, "color") and part.color is not None:
                part.color = _readable_ink(part.color)
        # Colourbar labels can have explicit light-theme ink of their own.
        marker = getattr(trace, "marker", None)
        bar = getattr(marker, "colorbar", None)
        if marker is not None and bar is not None and "colorbar" in marker.to_plotly_json():
            bar.tickfont.color = TEXT
            bar.title.font.color = TEXT
            bar.bgcolor = SURFACE
    return themed


def viewer_plotly_chart(figure: go.Figure, *, target: Any = None, **kwargs: Any) -> Any:
    """Apply appearance at the chart boundary without modifying scientific data."""
    theme = current_theme()
    if theme == "Midnight":
        kwargs["theme"] = None
    return (st if target is None else target).plotly_chart(
        themed_viewer_figure(figure, theme), **kwargs)


def browser_viewer_tokens(theme: str) -> dict[str, str]:
    """Appearance tokens for the isolated annotation document, using this adapter's palette."""
    if theme == "Midnight":
        return {"--canvas": "#1a2e3a", "--panel": SURFACE, "--field": "#1a3242",
                "--ink": TEXT, "--muted": MUTED, "--border": BORDER,
                "--accent": "#0a84ff", "--pressed": "#2797ff"}
    return {"--canvas": "#fff", "--panel": "#f6f6f8", "--field": "#fff",
            "--ink": "#1d1d1f", "--muted": "#6e6e73", "--border": "#d1d1d6",
            "--accent": "#006de5", "--pressed": "#2585ed"}
