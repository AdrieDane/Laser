"""Interactive RT–m/z grid browser for standard-lipidomics results.

This is the standard-lipidomics counterpart of the 4D MS viewer and mirrors it
panel for panel: the same three-column review layout, the same predicted-RT mesh
with confirmable anchors, the same click and box selection, per-sample
statistics, diagnostic strips and same-lipid-ID projection.

Everything conventional lipidomics carries no evidence for is absent rather than
disabled: there is no CCS axis and no 3D view, and no MS/MS viewer or fragment
criteria. XCMS projects do carry centroided MS1 scans, so this browser can
extract diagnostic chromatograms on demand. MATLAB-engine results also carry
diagnostic-only m/z/RT grid-coherence scores. The RT grid comes from the guarded
automatic class models fitted during the run.

The review controls here are deliberately view-only. Tightening a tolerance,
confirming an anchor or deleting a row changes what this browser shows; the
automatic annotations and every export remain exactly what the run produced.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from presentation.viewer_theme import viewer_plotly_chart

from annotator.grid_coherence import GRID_COHERENCE_COLUMNS
from presentation.standard_grid.state import (
    apply_review_criteria,  # noqa: F401 - compatibility re-export
    build_class_grid_state,
    review_criteria_mask as _review_criteria_mask,  # noqa: F401
    select_plotted_species,  # noqa: F401 - compatibility re-export
)
from presentation.standard_grid.controls import (
    render_review_controls as _render_review_controls,
    review_criteria_signature as _review_criteria_signature,  # noqa: F401
)
from presentation.standard_grid.traces import (
    _feature_trace_colour_map,  # noqa: F401 - compatibility re-export
    _feature_trace_range,  # noqa: F401 - compatibility re-export
    _render_standard_feature_traces,
    _same_id_trace_candidates,
)
from presentation.standard_grid.diagnostics import build_standard_diagnostic_figures
from presentation.standard_grid.data import (
    StandardGridDataServices,
    prepare_standard_grid_data as _prepare_standard_grid_data,
)
from presentation.standard_grid.annotation_diagnostics import (
    StandardGridDiagnosticServices,
    _render_annotation_diagnostics as _render_annotation_diagnostics_presenter,
)
from presentation.standard_grid.figure import (
    StandardGridFigureServices,
    build_standard_grid_figure as _build_standard_grid_figure,
)
from presentation.standard_grid.page import (
    StandardGridPageServices,
    render_standard_grid_browser as _render_standard_grid_page,
)
from viewer_common import (
    MATLAB_FONT,
    matlab_axis_config,
    plotly_click_bridge,
    row_sample_values,
)

# Session state is namespaced so a review in standard mode can never inherit
# selections, anchors or deletions left behind by the 4D viewer.
_SELECTION_STATE = "standard_selected_grid_keys"
_ANCHOR_STATE = "standard_confirmed_anchors"
_DELETED_STATE = "standard_deleted_annotations"
_PRIMARY_STATE = "standard_primary_feature_overrides"
_VIEW_RANGE_STATE = "standard_grid_view_ranges"
_PLOT_VIEW_STATE = "standard_grid_plot_views"
_PROJECTION_STATE = "standard_project_same_lipid_id_features"
_CONTROLS_STATE = "standard_review_controls_visible"
_AUTOMATIC_PRIMARY_LABEL = "Automatic (best database-grid fit)"
# Benchmark colours intentionally differ from the normal reviewer colours:
# they communicate agreement with the MS_data/MATLAB reference, not LASER's
# accepted/original-run/widened-window candidate provenance.
_BENCHMARK_STATUS_COLOURS = {
    "overlap_same_feature": "#16a34a",
    "overlap_different_feature": "#16a34a",
    "overlap_feature_not_evaluable": "#d97706",
    "laser_only": "#f97316",
    "matlab_only": "#7c3aed",
}
_BENCHMARK_STATUS_LABELS = {
    "overlap_same_feature": "both MATLAB + LASER",
    "overlap_different_feature": "both MATLAB + LASER",
    "overlap_feature_not_evaluable": ("both identities; MATLAB feature not evaluable"),
    "laser_only": "LASER-only",
    "matlab_only": "MATLAB-only",
}

# Columns forwarded to the browser as Plotly ``customdata``. The key is last:
# the click bridge identifies a clicked point by the final entry.
_CD_COLS = [
    "feat_feature_id",
    "name",
    "adduct",
    "ppm_error",
    "rt_error",
    "feat_mz",
    "feat_rt",
    "auto_annotation_status",
    "carbon",
    "db",
    "grid_resid",
    "np_rt_identity",
    "grid_coherence_score",
    "key",
]

_HOVER = (
    "<b>%{customdata[1]}</b> %{customdata[2]}<br>"
    "feature %{customdata[0]}<br>"
    "m/z %{customdata[5]:.4f}  (Δ %{customdata[3]:.2f} ppm)<br>"
    "RT %{customdata[6]:.2f}  (Δgrid %{customdata[10]:.3f} min)<br>"
    "grid coherence %{customdata[12]:.3f}<br>"
    "automatic status: %{customdata[7]}<extra></extra>"
)

_DIMENSION_TITLES = {"RT": "Retention time (min)", "m/z": "m/z"}

# Four diagnostic strips instead of the 4D viewer's five: conventional data has
# no CCS deviation, while MATLAB-engine m/z/RT coherence remains diagnostic.
_DIAGNOSTIC_CLASS_LEFT_MARGIN = 135
_DIAGNOSTIC_FEATURE_LEFT_MARGIN = 175

_FEATURE_METADATA_COLUMNS = {
    "feature_id",
    "mz",
    "rt",
    "mobility",
    "mobility_unit",
    "npeaks",
    "intensity",
    "quantification_measure",
    "mzmin",
    "mzmax",
    "rtmin",
    "rtmax",
    "into",
    "intb",
    "maxo",
    "sn",
}

# Enough of the candidate schema for the browser to lay out an empty grid
# rather than raise when a stream produced no m/z match at all.
_EMPTY_CANDIDATE_COLUMNS = [
    "feat_feature_id",
    "name",
    "class",
    "base",
    "adduct",
    "carbon",
    "db",
    "feat_rt",
    "feat_mz",
    "ref_mz",
    "ref_rt",
    "ppm_error",
    "rt_error",
    "grid_class",
    "key",
    "deletion_key",
    "automatic_candidate",
    "automatic_rt_anchor",
    "auto_annotation_status",
    "np_rt_identity",
    *GRID_COHERENCE_COLUMNS,
    "grid_coherence_threshold",
]


def _safe_species_label(df: pd.DataFrame) -> pd.Series:
    """``C:DB`` labels that tolerate reference rows without a composition."""
    carbon = pd.to_numeric(df.get("carbon"), errors="coerce")
    db = pd.to_numeric(df.get("db"), errors="coerce")
    return pd.Series(
        [
            f"{int(c)}:{int(d)}" if np.isfinite(c) and np.isfinite(d) else str(name)
            for c, d, name in zip(carbon, db, df.get("name", carbon))
        ],
        index=df.index,
        dtype=object,
    )


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def standard_sample_columns(features: pd.DataFrame) -> list[str]:
    """Per-sample intensity columns of a MATLAB/XCMS peak-group list.

    ``load_standard_feature_list`` records the same list in ``features.attrs``,
    but DataFrame attrs do not reliably survive Streamlit's result cache, so the
    browser recomputes them from the columns themselves.
    """
    if features is None or features.empty:
        return []
    stored = features.attrs.get("sample_columns")
    if stored:
        return [column for column in stored if column in features.columns]
    return [
        str(column)
        for column in features.columns
        if column not in _FEATURE_METADATA_COLUMNS
        and not str(column).endswith("::detected")
        and not str(column).casefold().startswith("group_")
    ]


def _grid_class(data: pd.DataFrame) -> pd.Series:
    """Keep ether, plasmalogen, and sphingoid subclasses on separate grids."""
    lipid_class = data["class"].fillna("").astype(str)
    base = data.get("base", pd.Series("", index=data.index)).fillna("").astype(str)
    return pd.Series(
        [
            f"{cls}-{subclass}" if subclass else cls
            for cls, subclass in zip(lipid_class, base)
        ],
        index=data.index,
        dtype=object,
    )


def annotation_identity(frame: pd.DataFrame) -> pd.Series:
    """The candidate key: which feature was called which lipid ion.

    On an anchored NP stream the exported ``name`` is the identity-level label
    and the reference identity is preserved separately; the grid is drawn from
    reference identities, so the key uses those. The MATLAB engine additionally
    resolves isotopologues of the same species and adduct as separate features,
    so its ``isotopologue`` column joins the key to keep them distinct.
    """
    if frame is None or frame.empty:
        return pd.Series(dtype=object)
    identity = (
        frame["reference_identity"] if "reference_identity" in frame else frame["name"]
    )
    key = (
        frame["feat_feature_id"].astype(str)
        + "|"
        + identity.astype(str)
        + "|"
        + frame["adduct"].astype(str)
    )
    if "isotopologue" in frame:
        key = key + "|" + frame["isotopologue"].fillna("").astype(str)
    return key


def _accepted_annotation_keys(annotations: pd.DataFrame) -> set[str]:
    """Keys of the candidates the run accepted automatically."""
    if annotations is None or annotations.empty:
        return set()
    accepted = (
        annotations["automatic_accepted"].fillna(False).astype(bool)
        if "automatic_accepted" in annotations
        else pd.Series(False, index=annotations.index)
    )
    return set(annotation_identity(annotations)[accepted])


def _status_lookup(annotations: pd.DataFrame) -> dict[str, str]:
    """Map every reviewed candidate key to the run's automatic status text."""
    if annotations is None or annotations.empty:
        return {}
    if "auto_annotation_status" not in annotations:
        return {}
    return dict(
        zip(
            annotation_identity(annotations),
            annotations["auto_annotation_status"].astype(str),
        )
    )


def _normalise_benchmark_species(value: object) -> str:
    """Match standard reviewer names to the benchmark identity key spelling."""
    text = str(value or "").strip()
    if text.startswith("IS: "):
        text = text[4:]
    if text.endswith(" (IS)"):
        text = text[:-5]
    text = " ".join(text.split())
    return re.sub(r"\b([12]-acyl)\s+(LPC|LPE)\b", r"\1\2", text)


def _normalise_benchmark_class(value: object) -> str:
    """Use the benchmark's canonical spelling for positional lyso classes."""
    text = " ".join(str(value or "").strip().split())
    return re.sub(r"\b([12]-acyl)\s+(LPC|LPE)\b", r"\1\2", text)


def _benchmark_identity_series(frame: pd.DataFrame) -> pd.Series:
    """Construct the class/species/adduct identity used by benchmarking."""
    if frame is None or frame.empty:
        return pd.Series(dtype=object, index=frame.index if frame is not None else None)
    species = frame["name"].map(_normalise_benchmark_species)
    if "reference_identity" in frame:
        reference = frame["reference_identity"].map(_normalise_benchmark_species)
        use_reference = reference.ne("") & reference.str.lower().ne("nan")
        species = reference.where(use_reference, species)
    lipid_class = frame.get("class", pd.Series("", index=frame.index)).map(
        _normalise_benchmark_class
    )
    adduct = (
        frame.get("adduct", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    return lipid_class + "|" + species + "|" + adduct


def _benchmark_point_colours(
    frame: pd.DataFrame,
    fallback: str | None = None,
) -> str | list[str] | None:
    """Return benchmark fill colours, or the normal reviewer fallback."""
    if frame is None or frame.empty or "benchmark_status" not in frame:
        return fallback
    statuses = frame["benchmark_status"].astype(str)
    colours = [
        _BENCHMARK_STATUS_COLOURS.get(status, fallback or "#64748b")
        for status in statuses
    ]
    if not statuses.isin(_BENCHMARK_STATUS_COLOURS).any():
        return fallback
    return colours


def _benchmark_visible_plot_candidates(
    frame: pd.DataFrame,
    *,
    show_unbenchmarked: bool,
) -> pd.DataFrame:
    """Hide only unclassified review candidates in benchmark mode.

    The browser retains the complete XCMS candidate pool for diagnostics and
    same-ID trace review.  This display filter affects only the species points
    drawn in the main grid: benchmark-coloured points, accepted annotations and
    anchors remain visible, while statusless review points can be turned off
    without mislabelling them as LASER-only.
    """
    if frame is None or frame.empty or show_unbenchmarked:
        return frame
    if "benchmark_status" not in frame:
        return frame
    status = frame["benchmark_status"].astype(str)
    benchmark_known = status.isin(_BENCHMARK_STATUS_COLOURS)
    automatic = (
        frame.get("automatic_candidate", pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(bool)
    )
    anchor = (
        frame.get("anchor", pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(bool)
    )
    confirmed = (
        frame.get("confirmed", pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(bool)
    )
    return frame[benchmark_known | automatic | anchor | confirmed].copy()


def _annotation_column_lookup(
    annotations: pd.DataFrame,
    column: str,
) -> dict[str, object]:
    if annotations is None or annotations.empty or column not in annotations:
        return {}
    return dict(zip(annotation_identity(annotations), annotations[column]))


def _grid_data_services() -> StandardGridDataServices:
    return StandardGridDataServices(
        empty_candidate_columns=tuple(_EMPTY_CANDIDATE_COLUMNS),
        grid_class=_grid_class,
        annotation_identity=annotation_identity,
        normalise_benchmark_species=_normalise_benchmark_species,
        benchmark_identity_series=_benchmark_identity_series,
        accepted_annotation_keys=_accepted_annotation_keys,
        status_lookup=_status_lookup,
        annotation_column_lookup=_annotation_column_lookup,
    )


def prepare_standard_grid_data(
    stream_result: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility façade for focused grid-data preparation."""
    return _prepare_standard_grid_data(
        stream_result,
        services=_grid_data_services(),
    )


# --------------------------------------------------------------------------- #
# Selection plumbing (shared with the 4D viewer's click bridge)
# --------------------------------------------------------------------------- #
def _point_key_from_component(component_key):
    component_state = st.session_state.get(component_key, {})
    point = component_state.get("point") if component_state else None
    customdata = point.get("customdata") if point else None
    return customdata[-1] if customdata and len(customdata) >= 3 else None


def _store_component_view(payload, view_key):
    view = payload.get("view") if payload else None
    if view_key and view:
        st.session_state.setdefault(_PLOT_VIEW_STATE, {})[view_key] = view


def _sync_component_selection(component_key, grid_key, valid_keys, view_key=None):
    """Run before the component-triggered rerun so the new plot is in sync."""
    component_state = st.session_state.get(component_key, {})
    point = component_state.get("point") if component_state else None
    clicked_key = _point_key_from_component(component_key)
    if clicked_key in valid_keys:
        st.session_state.setdefault(_SELECTION_STATE, {})[grid_key] = [clicked_key]
    _store_component_view(point, view_key)


def _sync_component_multi_selection(component_key, grid_key, valid_keys, view_key):
    """Copy box/lasso selections and their current axes into shared state."""
    component_state = st.session_state.get(component_key, {})
    selection = component_state.get("selection") if component_state else None
    keys = []
    for customdata in selection.get("customdata", []) if selection else []:
        if (
            customdata
            and len(customdata) >= len(_CD_COLS)
            and customdata[-1] in valid_keys
        ):
            keys.append(customdata[-1])
    st.session_state.setdefault(_SELECTION_STATE, {})[grid_key] = list(
        dict.fromkeys(keys)
    )
    _store_component_view(selection, view_key)


def _toggle_same_lipid_id_projection():
    st.session_state[_PROJECTION_STATE] = not bool(
        st.session_state.get(_PROJECTION_STATE, False)
    )


def _primary_feature_overrides() -> dict:
    return st.session_state.setdefault(_PRIMARY_STATE, {})


def _apply_primary_feature_choice(species_id: str, widget_key: str, context: dict):
    """Store or clear the main annotation feature chosen for one lipid ID."""
    choice = st.session_state.get(widget_key)
    overrides = _primary_feature_overrides()
    use_automatic = not choice or choice == _AUTOMATIC_PRIMARY_LABEL
    if use_automatic:
        overrides.pop(str(species_id), None)
    else:
        overrides[str(species_id)] = str(choice)
    target = context.get("automatic_feature_id") if use_automatic else str(choice)
    plot_key = context.get("grid_feature_keys", {}).get(str(target))
    if plot_key is not None:
        st.session_state.setdefault(_SELECTION_STATE, {})[context["chart_key"]] = [
            plot_key
        ]


def _set_anchor(chart_key: str, name: str, add: bool):
    """Add/remove a species from the confirmed-anchor set for this class."""
    store = st.session_state.setdefault(_ANCHOR_STATE, {})
    names = store.setdefault(chart_key, set())
    (names.add if add else names.discard)(name)


def _delete_annotations(chart_key: str, deletion_keys):
    """Hide lipid annotations from this browsing session only."""
    deleted = st.session_state.setdefault(_DELETED_STATE, set())
    deleted.update(deletion_keys)
    st.session_state.setdefault(_SELECTION_STATE, {})[chart_key] = []


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _observed_dimension(data, dimension):
    return {"RT": data["feat_rt"], "m/z": data["feat_mz"]}[dimension]


def _node_dimension(data, dimension):
    return {"RT": data["node_rt"], "m/z": data["node_mz"]}[dimension]


def _cloud_dimension(data, dimension):
    return {"RT": data["rt"], "m/z": data["mz"]}[dimension]


def _marker_groups(plot_df: pd.DataFrame):
    automatic_rt_anchor = (
        plot_df.get("automatic_rt_anchor", pd.Series(False, index=plot_df.index))
        .fillna(False)
        .astype(bool)
    )
    confirmed = plot_df["confirmed"].fillna(False).astype(bool)
    automatic = plot_df["automatic_candidate"].fillna(False).astype(bool)
    run_candidate = (
        plot_df.get("run_candidate", pd.Series(True, index=plot_df.index))
        .fillna(False)
        .astype(bool)
    )
    normal_groups = [
        (
            ~automatic_rt_anchor & ~confirmed & automatic,
            dict(color="#2e9f64", size=8, line=dict(width=0.4, color="white")),
            "accepted annotation",
        ),
        (
            ~automatic_rt_anchor & ~confirmed & ~automatic & run_candidate,
            dict(color="#2677c9", size=8, line=dict(width=0.4, color="white")),
            "candidate",
        ),
        (
            ~automatic_rt_anchor & ~confirmed & ~automatic & ~run_candidate,
            dict(color="#83bbed", size=8, line=dict(width=0.4, color="white")),
            "widened-window candidate",
        ),
        (
            automatic_rt_anchor,
            dict(color="#ffd400", size=11, line=dict(width=1.4, color="#8a5a00")),
            "engine-selected RT anchor",
        ),
        (
            confirmed & ~automatic_rt_anchor,
            dict(
                color="#af52de",
                size=12,
                symbol="diamond",
                line=dict(width=1.4, color="#5e2a84"),
            ),
            "manually confirmed review anchor",
        ),
    ]
    benchmark = plot_df.get(
        "benchmark_status", pd.Series("", index=plot_df.index)
    ).astype(str)
    benchmark_mask = benchmark.isin(_BENCHMARK_STATUS_COLOURS)
    if not benchmark_mask.any():
        return normal_groups

    benchmark_groups = []
    for status, colour in _BENCHMARK_STATUS_COLOURS.items():
        benchmark_groups.append(
            (
                benchmark.eq(status),
                dict(color=colour, size=9, line=dict(width=0.8, color="white")),
                _BENCHMARK_STATUS_LABELS[status],
            )
        )
    # A candidate can be visible in the reviewer without being part of the
    # benchmark comparison (for example a deliberately widened review window).
    # Keep those points on the normal reviewer palette rather than silently
    # assigning them a benchmark category.
    benchmark_fallback = ~benchmark_mask
    benchmark_groups.extend(
        (mask & benchmark_fallback, marker, name)
        for mask, marker, name in normal_groups
    )
    return benchmark_groups


def _axis_ranges(grid_state: dict, features: pd.DataFrame):
    """Plot extent from the mesh, falling back to species and reference."""

    def finite(values):
        return [value for value in values if value is not None and np.isfinite(value)]

    mesh_lines = grid_state["mesh_lines"]
    plot_df = grid_state["plot_df"]
    reference = grid_state["reference"]
    if mesh_lines.get("db_lines"):
        xs = finite(mesh_lines["db_lines"][0])
        ys = finite(mesh_lines["db_lines"][1])
    else:
        xs = list(reference["grid_rt"].dropna())
        ys = list(reference["ref_mz"].dropna())
    xs += list(plot_df["feat_rt"].dropna())
    ys += list(plot_df["feat_mz"].dropna())
    if xs and ys:
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    elif features is not None and not features.empty:
        rt = pd.to_numeric(features["rt"], errors="coerce")
        mz = pd.to_numeric(features["mz"], errors="coerce")
        x0, x1, y0, y1 = rt.min(), rt.max(), mz.min(), mz.max()
    else:
        return None
    xpad = max((x1 - x0) * 0.08, 0.3)
    ypad = max((y1 - y0) * 0.08, 10)
    return x0 - xpad, x1 + xpad, y0 - ypad, y1 + ypad


def _render_statistics(active_row, sample_cols: list[str]):
    """Per-sample intensities for the selected lipid."""
    label = active_row["name"] if active_row is not None else "no selection"
    st.markdown(
        f'<div class="matlab-title">Statistics: {label}</div>',
        unsafe_allow_html=True,
    )
    if active_row is None or not sample_cols:
        st.caption("Select a lipid to show per-sample intensities.")
        return
    values = row_sample_values(active_row, sample_cols)
    stat_df = pd.DataFrame({"sample": sample_cols, "value": values})
    stat_df = stat_df[np.isfinite(stat_df["value"])]
    fig = go.Figure()
    # XCMS peak-group lists carry no sample metadata, so every sample is drawn
    # under its own column name in one series rather than invented groups.
    fig.add_trace(
        go.Bar(
            x=stat_df["sample"],
            y=stat_df["value"],
            name="sample",
            marker_color="#636366",
            hovertemplate="<b>%{x}</b><br>%{y:.3g}<extra></extra>",
        )
    )
    fig.update_layout(
        height=245,
        margin=dict(l=54, r=118, t=36, b=64),
        title=dict(
            text=f"{active_row['name']} {active_row['adduct']}",
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=MATLAB_FONT,
        title_font=dict(family=MATLAB_FONT["family"], size=12, color="#1d1d1f"),
        xaxis=matlab_axis_config(tickangle=-45),
        yaxis=matlab_axis_config(
            title=dict(text="intensity"),
            rangemode="tozero",
            tickformat=".2s",
        ),
        showlegend=False,
        bargap=0.12,
    )
    viewer_plotly_chart(
        fig,
        width="stretch",
        key="standard-stats::viewer",
        config={"displayModeBar": False},
    )


def _diagnostic_services() -> StandardGridDiagnosticServices:
    return StandardGridDiagnosticServices(
        benchmark_status_colours=_BENCHMARK_STATUS_COLOURS,
        left_margin=_DIAGNOSTIC_CLASS_LEFT_MARGIN,
        plotly_click_bridge=plotly_click_bridge,
        sync_component_selection=_sync_component_selection,
    )


def _render_annotation_diagnostics(
    plot_df: pd.DataFrame,
    sample_cols: list[str],
    *,
    chart_key: str,
    active_keys: set[str],
):
    """Compatibility façade for the focused diagnostic presenter."""
    return _render_annotation_diagnostics_presenter(
        plot_df,
        sample_cols,
        chart_key=chart_key,
        active_keys=active_keys,
        services=_diagnostic_services(),
    )


def _render_primary_feature_control(
    candidates: pd.DataFrame, species_id: str, *, key_prefix: str, context: dict
):
    """Let the user assign the main annotation feature for one lipid species ID."""
    feature_ids = list(dict.fromkeys(candidates["feat_feature_id"].astype(str)))
    overrides = _primary_feature_overrides()
    assigned = str(overrides.get(str(species_id), "")) or None
    if len(feature_ids) < 2 and assigned is None:
        return

    widget_key = f"{key_prefix}-primary-feature::{species_id}"
    options = [_AUTOMATIC_PRIMARY_LABEL, *feature_ids]
    # The override store is the single source of truth; the selector only
    # reflects it, so a cleared or stale assignment cannot survive in the widget.
    st.session_state[widget_key] = (
        assigned if assigned in feature_ids else _AUTOMATIC_PRIMARY_LABEL
    )
    plottable = set(context.get("grid_feature_keys", {}))

    def option_label(value: str) -> str:
        if value == _AUTOMATIC_PRIMARY_LABEL:
            return value
        if value in plottable:
            return f"feature {value}"
        return f"feature {value} · not in this class/adduct view"

    st.selectbox(
        "Main annotation feature",
        options,
        key=widget_key,
        format_func=option_label,
        on_change=_apply_primary_feature_choice,
        args=(species_id, widget_key, context),
        help=(
            "Several features share this exact lipid ID. The assigned feature "
            "becomes the one the grid plots for this species, and the one its "
            "grid connector and anchor correction use."
        ),
    )
    if assigned in feature_ids and assigned not in plottable:
        st.caption(
            f"Feature {assigned} is assigned as the main annotation feature "
            f"for {species_id}, but the current review window does not retain "
            "it. The assignment applies where that feature is shown."
        )
    elif assigned in feature_ids:
        st.caption(
            f"Feature {assigned} is assigned as the main annotation feature "
            f"for {species_id}."
        )
    elif context.get("automatic_feature_id"):
        st.caption(
            f"Feature {context['automatic_feature_id']} is the automatic best "
            f"grid fit for {species_id}."
        )


def _render_species_candidate_diagnostics(
    candidates: pd.DataFrame,
    species_id: str,
    sample_cols: list[str],
    *,
    heading: str,
    key_prefix: str,
    primary_assignment: dict | None = None,
):
    """One diagnostic row per feature carrying the selected lipid ID."""
    st.markdown(f'<div class="matlab-title">{heading}</div>', unsafe_allow_html=True)
    if primary_assignment is not None:
        _render_primary_feature_control(
            candidates,
            species_id,
            key_prefix=key_prefix,
            context=primary_assignment,
        )
    if not sample_cols or candidates.empty:
        st.caption("No sample-level intensity columns are available.")
        return

    detail = candidates.copy().reset_index(drop=True)
    detail["row_id"] = [
        f"{index:04d}|{feature_id}|{adduct}"
        for index, (feature_id, adduct) in enumerate(
            zip(detail["feat_feature_id"], detail["adduct"])
        )
    ]
    assigned_feature = str(_primary_feature_overrides().get(str(species_id), ""))
    primary_suffix = pd.Series("", index=detail.index, dtype=object)
    if assigned_feature:
        primary_suffix = primary_suffix.mask(
            detail["feat_feature_id"].astype(str).eq(assigned_feature),
            " ★ primary",
        )
    accepted = detail["automatic_candidate"].fillna(False).astype(bool)
    run_candidate = (
        detail.get("run_candidate", pd.Series(True, index=detail.index))
        .fillna(False)
        .astype(bool)
    )
    status_suffix = np.select(
        [accepted, run_candidate],
        ["", " [candidate]"],
        default=" [widened window]",
    )
    row_labels = (
        "feature "
        + detail["feat_feature_id"].astype(str)
        + " "
        + detail["adduct"].astype(str)
        + status_suffix
        + primary_suffix
    ).tolist()
    heat_custom: np.ndarray = np.empty((len(detail), len(sample_cols), 2), dtype=object)
    heat_custom[:, :, 0] = detail["feat_feature_id"].astype(str).to_numpy()[:, None]
    heat_custom[:, :, 1] = detail["adduct"].astype(str).to_numpy()[:, None]
    normal_marker_colors = [
        "#2e9f64" if automatic else "#2677c9" if original else "#83bbed"
        for automatic, original in zip(accepted, run_candidate)
    ]
    benchmark_status = detail.get(
        "benchmark_status", pd.Series("", index=detail.index)
    ).astype(str)
    marker_colors = [
        _BENCHMARK_STATUS_COLOURS.get(status, fallback)
        for status, fallback in zip(benchmark_status, normal_marker_colors)
    ]
    fig_header, fig_body = build_standard_diagnostic_figures(
        detail,
        sample_cols,
        row_ids=detail["row_id"].tolist(),
        row_labels=row_labels,
        marker_colors=marker_colors,
        marker_sizes=[8] * len(detail),
        point_custom=np.column_stack(
            [
                detail["feat_feature_id"].astype(str),
                detail["adduct"].astype(str),
            ]
        ),
        heat_custom=heat_custom,
        heat_hovertemplate=(
            "feature %{customdata[0]} %{customdata[1]}<br>"
            "%{x}<br>log10 %{z:.2f}<extra></extra>"
        ),
        point_hover_prefix="feature %{customdata[0]} %{customdata[1]}",
        left_margin=_DIAGNOSTIC_FEATURE_LEFT_MARGIN,
        body_height=max(220, 32 * len(detail) + 45),
    )
    with st.container(border=True, key=f"{key_prefix}-diagnostics-shell"):
        viewer_plotly_chart(
            fig_header,
            width="stretch",
            key=f"{key_prefix}-detail-header",
            config={
                "displayModeBar": False,
                "scrollZoom": False,
                "doubleClick": False,
                "responsive": True,
            },
        )
        with st.container(height=360, border=False):
            viewer_plotly_chart(
                fig_body,
                width="stretch",
                key=f"{key_prefix}-detail-body",
                config={
                    "displayModeBar": False,
                    "scrollZoom": False,
                    "doubleClick": False,
                    "responsive": True,
                },
            )


def _render_selection_actions(chart_key: str, selected_rows: pd.DataFrame):
    """Anchor confirmation and session-only deletion for the current selection."""
    if selected_rows.empty:
        st.caption("Click a species on the grid to select its diagnostic row.")
        return
    names = list(selected_rows["name"].astype(str))
    if len(selected_rows) > 1:
        st.caption(f"{len(selected_rows)} lipids selected")
        confirm_col, unconfirm_col, delete_col = st.columns(3)
        if confirm_col.button(
            "⚓ Confirm all as anchors", type="primary", key="standard_bulk_anchor"
        ):
            for name in names:
                _set_anchor(chart_key, name, True)
            st.rerun()
        if unconfirm_col.button("✗ Unconfirm all", key="standard_bulk_unanchor"):
            for name in names:
                _set_anchor(chart_key, name, False)
            st.rerun()
        delete_col.button(
            "Delete selected annotations",
            key="standard_bulk_delete",
            on_click=_delete_annotations,
            args=(chart_key, tuple(selected_rows["deletion_key"].unique())),
            help=(
                "Removes these lipid annotations from this browsing session; "
                "the run's exports are unchanged."
            ),
        )
        return
    row = selected_rows.iloc[0]
    name = str(row["name"])
    st.caption(f"Selected: {name} {row['adduct']}")
    action_col, delete_col = st.columns(2)
    if bool(row["confirmed"]):
        action_col.button(
            "✗ Unconfirm anchor",
            key=f"standard_unanchor_{row['key']}",
            on_click=_set_anchor,
            args=(chart_key, name, False),
        )
    else:
        action_col.button(
            "⚓ Confirm as anchor",
            type="primary",
            key=f"standard_anchor_{row['key']}",
            on_click=_set_anchor,
            args=(chart_key, name, True),
        )
    delete_col.button(
        "Delete annotation",
        key=f"standard_delete_{row['key']}",
        on_click=_delete_annotations,
        args=(chart_key, (row["deletion_key"],)),
        help=(
            "Removes this lipid annotation from this browsing session; the "
            "run's exports are unchanged."
        ),
    )


def _render_anchor_tables(
    plot_df: pd.DataFrame, chart_key: str, rt_tolerance: float, confirmed: set[str]
):
    """Inlier/outlier split and engine/manual anchor provenance."""
    inlier_mask = plot_df["grid_resid"].abs() <= rt_tolerance
    inliers = plot_df[inlier_mask.fillna(False)]
    outliers = plot_df[~inlier_mask.fillna(False)]
    left, right = st.columns(2)
    with left:
        st.caption(f"Inliers ({len(inliers)})")
        st.dataframe(
            pd.DataFrame(
                {
                    "lipid": _safe_species_label(inliers),
                    "m/z": inliers["feat_mz"].round(3),
                    "Δs": (inliers["grid_resid"] * 60).round(1),
                }
            ),
            hide_index=True,
            height=130,
            width="stretch",
        )
    with right:
        st.caption(f"Outliers ({len(outliers)})")
        st.dataframe(
            pd.DataFrame(
                {
                    "lipid": _safe_species_label(outliers),
                    "m/z": outliers["feat_mz"].round(3),
                    "Δs": (outliers["grid_resid"] * 60).round(1),
                }
            ),
            hide_index=True,
            height=130,
            width="stretch",
        )

    anchors = plot_df[plot_df["anchor"] | plot_df["confirmed"]].copy()
    anchors = anchors.sort_values(["confirmed", "grid_resid"], ascending=[False, True])
    engine_selected = (
        plot_df.get("automatic_rt_anchor", pd.Series(False, index=plot_df.index))
        .fillna(False)
        .astype(bool)
    )
    st.markdown(
        f"**Anchor lipids ({int(engine_selected.sum())} selected by engine / "
        f"{int((plot_df['confirmed'] & ~engine_selected).sum())} manual)**"
    )
    if st.button(
        "Clear manually confirmed anchors",
        disabled=not confirmed,
        key=f"standard_clear_anchors_{chart_key}",
    ):
        st.session_state.setdefault(_ANCHOR_STATE, {})[chart_key] = set()
        st.rerun()
    anchor_automatic = (
        anchors.get("automatic_rt_anchor", pd.Series(False, index=anchors.index))
        .fillna(False)
        .astype(bool)
    )
    st.dataframe(
        pd.DataFrame(
            {
                "": anchors["confirmed"].map({True: "⚓", False: ""}),
                "source": np.where(
                    anchor_automatic, "engine-selected RT", "manual review"
                ),
                "lipid": _safe_species_label(anchors),
                "m/z": anchors["feat_mz"].round(3),
                "ΔRT grid (min)": anchors["grid_resid"].round(3),
            }
        ),
        hide_index=True,
        height=260,
        width="stretch",
    )


def _render_np_alignment_status(stream: str, stream_result: dict):
    if not stream.startswith("NP_"):
        return
    criteria = stream_result.get("criteria", {})
    if criteria.get("rt_model_source") == "matlab_np_rtdb":
        st.success(
            "Independent NP MATLAB RTDB grid active. No RP RT model or "
            "positional LPC/LPE resolver is used."
        )
        return
    alignment = stream_result.get("np_rt_alignment", pd.DataFrame())
    validated = bool(
        alignment is not None
        and not alignment.empty
        and alignment.iloc[0].get("validated", False)
    )
    if validated:
        offset = float(alignment.iloc[0].get("offset_min", 0.0))
        st.success(f"Anchored NP grid active · dataset offset {offset:+.3f} min.")
    elif alignment is not None and not alignment.empty:
        st.warning(
            "NP anchor calibration did not validate. The browser shows the "
            "unshifted reference for diagnosis, but LASER did not use that RT "
            "grid for automatic annotation."
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _figure_services() -> StandardGridFigureServices:
    return StandardGridFigureServices(
        cloud_dimension=_cloud_dimension,
        observed_dimension=_observed_dimension,
        node_dimension=_node_dimension,
        marker_groups=_marker_groups,
        safe_species_label=_safe_species_label,
        benchmark_point_colours=_benchmark_point_colours,
        benchmark_visible_plot_candidates=_benchmark_visible_plot_candidates,
        benchmark_status_colours=_BENCHMARK_STATUS_COLOURS,
        candidate_columns=tuple(_CD_COLS),
        hover_template=_HOVER,
        dimension_titles=_DIMENSION_TITLES,
    )


def build_standard_grid_figure(*args, **kwargs):
    """Compatibility façade injecting the grid figure's narrow services."""
    kwargs.setdefault("services", _figure_services())
    return _build_standard_grid_figure(*args, **kwargs)


def _page_services() -> StandardGridPageServices:
    return StandardGridPageServices(
        render_np_alignment_status=_render_np_alignment_status,
        prepare_standard_grid_data=prepare_standard_grid_data,
        standard_sample_columns=standard_sample_columns,
        build_class_grid_state=build_class_grid_state,
        primary_feature_overrides=_primary_feature_overrides,
        benchmark_visible_plot_candidates=_benchmark_visible_plot_candidates,
        axis_ranges=_axis_ranges,
        build_standard_grid_figure=build_standard_grid_figure,
        plotly_click_bridge=plotly_click_bridge,
        sync_component_selection=_sync_component_selection,
        sync_component_multi_selection=_sync_component_multi_selection,
        same_id_trace_candidates=_same_id_trace_candidates,
        toggle_same_lipid_id_projection=_toggle_same_lipid_id_projection,
        render_standard_feature_traces=_render_standard_feature_traces,
        render_species_candidate_diagnostics=_render_species_candidate_diagnostics,
        render_statistics=_render_statistics,
        render_annotation_diagnostics=_render_annotation_diagnostics,
        render_selection_actions=_render_selection_actions,
        render_anchor_tables=_render_anchor_tables,
        render_review_controls=_render_review_controls,
        selection_state=_SELECTION_STATE,
        anchor_state=_ANCHOR_STATE,
        deleted_state=_DELETED_STATE,
        view_range_state=_VIEW_RANGE_STATE,
        plot_view_state=_PLOT_VIEW_STATE,
        projection_state=_PROJECTION_STATE,
    )


def render_standard_grid_browser(result: dict) -> None:
    """Render the standard browser through its focused page coordinator."""
    _render_standard_grid_page(result, services=_page_services())
