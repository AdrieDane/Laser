"""Thin page coordinator for the standard-lipidomics grid workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
import streamlit as st
from presentation.standard_grid.persistent_plot import persistent_review_plot

from scientific_defaults import SCIENTIFIC_DEFAULTS
from standard_adducts import preferred_standard_adduct, primary_adduct_rows
from presentation.standard_grid.controls import CONTROLS_STATE


def _browser_help(benchmark: bool) -> str:
    if benchmark:
        return (
            "Benchmark colours: green = MATLAB and LASER, orange = LASER-only, "
            "purple = MATLAB-only. The reviewer has no CCS or MS/MS evidence."
        )
    return (
        "Conventional-data reviewer: green points are accepted, blue passed the "
        "run criteria, light blue appears only after relaxing viewer criteria, "
        "yellow is engine-selected RT anchor, and purple is manually confirmed."
    )


@dataclass(frozen=True)
class StandardGridPageServices:
    """Explicit dependencies used by the standard-grid page coordinator."""

    render_np_alignment_status: Callable
    prepare_standard_grid_data: Callable
    standard_sample_columns: Callable
    build_class_grid_state: Callable
    primary_feature_overrides: Callable
    benchmark_visible_plot_candidates: Callable
    axis_ranges: Callable
    build_standard_grid_figure: Callable
    plotly_click_bridge: Callable
    sync_component_selection: Callable
    sync_component_multi_selection: Callable
    same_id_trace_candidates: Callable
    toggle_same_lipid_id_projection: Callable
    render_standard_feature_traces: Callable
    render_species_candidate_diagnostics: Callable
    render_statistics: Callable
    render_annotation_diagnostics: Callable
    render_selection_actions: Callable
    render_anchor_tables: Callable
    render_review_controls: Callable
    selection_state: str
    anchor_state: str
    deleted_state: str
    view_range_state: str
    plot_view_state: str
    projection_state: str


@dataclass
class StandardGridPage:
    result: dict
    services: StandardGridPageServices
    stream: str = ""
    stream_result: dict = field(default_factory=dict)
    reference: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    sample_cols: list[str] = field(default_factory=list)
    run_ppm: float = 0.0
    run_rt: float = 0.0
    uses_rt_grid: bool = True
    run_use_grid: bool = False
    run_grid_threshold: float = 0.45

    def initialise(self) -> bool:
        streams = [
            name for name, stream_result in self.result.get("streams", {}).items()
            if not stream_result.get("reference", pd.DataFrame()).empty
        ]
        if not streams:
            st.info("No reference-supported streams are available for the grid browser.")
            return False
        with st.container(horizontal=True, vertical_alignment="center", key="standard-grid-stream-row"):
            st.subheader("Lipid class grid browser", anchor=False,
                         help=_browser_help(bool(self.result.get("benchmark_context"))))
            self.stream = st.selectbox("Stream", streams, key="standard_grid_stream", label_visibility="collapsed", width=180)
        self.stream_result = self.result["streams"][self.stream]
        status = self.result.get("benchmark_context", {}).get(
            "status_by_stream", {}
        ).get(self.stream, {})
        if status:
            self.stream_result = dict(self.stream_result)
            self.stream_result["benchmark_status_by_identity"] = status
        criteria = self.stream_result.get("criteria", {})
        self.run_ppm = float(criteria.get(
            "ppm_tolerance", SCIENTIFIC_DEFAULTS.standard_ppm_tolerance
        ))
        self.run_rt = float(criteria.get(
            "rt_tolerance_min", SCIENTIFIC_DEFAULTS.rt_tolerance_min
        ))
        self.uses_rt_grid = bool(criteria.get("uses_rt_grid", True))
        anchors = self.result.get("anchor_settings", {})
        self.run_use_grid = bool(criteria.get(
            "use_grid_coherence", anchors.get("use_grid_coherence", False)
        ))
        self.run_grid_threshold = float(criteria.get(
            "grid_coherence_threshold",
            anchors.get("grid_coherence_threshold", SCIENTIFIC_DEFAULTS.grid_coherence_threshold),
        ))
        self.services.render_np_alignment_status(self.stream, self.stream_result)
        prepare_key = (id(self.result), self.stream, id(self.stream_result.get("reference")),
                       id(self.stream_result.get("annotations")), id(self.stream_result.get("all_pairs")))
        cached = st.session_state.get("standard_prepared_grid_cache")
        if cached is None or cached[0] != prepare_key:
            cached = (prepare_key, self.services.prepare_standard_grid_data(self.stream_result))
            st.session_state.standard_prepared_grid_cache = cached
        self.reference, self.candidates = cached[1]
        if self.reference.empty:
            st.info("The selected stream has no policy-compatible reference grid.")
            return False
        deleted = st.session_state.setdefault(self.services.deleted_state, set())
        if not self.candidates.empty and deleted:
            self.candidates = self.candidates[
                ~self.candidates["deletion_key"].isin(deleted)
            ].copy()
        self.features = self.stream_result.get("features", pd.DataFrame())
        self.sample_cols = self.services.standard_sample_columns(self.features)
        return True

    def class_selection(self) -> dict:
        primary = primary_adduct_rows(
            self.candidates, class_column="grid_class", adduct_column="adduct",
            stream_column=None,
        )
        counts = (
            primary[primary["automatic_candidate"]].groupby("grid_class")["name"].nunique()
            if not self.candidates.empty else pd.Series(dtype=int)
        )
        anchor_counts = (
            primary[primary.get(
                "automatic_rt_anchor", pd.Series(False, index=primary.index)
            ).fillna(False).astype(bool)].groupby("grid_class")["name"].nunique()
            if not primary.empty else pd.Series(dtype=int)
        )
        all_classes = sorted(self.reference["grid_class"].dropna().astype(str).unique())
        options = (
            [item for item in all_classes if int(counts.get(item, 0)) > 0]
            + [item for item in all_classes if int(counts.get(item, 0)) == 0]
        )
        class_key = f"standard_focus_grid_class::{self.stream}"
        if st.session_state.get(class_key) not in all_classes:
            st.session_state[class_key] = "PC" if "PC" in all_classes else options[0]
        focus_class = st.session_state[class_key]
        adducts = sorted(self.reference.loc[
            self.reference["grid_class"].eq(focus_class), "adduct"
        ].dropna().astype(str).unique())
        adduct_key = f"standard_focus_adduct::{self.stream}::{focus_class}"
        if st.session_state.get(adduct_key) not in adducts:
            st.session_state[adduct_key] = preferred_standard_adduct(adducts)
        return {
            "primary": primary, "counts": counts, "anchor_counts": anchor_counts,
            "options": options, "class_key": class_key, "focus_class": focus_class,
            "adducts": adducts, "adduct_key": adduct_key,
            "adduct": st.session_state[adduct_key],
        }

    def build_grid_state(self, selection: dict, controls: dict) -> dict:
        chart_key = (
            f"standard-grid::{self.stream}::{selection['focus_class']}::{selection['adduct']}"
        )
        confirmed = st.session_state.setdefault(self.services.anchor_state, {}).get(chart_key, set())
        cache_key = (id(self.result), self.stream, selection["focus_class"], selection["adduct"],
                     tuple(sorted(confirmed)), repr(self.services.primary_feature_overrides()),
                     tuple((key, repr(value)) for key,value in controls.items() if key not in {"x_dimension", "y_dimension", "show_mesh", "show_labels", "show_cloud"}),
                     tuple(sorted(st.session_state.get(self.services.deleted_state, set()))))
        cached = st.session_state.get("standard_class_grid_state_cache")
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        state = self.services.build_class_grid_state(
            self.reference, self.candidates,
            lipid_class=selection["focus_class"], adduct=selection["adduct"],
            confirmed=confirmed, primary_overrides=self.services.primary_feature_overrides(),
            review={
                "ppm_tolerance": controls["ppm_tolerance"],
                "rt_tolerance": controls["rt_tolerance"],
                "use_rt": self.uses_rt_grid,
                "use_grid_coherence": controls["use_grid_coherence"],
                "grid_coherence_threshold": controls["grid_coherence_threshold"],
            },
            run_criteria={
                "ppm_tolerance": self.run_ppm, "rt_tolerance": self.run_rt,
                "use_rt": self.uses_rt_grid,
                "use_grid_coherence": self.run_use_grid,
                "grid_coherence_threshold": self.run_grid_threshold,
            },
        )
        state["chart_key"] = chart_key
        state["confirmed"] = confirmed
        state["plot_df"] = self.services.benchmark_visible_plot_candidates(
            state["plot_df"], show_unbenchmarked=controls["show_unbenchmarked"],
        )
        st.session_state.standard_class_grid_state_cache = (cache_key, state)
        return state

    def projection(self, state: dict) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
        chart_key = state["chart_key"]
        store = st.session_state.setdefault(self.services.selection_state, {})
        prior_keys = set(store.get(chart_key, []))
        selected = state["candidates"][
            state["candidates"]["key"].isin(prior_keys & set(state["plot_df"]["key"]))
        ]
        alternatives = state["candidates"].iloc[0:0]
        best = state["plot_df"].iloc[0:0]
        if st.session_state.setdefault(self.services.projection_state, False) and not selected.empty:
            species = str(selected.iloc[0]["name"])
            same = state["candidates"][state["candidates"]["name"].astype(str).eq(species)]
            best = state["plot_df"][state["plot_df"]["name"].astype(str).eq(species)].head(1)
            if not best.empty:
                alternatives = same[~same["key"].astype(str).eq(str(best.iloc[0]["key"]))]
        return alternatives, best, prior_keys

    def render_chart(self, plot_col, state: dict, controls: dict, alternatives, best, prior_keys) -> set[str]:
        ranges = self.services.axis_ranges(state, self.features)
        view_key = f"{self.stream}::{state['chart_key']}::{controls['x_dimension']}::{controls['y_dimension']}"
        if ranges is not None:
            ranges = st.session_state.setdefault(self.services.view_range_state, {}).setdefault(view_key, ranges)
        figure_key = (id(state), controls["x_dimension"], controls["y_dimension"], controls["show_mesh"],
                      controls["show_labels"], controls["show_cloud"], tuple(alternatives.get("key", [])), tuple(best.get("key", [])))
        cached_figure = st.session_state.get("standard_main_figure_cache")
        if cached_figure is not None and cached_figure[0] == figure_key:
            figure = cached_figure[1]
        else:
            figure = self.services.build_standard_grid_figure(
                state, self.features,
                x_dimension=controls["x_dimension"], y_dimension=controls["y_dimension"],
                show_mesh=controls["show_mesh"], show_labels=controls["show_labels"],
                show_cloud=controls["show_cloud"], selected_keys=set(),
                projected_alternatives=alternatives, projected_best=best,
                axis_ranges=ranges,
                saved_view=st.session_state.setdefault(self.services.plot_view_state, {}).get(view_key, {}),
                uirevision=view_key, show_unbenchmarked=controls["show_unbenchmarked"],
            )
            st.session_state.standard_main_figure_cache = (figure_key, figure)
        valid = set(state["plot_df"]["key"]) | set(alternatives["key"]) | set(best["key"])
        with plot_col:
            component_key = "standard_2d_interaction::grid-viewer"
            ordered = state["plot_df"].sort_values(["carbon", "db", "name", "adduct", "absresid"])["key"].tolist()
            persistent_review_plot(figure, key=component_key, channel=state["chart_key"],
                ordered_keys=ordered, selected_keys=prior_keys, valid_keys=valid,
                on_point_change=lambda: self.services.sync_component_selection(component_key, state["chart_key"], valid, view_key),
                config={"scrollZoom": True, "displaylogo": False, "doubleClick": "reset+autosize", "responsive": True,
                        "modeBarButtonsToAdd": ["select2d", "lasso2d"]})
            if state["mesh"] is None and controls["show_mesh"]:
                st.caption("No predicted grid: fewer than three reference RT nodes.")
            st.caption(state["correction"]["label"])
        return valid

    def render_selected(self, plot_col, state: dict, valid: set[str]) -> tuple[pd.DataFrame, pd.Series | None]:
        store = st.session_state.setdefault(self.services.selection_state, {})
        selected_keys = [key for key in store.get(state["chart_key"], []) if key in valid]
        selected = state["candidates"][state["candidates"]["key"].isin(selected_keys)]
        active_key = st.session_state.get("standard_active_grid_keys", {}).get(state["chart_key"])
        active_rows = selected[selected["key"].eq(active_key)] if not selected.empty else selected
        active = active_rows.iloc[0] if not active_rows.empty else selected.iloc[-1] if not selected.empty else None
        if active is None and not state["plot_df"].empty:
            active = state["plot_df"].sort_values(
                ["confirmed", "anchor", "absresid"], ascending=[False, False, True]
            ).iloc[0]
        if active is None or selected.empty:
            return selected, active
        with plot_col:
            species = str(active["name"])
            same_all = state["all_candidates"][
                state["all_candidates"]["name"].astype(str).eq(species)
            ]
            same, omitted = self.services.same_id_trace_candidates(same_all, active)
            if omitted:
                st.caption(f"{omitted} broader same-ID feature(s) are hidden from traces.")
            grid_rows = state["candidates"][state["candidates"]["name"].astype(str).eq(species)]
            automatic = grid_rows.sort_values(["absresid", "absppm"]).head(1)
            st.button(
                "Project/hide all same-ID features", width="stretch",
                key="standard-toggle-same-lipid-id-projection",
                on_click=self.services.toggle_same_lipid_id_projection,
            )
            if self.result.get("engine") == "matlab":
                self.services.render_standard_feature_traces(
                    self.result, self.stream, active, same, species,
                )
            self.services.render_species_candidate_diagnostics(
                same, species, self.sample_cols,
                heading=f"Same lipid ID diagnostics: {species} · {same['feat_feature_id'].nunique()} feature(s)",
                key_prefix="standard-review-same-lipid-id",
                primary_assignment={
                    "chart_key": state["chart_key"],
                    "grid_feature_keys": dict(zip(
                        grid_rows["feat_feature_id"].astype(str), grid_rows["key"].astype(str)
                    )),
                    "automatic_feature_id": (
                        str(automatic.iloc[0]["feat_feature_id"]) if not automatic.empty else None
                    ),
                },
            )
        return selected, active

    def render_panel(self, panel_col, state: dict, selection: dict, controls: dict, selected, active) -> None:
        with panel_col:
            self.services.render_statistics(active, self.sample_cols,
                self.features.attrs.get("sample_group_map", {}), self.features.attrs.get("sample_group_name_map", {}))
            with st.container(border=True, key="standard-ms-viewer-lipid-selector"):
                st.markdown("**Grid lipid selection**")
                class_col, adduct_col = st.columns(2, gap="small")
                class_col.selectbox(
                    "Lipid class", selection["options"], key=selection["class_key"],
                    format_func=lambda value: (
                        f"{value} ({int(selection['counts'].get(value, 0))} accepted)"
                        if int(selection["counts"].get(value, 0)) else f"{value} — not detected"
                    ),
                )
                adduct_col.selectbox("Adduct", selection["adducts"], key=selection["adduct_key"])
                index = selection["options"].index(selection["focus_class"])
                previous, following = st.columns(2, gap="small")
                previous.button("Previous class", key="standard-previous-class", width="stretch", disabled=index == 0,
                    on_click=_move_class, args=(selection["class_key"], selection["options"], -1))
                following.button("Next class", key="standard-next-class", width="stretch", disabled=index == len(selection["options"]) - 1,
                    on_click=_move_class, args=(selection["class_key"], selection["options"], 1))
            active_keys = set(selected["key"]) if not selected.empty else set()
            if active is not None:
                active_keys.add(active["key"])
            self.services.render_annotation_diagnostics(
                state["plot_df"], self.sample_cols,
                chart_key=state["chart_key"], active_keys=active_keys,
            )
            self.services.render_selection_actions(state["chart_key"], selected)
            if not state["plot_df"].empty:
                self.services.render_anchor_tables(
                    state["plot_df"], state["chart_key"],
                    controls["rt_tolerance"], state["confirmed"],
                )

    def render(self) -> None:
        if not self.initialise():
            return
        visible = bool(st.session_state.setdefault(CONTROLS_STATE, True))
        widths = [1.35, 5.25, 4.15] if visible else [0.28, 6.32, 4.15]
        ctrl_col, plot_col, panel_col = st.columns(widths, gap="small")
        with ctrl_col:
            controls = self.services.render_review_controls(
                annotation_run_key=(id(self.result), self.stream),
                run_ppm_tolerance=self.run_ppm, run_rt_tolerance=self.run_rt,
                uses_rt_grid=self.uses_rt_grid,
                run_use_grid_coherence=self.run_use_grid,
                run_grid_coherence_threshold=self.run_grid_threshold,
                benchmark_mode=bool(self.result.get("benchmark_context")),
            )
        selection = self.class_selection()
        state = self.build_grid_state(selection, controls)
        alternatives, best, prior_keys = self.projection(state)
        valid = self.render_chart(plot_col, state, controls, alternatives, best, prior_keys)
        selected, active = self.render_selected(plot_col, state, valid)
        self.render_panel(panel_col, state, selection, controls, selected, active)


def render_standard_grid_browser(
    result: dict, *, services: StandardGridPageServices,
) -> None:
    """Render the standard grid page through its focused coordinator."""
    StandardGridPage(result, services).render()


def _move_class(key: str, options: list[str], direction: int) -> None:
    index = options.index(st.session_state[key])
    st.session_state[key] = options[max(0, min(index + direction, len(options) - 1))]
