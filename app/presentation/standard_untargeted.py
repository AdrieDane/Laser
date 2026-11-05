"""Post-annotation untargeted workspace for standard lipidomics."""

from __future__ import annotations

import re
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from presentation.viewer_theme import viewer_plotly_chart
from presentation.untargeted_mode import render_mode_view

from standard_lipidomics.workflow import load_standard_feature_list
from standard_lipidomics.untargeted_adducts import EVIDENCE_VERSION
from scientific_defaults import SCIENTIFIC_DEFAULTS
from standard_lipidomics.untargeted import (
    UntargetedComparisonCriteria,
    attach_excel_sample_metadata,
    default_biological_groups,
    group_catalog,
    is_biological_group,
    paired_abundance_table,
)


def _restore_group_metadata_from_inputs(result: dict) -> None:
    """Hydrate cached/pre-existing results with the compact XCMS group map."""
    project = (
        st.session_state.get("standard_project_directory")
        or result.get("matlab_project_dir")
        or st.session_state.get("standard_headless_config", {})
        .get("inputs", {}).get("project")
    )
    audit = attach_excel_sample_metadata(result, project)
    if not audit.empty:
        result["untargeted_sample_metadata_audit"] = audit
    sources = st.session_state.get("standard_input_sources", {})
    for stream, stream_result in result.get("streams", {}).items():
        features = stream_result.get("features", pd.DataFrame())
        if not isinstance(features, pd.DataFrame):
            continue
        if features.attrs.get("sample_group_mapping_status") in {
            "excel_metadata_validated", "excel_metadata_partial",
        } or stream not in sources:
            continue
        try:
            metadata_source = load_standard_feature_list(sources[stream])
        except (OSError, TypeError, ValueError):
            continue
        for key in (
            "xcms_group_columns", "xcms_group_sizes", "sample_group_map",
            "sample_group_mapping_status",
        ):
            if key in metadata_source.attrs:
                features.attrs[key] = metadata_source.attrs[key]




def _untargeted_criteria(
    result: dict,
) -> tuple[UntargetedComparisonCriteria, dict[str, str]] | None:
    saved_criteria = result.get("untargeted_analysis", {}).get("criteria", {})
    catalog = group_catalog(result)
    defaults = default_biological_groups(result)
    if not catalog.empty:
        with st.expander("Sample-group mapping", expanded=False):
            st.dataframe(catalog, hide_index=True, width="stretch")
            audit = result.get("untargeted_sample_metadata_audit", pd.DataFrame())
            if isinstance(audit, pd.DataFrame) and not audit.empty:
                st.dataframe(audit, hide_index=True, width="stretch")
    if defaults is None:
        st.info(
            "Two validated biological sample groups were not found in the "
            "project Excel metadata (or validated XCMS fallback mapping). "
            "The annotation result is unchanged."
        )
        if not catalog.empty:
            st.dataframe(catalog, hide_index=True, width="stretch")
        return None
    biological = catalog[catalog["biological"]].copy()
    options = list(dict.fromkeys(biological["group"].astype(str)))
    labels = dict(zip(
        biological["group"].astype(str), biological["label"].astype(str)
    ))
    for key, default in zip(
        ("standard_untargeted_group_a", "standard_untargeted_group_b"), defaults
    ):
        if st.session_state.get(key) not in options:
            st.session_state[key] = default
    with st.expander("Comparison and significance criteria", expanded=False):
        columns = st.columns(2)
        group_a = columns[0].selectbox(
            "Reference group (A)", options, format_func=lambda value: labels.get(value, value),
            key="standard_untargeted_group_a",
        )
        group_b = columns[1].selectbox(
            "Comparison group (B)", options, format_func=lambda value: labels.get(value, value),
            key="standard_untargeted_group_b",
        )
        threshold_col, fold_col, detection_col = st.columns(3)
        p_value = threshold_col.number_input(
            "Raw p-value must be below", min_value=0.000001, max_value=1.0,
            value=float(saved_criteria.get(
                "p_value_threshold", SCIENTIFIC_DEFAULTS.untargeted_p_value_threshold
            )), step=0.001,
            format="%.6f", key="standard_untargeted_p_value",
            help="Unadjusted two-sided Welch-test p-value; no multiple-testing correction.",
        )
        fold = fold_col.number_input(
            "Minimum |log2 fold change|", min_value=0.0, max_value=20.0,
            value=float(saved_criteria.get(
                "min_abs_log2_fold_change",
                SCIENTIFIC_DEFAULTS.untargeted_min_abs_log2_fold_change,
            )),
            step=0.25, key="standard_untargeted_fold_change",
        )
        detection = detection_col.number_input(
            "Minimum detection in either group (%)", min_value=1.0, max_value=100.0,
            value=float(saved_criteria.get(
                "min_detection_percent",
                SCIENTIFIC_DEFAULTS.untargeted_min_detection_percent,
            )),
            step=5.0, key="standard_untargeted_detection",
            help="At least this percentage of samples in group A or B must carry a non-zero feature intensity.",
        )
    with st.expander("Feature-evidence search settings", expanded=False):
        ppm = st.number_input("Mass tolerance (ppm)", min_value=.1, max_value=100.,
            value=float(saved_criteria.get("cross_mode_ppm_tolerance", SCIENTIFIC_DEFAULTS.standard_ppm_tolerance)),
            key="standard-untargeted-evidence-ppm")
        coelution = st.number_input("Adduct co-elution tolerance (min)", min_value=.001, max_value=5.,
            value=float(saved_criteria.get("opposite_polarity_rt_tolerance_min", SCIENTIFIC_DEFAULTS.untargeted_opposite_polarity_rt_tolerance_min)),
            key="standard-untargeted-evidence-rt")
        isotope_rt = st.number_input("Isotope co-elution tolerance (min)", min_value=.001, max_value=5.,
            value=float(saved_criteria.get("isotope_rt_tolerance_min", SCIENTIFIC_DEFAULTS.untargeted_isotope_rt_tolerance_min)),
            key="standard-untargeted-isotope-rt")
        series_rt = st.number_input("CH₂-series RT neighbourhood (min)", min_value=.001, max_value=10.,
            value=float(saved_criteria.get("ch2_rt_tolerance_min", SCIENTIFIC_DEFAULTS.untargeted_ch2_rt_tolerance_min)),
            key="standard-untargeted-series-rt")
    if group_a == group_b:
        st.warning("Choose two different biological groups.")
        return None
    return UntargetedComparisonCriteria(
        group_a=str(group_a), group_b=str(group_b),
        p_value_threshold=float(p_value), min_abs_log2_fold_change=float(fold),
        min_detection_percent=float(detection),
        cross_mode_ppm_tolerance=float(ppm), opposite_polarity_rt_tolerance_min=float(coelution),
        isotope_rt_tolerance_min=float(isotope_rt), ch2_rt_tolerance_min=float(series_rt),
    ), labels






def _render_untargeted_volcano(
    analysis: dict, criteria: UntargetedComparisonCriteria,
) -> None:
    modes = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")
    by_stream = analysis.get("by_stream", {})
    figure = make_subplots(
        rows=2, cols=2, subplot_titles=modes, shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.08, vertical_spacing=0.13,
    )
    populated = sum(isinstance(by_stream.get(mode), pd.DataFrame)
                    and not by_stream.get(mode).empty for mode in modes)
    per_mode_limit = max(1_000, 12_000 // max(populated, 1))
    for index, mode in enumerate(modes):
        row, column = divmod(index, 2)
        table = by_stream.get(mode, pd.DataFrame())
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        mask = table["significant"].fillna(False).astype(bool)
        background = table[~mask]
        if len(background) > per_mode_limit:
            background = background.sample(
                n=per_mode_limit, random_state=20260814 + index)
        volcano = pd.concat([background, table[mask]], ignore_index=True, sort=False).copy()
        volcano["minus_log10_p"] = -np.log10(
            pd.to_numeric(volcano["p_value"], errors="coerce").clip(lower=1e-12))
        fold = pd.to_numeric(volcano["log2_fold_change_b_vs_a"], errors="coerce")
        significant = volcano["significant"].fillna(False).astype(bool)
        figure.add_trace(go.Scattergl(
            x=fold, y=volcano["minus_log10_p"], mode="markers",
            marker=dict(color=np.where(~significant, "#a9b1bb",
                        np.where(fold.ge(0.0), "#df3f3f", "#2670c9")),
                        size=np.where(significant, 7, 4), opacity=0.72),
            customdata=volcano[
                ["stream", "feature_id", "rt", "mz", "cluster_hypothesis"]
            ].to_numpy(dtype=object),
            hovertemplate=("%{customdata[0]} feature %{customdata[1]}<br>"
                           "RT %{customdata[2]:.3f} min · m/z %{customdata[3]:.4f}<br>"
                           "log2FC %{x:+.2f} · -log10(raw p) %{y:.2f}<br>"
                           "%{customdata[4]}<extra></extra>"),
            showlegend=False,
        ), row=row + 1, col=column + 1)
        figure.add_hline(y=-np.log10(criteria.p_value_threshold), line_dash="dot",
                         line_color="#8b5cf6", line_width=1, row=row + 1, col=column + 1)
        for boundary in (-criteria.min_abs_log2_fold_change,
                         criteria.min_abs_log2_fold_change):
            figure.add_vline(x=boundary, line_dash="dot", line_color="#8b5cf6",
                             line_width=1, row=row + 1, col=column + 1)
    figure.update_layout(height=670, margin=dict(l=55, r=25, t=45, b=55),
                         hovermode="closest")
    figure.update_xaxes(title_text="log2 fold change (B vs A)", row=2)
    figure.update_yaxes(title_text="−log10(raw p-value)", col=1)
    viewer_plotly_chart(figure, width="stretch", config={"displaylogo": False})


def _render_significant_features(significant: pd.DataFrame):
    columns = [
        "stream", "feature_id", "mz", "rt", "mobility", "mobility_unit", "ccs",
        "log2_fold_change_b_vs_a", "p_value", "significant", "detection_a_percent",
        "detection_b_percent", "detection_evidence", "cluster_class",
        "cluster_hypothesis", "homologous_series_neighbor_count",
        "homologous_series_hypothesis", "significant_ch2_series_id",
        "significant_ch2_series_size", "significant_ch2_neighbor_count",
        "significant_ch2_neighbor_feature_ids", "significant_ch2_cluster_hypothesis",
        "isotope_pattern", "isotope_pattern_assessment", "elemental_formula_candidates",
        "elemental_formula_candidate_count", "estimated_carbon_count_from_m_plus_1",
        "possible_isotope_of_feature_id", "neutral_mass_hypotheses",
        "chemical_composition_hypothesis", "cross_phase_match", "cross_phase_stream",
        "cross_phase_feature_id", "cross_phase_mass_error_ppm",
        "cross_phase_abundance_spearman", "cross_phase_abundance_pairs",
        "cross_phase_abundance_pairing", "opposite_polarity_match",
        "opposite_polarity_stream", "opposite_polarity_feature_id",
        "opposite_polarity_rule", "opposite_polarity_mass_error_ppm",
        "opposite_polarity_rt_delta_seconds", "opposite_polarity_abundance_spearman",
        "opposite_polarity_abundance_pairs", "opposite_polarity_abundance_pairing",
        "cross_acquisition_evidence", "same_mode_adduct_candidate_count", "all_mode_adduct_candidate_count",
        "lipid_series_id", "lipid_series_size", "lipid_series_class_context", "lipid_series_evidence",
    ]
    ordered = significant.sort_values(["p_value", "stream", "feature_id"]).reset_index(drop=True)
    event = st.dataframe(ordered[[column for column in columns if column in ordered]],
        hide_index=True, width="stretch", key="standard-untargeted-raw-p-list",
        on_select="rerun", selection_mode="single-row",
        column_config={"significant": st.column_config.CheckboxColumn("Passes extra filters")})
    selected = event.selection.rows
    return ordered.iloc[selected[0] if selected and selected[0] < len(ordered) else 0]


def _untargeted_sample_bar_figure(result, row):
    from standard_lipidomics.grid_browser import _sample_intensity_figure, standard_sample_columns
    features = result.get("streams", {}).get(row["stream"], {}).get("features", pd.DataFrame())
    selected = features[features.feature_id.astype(str).eq(str(row["feature_id"]))] if not features.empty else features
    if selected.empty:
        return None
    samples = standard_sample_columns(features)
    values = pd.Series({"name": f"Feature {row['feature_id']}", "adduct": "",
                        **{f"feat_{sample}": selected.iloc[0][sample] for sample in samples}})
    return _sample_intensity_figure(values, samples, features.attrs.get("sample_group_map", {}),
        features.attrs.get("sample_group_name_map", {}))


def _render_feature_chemical_evidence(row, analysis):
    st.subheader(f"Feature evidence: {row['stream']} · {row['feature_id']} · m/z {row['mz']:.4f}")
    with st.expander("Isotopes and candidate formulas", expanded=True):
        isotope_rows = []
        for number in (1, 2):
            prefix = f"isotope_m_plus_{number}"
            feature = row.get(prefix + "_feature_id", "")
            if pd.notna(feature) and str(feature):
                isotope_rows.append({"Isotope": f"M+{number}", "Feature": feature,
                    "m/z": row.get(prefix + "_mz"), "Relative intensity (%)": row.get(prefix + "_intensity_percent"),
                    "RT difference (s)": row.get(prefix + "_rt_delta_seconds")})
        if isotope_rows:
            st.dataframe(pd.DataFrame(isotope_rows), hide_index=True, width="stretch")
        else:
            st.write("No co-eluting isotope partner found.")
        st.write(row.get("chemical_composition_hypothesis", "Formula evidence unavailable."))
    for key, title in (("adduct_matches", "Adduct and cross-mode candidates"),
                       ("homologous_matches", "CH₂ / C₂H₄ series neighbours")):
        with st.expander(title, expanded=True):
            matches = analysis.get(key, pd.DataFrame())
            stream_col = "source_stream" if key == "adduct_matches" else "stream"
            if not matches.empty:
                matches = matches[matches[stream_col].eq(row["stream"]) & matches.source_feature_id.astype(str).eq(str(row["feature_id"]))]
            if matches.empty:
                st.write("No candidates within the search tolerances.")
            else:
                st.dataframe(matches, hide_index=True, width="stretch")
            if key == "homologous_matches":
                st.write(row.get("lipid_series_evidence", "No local series evidence."))


def _render_standard_untargeted(result: dict) -> dict:
    """View saved statistics only; the pipeline runs in its own module."""
    st.subheader("Untargeted differential features")
    analysis = result.get("untargeted_analysis", {})
    if not analysis:
        st.info("Run Untargeted analysis from the start page, then open its completed project here.")
        return {}
    if analysis.get("evidence_version", 0) < EVIDENCE_VERSION:
        st.info("This project contains the previously saved evidence. Run Untargeted analysis again to generate the expanded adduct and CH₂-series searches.")
    criteria = UntargetedComparisonCriteria(**analysis["criteria"])
    all_features = analysis.get("features", pd.DataFrame())
    significant = analysis.get("significant_features", pd.DataFrame())
    raw_p = analysis.get("raw_p_features", significant)
    metrics = st.columns(3)
    metrics[0].metric("Unannotated features tested", f"{len(all_features):,}")
    metrics[1].metric("Raw-p hits", f"{len(raw_p):,}")
    cluster_supported = (int(raw_p.get(
        "cluster_class", pd.Series(dtype=str)).fillna("").astype(str).ne("").sum())
        if not raw_p.empty else 0)
    metrics[2].metric("With local class context", f"{cluster_supported:,}")
    if raw_p.empty:
        st.info("No unannotated feature passes the saved raw-p-value threshold.")
        return analysis
    review_focus = render_mode_view(result, analysis)
    if review_focus:
        detections = analysis.get('compound_features', raw_p)
        selected = detections[detections.stream.eq(review_focus[0]) & detections.feature_id.astype(str).eq(str(review_focus[1]))]
        if not selected.empty:
            st.subheader(f"Review: {review_focus[0]} · feature {review_focus[1]}")
            _render_untargeted_review_controls(result, selected.iloc[0])
    with st.expander("Volcano plots", expanded=False):
        _render_untargeted_volcano(analysis, criteria)
    return analysis



def _untargeted_map_families(
    available: dict[str, pd.DataFrame],
) -> dict[str, list[tuple[str, str, pd.DataFrame]]]:
    """Pair positive and negative streams by chromatography name."""
    families: dict[str, list[tuple[str, str, pd.DataFrame]]] = {}
    for stream, stream_table in available.items():
        match = re.match(r"^(.*?)[_\- ]*(pos|neg)$", stream, flags=re.I)
        if match:
            family = match.group(1).rstrip("_- ") or stream
            polarity = match.group(2).casefold()
        else:
            family = stream
            polarity = "neg" if "neg" in stream.casefold() else "pos"
        families.setdefault(family, []).append((stream, polarity, stream_table))
    return families


def _mirrored_untargeted_table(
    streams: list[tuple[str, str, pd.DataFrame]],
) -> pd.DataFrame:
    """Combine one chromatography's streams using display-only signed m/z."""
    frames: list[pd.DataFrame] = []
    for stream, polarity, stream_table in streams:
        frame = stream_table.copy()
        frame["source_stream"] = stream
        frame["ion_mode"] = polarity.upper()
        frame["signed_mz"] = pd.to_numeric(
            frame["mz"], errors="coerce"
        ) * (-1.0 if polarity == "neg" else 1.0)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _p_value_marker_colours(
    p_values: pd.Series,
    base_colour: str,
    threshold: float,
) -> list[str]:
    """Map stronger raw-p evidence to a more saturated direction colour.

    Points immediately below the active threshold are deliberately pale. The
    base colour is reached two orders of magnitude below that threshold, which
    makes the scale reproducible across data sets instead of depending on the
    single smallest p-value in a run.
    """
    base = base_colour.lstrip("#")
    if len(base) != 6:
        raise ValueError("A six-digit hexadecimal base colour is required.")
    base_rgb = np.asarray(
        [int(base[index:index + 2], 16) for index in (0, 2, 4)],
        dtype=float,
    )
    threshold = float(np.clip(threshold, 1e-300, 1.0))
    values = pd.to_numeric(p_values, errors="coerce").fillna(1.0)
    scores = -np.log10(values.clip(lower=1e-300, upper=1.0))
    threshold_score = -np.log10(threshold)
    strength = np.clip((scores - threshold_score) / 2.0, 0.0, 1.0)
    white_fraction = 0.68 * (1.0 - strength)
    colours: list[str] = []
    for white_mix in white_fraction:
        rgb = np.rint(
            base_rgb * (1.0 - white_mix) + 255.0 * white_mix
        ).astype(int)
        colours.append("#" + "".join(f"{channel:02x}" for channel in rgb))
    return colours


def _selected_untargeted_point(
    table: pd.DataFrame,
    selection: tuple[str, str] | None,
) -> pd.DataFrame:
    """Resolve a stored map selection to one currently significant point."""
    if not selection or table.empty:
        return table.iloc[0:0].copy()
    stream, feature_id = map(str, selection)
    significant = table.get(
        "significant", pd.Series(False, index=table.index)
    ).fillna(False).astype(bool)
    return table[
        significant
        & table["source_stream"].astype(str).eq(stream)
        & table["feature_id"].astype(str).eq(feature_id)
    ].head(1)


def _untargeted_abundance_table(
    result: dict,
    stream: str,
    feature_id: str,
) -> pd.DataFrame:
    """Return sample-order intensities and metadata for one selected feature."""
    stream_result = result.get("streams", {}).get(stream, {})
    features = stream_result.get("features", pd.DataFrame())
    if not isinstance(features, pd.DataFrame) or features.empty:
        return pd.DataFrame()
    selected = features[
        features["feature_id"].astype(str).eq(str(feature_id))
    ]
    if selected.empty:
        return pd.DataFrame()
    row = selected.iloc[0]
    group_map = {
        str(key): str(value)
        for key, value in features.attrs.get("sample_group_map", {}).items()
    }
    group_name_map = {
        str(key): str(value)
        for key, value in features.attrs.get(
            "sample_group_name_map", {}
        ).items()
    }
    type_map = {
        str(key): str(value)
        for key, value in features.attrs.get("sample_type_map", {}).items()
    }
    configured = [
        str(column) for column in features.attrs.get("sample_columns", ())
        if str(column) in features.columns
    ]
    samples = configured or [
        sample for sample in group_map if sample in features.columns
    ]
    records: list[dict[str, object]] = []
    for injection_order, sample in enumerate(samples, start=1):
        intensity = pd.to_numeric(pd.Series([row.get(sample)]), errors="coerce").iloc[0]
        detected_column = f"{sample}::detected"
        detected = (
            bool(row.get(detected_column, False))
            if detected_column in features
            else bool(pd.notna(intensity) and intensity > 0.0)
        )
        internal_group = group_map.get(sample, "unclassified")
        records.append({
            "injection_order": injection_order,
            "sample": sample,
            "group": group_name_map.get(
                sample, re.sub(r"[_-]+", " ", internal_group).strip()
            ),
            "group_label": internal_group,
            "sample_type": type_map.get(sample, "not recorded"),
            "intensity": intensity,
            "detected": detected,
            "detection_evidence": (
                "MZmine DETECTED state"
                if detected_column in features else "non-zero intensity fallback"
            ),
            "biological_group": is_biological_group(internal_group),
        })
    return pd.DataFrame.from_records(records)


def _untargeted_group_summary(abundance: pd.DataFrame) -> pd.DataFrame:
    """Summarise detection and robust abundance within biological groups."""
    biological = abundance[abundance["biological_group"].fillna(False)].copy()
    records: list[dict[str, object]] = []
    for (group_label, group), table in biological.groupby(
        ["group_label", "group"], sort=False
    ):
        values = pd.to_numeric(table["intensity"], errors="coerce")
        detected = (
            table["detected"].fillna(False).astype(bool)
            if "detected" in table else values.gt(0.0) & values.notna()
        )
        detected_values = values[detected]
        records.append({
            "group": group,
            "group_label": group_label,
            "group_size": len(table),
            "detected_samples": int(detected.sum()),
            "detection_percent": 100.0 * float(detected.mean()) if len(table) else np.nan,
            "median": float(detected_values.median()) if not detected_values.empty else np.nan,
            "q1": float(detected_values.quantile(0.25)) if not detected_values.empty else np.nan,
            "q3": float(detected_values.quantile(0.75)) if not detected_values.empty else np.nan,
            "iqr": float(detected_values.quantile(0.75) - detected_values.quantile(0.25))
            if not detected_values.empty else np.nan,
        })
    return pd.DataFrame.from_records(records)


def _untargeted_review_outputs(result: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialise the persistent GUI review basket and reacquisition targets."""
    basket = st.session_state.setdefault("standard_untargeted_review_basket", {})
    review = pd.DataFrame.from_records(list(basket.values()))
    if review.empty:
        review = pd.DataFrame(columns=[
            "decision", "note", "stream", "phase", "polarity", "feature_id",
            "mz", "rt_min", "mobility", "mobility_unit", "ccs",
            "raw_p_value", "log2_fold_change_b_vs_a",
        ])
    reacquisition = review[
        review["decision"].isin(["keep", "follow up"])
    ].copy()
    if not reacquisition.empty:
        reacquisition = reacquisition.rename(columns={
            "mz": "target_mz", "rt_min": "target_rt_min",
        })
        reacquisition["rt_window_start_min"] = (
            pd.to_numeric(reacquisition["target_rt_min"], errors="coerce") - 0.10
        ).clip(lower=0.0)
        reacquisition["rt_window_end_min"] = (
            pd.to_numeric(reacquisition["target_rt_min"], errors="coerce") + 0.10
        )
        ordered_columns = [
            "decision", "note", "phase", "polarity", "stream", "feature_id",
            "target_mz", "target_rt_min", "rt_window_start_min",
            "rt_window_end_min", "mobility", "mobility_unit", "ccs",
            "raw_p_value", "log2_fold_change_b_vs_a",
        ]
        reacquisition = reacquisition[[
            column for column in ordered_columns if column in reacquisition
        ]]
    result["untargeted_review_basket"] = review
    result["untargeted_reacquisition_list"] = reacquisition
    return review, reacquisition


def _render_untargeted_review_controls(
    result: dict, selected_statistics: pd.Series,
) -> None:
    """Record a human review decision for one selected untargeted feature."""
    stream = str(selected_statistics.get("source_stream", selected_statistics.get("stream", "")))
    feature_id = str(selected_statistics.get("feature_id", ""))
    key = f"{stream}::{feature_id}"
    basket = st.session_state.setdefault("standard_untargeted_review_basket", {})
    current = basket.get(key, {})
    st.markdown("#### Feature review basket")
    decision_col, note_col = st.columns([1, 3])
    decisions = ["keep", "follow up", "reject"]
    current_decision = str(current.get("decision", "follow up"))
    decision = decision_col.selectbox(
        "Review decision", decisions,
        index=decisions.index(current_decision) if current_decision in decisions else 1,
        key=f"standard-untargeted-review-decision::{key}",
    )
    note = note_col.text_area(
        "Review note", value=str(current.get("note", "")),
        key=f"standard-untargeted-review-note::{key}",
        placeholder="Why keep, reject, or revisit this feature?",
        height=90,
    )
    save_col, remove_col = st.columns(2)
    if save_col.button(
        "Save to review basket", type="primary", width="stretch",
        key=f"standard-untargeted-review-save::{key}",
    ):
        phase_match = re.match(r"^(NP|RP)", stream, flags=re.I)
        basket[key] = {
            "decision": decision,
            "note": note,
            "stream": stream,
            "phase": phase_match.group(1).upper() if phase_match else "",
            "polarity": "NEG" if "neg" in stream.casefold() else "POS",
            "feature_id": feature_id,
            "mz": selected_statistics.get("mz", np.nan),
            "rt_min": selected_statistics.get("rt", np.nan),
            "mobility": selected_statistics.get("mobility", np.nan),
            "mobility_unit": selected_statistics.get("mobility_unit", ""),
            "ccs": selected_statistics.get("ccs", np.nan),
            "raw_p_value": selected_statistics.get("p_value", np.nan),
            "log2_fold_change_b_vs_a": selected_statistics.get(
                "log2_fold_change_b_vs_a", np.nan
            ),
            "cross_acquisition_evidence": selected_statistics.get(
                "cross_acquisition_evidence", ""
            ),
        }
        st.session_state.pop("standard_report_signature", None)
        st.success(f"Feature {feature_id} saved as {decision}.")
    if current and remove_col.button(
        "Remove from basket", width="stretch",
        key=f"standard-untargeted-review-remove::{key}",
    ):
        basket.pop(key, None)
        st.session_state.pop("standard_report_signature", None)
        st.rerun()
    _untargeted_review_outputs(result)


def _render_cross_mode_abundance_correlation(
    result: dict, selected_statistics: pd.Series,
) -> None:
    """Plot paired abundance for selected cross-phase/polarity candidates."""
    analysis = result.get("untargeted_analysis", {})
    criteria_values = analysis.get("criteria", {})
    try:
        criteria = UntargetedComparisonCriteria(**criteria_values)
    except (TypeError, ValueError):
        return
    source_stream = str(selected_statistics.get(
        "source_stream", selected_statistics.get("stream", "")
    ))
    source_feature = str(selected_statistics.get("feature_id", ""))
    comparisons = []
    for prefix, label in (
        ("cross_phase", "Cross-phase counterpart"),
        ("opposite_polarity", "Opposite-polarity counterpart"),
    ):
        target_stream = str(selected_statistics.get(f"{prefix}_stream", ""))
        target_feature = str(selected_statistics.get(f"{prefix}_feature_id", ""))
        if not target_stream or not target_feature:
            continue
        pairs, method = paired_abundance_table(
            result, source_stream, source_feature,
            target_stream, target_feature, criteria,
        )
        if pairs.empty:
            continue
        rho = pd.to_numeric(pd.Series([
            selected_statistics.get(f"{prefix}_abundance_spearman")
        ]), errors="coerce").iloc[0]
        comparisons.append((label, target_stream, target_feature, pairs, method, rho))
    if not comparisons:
        return
    with st.expander("Cross-mode abundance correlation", expanded=True):
        for label, target_stream, target_feature, pairs, method, rho in comparisons:
            plot = pairs.copy()
            plot["log2_source"] = np.log2(
                plot["intensity_a"].clip(lower=0.0) + 1.0
            )
            plot["log2_counterpart"] = np.log2(
                plot["intensity_b"].clip(lower=0.0) + 1.0
            )
            figure = go.Figure()
            for group, group_table in plot.groupby("group", sort=False):
                figure.add_trace(go.Scatter(
                    x=group_table["log2_source"],
                    y=group_table["log2_counterpart"],
                    mode="markers", name=str(group),
                    text=group_table["pair_key"], marker=dict(size=10),
                    customdata=group_table[["sample_a", "sample_b"]].to_numpy(dtype=object),
                    hovertemplate=(
                        "%{text}<br>%{customdata[0]} ↔ %{customdata[1]}<br>"
                        "source %{x:.3f}<br>counterpart %{y:.3f}<extra></extra>"
                    ),
                ))
            figure.update_layout(
                height=350, margin=dict(l=55, r=25, t=35, b=55),
                title=(
                    f"{label}: {target_stream} feature {target_feature} · "
                    f"Spearman ρ {rho:+.3f} · n={len(plot)} · {method}"
                    if pd.notna(rho) else
                    f"{label}: {target_stream} feature {target_feature} · n={len(plot)} · {method}"
                ),
                xaxis_title=f"{source_stream} feature {source_feature} · log₂(intensity + 1)",
                yaxis_title=f"{target_stream} feature {target_feature} · log₂(intensity + 1)",
                legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0.0),
            )
            viewer_plotly_chart(
                figure, width="stretch", config={"displaylogo": False},
                key=f"standard-untargeted-correlation::{source_stream}::{source_feature}::{target_stream}::{target_feature}",
            )
            st.dataframe(pairs, hide_index=True, width="stretch")


def _render_selected_untargeted_abundance(
    result: dict,
    stream: str,
    feature_id: str,
    selected_statistics: pd.Series | None = None,
) -> None:
    abundance = _untargeted_abundance_table(result, stream, feature_id)
    if abundance.empty:
        st.info("Sample-level intensities are unavailable for this feature.")
        return
    st.subheader(f"Sample abundance · {stream} feature {feature_id}")
    control_col, log_col = st.columns([2, 1])
    view = control_col.radio(
        "Abundance view", ["Sample order", "Group distribution"],
        horizontal=True, key=f"standard-untargeted-abundance-view::{stream}::{feature_id}",
    )
    log2_scale = log_col.toggle(
        "Log₂ intensity", value=False,
        key=f"standard-untargeted-abundance-log2::{stream}::{feature_id}",
        help="Transforms each displayed value as log₂(intensity + 1).",
    )
    abundance = abundance.copy()
    abundance["display_intensity"] = pd.to_numeric(
        abundance["intensity"], errors="coerce"
    )
    if log2_scale:
        abundance["display_intensity"] = np.log2(
            abundance["display_intensity"].clip(lower=0.0) + 1.0
        )
    y_title = "log₂(feature intensity + 1)" if log2_scale else "Feature intensity"
    figure = go.Figure()
    for group, group_table in abundance.groupby("group", sort=False):
        common = dict(
            name=str(group), text=group_table["sample"],
            customdata=group_table[["sample_type", "group_label"]].to_numpy(dtype=object),
            hovertemplate=(
                "%{text}<br>displayed intensity %{y:.4g}<br>"
                "%{customdata[1]} · %{customdata[0]}<extra></extra>"
            ),
        )
        if view == "Group distribution":
            figure.add_trace(go.Violin(
                x=[str(group)] * len(group_table), y=group_table["display_intensity"],
                box_visible=True, meanline_visible=True, points="all", jitter=0.18,
                pointpos=0.0, **common,
            ))
        else:
            figure.add_trace(go.Scatter(
                x=group_table["injection_order"], y=group_table["display_intensity"],
                mode="lines+markers", **common,
            ))
    figure.update_layout(
        height=360, margin=dict(l=55, r=25, t=25, b=55),
        xaxis_title=("Sample group" if view == "Group distribution" else "Sample order in XCMS input"),
        yaxis_title=y_title,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0.0),
    )
    viewer_plotly_chart(
        figure, width="stretch",
        key=f"standard-untargeted-abundance::{stream}::{feature_id}",
        config={"displaylogo": False},
    )
    summary = _untargeted_group_summary(abundance)
    metric_columns = st.columns(3)
    if selected_statistics is not None:
        fold_change = pd.to_numeric(
            pd.Series([selected_statistics.get("log2_fold_change_b_vs_a")]),
            errors="coerce",
        ).iloc[0]
        raw_p = pd.to_numeric(
            pd.Series([selected_statistics.get("p_value")]), errors="coerce"
        ).iloc[0]
        metric_columns[0].metric(
            "Log₂ fold change (B vs A)",
            f"{fold_change:+.3f}" if pd.notna(fold_change) else "n/a",
        )
        metric_columns[1].metric(
            "Raw p-value", f"{raw_p:.3g}" if pd.notna(raw_p) else "n/a"
        )
    metric_columns[2].metric("Samples shown", len(abundance))
    st.dataframe(summary, hide_index=True, width="stretch")
    with st.expander("Sample-level abundance data", expanded=False):
        st.dataframe(abundance, hide_index=True, width="stretch")
    if selected_statistics is not None:
        _render_cross_mode_abundance_correlation(result, selected_statistics)
        _render_untargeted_review_controls(result, selected_statistics)


def _prepare_untargeted_map_table(
    streams: list[tuple[str, str, pd.DataFrame]], analysis: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    table = _mirrored_untargeted_table(streams)
    for column in ("rt", "mz", "log2_fold_change_b_vs_a", "p_value"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    mobility = pd.to_numeric(
        table.get("mobility", pd.Series(np.nan, index=table.index)),
        errors="coerce",
    )
    units = table.get("mobility_unit", pd.Series("", index=table.index)).fillna(
        "").astype(str)
    table["mobility_display"] = [
        f"mobility {value:.5f} {unit}".strip()
        if pd.notna(value) else "mobility not available"
        for value, unit in zip(mobility, units)
    ]
    table = table.dropna(subset=["rt", "mz", "signed_mz"])
    significant = table[table["significant"].fillna(False).astype(bool)].copy()
    background = table[~table.index.isin(significant.index)]
    if len(background) > 20_000:
        background = background.sample(n=20_000, random_state=20260814)
    threshold = float(analysis.get("criteria", {}).get(
        "p_value_threshold", SCIENTIFIC_DEFAULTS.untargeted_p_value_threshold))
    return table, significant, background, threshold


def _add_significant_map_traces(
    figure: go.Figure, significant: pd.DataFrame, table: pd.DataFrame,
    p_value_threshold: float,
) -> None:
    group_a = str(table.get("group_a", pd.Series(["group A"])).iloc[0])
    group_b = str(table.get("group_b", pd.Series(["group B"])).iloc[0])
    for higher_in_b, colour, label in (
        (False, "#2670c9", f"higher in {group_a}"),
        (True, "#df3f3f", f"higher in {group_b}"),
    ):
        selected = significant[
            significant["log2_fold_change_b_vs_a"].ge(0.0) == higher_in_b
        ].copy()
        if selected.empty:
            continue
        optional = (
            "cluster_hypothesis", "homologous_series_hypothesis",
            "significant_ch2_cluster_hypothesis", "isotope_pattern",
            "cross_acquisition_evidence",
        )
        for column in optional:
            selected[column] = selected.get(
                column, pd.Series("", index=selected.index)).fillna("")
        custom_columns = [
            "feature_id", "source_stream", "ion_mode", "mz",
            "log2_fold_change_b_vs_a", "p_value", *optional, "mobility_display",
        ]
        figure.add_trace(go.Scattergl(
            x=selected["rt"], y=selected["signed_mz"], mode="markers", name=label,
            marker=dict(
                color=_p_value_marker_colours(
                    selected["p_value"], colour, p_value_threshold),
                size=9, opacity=0.95, symbol="diamond",
                line=dict(color="white", width=0.7),
            ),
            customdata=selected[custom_columns].to_numpy(dtype=object),
            hovertemplate=(
                "%{customdata[1]} · %{customdata[2]} feature %{customdata[0]}<br>"
                "RT %{x:.3f} min · measured m/z %{customdata[3]:.5f}<br>"
                "log2FC %{customdata[4]:+.2f} · raw p %{customdata[5]:.3g}<br>"
                "%{customdata[6]}<br>%{customdata[7]}<br>%{customdata[8]}<br>"
                "%{customdata[9]}<br>%{customdata[10]}<br>"
                "%{customdata[11]}<extra></extra>"
            ),
        ))


def _build_untargeted_map_figure(
    table: pd.DataFrame, significant: pd.DataFrame, background: pd.DataFrame,
    threshold: float, selection: tuple[str, str] | None,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scattergl(
        x=background["rt"], y=background["signed_mz"], mode="markers",
        name="unannotated, not significant",
        marker=dict(color="#aeb6c1", size=4, opacity=0.35),
        customdata=background[
            ["feature_id", "source_stream", "ion_mode", "mz", "mobility_display"]
        ].to_numpy(dtype=object),
        hovertemplate=("%{customdata[1]} · %{customdata[2]} feature "
                       "%{customdata[0]}<br>RT %{x:.3f} min · measured m/z "
                       "%{customdata[3]:.5f}<br>%{customdata[4]}"
                       "<extra>not significant</extra>"),
    ))
    _add_significant_map_traces(figure, significant, table, threshold)
    highlighted = _selected_untargeted_point(table, selection)
    if not highlighted.empty:
        row = highlighted.iloc[0]
        label = f"selected: {row['source_stream']} feature {row['feature_id']}"
        figure.add_trace(go.Scattergl(
            x=highlighted["rt"], y=highlighted["signed_mz"], mode="markers",
            name="selected feature",
            marker=dict(color="#7c3aed", size=18, opacity=1.0, symbol="diamond",
                        line=dict(color="#2e1065", width=2.2)),
            hovertemplate=f"{label}<br>RT %{{x:.3f}} min<extra></extra>",
        ))
        figure.add_annotation(
            x=float(row["rt"]), y=float(row["signed_mz"]), text=label,
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
            arrowcolor="#7c3aed", ax=45, ay=-42, bgcolor="#7c3aed",
            bordercolor="#2e1065", borderwidth=1,
            font=dict(color="white", size=12), opacity=0.97,
        )
    figure.update_layout(
        height=610, margin=dict(l=55, r=25, t=25, b=50),
        xaxis_title="Retention time (min)",
        yaxis_title="signed m/z (POS above · NEG below)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0.0),
        hovermode="closest",
    )
    figure.add_hline(line_color="#52606d", line_width=1.2, y=0.0)
    return figure


def _updated_map_selection(
    event: object, current: tuple[str, str] | None,
) -> tuple[str, str] | None:
    try:
        points = list(cast(Any, event).selection.points)
    except (AttributeError, TypeError):
        points = list(event.get("selection", {}).get("points", ())) if isinstance(
            event, dict) else []
    if not points:
        return current
    customdata = list(points[-1].get("customdata", ()) or ())
    return (str(customdata[1]), str(customdata[0])) if len(customdata) >= 2 else current


def _render_selected_map_feature(
    result: dict | None, selection: tuple[str, str] | None,
    significant: pd.DataFrame,
) -> None:
    if result is None or not selection:
        return
    stream, feature = selection
    mask = (significant["source_stream"].astype(str).eq(str(stream))
            & significant["feature_id"].astype(str).eq(str(feature)))
    if not mask.any():
        return
    _render_selected_untargeted_abundance(
        result, stream, feature, significant[mask].iloc[0])


def _render_standard_untargeted_feature_map(
    analysis: dict, result: dict | None = None,
) -> None:
    """Render a mirrored polarity RT-m/z map of unannotated features."""
    available = {
        str(stream): table for stream, table in analysis.get("by_stream", {}).items()
        if isinstance(table, pd.DataFrame) and not table.empty
    }
    if not available:
        return
    st.subheader("Untargeted MS feature map")
    families = _untargeted_map_families(available)
    family = st.selectbox(
        "Chromatography", list(families), key="standard_untargeted_map_chromatography",
        help="When both polarities exist, LASER draws them on one shared RT axis.",
    )
    selection_key = f"standard_untargeted_selected_feature::{family}"
    current = st.session_state.get(selection_key)
    table, significant, background, threshold = _prepare_untargeted_map_table(
        families[family], analysis)
    if table.empty:
        st.info("No unannotated features with valid RT and m/z are available.")
        return
    event = viewer_plotly_chart(
        _build_untargeted_map_figure(
            table, significant, background, threshold, current),
        width="stretch", key=f"standard-untargeted-map::{family}",
        on_select="rerun", selection_mode="points",
        config={"displaylogo": False},
    )
    updated = _updated_map_selection(event, current)
    if updated != current:
        st.session_state[selection_key] = updated
        st.rerun()
    _render_selected_map_feature(result, st.session_state.get(selection_key), significant)



def _render_untargeted_review_basket(result: dict) -> None:
    """Show and export all recorded review decisions and reacquisition targets."""
    review, reacquisition = _untargeted_review_outputs(result)
    st.subheader("Feature review basket")
    if review.empty:
        return
    counts = review["decision"].value_counts()
    columns = st.columns(3)
    for index, decision in enumerate(("keep", "follow up", "reject")):
        columns[index].metric(decision.title(), int(counts.get(decision, 0)))
    st.dataframe(review, hide_index=True, width="stretch")
    download_col, target_col = st.columns(2)
    download_col.download_button(
        "Download review basket CSV", review.to_csv(index=False).encode(),
        file_name="LASER_untargeted_feature_review_basket.csv",
        mime="text/csv", width="stretch",
    )
    target_col.download_button(
        "Download targeted reacquisition list CSV",
        reacquisition.to_csv(index=False).encode(),
        file_name="LASER_untargeted_targeted_reacquisition.csv",
        mime="text/csv", width="stretch",
        help=(
            "Contains keep and follow-up targets with m/z, RT, polarity, phase, "
            "feature ID, notes and a default ±0.10 min acquisition window."
        ),
    )


def render_untargeted_workspace(result: dict) -> dict:
    """Render the shared post-annotation untargeted workspace."""
    analysis = _render_standard_untargeted(result)
    if analysis:
        result["untargeted_analysis"] = analysis
        _render_untargeted_review_basket(result)
    return analysis
