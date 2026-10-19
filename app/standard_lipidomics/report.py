"""Self-contained evidence report for the standard-lipidomics domain.

The conventional XCMS peak-group lists contain useful feature-level evidence,
but they do not contain mobility, CCS, feature-linked MS/MS, or raw extracted
ion chromatograms.  This report deliberately mirrors the organization and
visual language of the 4D report while showing only evidence that was actually
available to the standard annotation engine.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from output_io import atomic_write_text
from report_sections import ReportSection, ReportSectionRegistry
from test_report import TestReport, render_report


_PALETTE = (
    "#2563eb",
    "#16a34a",
    "#d97706",
    "#7c3aed",
    "#db2777",
    "#0891b2",
    "#65a30d",
    "#dc2626",
    "#475569",
    "#0f766e",
)

_ANNOTATION_COLUMNS = (
    "identity",
    "lipid_class",
    "adduct",
    "stream",
    "feature_id",
    "role",
    "mz",
    "reference_mz",
    "ppm_error",
    "rt_min",
    "reference_rt_min",
    "rt_error_min",
    "rt_error_seconds",
    "annotation_score",
    "grid_coherence_score",
    "grid_coherence_neighbors",
    "reference_rt_observations",
    "rank_for_identity",
    "rank_for_feature",
    "status",
    "rt_evidence",
    "multiple_features",
    "feature_count",
    "all_feature_ids",
    "median_intensity",
    "intensity_q1",
    "intensity_q3",
    "maximum_intensity",
    "detected_fraction",
    "sample_intensities",
    "xcms_peak_count",
    "xcms_group_width_seconds",
)

_STYLE = """
<style>
.standard-report{font-size:15px}.standard-report h2{font-size:20px;margin:4px 0 12px}
.standard-report h3{font-size:16px;margin:22px 0 8px;text-transform:none;letter-spacing:0;color:#111827}
.standard-report .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:0 0 18px}
.standard-report .card{border:1px solid #dfe3e8;border-radius:9px;background:#fff;padding:11px 13px}
.standard-report .card .label{font-size:12px;color:#667085}.standard-report .card .value{font-size:25px;font-weight:750;color:#12345b}
.standard-report .card .detail{font-size:12px;color:#667085;margin-top:2px}
.standard-report .notice{border-left:4px solid #2563eb;background:#eff6ff;padding:10px 13px;margin:0 0 16px;border-radius:4px}
.standard-report .muted{color:#667085}.standard-report .scroll{overflow:auto;max-height:680px;border:1px solid #e5e7eb;border-radius:7px}
.standard-report table{width:100%;font-size:12px;margin:0}.standard-report th{position:sticky;top:0;z-index:1}
.standard-report .search{width:min(520px,100%);font-size:15px;padding:9px 11px;border:1px solid #cbd5e1;border-radius:7px;margin:0 0 12px}
.feature-list{display:flex;flex-direction:column;gap:9px}.feature-row{display:grid;grid-template-columns:minmax(240px,1.15fr) minmax(410px,2.2fr);gap:12px;border:1px solid #dfe3e8;border-left:5px solid #2563eb;border-radius:8px;padding:11px 13px;background:#fff}
.feature-row.equivalent{border-left-color:#d97706}.feature-row.review{border-left-color:#64748b}.feature-row h4{font-size:16px;margin:0 0 4px}.badge{display:inline-block;font-size:11px;font-weight:750;padding:2px 7px;border-radius:10px;background:#dbeafe;color:#1d4ed8;margin:0 5px 4px 0}.badge.alt{background:#fef3c7;color:#92400e}.badge.review{background:#e2e8f0;color:#475569}
.feature-meta{font-size:12px;color:#667085;line-height:1.45}.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(92px,1fr));gap:6px}.metric{border:1px solid #e5e7eb;border-radius:6px;padding:6px 8px}.metric .k{font-size:10px;color:#667085}.metric .v{font-size:14px;font-weight:700}.intensity{grid-column:1/-1;display:flex;align-items:center;gap:10px}.intensity svg{width:100%;height:56px;background:#f8fafc;border-radius:5px}.intensity .caption{font-size:11px;color:#667085;min-width:118px}
.class-bars svg,.scatter svg,.hist svg{width:100%;height:auto;max-width:980px}.class-bars text,.scatter text,.hist text{font:12px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#475569}.plot-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:12px}.plot{border:1px solid #e5e7eb;border-radius:8px;padding:10px;background:#fff}.plot h4{margin:0 0 6px;font-size:14px}
.availability td:first-child{font-weight:650}.yes{color:#087a4b;font-weight:700}.no{color:#9b1c1c;font-weight:700}.conditional{color:#9a6700;font-weight:700}
@media(max-width:850px){.feature-row{grid-template-columns:1fr}.metric-grid{grid-template-columns:repeat(2,1fr)}.plot-grid{grid-template-columns:1fr}}
</style>
"""


def _frame(value) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _first_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _series(
    frame: pd.DataFrame,
    names: Iterable[str],
    default=np.nan,
) -> pd.Series:
    column = _first_column(frame, names)
    if column is None:
        return pd.Series(default, index=frame.index)
    return frame[column]


def _lipid_class(identity: str) -> str:
    text = str(identity or "")
    match = re.match(r"\s*([^()]+?)\s*\(", text)
    return match.group(1).strip() if match else "(unknown)"


def _sample_names(result: dict) -> list[str]:
    names: list[str] = []
    for stream in result.get("streams", {}).values():
        features = _frame(stream.get("features"))
        for name in features.attrs.get("sample_columns", ()):
            if str(name) not in names:
                names.append(str(name))
    return names


def _required_feature_ids(result: dict) -> dict[str, set[str]]:
    required: dict[str, set[str]] = {}
    for key in ("identified_lipids", "equivalent_lipids", "all_candidates"):
        frame = _frame(result.get(key))
        if frame.empty:
            continue
        stream_column = _first_column(frame, ("stream", "standard_stream"))
        feature_column = _first_column(frame, ("feature_id", "feat_feature_id"))
        if stream_column is None or feature_column is None:
            continue
        for stream, group in frame.groupby(frame[stream_column].astype(str)):
            required.setdefault(str(stream), set()).update(
                group[feature_column].fillna("").astype(str)
            )
    return required


def _feature_metadata(result: dict) -> dict[tuple[str, str], dict]:
    lookup: dict[tuple[str, str], dict] = {}
    required = _required_feature_ids(result)
    for stream_name, stream in result.get("streams", {}).items():
        features = _frame(stream.get("features"))
        if features.empty or "feature_id" not in features:
            continue
        wanted = required.get(str(stream_name), set())
        subset = (
            features[features["feature_id"].fillna("").astype(str).isin(wanted)]
            if wanted
            else features.iloc[0:0]
        )
        for _, row in subset.iterrows():
            lookup[(str(stream_name), str(row["feature_id"]))] = {
                "npeaks": pd.to_numeric(row.get("npeaks"), errors="coerce"),
                "rtmin": pd.to_numeric(row.get("rtmin"), errors="coerce"),
                "rtmax": pd.to_numeric(row.get("rtmax"), errors="coerce"),
            }
    return lookup


def _normalise_annotations(
    frame: pd.DataFrame,
    *,
    role: str,
    sample_names: list[str],
    feature_metadata: dict[tuple[str, str], dict],
) -> pd.DataFrame:
    """Map LASER-reference and MATLAB-engine rows onto one report schema."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=_ANNOTATION_COLUMNS)
    source = frame.copy().reset_index(drop=True)
    out = pd.DataFrame(index=source.index)
    out["identity"] = (
        _series(source, ("feature_identity", "name", "species"), "")
        .fillna("")
        .astype(str)
    )
    class_values = (
        _series(source, ("lipid_class", "grid_class", "class"), "")
        .fillna("")
        .astype(str)
    )
    out["lipid_class"] = [
        value if value else _lipid_class(identity)
        for value, identity in zip(class_values, out["identity"])
    ]
    out["adduct"] = _series(source, ("adduct",), "").fillna("").astype(str)
    out["stream"] = (
        _series(source, ("stream", "standard_stream"), "").fillna("").astype(str)
    )
    out["feature_id"] = (
        _series(source, ("feature_id", "feat_feature_id"), "").fillna("").astype(str)
    )
    out["role"] = _series(source, ("quantitative_role",), role).fillna(role).astype(str)
    for output, names in {
        "mz": ("mz", "feat_mz"),
        "reference_mz": ("reference_mz", "ref_mz"),
        "ppm_error": ("ppm_error",),
        "rt_min": ("rt_min", "feat_rt"),
        "reference_rt_min": ("reference_rt_min", "rt_expected", "ref_rt"),
        "rt_error_min": ("rt_error_min", "rt_error"),
        "annotation_score": ("annotation_score", "combined_score", "score"),
        "grid_coherence_score": ("grid_coherence_score",),
        "grid_coherence_neighbors": ("grid_coherence_neighbors",),
        "reference_rt_observations": ("reference_rt_observations", "n_RT"),
        "rank_for_identity": ("candidate_rank_for_lipid", "rank_for_lipid"),
        "rank_for_feature": ("candidate_rank_for_feature", "rank_for_feature"),
    }.items():
        out[output] = pd.to_numeric(_series(source, names), errors="coerce")
    missing_ppm = (
        out["ppm_error"].isna()
        & out["mz"].notna()
        & out["reference_mz"].notna()
        & out["reference_mz"].ne(0)
    )
    out.loc[missing_ppm, "ppm_error"] = (
        (out.loc[missing_ppm, "mz"] - out.loc[missing_ppm, "reference_mz"])
        / out.loc[missing_ppm, "reference_mz"]
        * 1e6
    )
    missing_rt = (
        out["rt_error_min"].isna()
        & out["rt_min"].notna()
        & out["reference_rt_min"].notna()
    )
    out.loc[missing_rt, "rt_error_min"] = (
        out.loc[missing_rt, "rt_min"] - out.loc[missing_rt, "reference_rt_min"]
    )
    out["rt_error_seconds"] = out["rt_error_min"] * 60.0
    out["status"] = (
        _series(
            source,
            ("auto_annotation_status", "annotation_status", "annotation_source"),
            "",
        )
        .fillna("")
        .astype(str)
    )
    out["rt_evidence"] = (
        _series(source, ("rt_evidence_mode", "rt_model_type", "annotation_source"), "")
        .fillna("")
        .astype(str)
    )
    out["multiple_features"] = (
        _series(source, ("multiple_features_for_identity",), False)
        .fillna(False)
        .astype(bool)
    )
    out["feature_count"] = (
        pd.to_numeric(
            _series(source, ("feature_count_for_identity",), 1), errors="coerce"
        )
        .fillna(1)
        .astype(int)
    )
    out["all_feature_ids"] = (
        _series(source, ("all_feature_ids_for_identity",), "").fillna("").astype(str)
    )

    intensity_columns: list[tuple[str, str]] = []
    for sample in sample_names:
        for candidate in (f"intensity::{sample}", f"feat_{sample}", sample):
            if candidate in source.columns:
                intensity_columns.append((sample, candidate))
                break
    medians, q1s, q3s, maxima, detected, values_json = [], [], [], [], [], []
    for _, row in source.iterrows():
        pairs = [
            (sample, pd.to_numeric(row.get(column), errors="coerce"))
            for sample, column in intensity_columns
        ]
        finite = [(sample, float(value)) for sample, value in pairs if pd.notna(value)]
        positive = np.array([value for _, value in finite if value > 0], dtype=float)
        medians.append(float(np.median(positive)) if len(positive) else np.nan)
        q1s.append(float(np.quantile(positive, 0.25)) if len(positive) else np.nan)
        q3s.append(float(np.quantile(positive, 0.75)) if len(positive) else np.nan)
        maxima.append(float(np.max(positive)) if len(positive) else np.nan)
        detected.append(
            len(positive) / len(intensity_columns) if intensity_columns else np.nan
        )
        values_json.append(finite)
    out["median_intensity"] = medians
    out["intensity_q1"] = q1s
    out["intensity_q3"] = q3s
    out["maximum_intensity"] = maxima
    out["detected_fraction"] = detected
    out["sample_intensities"] = values_json
    npeaks, widths = [], []
    for stream, feature_id in zip(out["stream"], out["feature_id"]):
        metadata = feature_metadata.get((str(stream), str(feature_id)), {})
        npeaks.append(metadata.get("npeaks", np.nan))
        rtmin, rtmax = metadata.get("rtmin", np.nan), metadata.get("rtmax", np.nan)
        widths.append(
            (float(rtmax) - float(rtmin)) * 60.0
            if pd.notna(rtmin) and pd.notna(rtmax)
            else np.nan
        )
    out["xcms_peak_count"] = npeaks
    out["xcms_group_width_seconds"] = widths
    return out


def _fmt(value, digits: int = 2, suffix: str = "") -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return "n/a" if pd.isna(numeric) else f"{float(numeric):.{digits}f}{suffix}"


def _percent(value, digits: int = 1) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return "n/a" if pd.isna(numeric) else f"{float(numeric) * 100:.{digits}f}%"


def _compact_number(value) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    numeric = float(numeric)
    for divisor, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(numeric) >= divisor:
            return f"{numeric / divisor:.2f}{suffix}"
    return f"{numeric:.3g}"


def _cards(items: Iterable[tuple[str, str, str]]) -> str:
    return (
        "<div class='cards'>"
        + "".join(
            "<div class='card'><div class='label'>"
            + html.escape(label)
            + "</div><div class='value'>"
            + html.escape(value)
            + "</div>"
            + (f"<div class='detail'>{html.escape(detail)}</div>" if detail else "")
            + "</div>"
            for label, value, detail in items
        )
        + "</div>"
    )


def _table(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    data = _frame(frame).copy()
    if columns is not None:
        selected = [column for column in columns if column in data]
        data = data[selected]
    if data.empty:
        return "<p class='muted'>No rows available.</p>"
    return (
        "<div class='scroll'>"
        + data.to_html(index=False, border=0, escape=True, na_rep="")
        + "</div>"
    )


def _class_summary(selected: pd.DataFrame, equivalents: pd.DataFrame) -> pd.DataFrame:
    rows = []
    classes = sorted(
        set(selected.get("lipid_class", ())) | set(equivalents.get("lipid_class", ()))
    )
    for lipid_class in classes:
        primary = selected[selected["lipid_class"].eq(lipid_class)]
        alternate = equivalents[equivalents["lipid_class"].eq(lipid_class)]
        rows.append(
            {
                "lipid_class": lipid_class,
                "unique_primary_identities": int(primary["identity"].nunique()),
                "selected_features": int(primary["feature_id"].nunique()),
                "equivalent_features": int(alternate["feature_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _class_bars(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "<p class='muted'>No selected identities.</p>"
    data = summary.sort_values("unique_primary_identities", ascending=False)
    peak = max(float(data["unique_primary_identities"].max()), 1.0)
    width, label, row_h = 620, 130, 29
    height = 32 + len(data) * row_h
    rows = []
    for index, (_, row) in enumerate(data.iterrows()):
        y = 24 + index * row_h
        value = int(row["unique_primary_identities"])
        equivalent = int(row["equivalent_features"])
        bar = value / peak * width
        rows.append(
            f"<text x='{label - 7}' y='{y + 15}' text-anchor='end'>"
            f"{html.escape(str(row['lipid_class']))}</text>"
            f"<rect x='{label}' y='{y}' width='{max(bar, 1):.1f}' height='20' rx='3' fill='#6da6df'/>"
            f"<text x='{label + bar + 5:.1f}' y='{y + 15}'>{value}"
            + (f" · {equivalent} equivalent feature(s)" if equivalent else "")
            + "</text>"
        )
    return (
        "<div class='class-bars'><svg viewBox='0 0 930 "
        + str(height)
        + "' role='img' aria-label='Unique primary lipid identities per class'>"
        + "".join(rows)
        + "</svg></div>"
    )


def _intensity_svg(values: list[tuple[str, float]]) -> str:
    if not values:
        return "<span class='muted'>No sample intensities.</span>"
    shown = values[:80]
    positive = [max(float(value), 0.0) for _, value in shown]
    transformed = np.log1p(positive)
    peak = max(float(np.max(transformed)), 1.0)
    width, height = 520, 52
    gap = width / max(len(shown), 1)
    bars = []
    for index, ((sample, raw), value) in enumerate(zip(shown, transformed)):
        bar_h = float(value) / peak * 43.0
        bars.append(
            f"<rect x='{index * gap:.2f}' y='{height - bar_h:.2f}' "
            f"width='{max(gap - 0.8, 0.8):.2f}' height='{bar_h:.2f}' fill='#4b8bc8'>"
            f"<title>{html.escape(sample)}: {float(raw):.5g}</title></rect>"
        )
    return f"<svg viewBox='0 0 {width} {height}'>{''.join(bars)}</svg>"


def _feature_rows(frame: pd.DataFrame, *, kind: str) -> str:
    if frame.empty:
        return "<p class='muted'>No features in this category.</p>"
    search_id = f"search-{kind}"
    rows = []
    for _, row in frame.iterrows():
        role = str(row.get("role", kind))
        css = (
            "equivalent"
            if role == "equivalent"
            else "review"
            if kind == "review"
            else ""
        )
        badge_css = (
            "alt" if role == "equivalent" else "review" if kind == "review" else ""
        )
        badge = (
            "EQUIVALENT"
            if role == "equivalent"
            else "REVIEW CANDIDATE"
            if kind == "review"
            else "PRIMARY"
        )
        identity = str(row.get("identity", ""))
        stream = str(row.get("stream", ""))
        feature_id = str(row.get("feature_id", ""))
        search = " ".join(
            (
                identity,
                stream,
                feature_id,
                str(row.get("lipid_class", "")),
                str(row.get("adduct", "")),
            )
        ).casefold()
        feature_note = ""
        if bool(row.get("multiple_features", False)):
            feature_note = (
                f"<div class='feature-meta'>Selected among {int(row.get('feature_count', 1))} "
                f"feature matches: {html.escape(str(row.get('all_feature_ids', '')))}</div>"
            )
        metrics = (
            ("mass error", _fmt(row.get("ppm_error"), 2, " ppm")),
            ("RT delta", _fmt(row.get("rt_error_seconds"), 1, " s")),
            ("grid coherence", _fmt(row.get("grid_coherence_score"), 3)),
            ("annotation score", _fmt(row.get("annotation_score"), 3)),
            (
                "reference RT observations",
                _fmt(row.get("reference_rt_observations"), 0),
            ),
            ("XCMS peak count", _fmt(row.get("xcms_peak_count"), 0)),
            ("XCMS group RT width", _fmt(row.get("xcms_group_width_seconds"), 1, " s")),
            ("positive intensity fraction", _percent(row.get("detected_fraction"))),
        )
        metric_html = (
            "<div class='metric-grid'>"
            + "".join(
                f"<div class='metric'><div class='k'>{html.escape(label)}</div>"
                f"<div class='v'>{html.escape(value)}</div></div>"
                for label, value in metrics
            )
            + "</div>"
        )
        rows.append(
            f"<article class='feature-row {css}' data-search='{html.escape(search)}'>"
            "<div>"
            f"<span class='badge {badge_css}'>{badge}</span>"
            f"<h4>{html.escape(identity)} {html.escape(str(row.get('adduct', '')))}</h4>"
            f"<div class='feature-meta'>{html.escape(stream)} · feature {html.escape(feature_id)} · "
            f"m/z {_fmt(row.get('mz'), 6)} · RT {_fmt(row.get('rt_min'), 4, ' min')}</div>"
            f"<div class='feature-meta'>evidence: {html.escape(str(row.get('rt_evidence', '')))} · "
            f"status: {html.escape(str(row.get('status', '')))}</div>{feature_note}</div>"
            f"<div>{metric_html}<div class='intensity'><div class='caption'>median "
            f"{_compact_number(row.get('median_intensity'))}<br>log-scaled sample intensities</div>"
            f"{_intensity_svg(row.get('sample_intensities', []))}</div></div></article>"
        )
    script = (
        f"<script>function filter_{kind}(){{const q=document.getElementById('{search_id}').value.toLowerCase();"
        f"document.querySelectorAll('.panel-{kind} .feature-row').forEach(r=>r.style.display=r.dataset.search.includes(q)?'grid':'none');}}</script>"
    )
    return (
        f"<input id='{search_id}' class='search' type='search' placeholder='Filter by lipid, class, stream, adduct, or feature ID' oninput='filter_{kind}()'>"
        f"<div class='feature-list'>{''.join(rows)}</div>{script}"
    )


def _stream_summary(
    result: dict,
    selected: pd.DataFrame,
    equivalents: pd.DataFrame,
    review: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for stream_name, stream in result.get("streams", {}).items():
        features = _frame(stream.get("features"))
        rows.append(
            {
                "stream": stream_name,
                "input_XCMS_features": len(features),
                "sample_intensity_columns": len(
                    features.attrs.get("sample_columns", ())
                ),
                "primary_identities": int(selected["stream"].eq(stream_name).sum())
                if not selected.empty
                else 0,
                "equivalent_features": int(equivalents["stream"].eq(stream_name).sum())
                if not equivalents.empty
                else 0,
                "review_candidates": int(review["stream"].eq(stream_name).sum())
                if not review.empty
                else 0,
                "RT_grid_used": bool(
                    stream.get("criteria", {}).get("uses_rt_grid", True)
                ),
            }
        )
    return pd.DataFrame(rows)


def _metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "absolute mass error (ppm)": frame.get(
            "ppm_error", pd.Series(dtype=float)
        ).abs(),
        "absolute RT delta (s)": frame.get(
            "rt_error_seconds", pd.Series(dtype=float)
        ).abs(),
        "grid coherence": frame.get("grid_coherence_score", pd.Series(dtype=float)),
        "annotation score": frame.get("annotation_score", pd.Series(dtype=float)),
        "positive intensity fraction": frame.get(
            "detected_fraction", pd.Series(dtype=float)
        ),
        "XCMS group RT width (s)": frame.get(
            "xcms_group_width_seconds", pd.Series(dtype=float)
        ),
    }
    rows = []
    for lipid_class in [
        "ALL",
        *sorted(frame.get("lipid_class", pd.Series(dtype=str)).dropna().unique()),
    ]:
        mask = (
            pd.Series(True, index=frame.index)
            if lipid_class == "ALL"
            else frame["lipid_class"].eq(lipid_class)
        )
        for name, values in metrics.items():
            data = pd.to_numeric(values[mask], errors="coerce").dropna()
            if data.empty:
                continue
            rows.append(
                {
                    "lipid_class": lipid_class,
                    "metric": name,
                    "n": len(data),
                    "mean": round(float(data.mean()), 4),
                    "Q1": round(float(data.quantile(0.25)), 4),
                    "median": round(float(data.median()), 4),
                    "Q3": round(float(data.quantile(0.75)), 4),
                }
            )
    return pd.DataFrame(rows)


def _scatter_plot(frame: pd.DataFrame, title: str) -> str:
    finite = frame[
        pd.to_numeric(frame.get("rt_min"), errors="coerce").notna()
        & pd.to_numeric(frame.get("mz"), errors="coerce").notna()
    ].copy()
    if finite.empty:
        return "<p class='muted'>No finite RT/m/z coordinates.</p>"
    x = pd.to_numeric(finite["rt_min"], errors="coerce")
    y = pd.to_numeric(finite["mz"], errors="coerce")
    xmin, xmax, ymin, ymax = (
        float(x.min()),
        float(x.max()),
        float(y.min()),
        float(y.max()),
    )
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    points = []
    classes = sorted(finite["lipid_class"].unique())
    colors = {
        name: _PALETTE[index % len(_PALETTE)] for index, name in enumerate(classes)
    }
    for (_, row), xv, yv in zip(finite.iterrows(), x, y):
        px = 54 + (float(xv) - xmin) / dx * 710
        py = 310 - (float(yv) - ymin) / dy * 270
        points.append(
            f"<circle cx='{px:.2f}' cy='{py:.2f}' r='3.2' fill='{colors[row['lipid_class']]}' opacity='.72'>"
            f"<title>{html.escape(str(row['identity']))} · feature {html.escape(str(row['feature_id']))} · RT {float(xv):.4f} · m/z {float(yv):.6f}</title></circle>"
        )
    legend = " ".join(f"{name}" for name in classes[:12])
    return (
        f"<div class='plot scatter'><h4>{html.escape(title)}</h4>"
        "<svg viewBox='0 0 800 340'>"
        "<line x1='54' y1='310' x2='770' y2='310' stroke='#94a3b8'/><line x1='54' y1='35' x2='54' y2='310' stroke='#94a3b8'/>"
        f"<text x='400' y='334' text-anchor='middle'>retention time (min), {xmin:.2f}–{xmax:.2f}</text>"
        f"<text x='8' y='22'>m/z {ymin:.1f}–{ymax:.1f}</text>{''.join(points)}</svg>"
        f"<div class='feature-meta'>Classes: {html.escape(legend)}{' …' if len(classes) > 12 else ''}</div></div>"
    )


def _calibration_tables(result: dict) -> str:
    sections = []
    rt_models, np_alignments = [], []
    for stream_name, stream in result.get("streams", {}).items():
        models = _frame(stream.get("rt_models"))
        if not models.empty:
            models = models.copy()
            models.insert(0, "stream", stream_name)
            rt_models.append(models)
        alignment = _frame(stream.get("np_rt_alignment"))
        if not alignment.empty:
            alignment = alignment.copy()
            if "stream" not in alignment:
                alignment.insert(0, "stream", stream_name)
            np_alignments.append(alignment)
    if rt_models:
        sections.append(
            "<h3>RP class RT models</h3>"
            + _table(pd.concat(rt_models, ignore_index=True))
        )
    if np_alignments:
        sections.append(
            "<h3>NP anchor alignment</h3>"
            + _table(pd.concat(np_alignments, ignore_index=True))
        )
    grid = _frame(result.get("grid_coherence_audit"))
    if not grid.empty:
        sections.append("<h3>Grid-coherence audit</h3>" + _table(grid))
    for key, title in (
        ("matlab_seed_audit", "Automatic MATLAB anchor selection"),
        ("matlab_reference_seed_audit", "Independent reference-anchor evidence"),
        ("matlab_automatic_anchors", "Accepted automatic anchors"),
        ("matlab_internal_standard_audit", "Internal-standard recovery"),
        ("matlab_grid_coherence_audit", "MATLAB grid coherence"),
        ("matlab_rp_isomer_anchor_audit", "RP isomer-aware anchor refinement"),
        ("matlab_positional_isomer_audit", "RP positional-isomer resolution"),
        ("matlab_positional_refit_audit", "Experimental positional RT refit"),
    ):
        table = _frame(result.get(key))
        if not table.empty:
            sections.append(f"<h3>{title}</h3>" + _table(table))
    return (
        "".join(sections)
        or "<p class='muted'>No RT or anchor calibration tables were produced.</p>"
    )


def _policy_tables(result: dict) -> str:
    sections = []
    for key, title in (
        ("policy_audit", "MATLAB lipid-policy decisions"),
        ("np_rt_audit", "NP reference identity coverage"),
        ("matlab_compound_audit", "MATLAB lipid-definition audit"),
    ):
        table = _frame(result.get(key))
        if not table.empty:
            sections.append(f"<h3>{title}</h3>" + _table(table))
    settings = result.get("composition_range_settings")
    if settings:
        sections.append(
            "<h3>Carbon/double-bond limits</h3><pre>"
            + html.escape(str(settings))
            + "</pre>"
        )
    return "".join(sections) or "<p class='muted'>No policy audit was produced.</p>"


def _availability(result: dict) -> pd.DataFrame:
    has_samples = bool(_sample_names(result))
    uses_rt = any(
        bool(stream.get("criteria", {}).get("uses_rt_grid", True))
        for stream in result.get("streams", {}).values()
    )
    return pd.DataFrame(
        [
            ("Feature m/z and retention time", "available", "XCMS peak-group list"),
            (
                "Per-sample feature intensities",
                "available" if has_samples else "not available",
                "XCMS peak-group list",
            ),
            (
                "XCMS peak count and group boundaries",
                "available",
                "XCMS peak-group list",
            ),
            (
                "Reference/predicted RT residuals",
                "available" if uses_rt else "conditional",
                "only for a validated RP/NP RT grid",
            ),
            (
                "Local carbon/double-bond grid coherence",
                "conditional",
                "when the candidate has enough neighbouring grid nodes",
            ),
            (
                "Raw chromatogram traces",
                "not available",
                "requires mzML/mzXML; not stored in peakgrouplist",
            ),
            (
                "Ion-mobility traces and CCS",
                "not available",
                "standard workflow is not 4D",
            ),
            (
                "Feature-linked MS/MS spectra",
                "not available",
                "not stored in standard peakgrouplist",
            ),
            (
                "MZmine peak-shape metrics",
                "not available",
                "standard files originate from XCMS/MATLAB processing",
            ),
            (
                "IS-corrected QC RSD",
                "not available",
                "requires explicit sample metadata and IS normalization mapping",
            ),
            (
                "Two-group untargeted statistics",
                "available" if result.get("untargeted_analysis") else "conditional",
                "available after annotation when two biological groups are joined "
                "from project Excel metadata (or a validated XCMS fallback)",
            ),
        ],
        columns=["evidence", "status", "explanation"],
    )


@dataclass(frozen=True)
class StandardReportContext:
    """Precomputed evidence consumed by independent report sections."""

    result: Mapping[str, Any]
    data: Mapping[str, Any]


def _summary_section(context: StandardReportContext) -> str:
    data = context.data
    selected = data["selected"]
    equivalents = data["equivalents"]
    review = data["review"]
    streams = data["stream_summary"]
    differential = data["untargeted_differences"]
    return (
        _STYLE
        + "<div class='standard-report'>"
        + _cards(
            (
                (
                    "Input XCMS features",
                    f"{data['input_features']:,}",
                    f"across {len(streams)} stream(s)",
                ),
                (
                    "Primary lipid identities",
                    f"{len(selected):,}",
                    "one selected feature per lipid identity",
                ),
                (
                    "Equivalent features",
                    f"{len(equivalents):,}",
                    "additional matching features retained explicitly",
                ),
                (
                    "Review candidates",
                    f"{len(review):,}",
                    "not automatically selected",
                ),
                (
                    "Lipid classes",
                    f"{selected['lipid_class'].nunique() if not selected.empty else 0:,}",
                    "among primary identities",
                ),
                (
                    "Sample intensity columns",
                    f"{len(data['sample_names']):,}",
                    "union across available streams",
                ),
                (
                    "Differential unannotated features",
                    f"{len(differential):,}",
                    "after accepted annotation identities were excluded",
                ),
            )
        )
        + "<div class='notice'><b>Evidence boundary.</b> This report uses XCMS "
        "feature coordinates, sample intensities, annotation residuals and "
        "grid/anchor audits. Raw chromatograms, mobility, CCS and feature-linked "
        "MS/MS are not inferred because they are absent from the standard input "
        "files.</div>"
        + "<h3>Unique primary lipid identities by class</h3>"
        + _class_bars(data["class_summary"])
        + "<h3>Input and annotation counts by stream</h3>"
        + _table(streams)
        + "</div>"
    )


def _feature_section(key: str, kind: str):
    return lambda context: (
        "<div class='standard-report'>"
        + _feature_rows(context.data[key], kind=kind)
        + "</div>"
    )


def _untargeted_section(context: StandardReportContext) -> str:
    data = context.data
    untargeted = context.result.get("untargeted_analysis", {})
    return (
        "<div class='standard-report'><div class='notice'><b>Interpretation "
        "boundary.</b> "
        + str(
            untargeted.get(
                "interpretation_boundary",
                "Local cluster evidence is not a lipid identification.",
            )
        )
        + " Accepted primary and equivalent lipid features were excluded before "
        "testing.</div><h3>Recorded screening and evidence settings</h3>"
        + _table(data["untargeted_criteria"])
        + "<h3>Raw-p-screened unannotated features</h3>"
        + _table(data["untargeted_differences"])
        + "<h3>Human feature-review basket</h3>"
        + _table(data["untargeted_review"])
        + "<h3>Targeted reacquisition list</h3>"
        + _table(data["untargeted_reacquisition"])
        + "<h3>Comparison-group metadata</h3>"
        + _table(data["untargeted_groups"])
        + "<h3>Excel sample-metadata join audit</h3>"
        + _table(data["untargeted_metadata_audit"])
        + "</div>"
    )


def _maps_section(context: StandardReportContext) -> str:
    selected = context.data["selected"]
    return (
        "<div class='standard-report plot-grid'>"
        + "".join(
            _scatter_plot(group, stream)
            for stream, group in selected.groupby("stream", sort=False)
        )
        + "</div>"
    )


def _quality_section(context: StandardReportContext) -> str:
    return (
        "<div class='standard-report'><div class='notice'>These summaries "
        "describe annotation evidence and XCMS group metadata. They are not "
        "MZmine peak-shape or raw-trace quality measurements.</div>"
        + _table(context.data["metric_summary"])
        + "</div>"
    )


def _availability_section(context: StandardReportContext) -> str:
    content = (
        _table(context.data["availability"])
        .replace(">available<", "><span class='yes'>available</span><")
        .replace(">not available<", "><span class='no'>not available</span><")
        .replace(">conditional<", "><span class='conditional'>conditional</span><")
    )
    return (
        "<div class='standard-report'><div class='notice'>This tab prevents "
        "unavailable 4D evidence from being mistaken for a negative result.</div>"
        + content
        + "</div>"
    )


def _settings_section(context: StandardReportContext) -> str:
    data = context.data
    return (
        "<div class='standard-report'>"
        + _cards(
            (
                ("Annotation engine", data["engine"], "standard lipidomics"),
                ("Project", data["project"], "input location"),
            )
        )
        + _table(data["settings"])
        + "</div>"
    )


STANDARD_REPORT_SECTIONS = ReportSectionRegistry(
    (
        ReportSection("summary", "Summary", _summary_section),
        ReportSection(
            "primary",
            "Selected annotations",
            _feature_section("selected", "primary"),
            "One row per final primary lipid identity. The PRIMARY badge marks the "
            "feature selected when several features matched the same lipid.",
        ),
        ReportSection(
            "equivalent",
            "Equivalent features",
            _feature_section("equivalents", "equivalent"),
            "Additional accepted features for an identity remain explicit but do "
            "not inflate the unique-identity count.",
        ),
        ReportSection(
            "review",
            "Review candidates",
            _feature_section("review", "review"),
            "Mass/RT candidates that were not selected automatically. These are "
            "evidence for review, not final annotations.",
        ),
        ReportSection(
            "untargeted",
            "Untargeted differences",
            _untargeted_section,
            "Unannotated XCMS features passing the recorded raw-p, fold-change and "
            "detection gates. These are screening hits, not multiplicity-controlled "
            "statistical discoveries; no multiple-testing correction is applied.",
        ),
        ReportSection(
            "maps",
            "RT–m/z maps",
            _maps_section,
            "Only final primary identities are plotted; hover a point for its "
            "identity and feature ID.",
        ),
        ReportSection("quality", "Evidence distributions", _quality_section),
        ReportSection(
            "calibration",
            "RT, anchors & grid",
            lambda context: (
                "<div class='standard-report'>"
                + _calibration_tables(dict(context.result))
                + "</div>"
            ),
        ),
        ReportSection(
            "policy",
            "Policy & limits",
            lambda context: (
                "<div class='standard-report'>"
                + _policy_tables(dict(context.result))
                + "</div>"
            ),
        ),
        ReportSection("settings", "Inputs & settings", _settings_section),
        ReportSection(
            "availability",
            "Evidence availability",
            _availability_section,
        ),
    )
)


def render_standard_lipidomics_report(
    result: dict,
    *,
    settings: dict | None = None,
    project: str | Path | None = None,
) -> str:
    """Render one self-contained report for either standard annotation engine."""
    report_data = build_standard_report_data(result, settings=settings, project=project)
    context = StandardReportContext(result=result, data=report_data)
    return render_report(
        TestReport(
            title="LASER · standard lipidomics report",
            subtitle=(
                f"{report_data['engine']} annotation engine · conventional XCMS "
                "peak-group evidence"
            ),
            tabs=STANDARD_REPORT_SECTIONS.tabs(context),
            provenance={
                "workflow": "standard lipidomics",
                "engine": report_data["engine"],
                "project": report_data["project"],
            },
        )
    )


def build_standard_report_data(
    result: dict,
    *,
    settings: dict | None = None,
    project: str | Path | None = None,
) -> dict:
    """Return the authoritative data behind both native and HTML reports.

    Keeping this normalization in one place prevents the in-app project
    workspace and the optional HTML export from disagreeing about selected,
    equivalent, or review-only annotations.
    """
    sample_names = _sample_names(result)
    metadata = _feature_metadata(result)
    selected = _normalise_annotations(
        _frame(result.get("identified_lipids")),
        role="primary",
        sample_names=sample_names,
        feature_metadata=metadata,
    )
    equivalents = _normalise_annotations(
        _frame(result.get("equivalent_lipids")),
        role="equivalent",
        sample_names=sample_names,
        feature_metadata=metadata,
    )
    all_candidates = _normalise_annotations(
        _frame(result.get("all_candidates")),
        role="review",
        sample_names=sample_names,
        feature_metadata=metadata,
    )
    selected_keys = set(
        zip(
            selected.get("stream", ()),
            selected.get("feature_id", ()),
            selected.get("identity", ()),
            selected.get("adduct", ()),
        )
    )
    equivalent_keys = set(
        zip(
            equivalents.get("stream", ()),
            equivalents.get("feature_id", ()),
            equivalents.get("identity", ()),
            equivalents.get("adduct", ()),
        )
    )
    if all_candidates.empty:
        review = all_candidates
    else:
        keys = list(
            zip(
                all_candidates["stream"],
                all_candidates["feature_id"],
                all_candidates["identity"],
                all_candidates["adduct"],
            )
        )
        review = all_candidates[
            [key not in selected_keys | equivalent_keys for key in keys]
        ].copy()
        review["role"] = "review"
    untargeted = result.get("untargeted_analysis", {})
    settings_rows = pd.DataFrame(
        [
            {"setting": key, "value": value}
            for key, value in sorted((settings or {}).items())
            if not isinstance(value, (dict, list, tuple))
        ]
    )
    return {
        "sample_names": sample_names,
        "selected": selected,
        "equivalents": equivalents,
        "all_candidates": all_candidates,
        "review": review,
        "class_summary": _class_summary(selected, equivalents),
        "stream_summary": _stream_summary(result, selected, equivalents, review),
        "metric_summary": _metric_summary(selected),
        "availability": _availability(result),
        "settings": settings_rows,
        "engine": str(result.get("engine", "laser")).upper(),
        "project": str(project or "not recorded"),
        "input_features": sum(
            len(_frame(stream.get("features")))
            for stream in result.get("streams", {}).values()
        ),
        "untargeted_differences": _frame(untargeted.get("significant_features")),
        "untargeted_groups": _frame(untargeted.get("group_catalog")),
        "untargeted_metadata_audit": _frame(
            result.get("untargeted_sample_metadata_audit")
        ),
        "untargeted_review": _frame(result.get("untargeted_review_basket")),
        "untargeted_reacquisition": _frame(result.get("untargeted_reacquisition_list")),
        "untargeted_criteria": pd.DataFrame(
            [
                {"criterion": key, "value": value}
                for key, value in untargeted.get("criteria", {}).items()
            ]
        ),
    }


def write_standard_lipidomics_report(
    result: dict,
    output: str | Path,
    *,
    settings: dict | None = None,
    project: str | Path | None = None,
) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        destination,
        render_standard_lipidomics_report(result, settings=settings, project=project),
    )
    return destination


__all__ = [
    "STANDARD_REPORT_SECTIONS",
    "StandardReportContext",
    "build_standard_report_data",
    "render_standard_lipidomics_report",
    "write_standard_lipidomics_report",
]
