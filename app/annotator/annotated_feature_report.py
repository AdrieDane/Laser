"""Self-contained evidence report with one row per feature-identity match.

The trace galleries are intentionally lipid-centric contact sheets.  Method
optimization needs a different view: every feature that competed for an
annotation must remain visible beside the numerical evidence and the feature
that LASER preferred.  This module builds that review surface without
re-running or changing annotation decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import html
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from output_io import atomic_write_text
from report_lazy import (
    LAZY_REPORT_SCRIPT,
    LAZY_REPORT_STYLE,
    lazy_feature_rows_markup,
)
from scientific_defaults import SCIENTIFIC_DEFAULTS

from .exclusions import annotation_exclusion_mask, retained_annotations
from .feature_traces import (
    SummedFeatureChromatogram,
    load_feature_traces,
)
from .msms import (
    DEFAULT_FA_MIN_RELATIVE_INTENSITY_PERCENT,
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    MSMS_RULES,
    msms_preference,
    msms_rule_result,
)
from .species_viewer import class_evidence_text
from .trace_gallery import SampleTrace, _overlay, sample_colours


def _resolved_report_class(frame: pd.DataFrame) -> pd.Series:
    """Prefer the RP-resolved grid class while preserving legacy reports."""
    return (
        frame.get(
            "grid_class",
            frame.get("class", pd.Series("", index=frame.index, dtype=object)),
        )
        .fillna("")
        .astype(str)
    )


def assemble_feature_evidence_candidates(
    annotations: pd.DataFrame,
    *,
    msms_failures: pd.DataFrame | None = None,
    grid_failures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add excluded rivals for identities that have a surviving annotation.

    Entirely rejected lipid identities are deliberately not promoted into an
    annotation report.  A rule/grid-excluded feature is included only when the
    same lipid identity and adduct has at least one retained match, which makes
    the report complete without changing its meaning or class totals.
    """
    retained = annotations.copy()
    retained["rule_excluded"] = False
    retained["grid_excluded"] = False
    if retained.empty:
        return retained
    identity_keys = set(
        zip(retained["name"].astype(str), retained["adduct"].astype(str))
    )
    frames = [retained]
    for failures, flag in (
        (msms_failures, "rule_excluded"),
        (grid_failures, "grid_excluded"),
    ):
        if failures is None or failures.empty:
            continue
        extra = failures.copy()
        keep = [
            (str(name), str(adduct)) in identity_keys
            for name, adduct in zip(extra["name"], extra["adduct"])
        ]
        extra = extra.loc[keep].copy()
        if extra.empty:
            continue
        extra["rule_excluded"] = flag == "rule_excluded"
        extra["grid_excluded"] = flag == "grid_excluded"
        extra["automatic_candidate"] = False
        extra["status"] = "reject"
        extra["primary_feature_override"] = False
        frames.append(extra)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["_match_key"] = (
        combined["feat_feature_id"].astype(str)
        + "|"
        + combined["name"].astype(str)
        + "|"
        + combined["adduct"].astype(str)
    )
    # Retained rows were concatenated first and must win over a diagnostic copy
    # when filtering was advisory rather than active.
    return (
        combined.drop_duplicates("_match_key", keep="first")
        .drop(columns="_match_key")
        .reset_index(drop=True)
    )


def rank_identity_feature_matches(annotations: pd.DataFrame) -> pd.DataFrame:
    """Mark one reproducible primary feature per lipid identity and adduct.

    This is the same evidence order used by the interactive viewer: an explicit
    manual primary wins, then a confirming MS/MS rule, then proximity to the RT
    grid, ppm error and finally the combined annotation score.  No row is
    removed.
    """
    if annotations.empty:
        result = annotations.copy()
        result["identity_match_rank"] = pd.Series(dtype="Int64")
        result["primary_for_identity"] = pd.Series(dtype=bool)
        return result

    result = annotations.copy()
    result["_original_order"] = np.arange(len(result))
    excluded = annotation_exclusion_mask(result)
    result["_excluded_order"] = excluded.astype(int)
    manual = (
        result.get("primary_feature_override", pd.Series(False, index=result.index))
        .fillna(False)
        .astype(bool)
    )
    result["_manual_order"] = (~manual).astype(int)
    automatic = (
        result.get("automatic_candidate", pd.Series(True, index=result.index))
        .fillna(False)
        .astype(bool)
    )
    result["_automatic_order"] = (~automatic).astype(int)
    result["_msms_order"] = msms_preference(result)
    result["_rt_order"] = (
        pd.to_numeric(
            result.get("rt_error", pd.Series(np.nan, index=result.index)),
            errors="coerce",
        )
        .abs()
        .fillna(np.inf)
    )
    result["_ppm_order"] = (
        pd.to_numeric(
            result.get("ppm_error", pd.Series(np.nan, index=result.index)),
            errors="coerce",
        )
        .abs()
        .fillna(np.inf)
    )
    result["_score_order"] = pd.to_numeric(
        result.get("score", pd.Series(np.nan, index=result.index)), errors="coerce"
    ).fillna(np.inf)
    result["_feature_order"] = result.get(
        "feat_feature_id", pd.Series("", index=result.index)
    ).astype(str)

    group_columns = ["name", "adduct"]
    result = result.sort_values(
        group_columns
        + [
            "_excluded_order",
            "_manual_order",
            "_automatic_order",
            "_msms_order",
            "_rt_order",
            "_ppm_order",
            "_score_order",
            "_feature_order",
        ],
        kind="stable",
    )
    result["identity_match_rank"] = (
        result.groupby(group_columns, sort=False).cumcount() + 1
    ).astype("Int64")
    result["primary_for_identity"] = result["identity_match_rank"].eq(1)
    return (
        result.sort_values("_original_order", kind="stable")
        .drop(
            columns=[
                "_original_order",
                "_excluded_order",
                "_manual_order",
                "_automatic_order",
                "_msms_order",
                "_rt_order",
                "_ppm_order",
                "_score_order",
                "_feature_order",
            ]
        )
        .reset_index(drop=True)
    )


def _number(value, digits: int = 2, suffix: str = "") -> str:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed) or not np.isfinite(float(parsed)):
        return "n/a"
    return f"{float(parsed):+.{digits}f}{suffix}"


def _flag(value) -> bool:
    """Interpret nullable report flags without treating NaN as true."""
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _unsigned_number(value, digits: int = 2, suffix: str = "") -> str:
    """Format a non-directional measurement without a misleading plus sign."""
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed) or not np.isfinite(float(parsed)):
        return "n/a"
    return f"{float(parsed):.{digits}f}{suffix}"


def _significant_number(value, digits: int = 4, suffix: str = "") -> str:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed) or not np.isfinite(float(parsed)):
        return "n/a"
    return f"{float(parsed):.{digits}g}{suffix}"


def _peak_quality_summary(row: pd.Series) -> str:
    """Render the overall processing-QC verdict beside feature identity."""
    raw_quality = row.get("quantification_quality", "")
    quality = "" if pd.isna(raw_quality) else str(raw_quality).strip().upper()
    if quality not in {"PASS", "REVIEW", "FAIL"}:
        quality = "NOT ASSESSED"
    quality_class = quality.casefold().replace(" ", "-")
    raw_flags = row.get("processing_qc_flags", "")
    flags = "" if pd.isna(raw_flags) else str(raw_flags).strip()
    detail = html.escape(flags.replace(";", " · ")) if flags else "no QC flags"
    return (
        "<div class='identity-quality'><span>Peak quality</span>"
        f"<span class='quality-badge {quality_class}'>{html.escape(quality)}</span>"
        f"<small>{detail}</small></div>"
    )


def _peak_quality_strip(row: pd.Series) -> str:
    """Render two complete rows of quantitative feature-quality evidence."""
    fwhm_min = pd.to_numeric(row.get("median_fwhm_min"), errors="coerce")
    fwhm_seconds = float(fwhm_min) * 60 if pd.notna(fwhm_min) else np.nan
    duration_min = pd.to_numeric(row.get("median_peak_duration_min"), errors="coerce")
    duration_seconds = float(duration_min) * 60 if pd.notna(duration_min) else np.nan
    rt_error_min = pd.to_numeric(row.get("max_consensus_rt_error_min"), errors="coerce")
    rt_error_seconds = float(rt_error_min) * 60 if pd.notna(rt_error_min) else np.nan
    ms2_distance_min = pd.to_numeric(
        row.get("median_ms2_apex_distance_min"), errors="coerce"
    )
    ms2_distance_seconds = (
        float(ms2_distance_min) * 60 if pd.notna(ms2_distance_min) else np.nan
    )
    cards = [
        (
            "Median FWHM",
            _unsigned_number(fwhm_seconds, 1, " s"),
            "chromatographic width",
        ),
        (
            "FWHM CV",
            _unsigned_number(row.get("fwhm_cv_percent"), 1, "%"),
            "width consistency",
        ),
        (
            "Tailing factor",
            _unsigned_number(row.get("median_tailing_factor"), 2),
            "1.00 is symmetric",
        ),
        (
            "Double-peak fraction",
            _unsigned_number(
                pd.to_numeric(row.get("rt_double_peak_fraction"), errors="coerce")
                * 100,
                1,
                "%",
            ),
            "sample/QC traces",
        ),
        (
            "Detection rate",
            _unsigned_number(
                pd.to_numeric(row.get("biological_detected_fraction"), errors="coerce")
                * 100,
                1,
                "%",
            ),
            "biological samples",
        ),
        (
            "Sample/blank",
            _unsigned_number(row.get("sample_blank_ratio_height"), 1, "×"),
            "median peak height",
        ),
        (
            "Boundary/apex ratio",
            _unsigned_number(row.get("median_boundary_apex_ratio"), 3),
            "median boundary ÷ apex",
        ),
        (
            "Height–area correlation",
            _unsigned_number(row.get("area_height_spearman"), 2),
            "Spearman correlation across biological samples",
        ),
        (
            "Peak duration",
            _unsigned_number(duration_seconds, 1, " s"),
            "median integration-window width",
        ),
        (
            "Duration/FWHM",
            _unsigned_number(row.get("median_duration_fwhm_ratio"), 2),
            "integration width ÷ chromatographic FWHM",
        ),
        (
            "Maximum RT deviation",
            _unsigned_number(rt_error_seconds, 1, " s"),
            "largest sample/QC deviation from consensus RT",
        ),
        (
            "Mobility FWHM",
            _significant_number(row.get("median_mobility_fwhm"), 4),
            "median ion-mobility peak width",
        ),
        (
            "Mobility FWHM CV",
            _unsigned_number(row.get("mobility_fwhm_cv_percent"), 1, "%"),
            "mobility-width consistency",
        ),
        (
            "Mobility double-peak fraction",
            _unsigned_number(
                pd.to_numeric(
                    row.get("double_mobility_peak_sample_fraction"),
                    errors="coerce",
                )
                * 100,
                1,
                "%",
            ),
            "sample/QC mobilograms",
        ),
        (
            "MS/MS apex distance",
            _unsigned_number(ms2_distance_seconds, 1, " s"),
            "median acquisition distance from chromatographic apex",
        ),
        (
            "Asymmetry factor",
            _unsigned_number(row.get("median_asymmetry_factor"), 2),
            "1.00 is symmetric",
        ),
    ]
    return (
        "<div class='peak-quality-strip' aria-label='Peak quality metrics'>"
        + "".join(
            "<div class='peak-quality-metric'>"
            f"<span>{html.escape(label)}</span><b>{value}</b>"
            f"<small>{detail}</small></div>"
            for label, value, detail in cards
        )
        + "</div>"
    )


def _internal_standard_mask(frame: pd.DataFrame) -> pd.Series:
    """Recognise IS rows from current or legacy annotation tables."""
    names = frame.get("name", pd.Series("", index=frame.index)).astype(str)
    name_marker = names.str.contains("(IS)", regex=False, na=False)
    if "is_is" not in frame:
        return name_marker
    explicit = (
        frame["is_is"]
        .map(
            lambda value: (
                value
                if isinstance(value, (bool, np.bool_))
                else str(value).strip().casefold() in {"1", "true", "yes"}
            )
        )
        .fillna(False)
        .astype(bool)
    )
    return explicit | name_marker


def _json_records(value) -> list[dict]:
    """Return a safe list of JSON objects from an annotation-table cell."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return (
        [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, list)
        else []
    )


def _fatty_acyl_evidence_summary(row: pd.Series) -> str:
    """Render composition-compatible fatty-acyl fragment assignments.

    Molecular-species names deliberately use an underscore rather than sn-1/
    sn-2 notation: ordinary collision spectra support chain composition, not
    regiospecific placement.
    """
    status = row.get("fatty_acyl_status", "")
    if pd.isna(status) or str(status).casefold() != "supported":
        return ""
    detections = _json_records(row.get("fatty_acyl_evidence_json"))
    raw_groups = row.get("fatty_acyl_detected", "")
    groups = "" if pd.isna(raw_groups) else str(raw_groups).strip()
    raw_candidates = row.get("molecular_species_candidates", "")
    candidates = "" if pd.isna(raw_candidates) else str(raw_candidates).strip()
    raw_most_likely = row.get("most_likely_molecular_species", "")
    most_likely = "" if pd.isna(raw_most_likely) else str(raw_most_likely).strip()
    raw_likely_ambiguous = row.get("most_likely_molecular_species_ambiguous", False)
    likely_ambiguous = (
        False
        if pd.isna(raw_likely_ambiguous)
        else bool(raw_likely_ambiguous)
        if isinstance(raw_likely_ambiguous, (bool, np.bool_))
        else str(raw_likely_ambiguous).strip().casefold() in {"1", "true", "yes"}
    )
    raw_ambiguous = row.get("molecular_species_ambiguous", False)
    ambiguous = (
        False
        if pd.isna(raw_ambiguous)
        else bool(raw_ambiguous)
        if isinstance(raw_ambiguous, (bool, np.bool_))
        else str(raw_ambiguous).strip().casefold() in {"1", "true", "yes"}
    )
    details = []
    for item in detections:
        evidence_type = str(item.get("evidence_type", "")).replace("_", " ")
        details.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('fatty_acyl', '')))}</td>"
            f"<td>{html.escape(evidence_type)}</td>"
            f"<td>{_unsigned_number(item.get('observed_fragment_mz'), 4)}</td>"
            f"<td>{_unsigned_number(item.get('expected_mz_or_loss'), 4)}</td>"
            f"<td>{_number(item.get('error_da'), 4, ' Da')}</td>"
            f"<td>{_unsigned_number(item.get('intensity'), 4)}</td>"
            f"<td>{_unsigned_number(item.get('relative_intensity_percent'), 1, '%')}</td>"
            "</tr>"
        )
    likely = (
        f"<small><b>Most likely from FA-fragment intensity:</b> "
        f"{html.escape(most_likely)}</small>"
        if most_likely
        else "<small><b>Most likely from FA-fragment intensity:</b> "
        + (
            "unresolved because the leading candidates are tied</small>"
            if likely_ambiguous
            else "not available</small>"
        )
    )
    species = (
        f"<small><b>Molecular-species candidate{'s' if ambiguous else ''}:</b> "
        f"{html.escape(candidates)}"
        f"{' · ambiguous; no sn-position claim' if ambiguous else ' · no sn-position claim'}</small>"
        if candidates
        else "<small>No complete molecular species can be inferred from the detected chain evidence.</small>"
    )
    table = (
        "<table><thead><tr><th>FA</th><th>evidence</th><th>fragment m/z</th>"
        "<th>expected ion/loss</th><th>error</th><th>intensity</th>"
        "<th>relative intensity</th>"
        f"</tr></thead><tbody>{''.join(details)}</tbody></table>"
        if details
        else ""
    )
    return (
        "<div class='fatty-acyl-evidence'><b>FA FRAGMENT EVIDENCE: SUPPORTED</b>"
        f"<small><b>Detected:</b> {html.escape(groups)}</small>"
        f"{species}{likely}{table}</div>"
    )


def _spectrum_svg(
    spectrum: dict | None,
    *,
    fatty_acyl_evidence=None,
    width: int = 300,
    height: int = 150,
) -> str:
    peaks = (spectrum or {}).get("peaks") or []
    clean = []
    for peak in peaks:
        if not isinstance(peak, (list, tuple)) or len(peak) < 2:
            continue
        mz = pd.to_numeric(peak[0], errors="coerce")
        intensity = pd.to_numeric(peak[1], errors="coerce")
        if pd.notna(mz) and pd.notna(intensity) and float(intensity) >= 0:
            clean.append((float(mz), float(intensity)))
    if not clean:
        return (
            f'<svg class="evidence-plot empty" viewBox="0 0 {width} {height}">'
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle">'
            "no linked MS/MS spectrum</text></svg>"
        )

    left, right, top, bottom = 40, 7, 8, 20
    plot_width, plot_height = width - left - right, height - top - bottom
    low = min(0.0, min(mz for mz, _ in clean))
    high = max(mz for mz, _ in clean)
    precursor = pd.to_numeric((spectrum or {}).get("precursor_mz"), errors="coerce")
    if pd.notna(precursor):
        high = max(high, float(precursor))
    span = max(high - low, 1e-9)
    maximum = max(intensity for _, intensity in clean) or 1.0

    def to_x(value):
        return left + (value - low) / span * plot_width

    def to_y(value):
        return top + plot_height - value / maximum * plot_height

    parts = [
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
        f'x2="{left + plot_width}" y2="{top + plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top + plot_height}"/>',
    ]
    for fraction in (0.0, 0.5, 1.0):
        value = low + fraction * span
        x = to_x(value)
        anchor = "start" if fraction == 0 else "end" if fraction == 1 else "middle"
        parts.append(
            f'<text class="tick-label" x="{x:.1f}" y="{height - 5}" '
            f'text-anchor="{anchor}">{value:.0f}</text>'
        )
    for fraction in (0.0, 0.5, 1.0):
        value = fraction * 100
        y = top + plot_height - fraction * plot_height
        parts.append(
            f'<text class="tick-label" x="{left - 5}" y="{y + 3:.1f}" '
            f'text-anchor="end">{value:.0f}</text>'
        )
    supported_fragments = [
        float(item["observed_fragment_mz"])
        for item in _json_records(fatty_acyl_evidence)
        if pd.notna(pd.to_numeric(item.get("observed_fragment_mz"), errors="coerce"))
    ]
    for mz, intensity in clean:
        is_fatty_acyl = any(abs(mz - target) <= 0.001 for target in supported_fragments)
        parts.append(
            f'<line class="fragment{" fatty-acyl-fragment" if is_fatty_acyl else ""}" '
            f'x1="{to_x(mz):.1f}" y1="{to_y(0):.1f}" '
            f'x2="{to_x(mz):.1f}" y2="{to_y(intensity):.1f}">'
            f"<title>{'FA-supporting fragment · ' if is_fatty_acyl else ''}"
            f"m/z {mz:.4f}</title></line>"
        )
    if pd.notna(precursor) and low <= float(precursor) <= high:
        x = to_x(float(precursor))
        parts.append(
            f'<line class="precursor" x1="{x:.1f}" y1="{top}" '
            f'x2="{x:.1f}" y2="{top + plot_height}"/>'
        )
    parts.append(
        f'<text class="axis-title" x="{left + plot_width / 2:.0f}" y="{top + 7}" '
        'text-anchor="middle">m/z · relative intensity (%)</text>'
    )
    return (
        f'<svg class="evidence-plot" viewBox="0 0 {width} {height}">'
        + "".join(parts)
        + "</svg>"
    )


def _class_bar_chart(
    frame: pd.DataFrame,
    *,
    aria_label: str = "Unique lipid annotations per class",
    apply_exclusions: bool = True,
) -> str:
    population = retained_annotations(frame) if apply_exclusions else frame.copy()
    has_msms = (
        population.get("msms_has_spectrum", pd.Series(False, index=population.index))
        .fillna(False)
        .astype(bool)
    )
    identities = population.assign(
        _has_msms=has_msms,
        _report_class=_resolved_report_class(population),
    )
    if "name" in identities:
        # A lipid identity is counted once per class. It is marked as carrying
        # MS/MS when at least one contributing feature has a linked spectrum;
        # repeated features and adduct measurements never inflate either bar.
        identities = identities.groupby(
            ["_report_class", "name"],
            dropna=False,
            as_index=False,
        )["_has_msms"].max()
    counts = (
        identities.groupby("_report_class", dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    msms_counts = (
        identities[identities["_has_msms"]]
        .groupby("_report_class", dropna=False)
        .size()
        if not identities.empty
        else pd.Series(dtype=int)
    )
    if counts.empty:
        return "<p class='empty-message'>No unique lipid annotations.</p>"
    width = max(720, 70 * len(counts))
    # Keep the legend in its own band. The previous 24 px top margin placed the
    # total labels directly under the legend for the tallest classes.
    height, left, right, top, bottom = 275, 48, 12, 52, 55
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = max(int(counts.max()), 1)
    band = plot_width / len(counts)
    bar_width = min(42, band * 0.68)
    parts = [
        f'<svg class="class-chart" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{html.escape(aria_label)}">',
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
        f'x2="{left + plot_width}" y2="{top + plot_height}"/>',
        f'<g class="class-chart-legend" transform="translate({left},8)">'
        '<rect class="class-bar-key" x="0" y="0" width="12" height="8"/>'
        '<text x="17" y="8">unique annotations</text>'
        '<rect class="class-bar-msms" x="145" y="0" width="12" height="8"/>'
        '<text x="162" y="8">with MS/MS spectrum</text></g>',
    ]
    for index, (lipid_class, count) in enumerate(counts.items()):
        x = left + band * index + (band - bar_width) / 2
        bar_height = float(count) / maximum * plot_height
        y = top + plot_height - bar_height
        msms_count = int(msms_counts.get(lipid_class, 0))
        msms_height = float(msms_count) / maximum * plot_height
        msms_y = top + plot_height - msms_height
        label = html.escape(str(lipid_class) if str(lipid_class) else "unclassified")
        parts.extend(
            [
                f'<rect class="class-bar" x="{x:.1f}" y="{y:.1f}" '
                f'width="{bar_width:.1f}" height="{bar_height:.1f}">'
                f"<title>{label}: {int(count)} unique lipid IDs; "
                f"{msms_count} with MS/MS spectrum</title></rect>",
                f'<rect class="class-bar-msms" x="{x:.1f}" y="{msms_y:.1f}" '
                f'width="{bar_width:.1f}" height="{msms_height:.1f}">'
                f"<title>{label}: {msms_count} of {int(count)} unique lipid IDs "
                f"have an MS/MS spectrum</title></rect>",
                (
                    f'<text class="bar-msms-count" x="{x + bar_width / 2:.1f}" '
                    f'y="{msms_y + msms_height / 2 + 4:.1f}" '
                    f'text-anchor="middle">{msms_count}</text>'
                    if msms_count
                    else ""
                ),
                f'<text class="bar-count" x="{x + bar_width / 2:.1f}" y="{y - 5:.1f}" '
                f'text-anchor="middle">{int(count)}</text>',
                f'<text class="bar-label" x="{x + bar_width / 2:.1f}" '
                f'y="{top + plot_height + 16}" text-anchor="end" '
                f'transform="rotate(-45 {x + bar_width / 2:.1f} {top + plot_height + 16})">'
                f"{label}</text>",
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def _excluded_class_bar_chart(
    frame: pd.DataFrame,
    *,
    label: str = "Excluded lipid IDs per lipid class",
) -> str:
    """Count unique rejected lipid IDs per class, never rejected peak rows."""
    if frame.empty or not {"class", "name"}.issubset(frame.columns):
        return "<p class='empty-message'>No excluded lipid IDs.</p>"
    unique_identities = (
        frame.sort_values(
            "msms_has_spectrum" if "msms_has_spectrum" in frame else "name",
            ascending=False,
            kind="stable",
        )
        .drop_duplicates(["class", "name"], keep="first")
        .copy()
    )
    unique_identities["primary_for_identity"] = True
    return _class_bar_chart(
        unique_identities,
        aria_label=label,
    )


def _msms_rules_table(
    polarity: str,
    fragment_tolerance_da: float,
    fatty_acyl_min_relative_intensity_percent: float,
) -> str:
    """State every configured class-fragment rule active for this report."""
    selected = [
        rule
        for rule in MSMS_RULES
        if polarity not in {"pos", "neg"}
        or str(rule.get("polarity", "")).casefold() == polarity
    ]
    if not selected:
        return "<p class='empty-message'>No class-fragment rules are configured.</p>"
    rows = []
    for rule in selected:
        evidence = rule.get("evidence", {})
        evidence_type = str(evidence.get("type", "")).replace("_", " ")
        target = evidence.get("target_mz")
        requirement = f"{evidence_type}: {evidence.get('name', '')}".strip(": ")
        if target is not None:
            requirement += (
                f" at m/z {float(target):.4f} ± {fragment_tolerance_da:.3f} Da"
            )
        elif evidence_type == "fatty acyl":
            requirement += (
                " · ≥1 composition-compatible loss at ≥"
                f"{fatty_acyl_min_relative_intensity_percent:g}% relative intensity"
            )
        adducts = (
            ", ".join(map(str, rule.get("adducts", ()))) or "all configured adducts"
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(rule.get('id', '')))}</td>"
            f"<td>{html.escape(', '.join(map(str, rule.get('lipid_classes', ()))))}</td>"
            f"<td>{html.escape(adducts)}</td>"
            f"<td>{html.escape(requirement)}</td>"
            f"<td>{html.escape(str(rule.get('description', '')))}</td>"
            "</tr>"
        )
    return (
        "<div class='rule-table-wrap'><table class='rule-table'><thead><tr>"
        "<th>Rule ID</th><th>Lipid classes</th><th>Adducts</th>"
        "<th>Required evidence</th><th>Definition</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


_ROLE_LABELS = {
    "biological": "samples",
    "qc": "QC",
    "blank": "blanks",
    "calibration": "IS",
}


def _tic_panels(
    chromatograms: list[SummedFeatureChromatogram],
    *,
    sample_roles: dict[str, str],
    colours: dict[str, str],
) -> str:
    traces = [
        SampleTrace(
            sample_name=item.sample_name,
            is_representative=False,
            rt_values=item.rt_values,
            rt_intensities=item.intensities,
            mobility_values=np.empty(0),
            mobility_intensities=np.empty(0),
        )
        for item in chromatograms
    ]
    panels = [("TIC · all measurements", traces)]
    for role in ("biological", "qc", "blank", "calibration"):
        panels.append(
            (
                f"TIC · all {_ROLE_LABELS[role]}",
                [
                    trace
                    for trace in traces
                    if sample_roles.get(trace.sample_name) == role
                ],
            )
        )
    return "".join(
        "<div class='qc-panel'><h3>"
        + html.escape(label)
        + "</h3>"
        + _overlay(values, mobility=False, colours=colours, width=500, height=210)
        + f"<p class='subtle'>{len(values)} injection(s)</p></div>"
        for label, values in panels
    )


def _trend_svg(
    measurements: pd.DataFrame,
    value_column: str,
    y_label: str,
    *,
    role_colours: dict[str, str],
    width: int = 310,
    height: int = 180,
) -> str:
    values = measurements.copy()
    values["_x"] = pd.to_numeric(values.get("injection_order"), errors="coerce")
    values["_y"] = pd.to_numeric(values.get(value_column), errors="coerce")
    values = values[np.isfinite(values["_x"]) & np.isfinite(values["_y"])]
    if values.empty:
        return (
            f'<svg class="trend empty" viewBox="0 0 {width} {height}">'
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle">'
            f"no {html.escape(y_label)} data</text></svg>"
        )
    values = values.sort_values("_x", kind="stable")
    left, right, top, bottom = 47, 9, 18, 30
    plot_width, plot_height = width - left - right, height - top - bottom
    x_low, x_high = float(values["_x"].min()), float(values["_x"].max())
    y_low, y_high = float(values["_y"].min()), float(values["_y"].max())
    x_span = max(x_high - x_low, 1.0)
    y_pad = max((y_high - y_low) * 0.08, abs(y_high) * 0.01, 1e-9)
    y_low, y_high = y_low - y_pad, y_high + y_pad
    y_span = max(y_high - y_low, 1e-12)

    def to_x(value):
        return left + (value - x_low) / x_span * plot_width

    def to_y(value):
        return top + plot_height - (value - y_low) / y_span * plot_height

    points = " ".join(
        f"{to_x(float(x_value)):.1f},{to_y(float(y_value)):.1f}"
        for x_value, y_value in values[["_x", "_y"]].itertuples(
            index=False, name=None
        )
    )
    parts = [
        f'<svg class="trend" viewBox="0 0 {width} {height}">',
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
        f'x2="{left + plot_width}" y2="{top + plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top + plot_height}"/>',
        f'<polyline class="trend-line" points="{points}"/>',
    ]
    trend_rows = values.assign(
        sample_role=values.get("sample_role", "unknown"),
        state=values.get("state", "UNKNOWN"),
        sample=values.get("sample", ""),
    )[["_x", "_y", "sample_role", "state", "sample"]]
    for x_value, y_value, role_value, state, sample in trend_rows.itertuples(
        index=False, name=None
    ):
        role = str(role_value)
        detected = str(state).upper() == "DETECTED"
        parts.append(
            f'<circle cx="{to_x(float(x_value)):.1f}" cy="{to_y(float(y_value)):.1f}" '
            f'r="3.4" fill="{role_colours.get(role, "#777") if detected else "#fff"}" '
            f'stroke="{role_colours.get(role, "#777")}" stroke-width="1.2">'
            f"<title>{html.escape(str(sample))} · order "
            f"{float(x_value):g} · {html.escape(y_label)} {float(y_value):.5g}</title>"
            "</circle>"
        )
    for fraction in (0.0, 0.5, 1.0):
        x_value = x_low + fraction * x_span
        y_value = y_low + fraction * y_span
        parts.extend(
            [
                f'<text class="tick-label" x="{to_x(x_value):.1f}" y="{height - 8}" '
                f'text-anchor="middle">{x_value:g}</text>',
                f'<text class="tick-label" x="{left - 5}" y="{to_y(y_value) + 3:.1f}" '
                f'text-anchor="end">{y_value:.4g}</text>',
            ]
        )
    parts.extend(
        [
            f'<text class="axis-title" x="{left + plot_width / 2:.1f}" y="{height - 1}" '
            'text-anchor="middle">injection order</text>',
            f'<text class="axis-title" x="4" y="{top + plot_height / 2:.1f}" '
            f'transform="rotate(-90 4 {top + plot_height / 2:.1f})" '
            f'text-anchor="middle">{html.escape(y_label)}</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _is_trend_panels(trends: list[dict]) -> str:
    if not trends:
        return "<p class='empty-message'>No annotated internal-standard trends.</p>"
    role_colours = {
        "biological": "#2a78d6",
        "qc": "#1a9b61",
        "blank": "#7b7b78",
        "calibration": "#e49b00",
        "unknown": "#8b65c2",
    }
    parts = []
    for trend in trends:
        title = f"{trend['name']} {trend['adduct']} · feature #{trend['feature_id']}"
        parts.append(
            "<div class='is-trend'><h3>" + html.escape(title) + "</h3>"
            "<div class='trend-grid'>"
            + _trend_svg(
                trend["measurements"],
                "rt",
                "retention time (min)",
                role_colours=role_colours,
            )
            + _trend_svg(
                trend["measurements"],
                "height",
                "peak height",
                role_colours=role_colours,
            )
            + _trend_svg(
                trend["measurements"], "area", "peak area", role_colours=role_colours
            )
            + "</div></div>"
        )
    return "".join(parts)


def render_tic_report(
    summed_chromatograms: list[SummedFeatureChromatogram],
    *,
    sample_roles: dict[str, str] | None = None,
    sample_groups: dict[str, str] | None = None,
    title: str = "Feature-derived TIC",
    subtitle: str = "",
) -> str:
    """Render the five run-level TIC overlays as a standalone report."""
    roles = sample_roles or {}
    groups = sample_groups or {}
    sample_names = sorted(
        {trace.sample_name for trace in summed_chromatograms},
        key=lambda value: (groups.get(value, ""), value),
    )
    colours = sample_colours(sample_names, groups)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='meta'>{html.escape(subtitle)}</p>" if subtitle else "",
        "<p class='explanation'>These TIC panels are feature-derived: each "
        "line is the sum of all chromatograms exported by MZmine, not the raw "
        "scan-level TIC.</p>",
    ]
    if colours:
        parts.append("<div class='samples'>")
        for sample, colour in colours.items():
            label = f"{groups.get(sample)} · {sample}" if groups.get(sample) else sample
            parts.append(
                f"<span class='sample-key'><span class='swatch' "
                f"style='background:{colour}'></span>{html.escape(label)}</span>"
            )
        parts.append("</div>")
    parts.extend(
        [
            "<section class='run-qc'><div class='qc-grid'>",
            _tic_panels(summed_chromatograms, sample_roles=roles, colours=colours),
            "</div></section></body>",
        ]
    )
    return "".join(parts)


def write_tic_report(
    summed_chromatograms: list[SummedFeatureChromatogram],
    output: str | Path,
    **kwargs,
) -> Path:
    """Write the standalone feature-derived TIC report."""
    destination = Path(output)
    return atomic_write_text(
        destination, render_tic_report(summed_chromatograms, **kwargs)
    )


def render_injection_order_report(
    is_trends: list[dict],
    *,
    title: str = "Internal-standard injection-order trends",
    subtitle: str = "",
) -> str:
    """Render RT, peak-height and peak-area trends in a standalone report."""
    return "".join(
        [
            "<!doctype html><meta charset='utf-8'>",
            f"<title>{html.escape(title)}</title><style>{_STYLE}</style><body>",
            f"<h1>{html.escape(title)}</h1>",
            f"<p class='meta'>{html.escape(subtitle)}</p>" if subtitle else "",
            "<p class='explanation'>One panel per annotated internal standard. "
            "Filled points are DETECTED; hollow points are estimated or otherwise "
            "not explicitly detected.</p>",
            "<section class='run-qc'>",
            _is_trend_panels(is_trends),
            "</section></body>",
        ]
    )


def write_injection_order_report(
    is_trends: list[dict],
    output: str | Path,
    **kwargs,
) -> Path:
    """Write the standalone internal-standard injection-order report."""
    destination = Path(output)
    return atomic_write_text(
        destination, render_injection_order_report(is_trends, **kwargs)
    )


_RSD_COLOURS = {
    "<15%": "#15995b",
    "15–20%": "#8bc34a",
    "20–30%": "#f2b134",
    ">30%": "#dc554f",
    "not evaluable": "#b8b8b3",
}


def _rsd_category(value) -> str:
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value) or not np.isfinite(float(value)):
        return "not evaluable"
    if float(value) < 15:
        return "<15%"
    if float(value) <= 20:
        return "15–20%"
    if float(value) <= 30:
        return "20–30%"
    return ">30%"


def _pie_chart(counts: pd.Series, title: str, size: int = 190) -> str:
    total = int(counts.sum())
    if total <= 0:
        return "<div class='pie-card'><h3>" + html.escape(title) + "</h3>n/a</div>"
    cx = cy = size / 2
    radius = size * 0.36
    angle = -math.pi / 2
    paths = []
    legend = []
    for category in _RSD_COLOURS:
        count = int(counts.get(category, 0))
        if not count:
            continue
        sweep = 2 * math.pi * count / total
        end = angle + sweep
        x1, y1 = cx + radius * math.cos(angle), cy + radius * math.sin(angle)
        x2, y2 = cx + radius * math.cos(end), cy + radius * math.sin(end)
        if count == total:
            paths.append(
                f'<circle cx="{cx}" cy="{cy}" r="{radius}" '
                f'fill="{_RSD_COLOURS[category]}"><title>{category}: {count}</title></circle>'
            )
        else:
            large = 1 if sweep > math.pi else 0
            paths.append(
                f'<path d="M {cx} {cy} L {x1:.2f} {y1:.2f} A {radius} {radius} '
                f'0 {large} 1 {x2:.2f} {y2:.2f} Z" fill="{_RSD_COLOURS[category]}">'
                f"<title>{category}: {count} ({count / total * 100:.1f}%)</title></path>"
            )
        legend.append(
            f"<span><i style='background:{_RSD_COLOURS[category]}'></i>"
            f"{category}: {count}</span>"
        )
        angle = end
    return (
        "<div class='pie-card'><h3>" + html.escape(title) + "</h3>"
        f'<svg class="pie" viewBox="0 0 {size} {size}">{"".join(paths)}'
        f'<text x="{cx}" y="{cy + 4}" text-anchor="middle">n={total}</text></svg>'
        f"<div class='pie-legend'>{''.join(legend)}</div></div>"
    )


def _rsd_summary(frame: pd.DataFrame) -> str:
    if "qc_rsd_height_percent" not in frame:
        return "<p class='empty-message'>Corrected QC-RSD was not calculated.</p>"
    eligible = frame.copy()
    eligible = eligible[~_internal_standard_mask(eligible)].copy()
    excluded = annotation_exclusion_mask(eligible)
    automatic = (
        eligible.get("automatic_candidate", pd.Series(True, index=eligible.index))
        .fillna(False)
        .astype(bool)
    )
    accepted = (
        eligible.get("status", pd.Series("", index=eligible.index))
        .astype(str)
        .eq("accept")
    )
    eligible = eligible[(~excluded) & (automatic | accepted)].drop_duplicates(
        ["feat_feature_id", "name", "adduct"], keep="first"
    )
    if eligible.empty:
        return (
            "<p class='empty-message'>No endogenous annotated features for QC-RSD.</p>"
        )
    eligible["_rsd_category"] = eligible["qc_rsd_height_percent"].map(_rsd_category)
    eligible["_report_class"] = _resolved_report_class(eligible)
    panels = [
        _pie_chart(eligible["_rsd_category"].value_counts(), "All annotated features")
    ]
    for lipid_class, values in eligible.groupby(
        "_report_class",
        dropna=False,
        sort=True,
    ):
        panels.append(
            _pie_chart(
                values["_rsd_category"].value_counts(),
                str(lipid_class) or "unclassified",
            )
        )
    return "<div class='pie-grid'>" + "".join(panels) + "</div>"


_STYLE = """
body{font:14px -apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
 margin:0;padding:20px;color:#111;background:#f7f7f5}
h1{font-size:24px;margin:0 0 5px}.meta,.explanation{color:#575753;line-height:1.55}
.summary{background:#fff;border:1px solid #dfdfdc;border-radius:10px;padding:14px;
 margin:16px 0;overflow-x:auto}.summary h2{font-size:17px;margin:0 0 5px}
.class-chart{display:block;width:100%;min-width:700px;max-height:260px}
.class-chart .class-bar,.class-chart .class-bar-key{fill:#70a9e8}
.class-chart .class-bar-msms{fill:#155ca8}
.class-chart-legend text{font-size:12px;fill:#555}.axis{stroke:#b9b9b5;stroke-width:.8}
.bar-count{font-size:13px;font-weight:650;fill:#111}.bar-label{font-size:12px;fill:#555}
.bar-msms-count{font-size:13px;font-weight:700;fill:#fff;paint-order:stroke;
 stroke:#155ca8;stroke-width:1px}
.samples{display:flex;flex-wrap:wrap;gap:4px 13px;margin:12px 0 18px}
.sample-key{display:flex;align-items:center;gap:4px;color:#555;font-size:12px}
.swatch{width:13px;height:3px;border-radius:2px}.rows{min-width:1480px}
.run-qc{background:#fff;border:1px solid #dfdfdc;border-radius:10px;padding:14px;
 margin:16px 0}.run-qc h2{font-size:18px;margin:0 0 5px}.run-qc h3{font-size:14px;margin:0 0 6px}
.qc-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:10px}
.qc-panel,.is-trend,.pie-card{border:1px solid #e5e5e1;border-radius:7px;padding:9px;
 background:#fcfcfb}.is-trend{margin:9px 0}.trend-grid{display:grid;
 grid-template-columns:repeat(3,minmax(280px,1fr));gap:8px;overflow-x:auto}
.trend{display:block;width:100%;height:auto;max-height:190px}.trend-line{fill:none;
 stroke:#aaa;stroke-width:1}.trend .tick-label{fill:#777;font-size:9px}
.trend .axis-title{fill:#888;font-size:9px}.trend.empty text{fill:#aaa;font-size:12px}
.pie-grid{display:flex;flex-wrap:wrap;gap:9px}.pie-card{width:230px}.pie{display:block;
 width:160px;height:160px;margin:auto}.pie text{font-size:14px;font-weight:700;fill:#333}
.pie-legend{display:grid;gap:3px;font-size:11px}.pie-legend span{display:flex;align-items:center;gap:5px}
.pie-legend i{display:inline-block;width:9px;height:9px;border-radius:2px}
.summary-kpis{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0 18px}
.summary-kpi{min-width:210px;padding:12px 14px;border:1px solid #d8e1eb;
 border-radius:8px;background:#f3f7fb}.summary-kpi span{display:block;color:#5b6875;
 font-size:12px}.summary-kpi b{display:block;font-size:26px;color:#174f82;margin-top:3px}
.header,.feature-row{display:grid;grid-template-columns:225px 205px repeat(3,minmax(300px,1fr));
 gap:7px;align-items:start}.header{position:sticky;top:0;z-index:3;background:#ededeb;
 border:1px solid #d5d5d1;border-radius:8px 8px 0 0;padding:8px;font-weight:650}
.feature-row{background:#fff;border:1px solid #e0e0dc;border-top:0;padding:7px;
 break-inside:avoid}.feature-row.primary{border-left:5px solid #1baf7a}
.feature-row.accepted{border-left:5px solid #eda100}.feature-row.rejected{opacity:.78}
.identity{font-size:15px;font-weight:650}.subtle{color:#666;font-size:12px;line-height:1.45}
.badge{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.03em;
 padding:2px 5px;border-radius:4px;margin:3px 4px 5px 0}.badge.primary{background:#dff4e9;color:#086a43}
.badge.accepted{background:#fff1ca;color:#805800}.badge.review{background:#e8effb;color:#245ca0}
.badge.rejected{background:#efefec;color:#666}.metrics{display:grid;grid-template-columns:1fr 1fr;
 gap:4px}.metric{border:1px solid #ecece8;border-radius:5px;padding:3px 5px}
.metric span{display:block;color:#777;font-size:10px;line-height:1.2}.metric b{font-size:13px}
.msms-rule{grid-column:1/-1;border:1px solid #d8d8d4;border-radius:6px;padding:4px 5px}
.msms-rule b{display:inline-block;font-size:12px;padding:2px 6px;border-radius:4px;
 margin-bottom:2px}.msms-rule small{display:block;color:#666;font-size:10px;line-height:1.3}
.msms-rule.pass{background:#edf8f2;border-color:#b9dfcc}.msms-rule.pass b{background:#ccebdc;color:#086a43}
.msms-rule.fail{background:#fff0ef;border-color:#efc5c0}.msms-rule.fail b{background:#f4d3cf;color:#9b3028}
.msms-rule.no-spectrum{background:#fff8e7;border-color:#ead8a5}.msms-rule.no-spectrum b{background:#f8e8b8;color:#74540b}
.msms-rule.no-rule,.msms-rule.not-evaluated{background:#f3f3f1}.msms-rule.no-rule b,
.msms-rule.not-evaluated b{background:#e1e1dd;color:#555}
.fatty-acyl-evidence{grid-column:1/-1;border:1px solid #9dcfbc;border-radius:6px;
 padding:5px 6px;background:#eef9f4}.fatty-acyl-evidence>b{display:inline-block;
 font-size:11px;color:#075f42;margin-bottom:3px}.fatty-acyl-evidence small{display:block;
 color:#405d52;font-size:10px;line-height:1.35}.fatty-acyl-evidence table{width:100%;
 border-collapse:collapse;margin-top:4px;font-size:9px}.fatty-acyl-evidence th,
.fatty-acyl-evidence td{padding:2px 3px;border-top:1px solid #cbe5d9;text-align:right}
.fatty-acyl-evidence th:first-child,.fatty-acyl-evidence td:first-child,
.fatty-acyl-evidence th:nth-child(2),.fatty-acyl-evidence td:nth-child(2){text-align:left}
.identity-quality{margin-top:7px;border:1px solid #deded9;border-radius:6px;padding:5px 6px;
 background:#fafaf8}.identity-quality>span:first-child{display:block;color:#666;font-size:10px}
.identity-quality .quality-badge{margin:3px 0}.identity-quality small{display:block;color:#777;
 font-size:10px;line-height:1.3;overflow-wrap:anywhere}.peak-quality-strip{grid-column:1/-1;
 display:grid;grid-template-columns:repeat(8,minmax(135px,1fr));
 gap:5px;border-top:1px solid #ddddda;padding-top:8px;margin-top:2px}
.peak-quality-metric{border:1px solid #deded9;border-radius:6px;padding:6px 7px;
 background:#fafaf8;min-height:48px}.peak-quality-metric>span{display:block;color:#666;
 font-size:11px}.peak-quality-metric>b{display:block;font-size:14px;margin:3px 0 1px}
.peak-quality-metric small{display:block;color:#858580;font-size:10px;line-height:1.3}
.quality-badge{display:inline-block;padding:2px 6px;border-radius:4px;font-size:12px}
.quality-badge.pass{background:#ccebdc;color:#086a43}.quality-badge.review{background:#f8e8b8;color:#74540b}
.quality-badge.fail{background:#f4d3cf;color:#9b3028}.quality-badge.not-assessed{background:#e1e1dd;color:#555}
.rule-table-wrap{overflow-x:auto;margin:8px 0 16px}.rule-table{border-collapse:collapse;
 width:100%;font-size:12px;background:#fff}.rule-table th,.rule-table td{border:1px solid #ddddda;
 padding:5px 7px;text-align:left;vertical-align:top}.rule-table th{background:#f1f1ee}
.plot-panel{position:relative;min-width:0;align-self:start}.plot-title{position:absolute;
 top:4px;right:7px;z-index:2;font-size:10px;color:#5f666c;background:rgba(255,255,255,.82);
 border-radius:3px;padding:1px 4px;line-height:1.25;pointer-events:none}.evidence-plot,svg.spark{
 display:block;width:100%;height:auto;max-height:none}.evidence-plot .fragment{stroke:#2a78d6;stroke-width:1}
.evidence-plot .fatty-acyl-fragment{stroke:#e18118;stroke-width:2.2}
.evidence-plot .precursor{stroke:#e34948;stroke-width:1;stroke-dasharray:3 2}
.evidence-plot .tick-label,svg.spark .tick-label{fill:#777;font-size:9px}
.evidence-plot .axis-title,svg.spark .axis-title{fill:#999;font-size:9px}
.evidence-plot.empty text,svg.spark.empty text{fill:#aaa;font-size:12px}
svg.spark polyline{fill:none;stroke-width:1.2;stroke-linejoin:round}
svg.spark line.axis,svg.spark line.tick{stroke:#c7c7cc;stroke-width:.8}
svg.spark line.expected{stroke:#c7c7cc;stroke-width:1;stroke-dasharray:2 2}
.empty-message{color:#888}
.section-label{grid-column:1/-1;background:#e6eef8;border:1px solid #cad8ea;
 padding:7px 9px;font-size:14px;font-weight:700;color:#24527f}
.section-label.lipids{background:#f1f1ee;border-color:#ddddda;color:#555}
.lipid-map-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));
 gap:12px}.lipid-map-card{border:1px solid #deded9;border-radius:8px;padding:10px;
 background:#fff;min-width:0}.lipid-map-card h3{font-size:16px;margin:0 0 3px}
.lipid-map-card .subtle{margin:0 0 4px}.lipid-map{display:block;width:100%;height:auto}
.lipid-map .grid-line{stroke:#e8e8e4;stroke-width:.7}.lipid-map .tick-label{fill:#666;
 font-size:10px}.lipid-map .axis-title{fill:#444;font-size:11px;font-weight:600}
.lipid-map .map-point{opacity:.82}.lipid-map .map-point.primary{stroke:#111;stroke-width:1.8}
.lipid-map .msms-ring{fill:none;stroke:#111;stroke-width:1;stroke-dasharray:2 1}
.map-legend{display:flex;flex-wrap:wrap;align-items:center;gap:7px 14px;margin:10px 0 15px;
 color:#555;font-size:12px}.map-legend span{display:inline-flex;align-items:center;gap:5px}
.map-legend i{display:inline-block;width:10px;height:10px;border-radius:50%}
.map-legend .primary-key{background:#fff;border:2px solid #111;box-sizing:border-box}
.map-legend .msms-key{background:#fff;border:1px dashed #111;box-sizing:border-box}
"""

# Public styling contract for evidence-card consumers such as the paired-
# polarity reconciler.  Keeping the card renderer and its CSS here prevents a
# second, subtly different feature-report visual language from developing.
FEATURE_EVIDENCE_STYLE = _STYLE


def _report_text(value: object, default: str = "") -> str:
    """Return display text without evaluating nullable pandas scalars."""
    if value is None or value is pd.NA:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or default


def _retained_export_report_row(row: pd.Series) -> pd.Series:
    """Adapt a retained_features.csv row to the feature-card schema."""
    adapted = row.copy()
    aliases = {
        "feat_feature_id": "feature_id",
        "name": "feature_identity",
        "class": "lipid_class",
        "feat_mz": "mz",
        "feat_rt": "rt_min",
        "feat_mobility": "mobility",
        "status": "annotation_status",
        "score": "annotation_score",
        "rt_error": "rt_error_min",
        "ccs_error_percent": "ccs_delta_percent",
        "msms_rule_id": "class_rule_id",
        "fatty_acyl_detected": "detected_fatty_acyl_groups",
    }
    for target, source in aliases.items():
        if target not in adapted or pd.isna(adapted.get(target)):
            adapted[target] = adapted.get(source)
    if not _report_text(adapted.get("processing_qc_flags", "")):
        adapted["processing_qc_flags"] = adapted.get("quantification_qc_flags", "")
    return adapted


def _retained_card_metrics(row: pd.Series, polarity: str) -> str:
    rt_error = pd.to_numeric(row.get("rt_error"), errors="coerce")
    rt_seconds = float(rt_error) * 60 if pd.notna(rt_error) else np.nan
    coherence = pd.to_numeric(row.get("grid_coherence_score"), errors="coerce")
    rule = _report_text(row.get("class_rule_result", ""), "NOT EVALUATED").upper()
    rule_class = {
        "PASS": "pass",
        "FAIL": "fail",
        "NO_SPECTRUM": "no-spectrum",
        "NO_RULE": "no-rule",
    }.get(rule, "not-evaluated")
    return (
        "<div class='metrics'>"
        f"<div class='msms-rule {rule_class}'><b>MS/MS RULE: "
        f"{html.escape(rule.replace('_', ' '))}</b>"
        f"<small>{html.escape(_report_text(row.get('class_rule_description', '')))}</small></div>"
        f"<div class='metric'><span>polarity</span><b>{html.escape(polarity)}</b></div>"
        f"<div class='metric'><span>mass error</span><b>{_number(row.get('ppm_error'), 2, ' ppm')}</b></div>"
        f"<div class='metric'><span>ΔRT</span><b>{_number(rt_seconds, 1, ' s')}</b></div>"
        f"<div class='metric'><span>grid coherence</span><b>{'n/a' if pd.isna(coherence) else f'{float(coherence):.3f}'}</b></div>"
        f"<div class='metric'><span>ΔCCS</span><b>{_number(row.get('ccs_error_percent'), 2, '%')}</b></div>"
        f"<div class='metric'><span>annotation score</span><b>{_number(row.get('score'), 3)}</b></div>"
        f"<div class='metric'><span>IS-corrected QC RSD · height</span><b>{_number(row.get('qc_rsd_height_percent'), 1, '%')}</b></div>"
        f"<div class='metric'><span>IS-corrected QC RSD · area</span><b>{_number(row.get('qc_rsd_area_percent'), 1, '%')}</b></div>"
        "</div>"
    )


def render_retained_feature_evidence_card(
    export_row: pd.Series,
    *,
    polarity: str,
    traces: list[SampleTrace] | None = None,
    spectrum: dict | None = None,
    colours: dict[str, str] | None = None,
    sample_roles: dict[str, str] | None = None,
    selected_quantifier: bool = False,
) -> str:
    """Render one retained POS/NEG feature in the standard report format."""
    row = _retained_export_report_row(export_row)
    feature_traces = list(traces or [])
    roles = sample_roles or {}
    if roles:
        allowed = {
            "sample",
            "biological",
            "study",
            "case",
            "qc",
            "quality_control",
            "pooled_qc",
        }
        feature_traces = [
            trace
            for trace in feature_traces
            if str(roles.get(trace.sample_name, "unknown")).strip().casefold()
            in allowed
        ]
    feature_id = _report_text(row.get("feat_feature_id", ""))
    rt = pd.to_numeric(row.get("feat_rt"), errors="coerce")
    rt_error = pd.to_numeric(row.get("rt_error"), errors="coerce")
    expected_rt = (
        float(rt) - float(rt_error) if pd.notna(rt) and pd.notna(rt_error) else None
    )
    colour_map = colours or sample_colours(
        sorted({trace.sample_name for trace in feature_traces})
    )
    chromatogram = _overlay(
        feature_traces,
        mobility=False,
        colours=colour_map,
        marker=expected_rt,
        width=300,
        height=150,
    )
    mobilogram = _overlay(
        feature_traces,
        mobility=True,
        colours=colour_map,
        width=300,
        height=150,
    )
    mz = pd.to_numeric(row.get("feat_mz"), errors="coerce")
    mobility = pd.to_numeric(row.get("feat_mobility"), errors="coerce")
    badge = "SELECTED QUANTIFIER" if selected_quantifier else "COMPLEMENTARY EVIDENCE"
    badge_class = "primary" if selected_quantifier else "accepted"
    row_class = "primary" if selected_quantifier else "accepted"
    status = _report_text(row.get("status", "accept"), "accept").casefold()
    return (
        f"<div class='feature-row {row_class}'>"
        f"<div><div class='identity'>{html.escape(_report_text(row.get('name', '')))} "
        f"{html.escape(_report_text(row.get('adduct', '')))}</div>"
        f"<span class='badge {badge_class}'>{html.escape(polarity)} · {badge}</span>"
        f"<div class='subtle'>{html.escape(_report_text(row.get('class', '')))} · "
        f"feature #{html.escape(feature_id)}<br>"
        f"m/z {'n/a' if pd.isna(mz) else f'{float(mz):.5f}'} · "
        f"RT {'n/a' if pd.isna(rt) else f'{float(rt):.3f} min'} · "
        f"mobility {'n/a' if pd.isna(mobility) else f'{float(mobility):.4f}'}<br>"
        f"decision: {html.escape(status)} · quantification: "
        f"{html.escape(_report_text(row.get('quantification_quality', ''), 'not assessed'))}</div>"
        f"{_peak_quality_summary(row)}</div>"
        f"<div>{_retained_card_metrics(row, polarity)}</div>"
        f"<div class='plot-panel'><div class='plot-title'>{len(feature_traces)} sample trace(s)</div>{chromatogram}</div>"
        f"<div class='plot-panel'><div class='plot-title'>{len(feature_traces)} sample trace(s)</div>{mobilogram}</div>"
        f"<div class='plot-panel'><div class='plot-title'>feature-linked spectrum</div>"
        f"{_spectrum_svg(spectrum, fatty_acyl_evidence=row.get('fatty_acyl_evidence_json'))}</div>"
        f"{_fatty_acyl_evidence_summary(row)}"
        f"{_peak_quality_strip(row)}"
        "</div>"
    )


_KENDRICK_CH2_EXACT_MASS = 14.01565
_MAP_COLOURS = (
    "#2a78d6",
    "#e47732",
    "#1a9b61",
    "#9a67d5",
    "#d34f74",
    "#8a6d3b",
    "#18a0aa",
    "#6f7f32",
    "#c3563f",
    "#5267b7",
)


def build_lipid_class_map_data(annotations: pd.DataFrame) -> pd.DataFrame:
    """Return one selected retained feature per unique lipid identity.

    Both maps are final-result views rather than candidate diagnostics.  They
    therefore suppress alternative features after exclusions and retain the
    actual LASER primary for each class × lipid identity × adduct. LASER makes
    feature-selection decisions at that identity/adduct level; this report does
    not invent a second cross-adduct winner.
    """
    frame = retained_annotations(annotations)
    if "status" in frame:
        accepted = frame["status"].fillna("").astype(str).str.casefold().eq("accept")
        frame = frame.loc[accepted].copy()
    required = {"feat_feature_id", "name", "class", "adduct", "feat_rt", "feat_mz"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(
            columns=[
                "feat_feature_id",
                "name",
                "class",
                "adduct",
                "feat_rt",
                "feat_mz",
                "primary_for_identity",
                "msms_has_spectrum",
                "kendrick_mass",
                "nominal_kendrick_mass",
                "kendrick_mass_defect",
            ]
        )
    frame = frame.drop_duplicates(
        ["feat_feature_id", "name", "adduct"], keep="first"
    ).copy()
    for column in ("feat_feature_id", "name", "class", "adduct"):
        frame[column] = frame[column].fillna("").astype(str)
    frame["class"] = _resolved_report_class(frame)
    frame["feat_rt"] = pd.to_numeric(frame["feat_rt"], errors="coerce")
    frame["feat_mz"] = pd.to_numeric(frame["feat_mz"], errors="coerce")
    frame = frame[
        np.isfinite(frame["feat_rt"])
        & np.isfinite(frame["feat_mz"])
        & frame["feat_mz"].gt(0)
    ].copy()

    def boolean(values):
        return (
            values.fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
            .isin({"1", "true", "yes", "y"})
        )

    frame["primary_for_identity"] = boolean(
        frame.get("primary_for_identity", pd.Series(False, index=frame.index))
    )
    frame["msms_has_spectrum"] = boolean(
        frame.get("msms_has_spectrum", pd.Series(False, index=frame.index))
    )

    identity_keys = ["class", "name", "adduct"]
    # Keep the primary LASER actually selected whenever it is recorded. The
    # evidence order is only a compatibility fallback for legacy tables that
    # predate the explicit primary_for_identity export.
    has_primary = frame.groupby(identity_keys, dropna=False)[
        "primary_for_identity"
    ].transform("any")
    frame = frame.loc[~has_primary | frame["primary_for_identity"]].copy()
    frame["_msms_order"] = (~frame["msms_has_spectrum"]).astype(int)
    frame["_rt_order"] = (
        pd.to_numeric(
            frame.get("rt_error", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
        .abs()
        .fillna(np.inf)
    )
    frame["_ppm_order"] = (
        pd.to_numeric(
            frame.get("ppm_error", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
        .abs()
        .fillna(np.inf)
    )
    frame["_score_order"] = pd.to_numeric(
        frame.get("score", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).fillna(np.inf)
    frame = frame.sort_values(
        [
            "class",
            "name",
            "_msms_order",
            "_rt_order",
            "_ppm_order",
            "_score_order",
            "feat_feature_id",
            "adduct",
        ],
        kind="stable",
    ).drop_duplicates(identity_keys, keep="first")
    frame["primary_for_identity"] = True
    frame["kendrick_mass"] = frame["feat_mz"] * 14.0 / _KENDRICK_CH2_EXACT_MASS
    frame["nominal_kendrick_mass"] = np.rint(frame["kendrick_mass"]).astype(int)
    frame["kendrick_mass_defect"] = (
        frame["nominal_kendrick_mass"] - frame["kendrick_mass"]
    )
    return frame[
        [
            "feat_feature_id",
            "name",
            "class",
            "adduct",
            "feat_rt",
            "feat_mz",
            "primary_for_identity",
            "msms_has_spectrum",
            "kendrick_mass",
            "nominal_kendrick_mass",
            "kendrick_mass_defect",
        ]
    ].reset_index(drop=True)


def _map_axis_range(values: pd.Series, *, minimum_span: float) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, minimum_span)
    padding = span * 0.08
    centre = (low + high) / 2
    if high - low < minimum_span:
        low, high = centre - minimum_span / 2, centre + minimum_span / 2
    return low - padding, high + padding


def _lipid_class_scatter_svg(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    adduct_colours: dict[str, str],
    x_minimum_span: float,
    y_minimum_span: float,
    width: int = 560,
    height: int = 330,
) -> str:
    left, right, top, bottom = 66, 16, 16, 48
    plot_width, plot_height = width - left - right, height - top - bottom
    x_low, x_high = _map_axis_range(frame[x_column], minimum_span=x_minimum_span)
    y_low, y_high = _map_axis_range(frame[y_column], minimum_span=y_minimum_span)
    x_span, y_span = x_high - x_low, y_high - y_low

    def to_x(value):
        return left + (float(value) - x_low) / x_span * plot_width

    def to_y(value):
        return top + plot_height - (float(value) - y_low) / y_span * plot_height

    parts = [
        f'<svg class="lipid-map" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(y_label)} by {html.escape(x_label)}">',
    ]
    for fraction in np.linspace(0, 1, 6):
        x_value = x_low + float(fraction) * x_span
        y_value = y_low + float(fraction) * y_span
        x = to_x(x_value)
        y = to_y(y_value)
        parts.extend(
            [
                f'<line class="grid-line" x1="{x:.1f}" y1="{top}" '
                f'x2="{x:.1f}" y2="{top + plot_height}"/>',
                f'<line class="grid-line" x1="{left}" y1="{y:.1f}" '
                f'x2="{left + plot_width}" y2="{y:.1f}"/>',
                f'<text class="tick-label" x="{x:.1f}" y="{top + plot_height + 18}" '
                f'text-anchor="middle">{x_value:.3g}</text>',
                f'<text class="tick-label" x="{left - 7}" y="{y + 3:.1f}" '
                f'text-anchor="end">{y_value:.4g}</text>',
            ]
        )
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top + plot_height}"/>',
        ]
    )
    for row in frame.to_dict(orient="records"):
        x = to_x(row[x_column])
        y = to_y(row[y_column])
        primary = bool(row.get("primary_for_identity", False))
        has_msms = bool(row.get("msms_has_spectrum", False))
        radius = 5.2 if primary else 4.2
        tooltip = (
            f"{row.get('name', '')} {row.get('adduct', '')} · feature "
            f"#{row.get('feat_feature_id', '')} · RT {float(row['feat_rt']):.4f} min · "
            f"m/z {float(row['feat_mz']):.5f} · KMD "
            f"{float(row['kendrick_mass_defect']):.6f}"
        )
        if has_msms:
            parts.append(
                f'<circle class="msms-ring" cx="{x:.1f}" cy="{y:.1f}" '
                f'r="{radius + 2.2:.1f}"/>'
            )
        parts.append(
            f'<circle class="map-point{" primary" if primary else ""}" '
            f'cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
            f'fill="{adduct_colours[str(row.get("adduct", ""))]}">'
            f"<title>{html.escape(tooltip)}</title></circle>"
        )
    parts.extend(
        [
            f'<text class="axis-title" x="{left + plot_width / 2:.1f}" y="{height - 6}" '
            f'text-anchor="middle">{html.escape(x_label)}</text>',
            f'<text class="axis-title" x="14" y="{top + plot_height / 2:.1f}" '
            f'transform="rotate(-90 14 {top + plot_height / 2:.1f})" '
            f'text-anchor="middle">{html.escape(y_label)}</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _lipid_class_map_report(
    annotations: pd.DataFrame,
    *,
    kind: str,
    title: str,
    subtitle: str = "",
) -> str:
    frame = build_lipid_class_map_data(annotations)
    if kind not in {"rt_mz", "kendrick"}:
        raise ValueError("kind must be rt_mz or kendrick")
    adducts = sorted(frame.get("adduct", pd.Series(dtype=str)).astype(str).unique())
    adduct_colours = {
        adduct: _MAP_COLOURS[index % len(_MAP_COLOURS)]
        for index, adduct in enumerate(adducts)
    }
    explanation = (
        "Each panel contains the final retained feature LASER selected for each "
        "unique lipid identity and adduct after all exclusions. Alternative "
        "features for that same identification are omitted. Hover a point for "
        "its identity and exact values."
        if kind == "rt_mz"
        else "Each panel uses the conventional CH₂ Kendrick scale: Kendrick mass = "
        "m/z × 14 / 14.01565; nominal Kendrick mass is the nearest integer and "
        "Kendrick mass defect = nominal − Kendrick mass. Values are therefore "
        "ion/adduct-specific rather than neutral-mass defects. The final retained "
        "feature LASER selected for each unique lipid identity and adduct is shown "
        "after all exclusions. Hover a point for details."
    )
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='meta'>{html.escape(subtitle)}</p>" if subtitle else "",
        f"<p class='explanation'>{explanation}</p>",
    ]
    if frame.empty:
        parts.append(
            "<p class='empty-message'>No retained annotations with valid RT and m/z.</p></body>"
        )
        return "".join(parts)
    parts.append("<div class='map-legend'>")
    for adduct in adducts:
        parts.append(
            f"<span><i style='background:{adduct_colours[adduct]}'></i>"
            f"{html.escape(adduct or 'no adduct')}</span>"
        )
    parts.extend(
        [
            "<span><i class='primary-key'></i>final selected feature</span>",
            "<span><i class='msms-key'></i>linked MS/MS spectrum</span>",
            "</div><div class='lipid-map-grid'>",
        ]
    )
    for lipid_class, values in frame.groupby("class", dropna=False, sort=True):
        class_label = (
            str(lipid_class)
            if pd.notna(lipid_class) and str(lipid_class)
            else "unclassified"
        )
        if kind == "rt_mz":
            plot = _lipid_class_scatter_svg(
                values,
                x_column="feat_rt",
                y_column="feat_mz",
                x_label="retention time (min)",
                y_label="m/z",
                adduct_colours=adduct_colours,
                x_minimum_span=0.2,
                y_minimum_span=2.0,
            )
        else:
            plot = _lipid_class_scatter_svg(
                values,
                x_column="nominal_kendrick_mass",
                y_column="kendrick_mass_defect",
                x_label="nominal Kendrick mass (CH₂)",
                y_label="Kendrick mass defect (CH₂)",
                adduct_colours=adduct_colours,
                x_minimum_span=2.0,
                y_minimum_span=0.01,
            )
        parts.extend(
            [
                "<section class='lipid-map-card'>",
                f"<h3>{html.escape(class_label)}</h3>",
                f"<p class='subtle'>{len(values):,} final selected identification(s)</p>",
                plot,
                "</section>",
            ]
        )
    parts.append("</div></body>")
    return "".join(parts)


def render_rt_mz_map_report(
    annotations: pd.DataFrame,
    *,
    title: str = "RT–m/z maps by lipid class",
    subtitle: str = "",
) -> str:
    """Render retained annotations as one RT-versus-m/z panel per class."""
    return _lipid_class_map_report(
        annotations, kind="rt_mz", title=title, subtitle=subtitle
    )


def render_kendrick_map_report(
    annotations: pd.DataFrame,
    *,
    title: str = "Kendrick mass-defect maps by lipid class",
    subtitle: str = "",
) -> str:
    """Render retained annotations on conventional CH2 Kendrick axes."""
    return _lipid_class_map_report(
        annotations, kind="kendrick", title=title, subtitle=subtitle
    )


def write_rt_mz_map_report(
    annotations: pd.DataFrame,
    output: str | Path,
    **kwargs,
) -> Path:
    destination = Path(output)
    return atomic_write_text(
        destination, render_rt_mz_map_report(annotations, **kwargs)
    )


def write_kendrick_map_report(
    annotations: pd.DataFrame,
    output: str | Path,
    **kwargs,
) -> Path:
    destination = Path(output)
    return atomic_write_text(
        destination, render_kendrick_map_report(annotations, **kwargs)
    )


def render_annotation_summary_charts_report(
    annotations: pd.DataFrame,
    *,
    before_exclusions: pd.DataFrame | None = None,
    title: str = "Annotation summary charts",
    subtitle: str = "",
) -> str:
    """Render before/after class bars and corrected QC-RSD summaries."""
    retained = retained_annotations(annotations)
    before = (
        before_exclusions.copy() if before_exclusions is not None else retained.copy()
    )
    retained_feature_count = (
        retained["feat_feature_id"]
        .dropna()
        .astype(str)
        .replace("", np.nan)
        .dropna()
        .nunique()
        if "feat_feature_id" in retained
        else 0
    )
    retained_annotation_count = (
        retained[["class", "name"]].drop_duplicates().shape[0]
        if {"class", "name"}.issubset(retained.columns)
        else 0
    )
    return "".join(
        [
            "<!doctype html><meta charset='utf-8'>",
            f"<title>{html.escape(title)}</title><style>{_STYLE}</style>",
            "<body>",
            f"<h1>{html.escape(title)}</h1>",
            f"<p class='meta'>{html.escape(subtitle)}</p>" if subtitle else "",
            "<section class='summary'><h2>Unique annotations before exclusions</h2>",
            "<p class='subtle'>Each lipid identity is counted once per class before "
            "active MS/MS, grid-coherence, boundary/apex and review exclusions. "
            "Repeated features and adduct measurements do not inflate the bars.</p>",
            "<p class='subtle'>The light bar is the total; the darker overlay is "
            "the subset with a linked MS/MS spectrum.</p>",
            _class_bar_chart(
                before,
                aria_label="Unique lipid annotations before exclusions per class",
                apply_exclusions=False,
            ),
            "</section>",
            "<section class='summary'><h2>Unique annotations after all exclusions</h2>",
            "<p class='subtle'>Only accepted annotations that remain after every "
            "active criterion and review decision contribute to this chart. The "
            "darker overlay is the retained subset with a linked MS/MS spectrum.</p>",
            "<div class='summary-kpis'>",
            "<div class='summary-kpi'><span>Total unique annotated features after "
            "all exclusions</span><b>",
            f"{int(retained_feature_count):,}",
            "</b></div>",
            "<div class='summary-kpi'><span>Total unique lipid annotations after "
            "all exclusions</span><b>",
            f"{int(retained_annotation_count):,}",
            "</b></div>",
            "</div>",
            _class_bar_chart(
                retained,
                aria_label="Unique retained lipid annotations per class",
            ),
            "</section>",
            "<section class='run-qc'><h2>IS-corrected QC-RSD distributions</h2>",
            "<p class='subtle'>Overall and per-class pies use corrected peak height "
            "and preserve a separate not-evaluable category.</p>",
            _rsd_summary(retained),
            "</section></body>",
        ]
    )


def write_annotation_summary_charts_report(
    annotations: pd.DataFrame,
    output: str | Path,
    *,
    before_exclusions: pd.DataFrame | None = None,
    title: str = "Annotation summary charts",
    subtitle: str = "",
) -> Path:
    """Write the self-contained chart summary used by the combined report."""
    path = Path(output)
    return atomic_write_text(
        path,
        render_annotation_summary_charts_report(
            annotations,
            before_exclusions=before_exclusions,
            title=title,
            subtitle=subtitle,
        ),
    )


@dataclass(frozen=True)
class AnnotatedFeatureReportContext:
    """Cohesive input contract for one feature-evidence report."""

    annotations: pd.DataFrame
    traces: dict[str, list[SampleTrace]] = field(default_factory=dict)
    spectra: dict[str, dict] = field(default_factory=dict)
    title: str = "Annotated feature evidence"
    subtitle: str = ""
    sample_groups: dict[str, str] = field(default_factory=dict)
    sample_roles: dict[str, str] = field(default_factory=dict)
    summed_chromatograms: list[SummedFeatureChromatogram] = field(default_factory=list)
    is_trends: list[dict] = field(default_factory=list)
    report_kind: str = "annotations"
    polarity: str = ""
    fragment_tolerance_da: float = DEFAULT_FRAGMENT_TOLERANCE_DA
    fatty_acyl_min_relative_intensity_percent: float = (
        DEFAULT_FA_MIN_RELATIVE_INTENSITY_PERCENT
    )
    grid_coherence_threshold: float = SCIENTIFIC_DEFAULTS.grid_coherence_threshold
    maximum_boundary_apex_ratio: float = SCIENTIFIC_DEFAULTS.maximum_boundary_apex_ratio
    minimum_mzmine_shape_score: float = SCIENTIFIC_DEFAULTS.minimum_mzmine_shape_score
    lazy_rows: bool = False


def render_annotated_feature_report(
    annotations: pd.DataFrame,
    *,
    traces: dict[str, list[SampleTrace]] | None = None,
    spectra: dict[str, dict] | None = None,
    title: str = "Annotated feature evidence",
    subtitle: str = "",
    sample_groups: dict[str, str] | None = None,
    sample_roles: dict[str, str] | None = None,
    summed_chromatograms: list[SummedFeatureChromatogram] | None = None,
    is_trends: list[dict] | None = None,
    report_kind: str = "annotations",
    polarity: str = "",
    fragment_tolerance_da: float = DEFAULT_FRAGMENT_TOLERANCE_DA,
    fatty_acyl_min_relative_intensity_percent: float = (
        DEFAULT_FA_MIN_RELATIVE_INTENSITY_PERCENT
    ),
    grid_coherence_threshold: float = (SCIENTIFIC_DEFAULTS.grid_coherence_threshold),
    maximum_boundary_apex_ratio: float = (
        SCIENTIFIC_DEFAULTS.maximum_boundary_apex_ratio
    ),
    minimum_mzmine_shape_score: float = (
        SCIENTIFIC_DEFAULTS.minimum_mzmine_shape_score
    ),
    lazy_rows: bool = False,
) -> str:
    """Compatibility wrapper around the typed report context."""
    return render_annotated_feature_report_context(
        AnnotatedFeatureReportContext(
            annotations=annotations,
            traces=traces or {},
            spectra=spectra or {},
            title=title,
            subtitle=subtitle,
            sample_groups=sample_groups or {},
            sample_roles=sample_roles or {},
            summed_chromatograms=summed_chromatograms or [],
            is_trends=is_trends or [],
            report_kind=report_kind,
            polarity=polarity,
            fragment_tolerance_da=fragment_tolerance_da,
            fatty_acyl_min_relative_intensity_percent=(
                fatty_acyl_min_relative_intensity_percent
            ),
            grid_coherence_threshold=grid_coherence_threshold,
            maximum_boundary_apex_ratio=maximum_boundary_apex_ratio,
            minimum_mzmine_shape_score=minimum_mzmine_shape_score,
            lazy_rows=lazy_rows,
        )
    )


_REPORT_KINDS = {
    "annotations",
    "msms_exclusions",
    "grid_exclusions",
    "boundary_apex_exclusions",
    "mzmine_shape_exclusions",
}


def _feature_report_explanation(kind: str) -> str:
    explanations = {
        "msms_exclusions": (
            "One row is one feature × lipid identity × adduct match removed by the "
            "enforced class-fragment MS/MS rule. Every excluded match remains visible "
            "with its evidence; none is a selected annotation. The displayed rank is "
            "only the ordering among excluded candidates for that lipid ID."
        ),
        "grid_exclusions": (
            "One row is one feature × lipid identity × adduct match removed because "
            "its assessed grid-coherence score was below the active threshold. Every "
            "excluded match remains visible and none can become a primary annotation."
        ),
        "boundary_apex_exclusions": (
            "One row is one feature × lipid identity × adduct match removed because "
            "its median chromatographic boundary-to-apex intensity ratio exceeded the "
            "active maximum. Every excluded match remains visible and unselected."
        ),
        "mzmine_shape_exclusions": (
            "One row is one feature × lipid identity × adduct match removed because an "
            "available MZmine RT or mobility shape-fit score was below the active LASER "
            "minimum. Missing scores remain not evaluable and are not excluded."
        ),
    }
    return explanations.get(
        kind,
        (
            "One row is one feature × lipid identity × adduct match. No competing feature "
            "is collapsed. <b>SELECTED FOR LIPID ID</b> follows the viewer rule: manual "
            "primary, confirming MS/MS, smallest RT delta, smallest ppm error, then score. "
            "Other accepted, rejected, and review candidates remain visible."
        ),
    )


def _feature_report_summary(
    context: AnnotatedFeatureReportContext,
    frame: pd.DataFrame,
) -> list[str]:
    kind = context.report_kind
    if kind == "msms_exclusions":
        inconsistent = frame.apply(msms_rule_result, axis=1).ne("FAIL")
        parts = [
            "<section class='summary'><h2>Enforced class-fragment MS/MS rules</h2>",
            "<p class='subtle'>A linked spectrum is excluded only when required evidence is absent.</p>",
            _msms_rules_table(
                context.polarity.casefold(),
                float(context.fragment_tolerance_da),
                float(context.fatty_acyl_min_relative_intensity_percent),
            ),
            "</section><section class='summary'><h2>MS/MS-excluded lipid IDs per class</h2>",
            _excluded_class_bar_chart(
                frame, label="MS/MS-excluded lipid IDs per class"
            ),
            "</section>",
        ]
        if bool(inconsistent.any()):
            parts.append(
                "<section class='summary'><h2>Exclusion-record warning</h2>"
                f"<p class='subtle'><b>{int(inconsistent.sum())}</b> row(s) do not "
                "evaluate to FAIL and remain visible only for audit.</p></section>"
            )
        return parts
    settings = {
        "grid_exclusions": (
            "Grid-coherence exclusion rule",
            f"Candidates with an assessed score below <b>{float(context.grid_coherence_threshold):.2f}</b> are excluded.",
            "Grid-excluded lipid IDs per class",
        ),
        "boundary_apex_exclusions": (
            "Boundary-to-apex exclusion rule",
            f"Assessed candidates above <b>{float(context.maximum_boundary_apex_ratio):.3f}</b> are excluded.",
            "Boundary-excluded lipid IDs per class",
        ),
        "mzmine_shape_exclusions": (
            "MZmine shape-fit exclusion rule",
            f"Available scores below <b>{float(context.minimum_mzmine_shape_score):.2f}</b> are excluded.",
            "Shape-score-excluded lipid IDs per class",
        ),
    }
    if kind not in settings:
        return []
    heading, explanation, chart_label = settings[kind]
    return [
        f"<section class='summary'><h2>{heading}</h2>",
        f"<p class='subtle'>{explanation} Missing or unassessed values are retained.</p>",
        f"<h2>{chart_label}</h2>",
        _excluded_class_bar_chart(frame, label=chart_label),
        "</section>",
    ]


def _feature_report_empty_message(kind: str) -> str:
    return {
        "msms_exclusions": "No feature matches were excluded by the class-fragment MS/MS rule.",
        "grid_exclusions": "No feature matches were excluded by grid coherence.",
        "boundary_apex_exclusions": "No feature matches exceeded the boundary-to-apex maximum.",
        "mzmine_shape_exclusions": "No feature matches fell below the MZmine shape-score minimum.",
    }.get(kind, "No annotated feature matches.")


def _feature_row_badge(
    row: dict[str, object], rule_result: str
) -> tuple[str, str, str]:
    status = str(row.get("status", "review") or "review").casefold()
    if _flag(row.get("rule_excluded", False)):
        badge = (
            "NOT SELECTED · MS/MS RULE EXCLUDED"
            if rule_result == "FAIL"
            else "NOT SELECTED · INVALID MS/MS EXCLUSION"
        )
        return "rejected", "rejected", badge
    choices = (
        ("grid_excluded", "rejected", "rejected", "NOT SELECTED · GRID OUTLIER"),
        (
            "boundary_apex_excluded",
            "rejected",
            "rejected",
            "NOT SELECTED · BOUNDARY/APEX EXCLUDED",
        ),
        (
            "mzmine_shape_score_excluded",
            "rejected",
            "rejected",
            "NOT SELECTED · MZMINE SHAPE SCORE",
        ),
        ("primary_for_identity", "primary", "primary", "SELECTED FOR LIPID ID"),
    )
    for key, row_class, badge_class, badge in choices:
        if _flag(row.get(key, False)):
            return row_class, badge_class, badge
    if status == "accept":
        return "accepted", "accepted", "NOT SELECTED · ACCEPTED ALTERNATIVE"
    if status == "reject":
        return "rejected", "rejected", "NOT SELECTED · REJECTED MATCH"
    return "review", "review", "NOT SELECTED · REVIEW CANDIDATE"


def _feature_row_metrics(
    row: dict[str, object],
    rule_result: str,
    report_kind: str,
) -> str:
    rule_labels = {
        "PASS": "PASS",
        "FAIL": "FAIL",
        "NO_SPECTRUM": "NO MS/MS SPECTRUM",
        "NO_RULE": "NO RULE CONFIGURED",
        "NOT_EVALUATED": "NOT EVALUATED",
    }
    rule_classes = {
        "PASS": "pass",
        "FAIL": "fail",
        "NO_SPECTRUM": "no-spectrum",
        "NO_RULE": "no-rule",
        "NOT_EVALUATED": "not-evaluated",
    }
    rule_id = str(row.get("msms_rule_id", "") or "").strip()
    detail = class_evidence_text(row)
    if rule_result == "FAIL":
        detail += (
            " · candidate excluded by the active MS/MS rule filter"
            if _flag(row.get("rule_excluded", False))
            else " · candidate retained because the MS/MS rule filter is advisory"
        )
    rule_summary = (
        f"<div class='msms-rule {rule_classes[rule_result]}'>"
        f"<b>MS/MS RULE: {rule_labels[rule_result]}</b>"
        f"<small>{html.escape(rule_id + ' · ' if rule_id else '')}"
        f"{html.escape(detail)}</small></div>"
    )
    rt_error = pd.to_numeric(row.get("rt_error"), errors="coerce")
    rt_seconds = float(rt_error) * 60 if pd.notna(rt_error) else np.nan
    coherence = pd.to_numeric(row.get("grid_coherence_score"), errors="coerce")
    rank_label = (
        "rank among excluded" if report_kind != "annotations" else "selection rank"
    )
    return (
        f"<div class='metrics'>{rule_summary}"
        f"<div class='metric'><span>mass error</span><b>{_number(row.get('ppm_error'), 2, ' ppm')}</b></div>"
        f"<div class='metric'><span>ΔRT</span><b>{_number(rt_seconds, 1, ' s')}</b></div>"
        f"<div class='metric'><span>grid coherence</span><b>{'n/a' if pd.isna(coherence) else f'{float(coherence):.3f}'}</b></div>"
        f"<div class='metric'><span>MZmine RT shape</span><b>{_number(row.get('mzmine_rt_shape_score'), 3)}</b></div>"
        f"<div class='metric'><span>MZmine mobility shape</span><b>{_number(row.get('mzmine_mobility_shape_score'), 3)}</b></div>"
        f"<div class='metric'><span>ΔCCS</span><b>{_number(row.get('ccs_error_percent'), 2, '%')}</b></div>"
        f"<div class='metric'><span>annotation score</span><b>{_number(row.get('score'), 3)}</b></div>"
        f"<div class='metric'><span>{rank_label}</span><b>{int(str(row.get('identity_match_rank', 0)))}</b></div>"
        f"<div class='metric'><span>IS-corrected QC RSD · height</span><b>{_number(row.get('qc_rsd_height_percent'), 1, '%')}</b></div>"
        f"<div class='metric'><span>IS-corrected QC RSD · area</span><b>{_number(row.get('qc_rsd_area_percent'), 1, '%')}</b></div></div>"
    )


def _feature_row_markup(
    row: dict[str, object],
    *,
    traces: dict[str, list[SampleTrace]],
    spectra: dict[str, dict],
    roles: dict[str, str],
    colours: dict[str, str],
    shared_fragment: Callable[[str], str],
    report_kind: str,
) -> str:
    feature_id = str(row.get("feat_feature_id", ""))
    status = str(row.get("status", "review") or "review").casefold()
    rule_result = msms_rule_result(row)
    row_class, badge_class, badge = _feature_row_badge(row, rule_result)
    feature_traces = traces.get(feature_id, [])
    if roles:
        allowed_roles = {
            "sample",
            "biological",
            "study",
            "case",
            "qc",
            "quality_control",
            "pooled_qc",
        }
        feature_traces = [
            trace
            for trace in feature_traces
            if str(roles.get(trace.sample_name, "unknown")).strip().casefold()
            in allowed_roles
        ]
    expected_rt = pd.to_numeric(row.get("rt_expected"), errors="coerce")
    chromatogram = shared_fragment(
        _overlay(
            feature_traces,
            mobility=False,
            colours=colours,
            marker=float(expected_rt) if pd.notna(expected_rt) else None,
            width=300,
            height=150,
        )
    )
    mobilogram = shared_fragment(
        _overlay(
            feature_traces,
            mobility=True,
            colours=colours,
            width=300,
            height=150,
        )
    )
    spectrum = shared_fragment(
        _spectrum_svg(
            spectra.get(feature_id),
            fatty_acyl_evidence=row.get("fatty_acyl_evidence_json"),
        )
    )
    mz = pd.to_numeric(row.get("feat_mz"), errors="coerce")
    rt = pd.to_numeric(row.get("feat_rt"), errors="coerce")
    mobility = pd.to_numeric(row.get("feat_mobility"), errors="coerce")
    return (
        f"<div class='feature-row {row_class}'>"
        f"<div><div class='identity'>{html.escape(str(row.get('name', '')))} "
        f"{html.escape(str(row.get('adduct', '')))}</div>"
        f"<span class='badge {badge_class}'>{badge}</span>"
        f"<div class='subtle'>{html.escape(str(row.get('class', '')))} · feature #{html.escape(feature_id)}<br>"
        f"m/z {'n/a' if pd.isna(mz) else f'{float(mz):.5f}'} · "
        f"RT {'n/a' if pd.isna(rt) else f'{float(rt):.3f} min'} · "
        f"mobility {'n/a' if pd.isna(mobility) else f'{float(mobility):.4f}'}<br>"
        f"decision: {html.escape(status)} · MS/MS rule: {html.escape(rule_result)}<br>"
        f"QC normalizer: {html.escape(str(row.get('qc_rsd_internal_standard', '') or 'n/a'))}"
        f"{' (PE fallback)' if bool(row.get('qc_rsd_used_fallback', False)) else ''}</div>"
        f"{_peak_quality_summary(row)}</div>"
        f"<div>{_feature_row_metrics(row, rule_result, report_kind)}</div>"
        f"<div class='plot-panel'><div class='plot-title'>{len(feature_traces)} sample trace(s)</div>{chromatogram}</div>"
        f"<div class='plot-panel'><div class='plot-title'>{len(feature_traces)} sample trace(s)</div>{mobilogram}</div>"
        f"<div class='plot-panel'><div class='plot-title'>feature-linked spectrum</div>{spectrum}</div>"
        f"{_fatty_acyl_evidence_summary(row)}{_peak_quality_strip(row)}</div>"
    )


def _feature_section_label(kind: str, section: str) -> str:
    prefix = {
        "msms_exclusions": "MS/MS-excluded",
        "grid_exclusions": "Grid-excluded",
        "boundary_apex_exclusions": "Boundary-excluded",
        "mzmine_shape_exclusions": "Shape-score-excluded",
    }.get(kind, "Annotated")
    subject = "internal standards" if section == "is" else "endogenous lipids"
    return f"{prefix} {subject}"


@dataclass(frozen=True)
class _PreparedFeatureReport:
    frame: pd.DataFrame
    traces: dict[str, list[SampleTrace]]
    spectra: dict[str, dict]
    groups: dict[str, str]
    roles: dict[str, str]
    colours: dict[str, str]
    parts: list[str]


def _prepare_feature_report(
    context: AnnotatedFeatureReportContext,
) -> _PreparedFeatureReport:
    if context.report_kind not in _REPORT_KINDS:
        raise ValueError(
            "report_kind must be annotations, msms_exclusions, grid_exclusions, "
            "boundary_apex_exclusions or mzmine_shape_exclusions"
        )
    frame = rank_identity_feature_matches(context.annotations)
    traces, spectra = context.traces or {}, context.spectra or {}
    groups, roles = context.sample_groups or {}, context.sample_roles or {}
    sample_names = sorted(
        {trace.sample_name for values in traces.values() for trace in values},
        key=lambda value: (groups.get(value, ""), value),
    )
    colours = sample_colours(sample_names, groups)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(context.title)}</title><style>{_STYLE}{LAZY_REPORT_STYLE}</style>",
        "<body>",
        f"<h1>{html.escape(context.title)}</h1>",
        f"<p class='meta'>{html.escape(context.subtitle)}</p>"
        if context.subtitle
        else "",
        f"<p class='explanation'>{_feature_report_explanation(context.report_kind)}</p>",
        *_feature_report_summary(context, frame),
    ]
    if colours:
        parts.append("<div class='samples'>")
        for sample, colour in colours.items():
            label = f"{groups.get(sample)} · {sample}" if groups.get(sample) else sample
            parts.append(
                f"<span class='sample-key'><span class='swatch' "
                f"style='background:{colour}'></span>{html.escape(label)}</span>"
            )
        parts.append("</div>")
    return _PreparedFeatureReport(frame, traces, spectra, groups, roles, colours, parts)


def render_annotated_feature_report_context(
    context: AnnotatedFeatureReportContext,
) -> str:
    """Render every feature-identity match from one typed context."""
    report_kind = context.report_kind
    lazy_rows = context.lazy_rows
    prepared = _prepare_feature_report(context)
    frame, traces, spectra = prepared.frame, prepared.traces, prepared.spectra
    roles, colours, parts = prepared.roles, prepared.colours, prepared.parts

    if frame.empty:
        empty_message = _feature_report_empty_message(report_kind)
        parts.append(f"<p class='empty-message'>{empty_message}</p></body>")
        return "".join(parts)

    frame["_is_internal_standard"] = _internal_standard_mask(frame)
    frame["_is_sort"] = (~frame["_is_internal_standard"]).astype(int)
    frame["_class"] = frame.get("class", "").astype(str)
    frame["_name"] = frame.get("name", "").astype(str)
    frame["_adduct"] = frame.get("adduct", "").astype(str)
    frame = frame.sort_values(
        ["_is_sort", "_class", "_name", "_adduct", "identity_match_rank"],
        kind="stable",
    )
    parts.extend(
        [
            "<div class='rows'>",
            "<div class='header'><div>Annotation and feature</div><div>Evidence metrics</div>"
            "<div>Chromatogram trace</div><div>Ion-mobility trace</div>"
            "<div>MS/MS spectrum</div></div>",
        ]
    )
    lazy_row_payload: list[dict[str, str]] = []
    lazy_sections: dict[str, str] = {}
    lazy_fragments: dict[str, str] = {}
    lazy_fragment_keys: dict[str, str] = {}

    def shared_fragment(markup: str) -> str:
        """Store repeated SVG evidence once in a compressed lazy payload."""
        if not lazy_rows:
            return markup
        key = lazy_fragment_keys.get(markup)
        if key is None:
            key = str(len(lazy_fragments))
            lazy_fragment_keys[markup] = key
            lazy_fragments[key] = markup
        return f"@@LASER_FRAGMENT:{key}@@"

    section_names: list[str] = []
    current_section = None
    for row in frame.to_dict(orient="records"):
        section = "is" if bool(row["_is_internal_standard"]) else "lipids"
        if section != current_section:
            section_label = _feature_section_label(report_kind, section)
            section_markup = (
                f"<div class='section-label {section}'>{section_label}</div>"
            )
            lazy_sections[section] = section_markup
            section_names.append(section_label)
            if not lazy_rows:
                parts.append(section_markup)
            current_section = section
        feature_markup = _feature_row_markup(
            row,
            traces=traces,
            spectra=spectra,
            roles=roles,
            colours=colours,
            shared_fragment=shared_fragment,
            report_kind=report_kind,
        )
        if lazy_rows:
            lazy_row_payload.append({"section": section, "html": feature_markup})
        else:
            parts.append(feature_markup)
    if lazy_rows:
        parts.append(
            lazy_feature_rows_markup(
                lazy_row_payload,
                lazy_sections,
                fragments=lazy_fragments,
                heading=" / ".join(section_names),
            )
        )
        parts.append(f"<script>{LAZY_REPORT_SCRIPT}</script>")
    parts.append("</div></body>")
    return "".join(parts)


def write_annotated_feature_report(
    annotations: pd.DataFrame,
    output: str | Path,
    *,
    trace_database: str | Path | None = None,
    spectra: dict[str, dict] | None = None,
    title: str = "Annotated feature evidence",
    subtitle: str = "",
    sample_groups: dict[str, str] | None = None,
    sample_roles: dict[str, str] | None = None,
    summed_chromatograms: list[SummedFeatureChromatogram] | None = None,
    is_trends: list[dict] | None = None,
    report_kind: str = "annotations",
    polarity: str = "",
    fragment_tolerance_da: float = DEFAULT_FRAGMENT_TOLERANCE_DA,
    fatty_acyl_min_relative_intensity_percent: float = (
        DEFAULT_FA_MIN_RELATIVE_INTENSITY_PERCENT
    ),
    grid_coherence_threshold: float = (SCIENTIFIC_DEFAULTS.grid_coherence_threshold),
    maximum_boundary_apex_ratio: float = (
        SCIENTIFIC_DEFAULTS.maximum_boundary_apex_ratio
    ),
    minimum_mzmine_shape_score: float = (
        SCIENTIFIC_DEFAULTS.minimum_mzmine_shape_score
    ),
) -> Path:
    """Write the report, including an explicit empty report for zero matches."""
    trace_cache: dict[str, list[SampleTrace]] = {}
    if (
        not annotations.empty
        and "feat_feature_id" in annotations
        and trace_database
        and Path(trace_database).is_file()
    ):
        for feature_id in annotations["feat_feature_id"].astype(str).unique():
            trace_cache[feature_id] = [
                SampleTrace(
                    sample_name=trace.sample_name,
                    is_representative=trace.is_representative,
                    rt_values=trace.rt_values,
                    rt_intensities=trace.rt_intensities,
                    mobility_values=trace.mobility_values,
                    mobility_intensities=trace.mobility_intensities,
                )
                for trace in load_feature_traces(trace_database, feature_id)
            ]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        destination,
        render_annotated_feature_report(
            annotations,
            traces=trace_cache,
            spectra=spectra,
            title=title,
            subtitle=subtitle,
            sample_groups=sample_groups,
            sample_roles=sample_roles,
            summed_chromatograms=summed_chromatograms,
            is_trends=is_trends,
            report_kind=report_kind,
            polarity=polarity,
            fragment_tolerance_da=fragment_tolerance_da,
            fatty_acyl_min_relative_intensity_percent=(
                fatty_acyl_min_relative_intensity_percent
            ),
            grid_coherence_threshold=grid_coherence_threshold,
            maximum_boundary_apex_ratio=maximum_boundary_apex_ratio,
            minimum_mzmine_shape_score=minimum_mzmine_shape_score,
            lazy_rows=True,
        ),
    )
    return destination


def write_msms_excluded_feature_report(
    annotations: pd.DataFrame,
    output: str | Path,
    **kwargs,
) -> Path:
    """Write the evidence list for enforced class-fragment exclusions."""
    return write_annotated_feature_report(
        annotations,
        output,
        report_kind="msms_exclusions",
        summed_chromatograms=[],
        is_trends=[],
        **kwargs,
    )


def write_grid_excluded_feature_report(
    annotations: pd.DataFrame,
    output: str | Path,
    **kwargs,
) -> Path:
    """Write the evidence list for active grid-coherence exclusions."""
    return write_annotated_feature_report(
        annotations,
        output,
        report_kind="grid_exclusions",
        summed_chromatograms=[],
        is_trends=[],
        **kwargs,
    )


def write_boundary_apex_excluded_feature_report(
    annotations: pd.DataFrame,
    output: str | Path,
    **kwargs,
) -> Path:
    """Write evidence for active boundary-to-apex-ratio exclusions."""
    return write_annotated_feature_report(
        annotations,
        output,
        report_kind="boundary_apex_exclusions",
        summed_chromatograms=[],
        is_trends=[],
        **kwargs,
    )


def write_mzmine_shape_excluded_feature_report(
    annotations: pd.DataFrame,
    output: str | Path,
    **kwargs,
) -> Path:
    """Write evidence for the exported MZmine shape-score criterion."""
    return write_annotated_feature_report(
        annotations,
        output,
        report_kind="mzmine_shape_exclusions",
        summed_chromatograms=[],
        is_trends=[],
        **kwargs,
    )
