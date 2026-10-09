"""The standard LASER test report: one tabbed HTML for a whole run.

A run produces a dozen artefacts — two trace galleries, a species table, the
processing-QC report, the MZmine batch, the annotation criteria. Reviewing them
means opening a dozen files and remembering which settings produced which
numbers. This assembles all of it into one page, so a run can be judged, filed
and compared as a single object.

Tabs use hidden radio buttons and sibling selectors. A small embedded script
only decompresses and paginates large feature lists when their tab is opened;
there are still no external requests, so the page renders from a ZIP, email
attachment or shared drive years later.

The settings tabs are not decoration. A trace gallery is only interpretable
against the parameters that produced it, and keeping them in the same file is
what makes two runs comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import html
from pathlib import Path
import re

import pandas as pd

from output_io import atomic_write_text
from report_lazy import LAZY_REPORT_SCRIPT, LAZY_REPORT_STYLE

# Categorical slots 1 and 2 of the validated palette: two measures, one chart.
SPECIES_HUE = "#2a78d6"
FEATURES_HUE = "#eb6834"


@dataclass
class ReportTab:
    """One tab: a label, an anchor id, and a block of ready HTML."""

    key: str
    label: str
    body: str
    note: str = ""


@dataclass
class TestReport:
    """Everything one run wants to show, in display order."""

    __test__ = False

    title: str
    subtitle: str = ""
    tabs: list[ReportTab] = field(default_factory=list)
    # The verdict sits above the tabs, not inside one. A primary outcome
    # measure has to answer "did this run pass?" before anything else.
    status: str = ""
    headline: dict = field(default_factory=dict)
    gates: list = field(default_factory=list)
    comparison: pd.DataFrame | None = None
    provenance: dict = field(default_factory=dict)


def _table_html(frame: pd.DataFrame, *, highlight_last: bool = False) -> str:
    if frame.empty:
        return "<p class='empty'>No rows.</p>"
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    rows = []
    for position, (_, row) in enumerate(frame.iterrows()):
        css = (
            " class='total'"
            if highlight_last and position == len(frame) - 1 else ""
        )
        cells = "".join(
            f"<td>{html.escape('' if pd.isna(v) else str(v))}</td>" for v in row
        )
        rows.append(f"<tr{css}>{cells}</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def species_bar_chart(
    table: pd.DataFrame,
    *,
    species_column: str = "identified_species",
    features_column: str = "identified_features",
    class_column: str = "class",
    top: int = 30,
) -> str:
    """Horizontal bars containing only unique lipid IDs per class."""
    data = table[table[class_column].astype(str).ne("TOTAL")].copy()
    if data.empty:
        return "<p class='empty'>No annotated classes.</p>"
    data = data.sort_values(species_column, ascending=False).head(top)
    peak = float(max(data[species_column].max(), 1))
    row_height, gap, label_width, plot_width = 26, 6, 128, 520
    height = len(data) * (row_height + gap) + 34
    bars = []
    for index, (_, row) in enumerate(data.iterrows()):
        y = index * (row_height + gap) + 26
        name = html.escape(str(row[class_column]))
        value = float(row[species_column] or 0)
        width = value / peak * plot_width
        bars.append(
            f'<rect x="{label_width}" y="{y:.0f}" '
            f'width="{max(width, 0.6):.1f}" height="{row_height - 3:.1f}" '
            f'rx="2" fill="{SPECIES_HUE}"/>'
            f'<text class="value" x="{label_width + width + 4:.1f}" '
            f'y="{y + row_height - 8:.0f}">{int(value)}</text>'
        )
        bars.append(
            f'<text class="cat" x="{label_width - 6}" y="{y + row_height - 8}" '
            f'text-anchor="end">{name}</text>'
        )
    legend = (
        f'<rect x="{label_width}" y="4" width="9" height="9" rx="2" '
        f'fill="{SPECIES_HUE}"/>'
        f'<text class="legend" x="{label_width + 13}" y="12">unique lipid IDs</text>'
    )
    return (
        f'<svg class="bars" viewBox="0 0 {label_width + plot_width + 54} {height}" '
        f'role="img" aria-label="Unique lipid IDs per lipid '
        f'class">{legend}{"".join(bars)}</svg>'
    )


def feature_filter_bar_chart(
    table: pd.DataFrame,
    *,
    all_column: str = "all_unique_features",
    excluded_column: str = "excluded_unique_features",
    class_column: str = "class",
    top: int = 30,
) -> str:
    """Show all unique candidate features and the criteria-excluded subset."""
    required = {all_column, excluded_column, class_column}
    if not required.issubset(table.columns):
        return ""
    data = table[table[class_column].astype(str).ne("TOTAL")].copy()
    if data.empty:
        return "<p class='empty'>No candidate features.</p>"
    data = data.sort_values(all_column, ascending=False).head(top)
    peak = float(max(data[all_column].max(), 1))
    row_height, gap, label_width, plot_width = 26, 6, 128, 520
    height = len(data) * (row_height + gap) + 34
    bars = []
    for index, (_, row) in enumerate(data.iterrows()):
        y = index * (row_height + gap) + 26
        name = html.escape(str(row[class_column]))
        total = float(row[all_column] or 0)
        rejected = float(row[excluded_column] or 0)
        total_width = total / peak * plot_width
        rejected_width = rejected / peak * plot_width
        bars.append(
            f'<rect x="{label_width}" y="{y:.0f}" '
            f'width="{max(total_width, 0.6):.1f}" height="{row_height - 3:.1f}" '
            'rx="2" fill="#9fc3e8"/>'
            f'<rect x="{label_width}" y="{y:.0f}" '
            f'width="{max(rejected_width, 0.0):.1f}" height="{row_height - 3:.1f}" '
            'rx="2" fill="#c94c4c"/>'
            f'<text class="value" x="{label_width + total_width + 4:.1f}" '
            f'y="{y + row_height - 8:.0f}">all {int(total)} · excluded '
            f'{int(rejected)}</text>'
            f'<text class="cat" x="{label_width - 6}" '
            f'y="{y + row_height - 8}" text-anchor="end">{name}</text>'
        )
    legend = (
        f'<rect x="{label_width}" y="4" width="9" height="9" rx="2" '
        'fill="#9fc3e8"/><text class="legend" '
        f'x="{label_width + 13}" y="12">all unique candidate features</text>'
        f'<rect x="{label_width + 205}" y="4" width="9" height="9" rx="2" '
        'fill="#c94c4c"/><text class="legend" '
        f'x="{label_width + 218}" y="12">fully criteria-excluded subset</text>'
    )
    return (
        f'<svg class="bars" viewBox="0 0 {label_width + plot_width + 120} {height}" '
        f'role="img" aria-label="All and criteria-excluded unique features per '
        f'lipid class">{legend}{"".join(bars)}</svg>'
    )


def _headline_block(headline: dict) -> str:
    """The pair a run is judged on.

    Species count alone rises when quality falls, so it is never shown without
    the confirmation rate beside it.
    """
    if not headline:
        return ""
    confirmation = headline.get("confirmation_fraction")
    cards = [
        ("Unique species", f"{headline.get('unique_species', 0):,}", ""),
        (
            "Class fragment confirmed",
            "—" if confirmation is None else f"{confirmation:.1%}",
            f"{headline.get('confirmed', 0):,} of "
            f"{headline.get('testable', 0):,} testable",
        ),
        (
            "Candidate features (all)",
            f"{headline.get('all_candidate_unique_features', 0):,}",
            "before active annotation criteria",
        ),
        (
            "Criteria-excluded features",
            f"{headline.get('criteria_excluded_unique_features', 0):,}",
            "unique features with no retained match",
        ),
        (
            "Annotated features (retained)",
            f"{headline.get('annotated_features', 0):,}",
            "reported after criteria",
        ),
    ]
    ambiguity = headline.get("ambiguity_fraction")
    if ambiguity is not None:
        cards.append(("Unresolved features", f"{ambiguity:.1%}", ""))
    expected = headline.get("internal_standards_expected") or 0
    if expected:
        cards.append((
            "Internal standards",
            f"{headline.get('internal_standards_found', 0)}/{expected}", "",
        ))
    return "<div class='cards'>" + "".join(
        f"<div class='card'><div class='k'>{html.escape(label)}</div>"
        f"<div class='v'>{html.escape(value)}</div>"
        + (f"<div class='sub'>{html.escape(sub)}</div>" if sub else "")
        + "</div>"
        for label, value, sub in cards
    ) + "</div>"


def _gate_block(gates: list) -> str:
    if not gates:
        return ""
    rows = []
    for gate in gates:
        status = gate.status
        value = gate.value
        shown = (
            "—" if value is None
            else f"{value:.1%}" if isinstance(value, float) and abs(value) <= 1
            else f"{value:,.0f}" if isinstance(value, (int, float))
            else str(value)
        )
        rows.append(
            f"<tr class='{status.lower()}'>"
            f"<td class='gate-status'><span class='pill {status.lower()}'>"
            f"{status}</span></td>"
            f"<td class='gate-name'>{html.escape(gate.label)}</td>"
            f"<td>{shown}</td>"
            f"<td class='gate-detail'>{html.escape(gate.detail)}</td></tr>"
        )
    return (
        "<table class='gates'><thead><tr><th></th><th>gate</th><th>value</th>"
        f"<th>why it matters</th></tr></thead><tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _provenance_block(provenance: dict) -> str:
    """What this run was, so two reports can be known to be comparable."""
    if not provenance:
        return ""
    rows = "".join(
        f"<tr><td class='key'>{html.escape(str(k))}</td>"
        f"<td>{html.escape(str(v))}</td></tr>"
        for k, v in provenance.items() if not isinstance(v, (dict, list))
    )
    return f"<table class='settings'><tbody>{rows}</tbody></table>"


def _settings_table(settings: dict, *, skip: tuple[str, ...] = ()) -> str:
    rows = [
        (key, value) for key, value in sorted(settings.items())
        if key not in skip and not isinstance(value, (dict, list))
    ]
    if not rows:
        return "<p class='empty'>Not recorded for this run.</p>"
    body = "".join(
        f"<tr><td class='key'>{html.escape(str(k))}</td>"
        f"<td>{html.escape(str(v))}</td></tr>" for k, v in rows
    )
    return f"<table class='settings'><tbody>{body}</tbody></table>"


def embedded_document(path: Path | None) -> str:
    """Inline another self-contained report, keeping only its body.

    Nesting a whole document would give the page a second <head> and a second
    set of global styles that would leak into every tab.
    """
    if path is None or not Path(path).is_file():
        return "<p class='empty'>Not produced for this run.</p>"
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    body = re.search(r"<body[^>]*>(.*)</body>", text, re.S | re.I)
    inner = body.group(1) if body else text
    # The embedded report's own styles are namespaced into the tab so they
    # cannot restyle the surrounding page.
    styles = "\n".join(
        re.findall(r"<style[^>]*>(.*?)</style>", text, re.S | re.I)
    )
    scoped = re.sub(
        r"(^|\})\s*([^@{}]+)\{", lambda m: f"{m.group(1)} .embedded {m.group(2)}{{",
        styles,
    ) if styles else ""
    inner = re.sub(r"<script.*?</script>", "", inner, flags=re.S | re.I)
    return f"<style>{scoped}</style><div class='embedded'>{inner}</div>"


_STYLE = """
:root{color-scheme:light}
body{font:15px -apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",
 "Segoe UI",Arial,sans-serif;margin:0;color:#0b0b0b;background:#fcfcfb}
header{padding:18px 26px 0}
h1{font-size:23px;margin:0 0 3px}
.meta{color:#52514e;margin:0 0 14px}
.tabs{display:flex;flex-wrap:wrap;gap:2px;padding:0 26px;
 border-bottom:1px solid #e5e5ea;background:#fcfcfb;position:sticky;top:0;z-index:5}
.tabs label{padding:10px 16px;font-size:14px;font-weight:600;color:#52514e;
 cursor:pointer;border:1px solid transparent;border-bottom:none;
 border-radius:7px 7px 0 0;margin-bottom:-1px}
.tabs label:hover{color:#0b0b0b;background:#f2f2ef}
input.tab{position:absolute;opacity:0;pointer-events:none}
.panel{display:none;padding:20px 26px 60px}
.panel .note{color:#52514e;max-width:74em;line-height:1.55;margin:0 0 16px}
table{border-collapse:collapse;font-size:14px;margin:0 0 18px}
th,td{border:1px solid #e5e5ea;padding:4px 9px;text-align:right}
th{background:#f2f2ef;font-weight:600;text-align:right}
th:first-child,td:first-child{text-align:left}
tr.total td{font-weight:700;background:#f7f7f4}
table.settings td.key{color:#52514e;text-align:left}
svg.bars{max-width:760px;width:100%;height:auto;margin-bottom:20px}
svg.bars text.cat{font-size:13px;fill:#0b0b0b}
svg.bars text.value{font-size:12px;fill:#52514e}
svg.bars text.legend{font-size:13px;fill:#52514e}
.empty{color:#8a8a85}
.embedded{border-top:1px solid #e5e5ea;padding-top:14px}
.verdict{padding:0 26px 16px}
.status{display:inline-block;font-size:15px;font-weight:700;letter-spacing:.04em;
 padding:5px 13px;border-radius:6px;margin-bottom:12px}
.status.pass{background:#e3f5ec;color:#0d6b46}
.status.fail{background:#fde8e7;color:#a52725}
.status.not_evaluable{background:#f2f2ef;color:#52514e}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:14px}
.card{border:1px solid #e5e5ea;border-radius:7px;padding:9px 14px;background:#fff;
 min-width:132px}
.card .k{font-size:13px;color:#52514e}
.card .v{font-size:25px;font-weight:700;line-height:1.25}
.card .sub{font-size:12px;color:#8a8a85}
table.gates{font-size:14px;width:100%;max-width:1000px}
table.gates td,table.gates th{text-align:left}
table.gates td.gate-detail{color:#52514e;font-size:13px}
.pill{font-size:12px;font-weight:700;padding:2px 7px;border-radius:3px}
.pill.pass{background:#e3f5ec;color:#0d6b46}
.pill.fail{background:#fde8e7;color:#a52725}
.pill.not_evaluable{background:#f2f2ef;color:#7a7a75}
tr.fail td.gate-name{font-weight:700}
h3{font-size:14px;margin:16px 0 6px;color:#52514e;text-transform:uppercase;
 letter-spacing:.05em}
"""


def render_report(report: TestReport) -> str:
    """One page, one radio group, one panel visible at a time."""
    if not report.tabs:
        report = TestReport(report.title, report.subtitle, [
            ReportTab("empty", "No content", "<p class='empty'>Nothing to show.</p>")
        ])
    rules = "\n".join(
        f"#tab-{tab.key}:checked ~ .panel-{tab.key}{{display:block}}\n"
        f"#tab-{tab.key}:checked ~ .tabs label[for=tab-{tab.key}]"
        "{color:#0b0b0b;background:#fff;border-color:#e5e5ea}"
        for tab in report.tabs
    )
    inputs = "".join(
        f'<input class="tab" type="radio" name="tab" id="tab-{tab.key}"'
        f'{" checked" if index == 0 else ""}>'
        for index, tab in enumerate(report.tabs)
    )
    labels = "".join(
        f'<label for="tab-{tab.key}">{html.escape(tab.label)}</label>'
        for tab in report.tabs
    )
    panels = "".join(
        f'<section class="panel panel-{tab.key}">'
        + (f"<p class='note'>{tab.note}</p>" if tab.note else "")
        + tab.body + "</section>"
        for tab in report.tabs
    )
    verdict = ""
    if report.status or report.headline or report.gates:
        parts = []
        if report.status:
            parts.append(
                f"<div class='status {report.status.lower()}'>"
                f"{html.escape(report.status)}</div>"
            )
        parts.append(_headline_block(report.headline))
        if report.comparison is not None and not report.comparison.empty:
            parts.append("<h3>Against the previous run</h3>")
            parts.append(_table_html(report.comparison))
        if report.gates:
            parts.append("<h3>Release gates</h3>")
            parts.append(_gate_block(report.gates))
        if report.provenance:
            parts.append("<h3>Provenance</h3>")
            parts.append(_provenance_block(report.provenance))
        verdict = f"<div class='verdict'>{''.join(parts)}</div>"

    return (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{html.escape(report.title)}</title>"
        f"<style>{_STYLE}\n{LAZY_REPORT_STYLE}\n{rules}</style>"
        f"<header><h1>{html.escape(report.title)}</h1>"
        + (f"<p class='meta'>{html.escape(report.subtitle)}</p>"
           if report.subtitle else "")
        + "</header>"
        + verdict
        # The radios precede the tab bar and the panels so sibling selectors
        # can reach both.
        + inputs
        + f'<nav class="tabs">{labels}</nav>'
        + panels
        + "<script>window.LASER_LAZY_DEFER_HIDDEN=true;</script>"
        + f"<script>{LAZY_REPORT_SCRIPT}</script>"
    )


def write_report(report: TestReport, output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, render_report(report))
    return destination
