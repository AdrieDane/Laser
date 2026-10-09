"""A scrollable contact sheet of traces for annotated lipids.

Reading peak shape off a table is impossible, and opening traces one at a time
in the viewer does not scale to a thousand annotations. This renders every
annotated lipid in one self-contained HTML page, so a whole run can be scanned
in a couple of minutes.

Four decisions make the page worth scrolling:

**Every sample is drawn, on its own axes.** Each panel carries a retention-time
axis and an intensity axis in scientific notation, so a trace can be read
absolutely rather than only compared with its neighbours. Samples share one
intensity scale within a panel, so their relative heights are real.

**Colour carries the study group, shade carries the sample.** Ten samples cannot
be given ten distinguishable hues — past a handful they stop being separable,
and colourblind readers lose them sooner. Hue therefore encodes the group and
lightness separates injections within it, which also makes a group difference
visible at a glance. A legend at the top names every sample.

**Every competing feature is drawn.** When several features match the same lipid
ID, all of them appear and the one the annotation picked is marked. A picked
feature that looks worse than its rivals is the most useful thing this page can
surface.

**Homologues sit next to each other.** Tiles are ordered by class, then carbon
number and double bonds, so a series reads as a row and a bad integration breaks
an otherwise regular run of shapes.

Everything is inline SVG. No JavaScript, no external requests, no image files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import html
from pathlib import Path

import numpy as np
import pandas as pd

from output_io import atomic_write_text

from .feature_traces import load_feature_traces

# Room for axis labels on every panel, while still fitting several per row.
TILE_WIDTH = 236
TILE_HEIGHT = 118
PLOT_LEFT = 44
PLOT_BOTTOM = 17
PLOT_TOP = 6
PLOT_RIGHT = 6

# One hue per sample, assigned in fixed order and never cycled while the list
# lasts. The first eight are the validated categorical palette; the last two
# extend it for a ten-injection batch. Past eight, hues stop being reliably
# separable for colour-vision deficiency, so the legend and the per-panel axes
# carry identity too rather than colour alone.
SAMPLE_HUES = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#8a5a2b",  # brown
    "#00868b",  # teal
)
# Beyond this many candidates for one lipid the tile stops being scannable, and
# the tail is almost always poor matches anyway. The count is still reported.
MAX_PANELS_PER_LIPID = 4


@dataclass(frozen=True)
class SampleTrace:
    """One raw file's trace for one feature."""

    sample_name: str
    is_representative: bool
    rt_values: np.ndarray
    rt_intensities: np.ndarray
    mobility_values: np.ndarray
    mobility_intensities: np.ndarray


@dataclass(frozen=True)
class TracePanel:
    """One candidate feature for a lipid, with every sample that carries it."""

    feature_id: str
    picked: bool
    detail: str
    samples: list[SampleTrace]
    expected_rt: float | None = None
    # The class-fragment verdict belongs on the panel rather than the tile:
    # competing features for one lipid frequently disagree, and which of them
    # carries the fragment is exactly what decides between them.
    msms_status: str = ""
    msms_rule: str = ""


@dataclass(frozen=True)
class GalleryTile:
    """One lipid identity and every feature that matched it."""

    label: str
    lipid_class: str
    sort_key: tuple
    flag: str = ""
    panels: list[TracePanel] = field(default_factory=list)
    candidate_count: int = 0


def _shade(hex_colour: str, factor: float) -> str:
    """Lighten or darken a hue, keeping it inside a readable band.

    Samples within a group are separated by lightness rather than hue. The
    factor stays between roughly 0.7 and 1.3 so no injection is rendered so pale
    it disappears against the surface or so dark it reads as a different series.
    """
    value = hex_colour.lstrip("#")
    red, green, blue = (float(int(value[i:i + 2], 16)) for i in (0, 2, 4))
    if factor >= 1.0:
        weight = min(factor - 1.0, 1.0)
        red, green, blue = (
            channel + (255 - channel) * weight * 0.72
            for channel in (red, green, blue)
        )
    else:
        red, green, blue = (channel * max(factor, 0.35) for channel in (red, green, blue))
    return "#%02x%02x%02x" % tuple(int(round(min(max(c, 0), 255))) for c in (red, green, blue))


def sample_colours(
    samples: list[str], groups: dict[str, str] | None = None
) -> dict[str, str]:
    """Assign one hue per sample, in fixed order.

    Colour follows the sample, not its rank in any particular panel, so a trace
    keeps the same hue on every tile in the page and a reader learns it once.
    Samples are ordered by study group first so the legend reads in blocks.

    Beyond ten samples the palette has to repeat; a repeated hue is darkened so
    two injections are never drawn identically, but at that point the legend is
    doing the real work.
    """
    groups = groups or {}
    ordered = sorted(samples, key=lambda name: (groups.get(name, ""), name))
    colours: dict[str, str] = {}
    for index, sample in enumerate(ordered):
        base = SAMPLE_HUES[index % len(SAMPLE_HUES)]
        lap = index // len(SAMPLE_HUES)
        colours[sample] = base if not lap else _shade(base, 0.68 ** lap)
    return colours


def _scientific(value: float) -> str:
    """Axis label in scientific notation, e.g. 2.4e5."""
    if not np.isfinite(value) or value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return f"{mantissa:.1f}e{exponent}"


def _overlay(
    samples: list[SampleTrace],
    *,
    mobility: bool,
    colours: dict[str, str],
    width: int = TILE_WIDTH,
    height: int = TILE_HEIGHT,
    marker: float | None = None,
) -> str:
    """Draw every sample on one pair of labelled axes.

    The intensity scale is shared across the samples in a panel, so a sample
    that is genuinely weaker looks weaker. Scaling each sample to its own
    maximum would make ten injections of different abundance look identical,
    which is the opposite of what a reviewer needs.
    """
    series = []
    for sample in samples:
        x = sample.mobility_values if mobility else sample.rt_values
        y = sample.mobility_intensities if mobility else sample.rt_intensities
        if len(x) < 2 or len(y) < 2:
            continue
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 2:
            continue
        series.append((x[finite], y[finite], sample.sample_name))
    if not series:
        return (
            f'<svg class="spark empty" viewBox="0 0 {width} {height}">'
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle">'
            "no trace</text></svg>"
        )
    low = min(float(x.min()) for x, _, _ in series)
    high = max(float(x.max()) for x, _, _ in series)
    peak = max(float(y.max()) for _, y, _ in series)
    span = max(high - low, 1e-9)
    peak = max(peak, 1e-9)

    plot_w = width - PLOT_LEFT - PLOT_RIGHT
    plot_h = height - PLOT_TOP - PLOT_BOTTOM
    def to_x(v):
        return PLOT_LEFT + (v - low) / span * plot_w
    def to_y(v):
        return PLOT_TOP + plot_h - (v / peak) * plot_h

    parts = [
        f'<line class="axis" x1="{PLOT_LEFT}" y1="{PLOT_TOP + plot_h}" '
        f'x2="{PLOT_LEFT + plot_w}" y2="{PLOT_TOP + plot_h}"/>',
        f'<line class="axis" x1="{PLOT_LEFT}" y1="{PLOT_TOP}" '
        f'x2="{PLOT_LEFT}" y2="{PLOT_TOP + plot_h}"/>',
    ]
    for fraction in (0.0, 0.5, 1.0):
        value = low + span * fraction
        x = to_x(value)
        anchor = "start" if fraction == 0 else "end" if fraction == 1 else "middle"
        parts.append(
            f'<line class="tick" x1="{x:.1f}" y1="{PLOT_TOP + plot_h}" '
            f'x2="{x:.1f}" y2="{PLOT_TOP + plot_h + 3}"/>'
            f'<text class="tick-label" x="{x:.1f}" y="{height - 5}" '
            f'text-anchor="{anchor}">{value:.2f}</text>'
        )
    for fraction in (0.0, 0.5, 1.0):
        value = peak * fraction
        y = to_y(value)
        parts.append(
            f'<line class="tick" x1="{PLOT_LEFT - 3}" y1="{y:.1f}" '
            f'x2="{PLOT_LEFT}" y2="{y:.1f}"/>'
            f'<text class="tick-label" x="{PLOT_LEFT - 5}" y="{y + 3:.1f}" '
            f'text-anchor="end">{_scientific(value)}</text>'
        )
    if marker is not None and np.isfinite(marker) and low <= marker <= high:
        mx = to_x(marker)
        parts.append(
            f'<line class="expected" x1="{mx:.1f}" y1="{PLOT_TOP}" '
            f'x2="{mx:.1f}" y2="{PLOT_TOP + plot_h}"/>'
        )
    for x, y, sample_name in series:
        points = " ".join(
            f"{to_x(a):.1f},{to_y(b):.1f}" for a, b in zip(x, y)
        )
        colour = colours.get(sample_name, "#2a78d6")
        parts.append(
            f'<polyline points="{points}" stroke="{colour}"/>'
        )
    axis_label = "1/K0" if mobility else "RT (min)"
    parts.append(
        f'<text class="axis-title" x="{PLOT_LEFT + plot_w / 2:.0f}" '
        f'y="{PLOT_TOP + 7}" text-anchor="middle">{axis_label}</text>'
    )
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}">'
        + "".join(parts) + "</svg>"
    )


def collect_tiles(
    annotations: pd.DataFrame,
    trace_database: str | Path,
    *,
    limit: int | None = None,
    max_panels: int = MAX_PANELS_PER_LIPID,
) -> list[GalleryTile]:
    """Group annotations by lipid, loading every sample trace for each feature."""
    if annotations.empty:
        return []
    frame = annotations.copy()
    for source, target in (("feat_feature_id", "feature_id"),
                           ("feat_rt", "rt"), ("feat_mz", "mz")):
        if source in frame and target not in frame:
            frame[target] = frame[source]
    if "feature_id" not in frame or "name" not in frame:
        return []
    # "class" is a Python keyword, so itertuples renames it and reading it back
    # by name silently yields nothing.
    frame = frame.rename(columns={"class": "lipid_class"})
    if "score" in frame:
        frame = frame.sort_values("score")

    cache: dict[str, list[SampleTrace]] = {}

    def traces_for(feature_id: str) -> list[SampleTrace]:
        if feature_id not in cache:
            cache[feature_id] = [
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
        return cache[feature_id]

    tiles: list[GalleryTile] = []
    for name, group in frame.groupby("name", sort=False):
        panels: list[TracePanel] = []
        for position, row in enumerate(group.itertuples(index=False)):
            if len(panels) >= max_panels:
                break
            feature_id = str(getattr(row, "feature_id", ""))
            samples = traces_for(feature_id)
            if not samples:
                continue
            mz = getattr(row, "mz", np.nan)
            rt = getattr(row, "rt", np.nan)
            expected = getattr(row, "rt_expected", np.nan)
            status = str(getattr(row, "msms_status", "") or "")
            panels.append(TracePanel(
                feature_id=feature_id,
                # The first row survives the score sort, so it is the feature
                # the annotation actually reports for this lipid.
                picked=position == 0,
                detail=" · ".join(part for part in (
                    f"#{feature_id}",
                    f"{float(rt):.2f} min" if np.isfinite(rt) else "",
                    f"{float(mz):.4f}" if np.isfinite(mz) else "",
                    f"{len(samples)} samples",
                ) if part),
                samples=samples,
                expected_rt=float(expected) if np.isfinite(expected) else None,
                msms_status=status,
                msms_rule=str(getattr(row, "msms_rule_id", "") or ""),
            ))
        if not panels:
            continue
        first = group.iloc[0]
        lipid_class = str(first.get("lipid_class", "") or "")
        carbon = pd.to_numeric(first.get("carbon"), errors="coerce")
        double_bonds = pd.to_numeric(first.get("db"), errors="coerce")
        status = str(first.get("msms_status", "") or "")
        tiles.append(GalleryTile(
            label=str(name),
            lipid_class=lipid_class,
            sort_key=(
                lipid_class,
                str(first.get("adduct", "") or ""),
                float(carbon) if pd.notna(carbon) else 1e9,
                float(double_bonds) if pd.notna(double_bonds) else 1e9,
            ),
            flag={"supported": "confirmed",
                  "not_detected": "contradicted"}.get(status, ""),
            panels=panels,
            candidate_count=len(group),
        ))
        if limit and len(tiles) >= limit:
            break
    tiles.sort(key=lambda tile: tile.sort_key)
    return tiles


_STYLE = """
body{font:12px -apple-system,BlinkMacSystemFont,'SF Pro Text',Arial,sans-serif;
 margin:0;padding:20px 24px;color:#0b0b0b;background:#fcfcfb}
h1{font-size:17px;margin:0 0 2px}
.meta{color:#52514e;margin-bottom:14px}
.legend{margin-bottom:14px;color:#52514e;max-width:74em;line-height:1.5}
.legend b{color:#0b0b0b;font-weight:600}
.samples{display:flex;flex-wrap:wrap;gap:4px 14px;margin:0 0 20px;
 padding:9px 12px;border:1px solid #e5e5ea;border-radius:7px;background:#fff}
.samples .key{display:flex;align-items:center;gap:5px;font-size:11px;color:#52514e}
.samples .swatch{width:14px;height:3px;border-radius:2px;display:inline-block}
.samples .group{font-weight:600;color:#0b0b0b;margin-right:2px}
h2{font-size:13px;margin:26px 0 8px;padding-bottom:4px;
 border-bottom:1px solid #e5e5ea;position:sticky;top:0;background:#fcfcfb}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.tile{border:1px solid #e5e5ea;border-radius:7px;padding:8px 9px 7px;
 background:#fff;break-inside:avoid}
.tile.confirmed{border-left:3px solid #1baf7a}
.tile.contradicted{border-left:3px solid #e34948}
.name{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rivals{color:#b25000;font-size:10px;margin-bottom:2px}
.panel{margin-top:5px}
.panel+.panel{border-top:1px dashed #e5e5ea;padding-top:5px}
.detail{color:#52514e;font-size:10px;white-space:nowrap;overflow:hidden;
 text-overflow:ellipsis}
.panel.picked .detail{color:#0b0b0b;font-weight:600}
.panel.picked .detail:before{content:"\2713 ";color:#1baf7a}
.panel.rival{opacity:.78}
.badge{float:right;font-size:9px;font-weight:600;padding:0 4px;border-radius:3px;
 line-height:14px;margin-left:6px}
.badge.pass{background:#e3f5ec;color:#0d6b46}
.badge.fail{background:#fde8e7;color:#a52725}
.badge.none{background:#f2f2ef;color:#7a7a75}
.badge.na{background:#f7f7f4;color:#a3a39d}
svg.spark{display:block;width:100%;height:auto}
svg.spark polyline{fill:none;stroke-width:1.2;stroke-linejoin:round}
svg.spark line.axis{stroke:#c7c7cc;stroke-width:.8}
svg.spark line.tick{stroke:#c7c7cc;stroke-width:.8}
svg.spark text.tick-label{fill:#7a7a75;font-size:7px}
svg.spark text.axis-title{fill:#a3a39d;font-size:7px}
svg.spark line.expected{stroke:#c7c7cc;stroke-width:1;stroke-dasharray:2 2}
svg.spark.empty text{fill:#c7c7cc;font-size:9px}
"""


_MSMS_BADGE = {
    "supported": ("pass", "MS2 rule passed"),
    "not_detected": ("fail", "MS2 acquired, class fragment absent"),
    "no_msms": ("none", "no MS2 spectrum"),
}


def _msms_badge(panel: "TracePanel") -> str:
    """State the class-fragment verdict on the panel itself.

    An edge colour on the tile cannot say which of several competing features
    carried the fragment, and that is usually the deciding evidence.
    """
    css, description = _MSMS_BADGE.get(
        panel.msms_status, ("na", "no rule configured for this class")
    )
    label = {"pass": "MS2 \u2713", "fail": "MS2 \u2717",
             "none": "MS2 \u2013", "na": "MS2 n/a"}[css]
    rule = f" ({html.escape(panel.msms_rule)})" if panel.msms_rule else ""
    return (
        f"<span class='badge {css}' title='{description}{rule}'>{label}</span>"
    )


def _sample_legend(colours: dict[str, str], groups: dict[str, str]) -> str:
    """Name every sample once, at the top, since tiles have no room for it."""
    if not colours:
        return ""
    keys = []
    current = None
    for sample, colour in colours.items():
        group = groups.get(sample, "")
        if group != current:
            current = group
            if group:
                keys.append(f"<span class='group'>{html.escape(group)}</span>")
        keys.append(
            f"<span class='key'><span class='swatch' style='background:{colour}'>"
            f"</span>{html.escape(sample)}</span>"
        )
    return "<div class='samples'>" + "".join(keys) + "</div>"


def render_gallery(
    tiles: list[GalleryTile],
    *,
    title: str = "Annotated lipid traces",
    subtitle: str = "",
    dimension: str = "chromatogram",
    sample_groups: dict[str, str] | None = None,
) -> str:
    """Render one dimension of the tiles as a self-contained HTML page.

    Chromatograms and mobilograms go to separate pages: they are read for
    different things, and interleaving them halves how many lipids fit on
    screen.
    """
    mobility = dimension == "mobilogram"
    order = list(dict.fromkeys(
        sample.sample_name
        for tile in tiles for panel in tile.panels for sample in panel.samples
    ))
    groups = sample_groups or {}
    order.sort(key=lambda name: (groups.get(name, ""), name))
    colours = sample_colours(order, groups)
    if not tiles:
        return (
            f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}"
            f"</title><style>{_STYLE}</style><h1>{html.escape(title)}</h1>"
            "<p class='meta'>No annotated lipid had a stored trace.</p>"
        )
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='meta'>{html.escape(subtitle)}</p>" if subtitle else "",
        "<p class='legend'>One tile per lipid, ordered by class then carbon "
        "number and double bonds, so homologues sit together and a bad "
        "integration breaks an otherwise regular run of shapes.<br>"
        "<b>Every sample is overlaid</b> on its own labelled axes — retention "
        "time across, intensity up in scientific notation — sharing one "
        "intensity scale per panel so relative heights are real. Each sample "
        "has its own colour, the same on every tile; the legend below names "
        "them.<br>"
        "<b>Where several features matched the same lipid</b>, each is shown "
        "and the one the annotation reports is ticked ✓ — a picked feature "
        "that looks worse than its rivals is worth investigating.<br>"
        "Each panel carries its class-fragment verdict: <b>MS2 \u2713</b> passed, "
        "<b>MS2 \u2717</b> spectrum acquired but fragment absent, "
        "<b>MS2 \u2013</b> no spectrum, <b>n/a</b> no rule for this class."
        + ("" if mobility else " Dashed line: predicted retention time.")
        + "</p>",
        _sample_legend(colours, groups),
    ]
    current_class = None
    for tile in tiles:
        if tile.lipid_class != current_class:
            if current_class is not None:
                parts.append("</div>")
            current_class = tile.lipid_class
            parts.append(
                f"<h2>{html.escape(current_class or 'unclassified')}</h2>"
                "<div class='grid'>"
            )
        parts.append(
            f"<div class='tile {tile.flag}'>"
            f"<div class='name'>{html.escape(tile.label)}</div>"
        )
        if len(tile.panels) > 1 or tile.candidate_count > len(tile.panels):
            shown = len(tile.panels)
            extra = (
                f", {tile.candidate_count - shown} more not shown"
                if tile.candidate_count > shown else ""
            )
            parts.append(
                f"<div class='rivals'>{tile.candidate_count} features "
                f"matched{extra}</div>"
            )
        for panel in tile.panels:
            css = "panel picked" if panel.picked else "panel rival"
            badge = _msms_badge(panel)
            parts.append(
                f"<div class='{css}'>"
                f"<div class='detail'>{html.escape(panel.detail)}{badge}</div>"
                + _overlay(
                    panel.samples, mobility=mobility, colours=colours,
                    marker=None if mobility else panel.expected_rt,
                )
                + "</div>"
            )
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def write_gallery_page(
    tiles: list[GalleryTile],
    output: str | Path,
    *,
    title: str,
    subtitle: str = "",
    dimension: str = "chromatogram",
    sample_groups: dict[str, str] | None = None,
) -> Path | None:
    """Write already-collected tiles, so both pages share one database read."""
    if not tiles:
        return None
    if dimension == "mobilogram" and not any(
        len(sample.mobility_values) > 1
        for tile in tiles for panel in tile.panels for sample in panel.samples
    ):
        return None
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        destination,
        render_gallery(
            tiles,
            title=title,
            subtitle=subtitle or f"{len(tiles):,} annotated lipids",
            dimension=dimension,
            sample_groups=sample_groups,
        ),
    )
    return destination


def write_trace_gallery(
    annotations: pd.DataFrame,
    trace_database: str | Path,
    output: str | Path,
    *,
    title: str = "Annotated lipid traces",
    subtitle: str = "",
    limit: int | None = None,
    dimension: str = "chromatogram",
) -> Path | None:
    """Build one gallery page and write it, returning the path or None."""
    tiles = collect_tiles(annotations, trace_database, limit=limit)
    return write_gallery_page(
        tiles, output, title=title, subtitle=subtitle, dimension=dimension
    )
