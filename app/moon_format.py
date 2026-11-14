"""Moon annotation documents (``.moon``, ``moon-annotation/1``) for LASER.

Moon (Python around the MATLAB MS pipeline) exchanges annotations as a small
JSON document; ``.moon`` is the same JSON with a user-facing extension. This
module lets LASER write and read that document directly:

* **export**: final LASER annotations -> ``.moon`` that Moon's
  ``setannotation`` puts into the MATLAB project as seed peaks (inliers);
* **import**: a ``.moon`` (from MATLAB through Moon ``getannotation``, or from
  LASER itself) -> LASER-style rows, or the manual crosswalk read by
  ``standard_lipidomics.validation.compare_manual_annotations``.

Shared contract (Moon ``docs/annotation_json.md`` and ``docs/moon_file.md``):
a UTF-8 JSON object, optional ``"format": "moon-annotation/1"`` and a
``features`` list. Each feature needs ``mode`` (RP_Pos, RP_Neg, NP_Pos,
NP_Neg), ``feature`` (Moon's ``FT<n>``), ``name`` (pipeline name; suffix
``(IS)`` marks an internal standard) and ``adduct`` unless ``set_as`` is
``inlier``. Any other field is allowed and kept; LASER-specific details go
under ``laser`` (per feature and per document). Moon's validator remains
authoritative; :func:`validate_moon` checks only this shared minimum.

Feature IDs: Moon's ``FT<n>`` is the 1-based row of the peak-group list (or
its ``feature`` column when present); LASER's ``feat_feature_id`` is the first
``name``/``feature``/``feature_id``/``id`` column. Both are mapped through the
same peak-group lists, so documents only work with the project they were
made for. The export mirrors Moon's own converter (``moon/laser.py``) so both
give the same document for the same run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import pandas as pd

from output_io import atomic_write_text


FORMAT = "moon-annotation/1"
SOURCE = "Laser"
STREAMS = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")
SET_AS = ("inlier", "accepted", "data")
ROLES = ("primary", "equivalent", "other")
MZ_TOL = 0.001          # Da: LASER and Moon read the same peak-group list
RT_TOL = 0.6            # s
MAPPED_STATUS = "exact_feature_id"   # crosswalk status for Moon feature IDs

_IS_SUFFIX = re.compile(r"\s*\(IS\)\s*$", re.IGNORECASE)
_ADDUCT = re.compile(r"^\[.+\](?:\d*[+-]|\(\d+[+-]\))$")
_FT = re.compile(r"\s*FT0*(\d+)\s*", re.IGNORECASE)
_STREAM = re.compile(r"\s*(NP|RP)[\s_]*(pos|positive|\+|neg|negative|-)\s*", re.IGNORECASE)


# ----------------------------------------------------------------- helpers
def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def _first(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = _text(row.get(name))
        if value != "":
            return value
    return ""


def _float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def split_is(name: str) -> tuple[str, bool]:
    """``'PC(28:0) (IS)'`` -> ``('PC(28:0)', True)``, as Moon does."""
    stripped = _IS_SUFFIX.sub("", name)
    return stripped.strip(), stripped != name


def stream_name(value: str) -> str | None:
    """``RP_Pos``, ``RP+``, ``NP negative`` ... -> ``RP_Pos``/``NP_Neg``; else None."""
    m = _STREAM.fullmatch(_text(value))
    if not m:
        return None
    sign = m.group(2).lower()
    return f"{m.group(1).upper()}_{'Pos' if sign in ('pos', 'positive', '+') else 'Neg'}"


def feature_number(feature: str) -> int | None:
    m = _FT.fullmatch(_text(feature))
    return int(m.group(1)) if m else None


def moon_feature_id(n: int, total: int) -> str:
    """Moon's ID for row ``n`` of ``total``: ``FT`` + number padded to the digits of ``total``."""
    return f"FT{n:0{len(str(total))}d}"


# ----------------------------------------------------------- peak-group lists
@dataclass(frozen=True)
class PeakRow:
    stream: str
    feature: str        # Moon FT<n>
    laser_id: str       # LASER feat_feature_id
    mz: float
    rt_seconds: float
    rt_raw: float       # as in the list (LASER's feat_rt unit)
    row: int            # 1-based row in the list


def _cells(line: str) -> list[str]:
    return [re.sub(r'^"|"$', "", c) for c in line.split("\t")]


def read_peaklist(path: str | Path, stream: str) -> list[PeakRow]:
    """Rows of one tab-separated XCMS peak-group list, with both IDs."""
    with Path(path).open("rb") as handle:
        header = _cells(handle.readline().decode("latin-1").rstrip("\r\n"))
        lines = [raw.decode("latin-1").rstrip("\r\n") for raw in handle]
    lines = [line for line in lines if line]
    low = [h.strip().casefold() for h in header]
    i_laser = next((i for i, h in enumerate(low) if h in ("name", "feature", "feature_id", "id")), None)
    i_moon = header.index("feature") if "feature" in header else None   # Moon: exact header
    i_mz, i_rt = header.index("mzmed"), header.index("rtmed")
    parsed = []
    for line in lines:
        cells = _cells(line)
        mz = float(cells[i_mz]) if cells[i_mz] != "NA" else 0.0   # MATLAB: NA -> 0
        rt = float(cells[i_rt]) if cells[i_rt] != "NA" else 0.0
        parsed.append((cells, mz, rt))
    scale = 60.0 if parsed and max(rt for _, _, rt in parsed) < 60 else 1.0   # minutes -> s
    rows = []
    for k, (cells, mz, rt) in enumerate(parsed, start=1):
        feature = cells[i_moon] if i_moon is not None else moon_feature_id(k, len(parsed))
        laser_id = cells[i_laser].strip() if i_laser is not None and i_laser < len(cells) else ""
        rows.append(PeakRow(stream, feature, laser_id, mz, rt * scale, rt, k))
    return rows


class PeakIndex:
    """Both ID systems of a project's peak-group lists, per stream."""

    def __init__(self, peaklists: Mapping[str, str | Path]):
        self.by_laser: dict[str, dict[str, PeakRow]] = {}
        self.by_number: dict[str, dict[int, PeakRow]] = {}
        for stream, path in peaklists.items():
            rows = read_peaklist(path, stream)
            self.by_laser[stream] = {r.laser_id.removesuffix(".0"): r for r in rows if r.laser_id}
            self.by_number[stream] = {n: r for r in rows if (n := feature_number(r.feature)) is not None}

    @classmethod
    def from_project(cls, project: str | Path) -> "PeakIndex":
        from standard_lipidomics.workflow import discover_standard_streams
        return cls(discover_standard_streams(project))

    def from_laser(self, stream: str, laser_id: str) -> PeakRow | None:
        return self.by_laser.get(stream, {}).get(_text(laser_id).removesuffix(".0"))

    def from_moon(self, stream: str, feature: str) -> PeakRow | None:
        n = feature_number(feature)
        return None if n is None else self.by_number.get(stream, {}).get(n)


def _index(peaks: "PeakIndex | Mapping[str, str | Path] | str | Path") -> PeakIndex:
    if isinstance(peaks, PeakIndex):
        return peaks
    if isinstance(peaks, Mapping):
        return PeakIndex(peaks)
    return PeakIndex.from_project(peaks)


# ------------------------------------------------------------------- export
def annotations_to_moon(
    annotations: pd.DataFrame | Iterable[Mapping[str, Any]],
    peaks: "PeakIndex | Mapping[str, str | Path] | str | Path",
    *,
    source: str = SOURCE,
    laser_info: Mapping[str, Any] | None = None,
    created: str | None = None,
) -> tuple[dict, dict]:
    """Final LASER annotations -> (``.moon`` document, report).

    Only M+0 rows are written, as seed peaks (``set_as: inlier``) with
    interpolation off; Moon maps and the user accepts in the MATLAB GUI.
    Rows that cannot be placed on a peak-group-list row are reported, never
    guessed. The document matches Moon's own LASER converter.
    """
    rows = (annotations.to_dict("records") if isinstance(annotations, pd.DataFrame)
            else [dict(r) for r in annotations])
    index = _index(peaks)
    feats: list[dict] = []
    skipped: list[dict] = []
    seen: dict[tuple[str, str], int] = {}
    per_mode: dict[str, int] = {}

    def skip(row, reason, **extra):
        skipped.append({"reason": reason, "stream": _text(row.get("stream")) or None,
                        "feature_id": _first(row, "feature_id", "feat_feature_id") or None,
                        "name": _first(row, "name", "feature_identity", "species") or None,
                        "adduct": _text(row.get("adduct")) or None, **extra})

    for row in rows:
        mode = next((m for v in (row.get("stream"), row.get("standard_stream"), row.get("mode"))
                     if _text(v) and (m := stream_name(_text(v)))), None)
        if mode is None:
            skip(row, "unknown_stream")
            continue
        raw_name = _first(row, "name", "feature_identity", "species")
        adduct = _first(row, "adduct")
        if not raw_name or not adduct:
            skip(row, "no_name_or_adduct")
            continue
        isotope = _first(row, "isotopologue", "isotope")
        if isotope not in ("", "M+0", "M0"):
            skip(row, "isotope", isotope=isotope)
            continue
        peak = index.from_laser(mode, _first(row, "feature_id", "feat_feature_id"))
        if peak is None:
            skip(row, "unknown_feature")
            continue
        mz = _float(_first(row, "feat_mz", "mz", "mzmed"))
        rt = _float(_first(row, "feat_rt", "rt_min", "rt", "rtmed"))
        if mz is not None and abs(mz - peak.mz) > MZ_TOL:
            skip(row, "mz_mismatch", feature=peak.feature, mz=mz, peaklist_mz=peak.mz)
            continue
        if rt is not None and min(abs(rt * 60 - peak.rt_seconds), abs(rt - peak.rt_seconds)) > RT_TOL:
            skip(row, "rt_mismatch", feature=peak.feature, rt=rt, peaklist_rt_s=peak.rt_seconds)
            continue
        name, is_from_name = split_is(raw_name)
        is_is = is_from_name or _truthy(_first(row, "is_is", "internal_standard"))
        key = (mode, peak.feature)
        if key in seen:
            prev = feats[seen[key]]
            if (prev["name"], prev["adduct"]) != (name, adduct):
                skip(row, "duplicate_feature", feature=peak.feature, kept=prev["name"])
            continue
        laser = {k: v for k, v in {
            "role": _first(row, "quantitative_role"),
            "status": _first(row, "auto_annotation_status", "annotation_status", "status"),
            "ppm": _float(_first(row, "ppm_error", "ppm")),
            # LASER's own MATLAB coding (for checking only; MATLAB codes itself)
            "matlab_compound": _float(_first(row, "matlab_compound")),
            "matlab_code": _float(_first(row, "matlab_code")),
            "name": raw_name if raw_name != name else None,
        }.items() if v not in (None, "")}
        seen[key] = len(feats)
        feats.append({"mode": mode, "feature": peak.feature, "name": name, "adduct": adduct,
                      "internal_standard": is_is, "mz": round(peak.mz, 5), "rt": round(peak.rt_seconds, 3),
                      "set_as": "inlier", "laser": laser})
        per_mode[mode] = per_mode.get(mode, 0) + 1
    reasons: dict[str, int] = {}
    for s in skipped:
        reasons[s["reason"]] = reasons.get(s["reason"], 0) + 1
    report = {"rows": len(rows), "features": len(feats), "per_mode": dict(sorted(per_mode.items())),
              "internal_standards": sum(f["internal_standard"] for f in feats),
              "skipped_by_reason": reasons, "skipped": skipped}
    info = {**dict(laser_info or {}), "rows": report["rows"], "skipped_by_reason": reasons}
    doc = {"format": FORMAT, "source": source,
           "created": created or datetime.now().isoformat(timespec="seconds"),
           "set_as": "inlier", "interpolation": False, "laser": info, "features": feats}
    return doc, report


def write_moon(document: Mapping[str, Any], path: str | Path) -> Path:
    """Write as Moon does: UTF-8, indent 2, non-ASCII kept. ``.moon`` or ``.json``."""
    problems = validate_moon(document)
    if problems:
        raise ValueError("Not a valid Moon annotation: " + "; ".join(problems[:5]))
    return atomic_write_text(Path(path), json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------- import
def validate_moon(data: Any) -> list[str]:
    """Problems with the shared minimum contract (empty list: fine)."""
    if not isinstance(data, Mapping):
        return [f"expected a JSON object, not {type(data).__name__}"]
    problems = []
    if data.get("format", FORMAT) != FORMAT:
        problems.append(f"unknown format {data.get('format')!r} (expected {FORMAT})")
    if not isinstance(data.get("features"), list):
        return problems + ["'features' must be a list"]
    default = data.get("set_as") or "inlier"
    if default not in SET_AS:
        problems.append(f"set_as {default!r}: expected one of {SET_AS}")
    for i, f in enumerate(data["features"]):
        if not isinstance(f, Mapping):
            problems.append(f"features[{i}]: expected an object")
            continue
        set_as = f.get("set_as") or ("accepted" if f.get("accepted") is True else default)
        required = ("mode", "feature", "name") + (() if set_as == "inlier" else ("adduct",))
        missing = [k for k in required if not f.get(k)]
        where = f"features[{i}] ({f.get('mode')} {f.get('feature')})"
        if missing:
            problems.append(f"{where}: missing {', '.join(missing)}")
        if f.get("set_as") is not None and f["set_as"] not in SET_AS:
            problems.append(f"{where}: set_as {f['set_as']!r}")
        if f.get("role") is not None and f["role"] not in ROLES:
            problems.append(f"{where}: role {f['role']!r}")
        adduct = _text(f.get("adduct")).strip()
        if adduct and not _ADDUCT.match(adduct):
            problems.append(f"{where}: adduct {adduct!r} is not like [M+H]+ / [M-2H](2-)")
        if f.get("mode") and stream_name(_text(f["mode"])) is None:
            problems.append(f"{where}: unknown mode {f['mode']!r}")
    return problems


def read_moon(path: str | Path) -> dict:
    """Read a ``.moon``/``.json`` annotation (BOM allowed) and check the shared minimum."""
    raw = Path(path).read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    problems = validate_moon(data)
    if problems:
        raise ValueError(f"{Path(path).name}: not a valid Moon annotation: " + "; ".join(problems[:5]))
    return data


def _class_names(doc: Mapping[str, Any]) -> dict[int, str]:
    """``i_compound`` -> class name from a getannotation export's ``classes``."""
    out = {}
    for c in list(doc.get("classes") or []) + list(doc.get("class_state") or []):
        if isinstance(c, Mapping) and c.get("i_compound") is not None and c.get("class"):
            out[int(c["i_compound"])] = str(c["class"])
    return out


def lipid_class(species: str) -> str:
    """Class from a pipeline name: the part before the first ``(``."""
    return species.split("(", 1)[0].strip() or species


def moon_to_frame(doc: Mapping[str, Any], peaks: "PeakIndex | Mapping[str, str | Path] | str | Path") -> pd.DataFrame:
    """One row per document feature in LASER terms (``stream``, ``name``,
    ``class``, ``adduct``, ``feat_feature_id`` ...). ``moon_status`` is
    ``mapped`` or ``unknown_feature`` (no such row in this project's lists)."""
    index = _index(peaks)
    classes = _class_names(doc)
    default_source = doc.get("source")
    records = []
    for f in doc.get("features") or []:
        stream = stream_name(_text(f.get("mode"))) or _text(f.get("mode"))
        species, is_from_name = split_is(_text(f.get("name")))
        peak = index.from_moon(stream, _text(f.get("feature")))
        matlab = f.get("matlab") if isinstance(f.get("matlab"), Mapping) else {}
        isotope = _text(f.get("isotope")).strip()
        records.append({
            "stream": stream,
            "name": species,
            "class": classes.get(int(matlab["i_compound"])) if matlab.get("i_compound") is not None
                     and int(matlab["i_compound"]) in classes else lipid_class(species),
            "adduct": _text(f.get("adduct")),
            "isotopologue": "" if isotope in ("", "M+0", "M0", "[]") else isotope,
            "is_is": bool(f.get("internal_standard")) or is_from_name,
            "feat_feature_id": peak.laser_id if peak else "",
            "feat_mz": peak.mz if peak else None,
            "feat_rt": peak.rt_raw if peak else None,
            "moon_feature": _text(f.get("feature")),
            "moon_status": "mapped" if peak else "unknown_feature",
            "moon_matlab_row": matlab.get("row"),
            "mz": f.get("mz"),
            "rt_seconds": f.get("rt"),
            "source": f.get("source") or default_source,
            "role": f.get("role"),
            "set_as": f.get("set_as") or doc.get("set_as"),
            "accepted": f.get("accepted"),
        })
    return pd.DataFrame.from_records(records, columns=[
        "stream", "name", "class", "adduct", "isotopologue", "is_is", "feat_feature_id", "feat_mz",
        "feat_rt", "moon_feature", "moon_status", "moon_matlab_row", "mz", "rt_seconds", "source",
        "role", "set_as", "accepted"])


def moon_to_crosswalk(doc: Mapping[str, Any], peaks: "PeakIndex | Mapping[str, str | Path] | str | Path") -> pd.DataFrame:
    """A (MATLAB-curated) ``.moon`` as the manual crosswalk read by
    ``compare_manual_annotations``. Only M+0 features (LASER annotates M+0).
    Moon's feature IDs are exact rows of the same lists, so mapped rows get
    ``crosswalk_status = exact_feature_id`` (no profile matching needed)."""
    frame = moon_to_frame(doc, peaks)
    frame = frame[frame.isotopologue.eq("")]
    mapped = frame.moon_status.eq("mapped")
    return pd.DataFrame({
        "stream": frame.stream,
        "matlab_species": frame.name,
        "matlab_class": frame["class"],
        "matlab_adduct": frame.adduct,
        "matlab_is_internal_standard": frame.is_is,
        "xcms_feature_id": frame.feat_feature_id,
        "crosswalk_status": mapped.map({True: MAPPED_STATUS, False: "unmapped"}),
        "crosswalk_reason": frame.moon_status.where(~mapped, "moon_feature_id"),
        "matlab_peak_row": frame.moon_matlab_row,
        "moon_feature": frame.moon_feature,
        "source": frame.source,
        "role": frame.role,
    }).reset_index(drop=True)


__all__ = [
    "FORMAT", "MAPPED_STATUS", "PeakIndex", "PeakRow", "annotations_to_moon", "lipid_class",
    "moon_to_crosswalk", "moon_to_frame", "read_moon", "read_peaklist", "split_is", "stream_name",
    "validate_moon", "write_moon",
]
