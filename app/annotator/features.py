"""Load an XCMS/CAMERA or MZmine modular feature list.

The primary format is the XCMS/CAMERA ``peakgrouplist`` export (tab separated)
with columns ``feature``, ``mzmed``, ``rtmed``, ``npeaks`` followed by group and
per-sample intensity columns.  Generic CSV/TSV feature tables are also supported
through flexible column detection / explicit column mapping.
"""

from __future__ import annotations

import io
import gzip
from pathlib import Path

import pandas as pd

DETECTED_COLUMN_SUFFIX = "::detected"
HEIGHT_COLUMN_PREFIX = "height::"
AREA_COLUMN_PREFIX = "area::"

_MZ_CANDIDATES = ["mzmed", "mz", "m/z", "mzmedian", "mz_med", "row m/z", "average mz"]
_RT_CANDIDATES = ["rtmed", "rt", "retention_time", "rt_med", "row retention time",
                  "average rt", "retentiontime"]
_ID_CANDIDATES = ["feature", "id", "feature_id", "name", "row id"]
_NPEAKS_CANDIDATES = ["npeaks", "n_peaks", "peaks"]
_MOBILITY_CANDIDATES = ["ion_mobility", "mobility", "average mobility"]
_MOBILITY_UNIT_CANDIDATES = ["ion_mobility_unit", "mobility_unit"]
_RT_SHAPE_SCORE_CANDIDATES = [
    "rt_shape_classification_summary:shape_class_score",
]
_MOBILITY_SHAPE_SCORE_CANDIDATES = [
    "mobility_shape_classification_summary:shape_class_score",
]

# Structural columns that are never sample intensities.
_META_COLS = {"feature", "mzmed", "mzmin", "mzmax", "rtmed", "rtmin", "rtmax",
              "npeaks", "id", "name", "isotopes", "adduct", "pcgroup",
              "into", "intb", "maxo", "sn", "integrated intensity",
              "peak area", "peak height", "apex intensity", "signal to noise",
              "baseline corrected area"}


def _detect(cols: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def read_raw(source) -> pd.DataFrame:
    """Read a CSV/TSV/txt feature table, auto-detecting the delimiter."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file() and Path(f"{path}.gz").is_file():
            path = Path(f"{path}.gz")
        opener = gzip.open if path.suffix == ".gz" else Path.open
        with opener(path, "rt", encoding="utf-8-sig", errors="replace") as handle:
            header = handle.readline()
        sep = "\t" if header.count("\t") >= header.count(",") else ","
        return pd.read_csv(path, sep=sep, low_memory=False)
    elif hasattr(source, "read"):
        data = source.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
    else:
        data = bytes(source)

    text = data.decode("utf-8-sig", errors="replace")
    # Sniff delimiter from the header line.
    header = text.split("\n", 1)[0]
    sep = "\t" if header.count("\t") >= header.count(",") else ","
    # MZmine modular exports may contain sparse annotation columns whose early
    # and late rows infer different dtypes. Parse once so large real examples do
    # not emit DtypeWarning noise or depend on chunk order.
    return pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)


def _finalize_features(out: pd.DataFrame, sample_names: list[str]) -> pd.DataFrame:
    """Drop unusable rows and declare authoritative sample measurement columns."""
    result = out.dropna(subset=["mz", "rt"]).reset_index(drop=True)
    result.attrs["sample_columns"] = tuple(str(name) for name in sample_names)
    return result


def load_feature_list(
    source,
    mz_col: str | None = None,
    rt_col: str | None = None,
    id_col: str | None = None,
    rt_unit: str = "minutes",
    intensity_agg: str = "max",
    exclude_intensity_columns: tuple[str, ...] | list[str] = (),
) -> pd.DataFrame:
    """Return a tidy feature table with columns:

    ``feature_id``, ``mz``, ``rt`` (minutes), ``mobility``, ``mobility_unit``,
    ``npeaks``, ``intensity``, ``quantification_measure`` and the original
    sample intensity columns. MZmine inputs additionally retain independent
    ``height::<sample>`` and ``area::<sample>`` columns so neither quantitative
    channel is lost when peak height is used as the default annotation signal.
    """
    df = read_raw(source)
    cols = list(df.columns)
    excluded_intensity_columns = {
        str(column).casefold() for column in exclude_intensity_columns
    }

    mz_col = mz_col or _detect(cols, _MZ_CANDIDATES)
    rt_col = rt_col or _detect(cols, _RT_CANDIDATES)
    id_col = id_col or _detect(cols, _ID_CANDIDATES)
    npeaks_col = _detect(cols, _NPEAKS_CANDIDATES)
    mobility_col = _detect(cols, _MOBILITY_CANDIDATES)
    mobility_unit_col = _detect(cols, _MOBILITY_UNIT_CANDIDATES)
    rt_shape_score_col = _detect(cols, _RT_SHAPE_SCORE_CANDIDATES)
    mobility_shape_score_col = _detect(cols, _MOBILITY_SHAPE_SCORE_CANDIDATES)

    if mz_col is None or rt_col is None:
        raise ValueError(
            "Could not detect m/z and RT columns automatically. "
            f"Available columns: {cols}"
        )

    out = pd.DataFrame()
    out["feature_id"] = (
        df[id_col].astype(str) if id_col else [f"F{i:06d}" for i in range(len(df))]
    )
    out["mz"] = pd.to_numeric(df[mz_col], errors="coerce")
    rt = pd.to_numeric(df[rt_col], errors="coerce")
    if rt_unit == "seconds":
        rt = rt / 60.0
    out["rt"] = rt
    # Preserve optional XCMS peak-group metadata when present.  These fields
    # are not needed for annotation, but they let the review browser show the
    # original RT/mass bounds alongside an on-demand chromatogram.
    optional_columns = {
        "mzmin": _detect(cols, ["mzmin", "mz_min"]),
        "mzmax": _detect(cols, ["mzmax", "mz_max"]),
        "rtmin": _detect(cols, ["rtmin", "rt_min"]),
        "rtmax": _detect(cols, ["rtmax", "rt_max"]),
        "into": _detect(cols, ["into", "integrated intensity", "peak area"]),
        "intb": _detect(cols, ["intb", "baseline corrected area"]),
        "maxo": _detect(cols, ["maxo", "peak height", "apex intensity"]),
        "sn": _detect(cols, ["sn", "signal to noise"]),
    }
    for target, source_column in optional_columns.items():
        if source_column is not None:
            values = pd.to_numeric(df[source_column], errors="coerce")
            if target.startswith("rt") and rt_unit == "seconds":
                values = values / 60.0
            out[target] = values
    out["mobility"] = (
        pd.to_numeric(df[mobility_col], errors="coerce")
        if mobility_col else float("nan")
    )
    out["mobility_unit"] = (
        df[mobility_unit_col].astype("string").fillna("")
        if mobility_unit_col else ""
    )
    out["npeaks"] = (
        pd.to_numeric(df[npeaks_col], errors="coerce") if npeaks_col else 1
    )
    # MZmine's Feature Filter exports the fit score used by its optional RT and
    # mobilogram shape gates. Preserve these row-level summaries as evidence;
    # they are metadata, never sample intensities. Processing QC can later
    # replace them with medians across DETECTED biological/QC measurements.
    if rt_shape_score_col:
        out["mzmine_rt_shape_score"] = pd.to_numeric(
            df[rt_shape_score_col], errors="coerce"
        )
    if mobility_shape_score_col:
        out["mzmine_mobility_shape_score"] = pd.to_numeric(
            df[mobility_shape_score_col], errors="coerce"
        )

    # Prefer MZmine peak height for quantification. Peak height is less sensitive
    # than integrated area to incorrect feature boundaries and split/incompletely
    # resolved peaks. Keep peak area as a compatibility fallback for older
    # exports, then turn the selected measure into ordinary sample columns;
    # annotation/alignment metrics must not be mistaken for intensities.
    mzmine_height_cols = [
        c for c in cols if c.startswith("datafile:") and c.endswith(":height")
    ]
    mzmine_area_cols = [
        c for c in cols if c.startswith("datafile:") and c.endswith(":area")
    ]
    if mzmine_height_cols:
        intensity_cols = mzmine_height_cols
        source_measure = "height"
        source_suffix = ":height"
        quantification_measure = "peak_height"
    elif mzmine_area_cols:
        intensity_cols = mzmine_area_cols
        source_measure = "area"
        source_suffix = ":area"
        quantification_measure = "peak_area_fallback"
    else:
        intensity_cols = [
            c for c in cols
            if c.lower() not in _META_COLS
            and c.casefold() not in excluded_intensity_columns
            and c not in (mz_col, rt_col, id_col, mobility_col)
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        source_measure = None
        source_suffix = None
        quantification_measure = "generic_intensity"
    if source_suffix:
        sample_names = [
            c[len("datafile:"):-len(source_suffix)] for c in intensity_cols
        ]
    else:
        sample_names = intensity_cols
    # MZmine marks each sample DETECTED, ESTIMATED (gap-filled) or UNKNOWN. Only
    # DETECTED is a real measurement, so keep that per sample as a companion to
    # the intensity columns. The "::detected" suffix keeps these out of the
    # intensity-column detection used downstream.
    detection = pd.DataFrame(index=df.index)
    for sample in sample_names:
        state_col = f"datafile:{sample}:feature_state"
        if state_col in df:
            detection[f"{sample}{DETECTED_COLUMN_SUFFIX}"] = (
                df[state_col].astype("string").str.upper().eq("DETECTED")
            )
    if intensity_cols:
        sub = df[intensity_cols].apply(pd.to_numeric, errors="coerce")
        sub.columns = sample_names
        out = pd.concat([out, sub], axis=1)
        if source_measure and source_measure in df:
            row_signal = pd.to_numeric(df[source_measure], errors="coerce")
            out["intensity"] = row_signal.fillna(getattr(sub, intensity_agg)(axis=1))
        else:
            out["intensity"] = getattr(sub, intensity_agg)(axis=1)
    else:
        out["intensity"] = out["npeaks"].astype(float)
    # Preserve both MZmine quantitative channels independently. These columns
    # are evidence, not additional samples; sample-column discovery explicitly
    # excludes their reserved prefixes. Never substitute one measurement for
    # the other here, because a missing height or area is scientifically useful
    # information in its own right.
    for columns, prefix, suffix in (
        (mzmine_height_cols, HEIGHT_COLUMN_PREFIX, ":height"),
        (mzmine_area_cols, AREA_COLUMN_PREFIX, ":area"),
    ):
        if not columns:
            continue
        measurements = df[columns].apply(pd.to_numeric, errors="coerce")
        measurements.columns = [
            f"{prefix}{column[len('datafile:'):-len(suffix)]}"
            for column in columns
        ]
        out = pd.concat([out, measurements], axis=1)
    if not detection.empty:
        out = pd.concat([out, detection], axis=1)
    out["quantification_measure"] = quantification_measure

    return _finalize_features(out, sample_names)
