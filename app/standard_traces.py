"""On-demand chromatograms for conventional XCMS projects.

The 4D workflow receives compact MZmine trace databases.  Conventional XCMS
projects instead carry centroided MS1 ``.mzXML``/``.mzML`` files next to the
peak-group lists.  This module reads only the scans needed for the selected
lipid ID and returns ordinary RT/intensity arrays for the Streamlit viewer.

The reader intentionally has no dependency on R, xcms, pyteomics, or vendor
software.  XCMS peak areas and boundaries remain the annotation source of
truth; these traces are a visual review aid and are never used to change an
annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import base64
import re
from pathlib import Path
from typing import Iterable, Iterator
import xml.etree.ElementTree as ET
import zlib

import numpy as np


@dataclass(frozen=True)
class StandardSampleTrace:
    """One XCMS feature's extracted-ion trace in one sample."""

    feature_id: str
    sample_name: str
    rt_values: np.ndarray
    rt_intensities: np.ndarray
    peak_height: float
    raw_path: str


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _parse_retention_time(value: str | None) -> float:
    """Return an mzXML/mzML retention time in minutes."""
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    # mzXML uses ISO-8601 durations such as PT120.5S.  A few converters write
    # a bare number; XCMS treats those as seconds.
    match = re.fullmatch(
        r"PT(?:(?P<h>[0-9.]+)H)?(?:(?P<m>[0-9.]+)M)?(?:(?P<s>[0-9.]+)S)?",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        seconds = (
            float(match.group("h") or 0.0) * 3600.0
            + float(match.group("m") or 0.0) * 60.0
            + float(match.group("s") or 0.0)
        )
        return seconds / 60.0
    try:
        return float(text) / 60.0
    except ValueError:
        return float("nan")


def _decode_binary_array(
    value: str | None,
    *,
    precision: int = 64,
    compression: str = "",
    endian: str = "<",
) -> np.ndarray:
    if not value:
        return np.asarray([], dtype=float)
    raw = base64.b64decode("".join(str(value).split()))
    if "zlib" in compression.casefold():
        raw = zlib.decompress(raw)
    dtype = np.dtype((">" if endian == ">" else "<") + ("f4" if precision == 32 else "f8"))
    # frombuffer is substantially faster than unpacking each point and returns
    # a copy so the cleared XML element cannot invalidate the result.
    return np.asarray(np.frombuffer(raw, dtype=dtype), dtype=float).copy()


def _decode_mzxml_peaks(peaks: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    attrs = {str(key).casefold(): str(value) for key, value in peaks.attrib.items()}
    precision = int(attrs.get("precision", "32"))
    endian = ">" if attrs.get("byteorder", "network").casefold() in {
        "network", "big", "bigendian", "big-endian"
    } else "<"
    values = _decode_binary_array(
        peaks.text,
        precision=precision,
        compression=attrs.get("compressiontype", ""),
        endian=endian,
    )
    if len(values) < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    values = values[: len(values) - len(values) % 2].reshape(-1, 2)
    pair_order = attrs.get("pairorder", "m/z-int").casefold()
    if pair_order.startswith("int"):
        return values[:, 1], values[:, 0]
    return values[:, 0], values[:, 1]


def _iter_mzxml_scans(path: Path) -> Iterator[tuple[float, np.ndarray, np.ndarray]]:
    for _, scan in ET.iterparse(path, events=("end",)):
        if _local_name(scan.tag) != "scan":
            continue
        attrs = {str(key).casefold(): str(value) for key, value in scan.attrib.items()}
        if attrs.get("mslevel", "1") != "1":
            scan.clear()
            continue
        peaks = next(
            (child for child in scan if _local_name(child.tag) == "peaks"),
            None,
        )
        if peaks is not None:
            mz, intensity = _decode_mzxml_peaks(peaks)
            rt = _parse_retention_time(attrs.get("retentiontime"))
            if np.isfinite(rt) and len(mz):
                yield rt, mz, intensity
        scan.clear()


def _cv_params(element: ET.Element) -> list[dict[str, str]]:
    return [
        {str(key): str(value) for key, value in child.attrib.items()}
        for child in element.iter()
        if _local_name(child.tag) == "cvParam"
    ]


def _param(params: Iterable[dict[str, str]], name: str) -> dict[str, str] | None:
    target = name.casefold()
    return next(
        (param for param in params if param.get("name", "").casefold() == target),
        None,
    )


def _iter_mzml_scans(path: Path) -> Iterator[tuple[float, np.ndarray, np.ndarray]]:
    for _, spectrum in ET.iterparse(path, events=("end",)):
        if _local_name(spectrum.tag) != "spectrum":
            continue
        params = _cv_params(spectrum)
        ms_level = _param(params, "ms level")
        if ms_level is not None and str(ms_level.get("value", "1")) != "1":
            spectrum.clear()
            continue
        scan_start = _param(params, "scan start time")
        if scan_start is None:
            spectrum.clear()
            continue
        rt = _parse_retention_time(scan_start.get("value"))
        unit = str(scan_start.get("unitName", scan_start.get("unitAccession", ""))).casefold()
        try:
            numeric_rt = float(scan_start.get("value", "nan"))
            if "second" in unit or unit.endswith(":1000010"):
                rt = numeric_rt / 60.0
            else:
                rt = numeric_rt
        except ValueError:
            numeric_rt = rt
        if not np.isfinite(rt):
            spectrum.clear()
            continue

        arrays: dict[str, np.ndarray] = {}
        for binary_array in spectrum.iter():
            if _local_name(binary_array.tag) != "binaryDataArray":
                continue
            array_params = _cv_params(binary_array)
            mz_param = next(
                (param for param in array_params if "m/z array" in param.get("name", "").casefold()),
                None,
            )
            intensity_param = next(
                (param for param in array_params if "intensity array" in param.get("name", "").casefold()),
                None,
            )
            selected = mz_param or intensity_param
            if selected is None:
                continue
            precision = 32 if any(
                "32-bit" in param.get("name", "") for param in array_params
            ) else 64
            compression = " ".join(param.get("name", "") for param in array_params)
            binary = next(
                (child for child in binary_array if _local_name(child.tag) == "binary"),
                None,
            )
            decoded = _decode_binary_array(
                binary.text if binary is not None else None,
                precision=precision,
                compression=compression,
                endian="<",
            )
            arrays["mz" if mz_param is not None else "intensity"] = decoded
        mz = arrays.get("mz", np.asarray([], dtype=float))
        intensity = arrays.get("intensity", np.asarray([], dtype=float))
        count = min(len(mz), len(intensity))
        if count:
            yield rt, mz[:count], intensity[:count]
        spectrum.clear()


def _iter_ms1_scans(path: Path) -> Iterator[tuple[float, np.ndarray, np.ndarray]]:
    suffix = path.suffix.casefold()
    if suffix == ".mzxml":
        yield from _iter_mzxml_scans(path)
    elif suffix == ".mzml":
        yield from _iter_mzml_scans(path)


def _resolve_casefold_path(base: Path, parts: tuple[str, ...]) -> Path | None:
    """Resolve a project path on case-sensitive and Windows disks."""
    current = base
    for part in parts:
        if not current.is_dir():
            return None
        exact = current / part
        if exact.is_dir():
            current = exact
            continue
        match = next(
            (
                child for child in current.iterdir()
                if child.is_dir() and child.name.casefold() == part.casefold()
            ),
            None,
        )
        if match is None:
            return None
        current = match
    return current


def discover_xcms_raw_files(
    project_dir: str | Path,
    stream_name: str,
) -> dict[str, Path]:
    """Map XCMS sample-column names to matching converted raw files."""
    root = Path(project_dir).expanduser().resolve()
    directories: list[Path] = []
    for ancestor in (root, *root.parents):
        for relative in (
            ("Data", "Processed", stream_name),
            ("Processed", stream_name),
            ("Data", "Raw", stream_name),
            ("Raw", stream_name),
        ):
            candidate = _resolve_casefold_path(ancestor, relative)
            if (
                candidate is not None
                and candidate.is_dir()
                and candidate not in directories
            ):
                directories.append(candidate)
    files: list[Path] = []
    for directory in directories:
        files.extend(directory.rglob("*.mzXML"))
        files.extend(directory.rglob("*.mzML"))
        files.extend(directory.rglob("*.mzxml"))
        files.extend(directory.rglob("*.mzml"))
    return {path.stem.casefold(): path for path in sorted(set(files))}


def _extract_target_traces(
    path: Path,
    target_mz: tuple[float, ...],
    ppm_tolerance: float,
) -> tuple[np.ndarray, ...]:
    rt_values: list[float] = []
    intensities: list[list[float]] = [[] for _ in target_mz]
    for rt, mz, intensity in _iter_ms1_scans(path):
        rt_values.append(float(rt))
        for index, target in enumerate(target_mz):
            tolerance = abs(float(target)) * float(ppm_tolerance) * 1e-6
            mask = np.abs(mz - target) <= tolerance
            intensities[index].append(
                float(np.sum(intensity[mask])) if np.any(mask) else 0.0
            )
    retention_times = np.asarray(rt_values, dtype=float)
    return tuple(
        np.column_stack([retention_times, np.asarray(values, dtype=float)])
        if len(retention_times) else np.empty((0, 2), dtype=float)
        for values in intensities
    )


def _feature_specs_key(feature_specs: tuple[tuple[str, float, float], ...]) -> tuple:
    return tuple((str(feature_id), float(mz), float(rt)) for feature_id, mz, rt in feature_specs)


@lru_cache(maxsize=8)
def _load_cached(
    project_dir: str,
    stream_name: str,
    feature_specs: tuple[tuple[str, float, float], ...],
    ppm_tolerance: float,
) -> tuple[StandardSampleTrace, ...]:
    files = discover_xcms_raw_files(project_dir, stream_name)
    if not files:
        return tuple()
    # The XCMS feature list supplies sample names, but this cache receives only
    # feature specs.  Matching all raw files is preferable to silently showing
    # one sample when a project has no metadata table; sample order is stable.
    targets = tuple(float(spec[1]) for spec in feature_specs)
    traces: list[StandardSampleTrace] = []
    for _, path in sorted(files.items()):
        extracted = _extract_target_traces(path, targets, ppm_tolerance)
        sample_name = path.stem
        for spec, values in zip(feature_specs, extracted, strict=True):
            if not len(values):
                continue
            traces.append(StandardSampleTrace(
                feature_id=str(spec[0]),
                sample_name=sample_name,
                rt_values=values[:, 0],
                rt_intensities=values[:, 1],
                peak_height=float(np.max(values[:, 1])) if len(values) else 0.0,
                raw_path=str(path),
            ))
    return tuple(traces)


def load_standard_feature_traces(
    project_dir: str | Path,
    stream_name: str,
    feature_specs: tuple[tuple[str, float, float], ...],
    *,
    ppm_tolerance: float = 5.0,
) -> dict[str, list[StandardSampleTrace]]:
    """Extract traces for all requested features in one pass per raw sample."""
    loaded = _load_cached(
        str(Path(project_dir).expanduser().resolve()),
        str(stream_name),
        _feature_specs_key(feature_specs),
        float(ppm_tolerance),
    )
    grouped: dict[str, list[StandardSampleTrace]] = {}
    for trace in loaded:
        grouped.setdefault(trace.feature_id, []).append(trace)
    return grouped
