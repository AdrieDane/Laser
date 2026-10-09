"""Construct the MATLAB annotation domain model directly from XCMS exports.

The vendored engine operates on a typed :class:`LegacyProject`; it does not
intrinsically require MATLAB files.  This adapter builds the same domain model
from LASER's standard-lipidomics peak-group lists and the bundled MATLAB
catalogue.  Negative-ion m/z values are stored with a negative sign internally,
matching the MATLAB application's convention, while the original XCMS tables
and feature identifiers are retained for the exported result.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import pandas as pd

from standard_lipidomics.workflow import (
    discover_standard_streams,
    load_standard_feature_list,
    parse_standard_stream,
)
from standard_lipidomics.mass_calibration import apply_standard_mass_calibration

from .vendor.composition_initializer import initialize_composition_attrs
from .vendor.project import (
    CompoundCatalog,
    LegacyProject,
    PhaseData,
    SampleMetadata,
    readonly_mapping,
)


_ISOTOPE_LABELS = ("2H", "13C", "15N", "17O", "18O", "33S", "34S", "36S")
_PHASE_STREAMS = {
    "RP": ("RP_Pos", "RP_Neg"),
    "NP": ("NP_Pos", "NP_Neg"),
}
_STANDARD_STREAMS = tuple(
    stream for streams in _PHASE_STREAMS.values() for stream in streams
)


@dataclass(frozen=True)
class XcmsProject:
    """One-phase MATLAB-domain project plus lossless XCMS row provenance."""

    project: LegacyProject
    features: Mapping[str, pd.DataFrame]
    row_stream: tuple[str, ...]
    row_feature_id: tuple[str, ...]
    row_local_index: np.ndarray


def _number(value: str) -> float:
    text = str(value).strip()
    return float(text) if text else float("nan")


def _composition_part(value: str, column: int) -> str:
    text = str(value).strip()
    if column == 1:
        if text == "*":
            return "'"
        if text.startswith("_"):
            return f"-{text[1:]}"
    return text


def _r_group_count(formula: str) -> int:
    match = re.search(r"R(\d*)", str(formula))
    if match is None:
        return 0
    return int(match.group(1) or 1)


def _object_vector(values) -> np.ndarray:
    """Create a MATLAB-cell-like one-dimensional object array."""
    items = list(values)
    result: np.ndarray = np.empty(len(items), dtype=object)
    result[:] = items
    return result


def load_bundled_compound_catalog(reference_root: str | Path) -> CompoundCatalog:
    """Load ``lipids.tsv`` into the immutable model used by the vendor core.

    The source table uses blank cells to inherit class-level labels and three
    adjacent composition-decoration columns.  Only the monoisotopic target is
    needed for species-level automatic annotation; molecular-species naming and
    every configured RP adduct remain identical to the MATLAB catalogue.
    """
    root = Path(reference_root).expanduser().resolve()
    path = root / "DataBase" / "lipids.tsv"
    if not path.is_file():
        raise ValueError(f"MATLAB lipid catalogue not found: {path}")
    with path.open("r", encoding="latin-1", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows or len(rows[0]) != 11:
        raise ValueError(f"Unexpected MATLAB lipid catalogue format: {path}")

    classes: list[str] = []
    class_names: list[str] = []
    r_groups: list[str] = []
    compositions: list[tuple[str, str, str]] = []
    priorities_np: list[float] = []
    priorities_rp: list[float] = []
    combined: list[str] = []
    measurements: list[object] = []
    chemistry: list[str] = []
    n_r: list[int] = []

    active_class = ""
    class_name = ""
    r_group = ""
    for row_number, row in enumerate(rows[1:], start=2):
        if not any(str(value).strip() for value in row):
            continue
        if len(row) != 11:
            raise ValueError(
                f"lipids.tsv row {row_number} has {len(row)} columns; expected 11"
            )
        if row[0].strip():
            active_class = row[0].strip()
            class_name = row[1].strip()
            r_group = row[2].strip()
        if not active_class:
            raise ValueError(f"lipids.tsv row {row_number} has no inherited class")
        classes.append(active_class)
        class_names.append(class_name)
        r_groups.append(r_group)
        compositions.append((
            _composition_part(row[3], 0),
            _composition_part(row[4], 1),
            _composition_part(row[5], 2),
        ))
        priorities_np.append(_number(row[6]))
        priorities_rp.append(_number(row[7]))
        combined.append(row[8].strip())
        values = tuple(
            value.strip() for value in row[9].split(";") if value.strip()
        )
        measurements.append(
            values[0] if len(values) == 1 else np.asarray(values, dtype=object)
        )
        chemistry.append(row[10].strip())
        n_r.append(_r_group_count(row[10]))

    count = len(classes)
    mono_counts: tuple[np.ndarray, ...] = tuple(
        np.zeros((1, len(_ISOTOPE_LABELS)), dtype=np.int16)
        for _ in range(count)
    )
    mono_mass = tuple(np.asarray([0.0], dtype=float) for _ in range(count))
    mono_abundance = tuple(np.asarray([1.0], dtype=float) for _ in range(count))
    priority = {
        "NP": np.asarray(priorities_np, dtype=float),
        "RP": np.asarray(priorities_rp, dtype=float),
    }
    # The project no longer supplies analyst-selected class flags.  Definitions
    # with priority >= 5 are the catalogue's standard discovery set; lower
    # priority variants remain available as siblings but are not swept alone.
    accepted = {
        phase: (np.isfinite(values) & (values >= 5)).astype(np.uint8)
        for phase, values in priority.items()
    }
    interpolation = {
        phase: (~np.isfinite(values) | (values < 7)).astype(np.uint8)
        for phase, values in priority.items()
    }
    composition_array = np.asarray(compositions, dtype=object)
    fields = {
        "class": np.asarray(classes, dtype=object),
        "class_name": np.asarray(class_names, dtype=object),
        "R_group": np.asarray(r_groups, dtype=object),
        "composition": composition_array,
        "nR": np.asarray(n_r, dtype=np.int16),
        "chem": np.asarray(chemistry, dtype=object),
        "measurement": _object_vector(measurements),
        "natural_isotopes": _object_vector(mono_counts),
        "natural_isotopes_dmz": _object_vector(mono_mass),
        "natural_isotopes_abundance": _object_vector(mono_abundance),
        "natural_isotopes_label": np.asarray(_ISOTOPE_LABELS, dtype=object),
        "priority_NP": priority["NP"],
        "priority_RP": priority["RP"],
        "accepted_NP": accepted["NP"],
        "accepted_RP": accepted["RP"],
        "interpolation_NP": interpolation["NP"],
        "interpolation_RP": interpolation["RP"],
        "combined": np.asarray(combined, dtype=object),
        "inliers": np.asarray([[] for _ in range(count)], dtype=object),
        "outliers": np.asarray([[] for _ in range(count)], dtype=object),
    }
    attrs = initialize_composition_attrs(fields, {"label_type": ""})
    fields["composition_attr"] = attrs
    return CompoundCatalog(
        count=count,
        classes=tuple(classes),
        class_names=tuple(class_names),
        combined=tuple(combined),
        priority=readonly_mapping(priority),
        accepted=readonly_mapping(accepted),
        interpolation=readonly_mapping(interpolation),
        composition_attr=attrs,
        inliers=tuple([] for _ in range(count)),
        outliers=tuple([] for _ in range(count)),
        legacy_fields=readonly_mapping(fields),
    )


def _normalise_sources(
    sources: str | Path | Mapping[str, str | Path | bytes],
) -> dict[str, str | Path | bytes]:
    if isinstance(sources, (str, Path)):
        discovered = discover_standard_streams(sources)
        if not discovered:
            raise ValueError(f"No XCMS peakgrouplist streams found under {sources}")
        found: dict[str, str | Path | bytes] = {}
        found.update(discovered)
        return found
    result: dict[str, str | Path | bytes] = {}
    for key, value in sources.items():
        stream = parse_standard_stream(
            key if key in _STANDARD_STREAMS else str(value)
        )
        result[stream.name] = value
    return result


def _load_feature_tables(
    available: Mapping[str, str | Path | bytes],
    calibrations: Mapping[str, object] | None,
) -> dict[str, pd.DataFrame]:
    return {
        name: apply_standard_mass_calibration(
            load_standard_feature_list(source), (calibrations or {}).get(name)
        )
        for name, source in available.items()
    }


def build_xcms_project(
    sources: str | Path | Mapping[str, str | Path | bytes],
    *,
    phase: str,
    reference_root: str | Path,
    mass_calibrations: Mapping[str, object] | None = None,
) -> XcmsProject:
    """Build a signed-mass NP or RP project from its XCMS POS/NEG lists."""
    phase_name = str(phase).upper()
    if phase_name not in _PHASE_STREAMS:
        raise ValueError(
            f"XCMS MATLAB mode requires NP or RP, not {phase_name!r}"
        )
    phase_streams = _PHASE_STREAMS[phase_name]
    resolved = _normalise_sources(sources)
    available = {
        name: resolved[name] for name in phase_streams if name in resolved
    }
    if not available:
        expected = " and/or ".join(phase_streams)
        raise ValueError(
            f"Automatic MATLAB mode requires {expected} XCMS data"
        )
    feature_tables = _load_feature_tables(available, mass_calibrations)
    sample_count = max(
        max(len(table.attrs.get("sample_columns", [])), 1)
        for table in feature_tables.values()
    )
    mz_parts: list[np.ndarray] = []
    rt_parts: list[np.ndarray] = []
    representative_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    intensity_parts: list[np.ndarray] = []
    row_stream: list[str] = []
    row_feature_id: list[str] = []
    row_local_index: list[int] = []

    for stream_name in phase_streams:
        table = feature_tables.get(stream_name)
        if table is None:
            continue
        sign = -1.0 if stream_name.endswith("Neg") else 1.0
        mz_parts.append(sign * pd.to_numeric(table["mz"], errors="coerce").to_numpy(float))
        rt_parts.append(pd.to_numeric(table["rt"], errors="coerce").to_numpy(float) * 60.0)
        representative = pd.to_numeric(table["intensity"], errors="coerce").to_numpy(float)
        representative_parts.append(np.maximum(np.nan_to_num(representative), 1.0))
        samples = list(table.attrs.get("sample_columns", []))
        matrix: np.ndarray = np.zeros((len(table), sample_count), dtype=float)
        if samples:
            values = table[samples].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
            matrix[:, : values.shape[1]] = values
            detected = np.count_nonzero(values > 0, axis=1)
        else:
            matrix[:, 0] = representative
            detected = (representative > 0).astype(int)
        intensity_parts.append(matrix)
        npeaks = pd.to_numeric(table.get("npeaks"), errors="coerce").fillna(0).to_numpy(float)
        count_parts.append(np.maximum(npeaks, detected))
        row_stream.extend([stream_name] * len(table))
        row_feature_id.extend(table["feature_id"].astype(str).tolist())
        row_local_index.extend(range(len(table)))

    mz = np.concatenate(mz_parts)
    rt = np.concatenate(rt_parts)
    representative = np.concatenate(representative_parts)
    grouped_count = np.concatenate(count_parts)
    intensity = np.vstack(intensity_parts)
    # MATLAB's MS_mapping contract is ascending signed m/z.  Keep provenance
    # arrays in the same order so the final rows can be restored losslessly.
    order = np.argsort(mz, kind="stable")
    mz = mz[order]
    rt = rt[order]
    representative = representative[order]
    grouped_count = grouped_count[order]
    intensity = intensity[order]
    row_stream = [row_stream[index] for index in order]
    row_feature_id = [row_feature_id[index] for index in order]
    row_local_index = [row_local_index[index] for index in order]
    finite_rt = rt[np.isfinite(rt)]
    rt_axis = (
        np.asarray([finite_rt.min(), finite_rt.max()], dtype=float)
        if len(finite_rt) else np.asarray([0.0, 1.0], dtype=float)
    )
    phase_data = PhaseData(
        name=phase_name,
        mz_grouped=mz,
        rt_grouped=rt,
        representative_intensity=representative,
        grouped_peak_count=grouped_count,
        predicted_rt=np.full(len(mz), np.nan, dtype=float),
        annotation=np.zeros((len(mz), 2), dtype=float),
        sample_intensity=intensity,
        rt_axis=rt_axis,
    )
    sample_names = np.asarray(
        [[f"sample_{index + 1}"] for index in range(sample_count)], dtype=object
    )
    columns = np.full(sample_names.shape, phase_name, dtype=object)
    metadata = SampleMetadata(
        sample_count=sample_count,
        acquisition_columns=(phase_name,),
        fields=readonly_mapping({"sample_name": sample_names, "column": columns}),
    )
    catalog = load_bundled_compound_catalog(reference_root)
    source_path = (
        Path(sources).expanduser().resolve()
        if isinstance(sources, (str, Path)) else Path("<xcms-input>")
    )
    digest = hashlib.sha256()
    for stream_name, table in feature_tables.items():
        digest.update(stream_name.encode())
        digest.update(pd.util.hash_pandas_object(table, index=True).values.tobytes())
    project = LegacyProject(
        source=source_path,
        compounds=catalog,
        phases=readonly_mapping({phase_name: phase_data}),
        metadata=metadata,
        source_sha256=readonly_mapping({"xcms": digest.hexdigest()}),
        load_seconds=0.0,
        array_bytes=sum(
            array.nbytes for array in (mz, rt, representative, grouped_count, intensity)
        ),
    )
    return XcmsProject(
        project=project,
        features=readonly_mapping(feature_tables),
        row_stream=tuple(row_stream),
        row_feature_id=tuple(row_feature_id),
        row_local_index=np.asarray(row_local_index, dtype=np.int64),
    )


def build_rp_xcms_project(
    sources: str | Path | Mapping[str, str | Path | bytes],
    *,
    reference_root: str | Path,
    mass_calibrations: Mapping[str, object] | None = None,
) -> XcmsProject:
    """Backward-compatible wrapper for an RP XCMS project."""
    return build_xcms_project(
        sources,
        phase="RP",
        reference_root=reference_root,
        mass_calibrations=mass_calibrations,
    )


def build_np_xcms_project(
    sources: str | Path | Mapping[str, str | Path | bytes],
    *,
    reference_root: str | Path,
    mass_calibrations: Mapping[str, object] | None = None,
) -> XcmsProject:
    """Build an NP XCMS project with NP priorities and RT calibration state."""
    return build_xcms_project(
        sources,
        phase="NP",
        reference_root=reference_root,
        mass_calibrations=mass_calibrations,
    )


def with_memberships(
    project: LegacyProject,
    inliers: tuple[list[list[object]], ...],
) -> LegacyProject:
    """Return a project carrying automatically derived inlier coordinates."""
    fields = dict(project.compounds.legacy_fields)
    fields["inliers"] = np.asarray(inliers, dtype=object)
    compounds = replace(
        project.compounds,
        inliers=inliers,
        legacy_fields=readonly_mapping(fields),
    )
    return replace(project, compounds=compounds)
