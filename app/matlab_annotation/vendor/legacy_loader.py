"""Read-only loader for shipped MATLAB 5/V6 lipidomics projects."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.io import loadmat

from .composition_initializer import initialize_composition_attrs
from .project import (
    CompoundCatalog,
    LegacyProject,
    PhaseData,
    SampleMetadata,
    readonly_mapping,
)


REQUIRED_FILES = ("MS.mat", "MS_data.mat", "MS_metadata.mat")
GROUP_DEFINITION_FILE = "group_definition.tsv"
PHASE_FIELDS = (
    "mz_grouped", "RT_grouped", "x_grouped", "n_grouped",
    "RTp_grouped", "annotation", "X_grouped", "RT_range",
)


class LegacyProjectError(ValueError):
    """Raised when a legacy project violates the supported structural contract."""


def load_legacy_project(source: str | Path) -> LegacyProject:
    """Load and validate a legacy project without modifying its source files."""
    started = perf_counter()
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_dir():
        raise LegacyProjectError(f"project directory does not exist: {source_path}")
    paths = {name: source_path / name for name in REQUIRED_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise LegacyProjectError(f"missing required project files: {', '.join(missing)}")
    group_path = source_path / GROUP_DEFINITION_FILE
    if group_path.is_file():
        paths[GROUP_DEFINITION_FILE] = group_path
    hashes_before = {name: _sha256(path) for name, path in paths.items()}

    ms = _load_variable(paths["MS.mat"], "MS")
    ms_data = _load_variable(paths["MS_data.mat"], "MS_data")
    metadata = _load_variable(paths["MS_metadata.mat"], "MS_metadata")

    _initialize_phase_state(ms, ms_data)
    if group_path.is_file():
        _initialize_group_metadata(metadata, group_path)
    sample_metadata = _load_metadata(metadata)
    compounds = _load_compounds(ms, metadata)
    phases = {
        phase: _load_phase(phase, value, compounds.count, sample_metadata.sample_count)
        for phase, value in ms_data.items()
    }
    if not phases:
        raise LegacyProjectError("MS_data contains no chromatography phase")
    _validate_phase_state(compounds, tuple(phases))

    hashes_after = {name: _sha256(path) for name, path in paths.items()}
    if hashes_after != hashes_before:
        raise LegacyProjectError("source files changed while loading")
    array_bytes = sum(_array_bytes(value) for value in (ms, ms_data, metadata))
    return LegacyProject(
        source=source_path,
        compounds=compounds,
        phases=readonly_mapping(phases),
        metadata=sample_metadata,
        source_sha256=readonly_mapping(hashes_before),
        load_seconds=perf_counter() - started,
        array_bytes=array_bytes,
    )


def _load_variable(path: Path, variable: str) -> dict[str, Any]:
    try:
        loaded = loadmat(path, variable_names=[variable], simplify_cells=True)
    except (OSError, ValueError, NotImplementedError) as exc:
        raise LegacyProjectError(f"cannot read {path.name}: {exc}") from exc
    value = loaded.get(variable)
    if not isinstance(value, dict):
        raise LegacyProjectError(f"{path.name} does not contain scalar struct {variable}")
    return value


def _load_compounds(ms: dict[str, Any], metadata: dict[str, Any]) -> CompoundCatalog:
    required = {"class", "class_name", "combined", "inliers", "outliers"}
    missing = sorted(required - ms.keys())
    if missing:
        raise LegacyProjectError(f"MS is missing fields: {', '.join(missing)}")
    classes = _string_tuple(ms["class"])
    count = len(classes)
    if count == 0:
        raise LegacyProjectError("MS contains no compounds")
    class_names = _fixed_tuple(_string_tuple(ms["class_name"]), count, "MS.class_name")
    combined = _fixed_tuple(_string_tuple(ms["combined"]), count, "MS.combined")
    if "composition_attr" in ms:
        composition_attr = _object_tuple(ms["composition_attr"], count, "MS.composition_attr")
    else:
        try:
            composition_attr = initialize_composition_attrs(ms, metadata)
        except ValueError as exc:
            raise LegacyProjectError(str(exc)) from exc
        ms["composition_attr"] = composition_attr
    inliers = _membership_tuple(ms["inliers"], count, "MS.inliers")
    outliers = _membership_tuple(ms["outliers"], count, "MS.outliers")
    priority = _phase_vectors(ms, "priority_", count)
    accepted = _phase_vectors(ms, "accepted_", count)
    interpolation = _phase_vectors(ms, "interpolation_", count)
    _freeze_arrays(ms)
    return CompoundCatalog(
        count=count,
        classes=classes,
        class_names=class_names,
        combined=combined,
        priority=readonly_mapping(priority),
        accepted=readonly_mapping(accepted),
        interpolation=readonly_mapping(interpolation),
        composition_attr=composition_attr,
        inliers=inliers,
        outliers=outliers,
        legacy_fields=readonly_mapping(ms),
    )


def _load_metadata(metadata: dict[str, Any]) -> SampleMetadata:
    if "sample_name" not in metadata or "column" not in metadata:
        raise LegacyProjectError("MS_metadata requires sample_name and column")
    sample_names = np.asarray(metadata["sample_name"], dtype=object)
    sample_count = 1 if sample_names.ndim == 0 else int(sample_names.shape[0])
    columns = tuple(dict.fromkeys(str(item) for item in np.asarray(metadata["column"], dtype=object).flat))
    if sample_count < 1 or not columns:
        raise LegacyProjectError("MS_metadata has no samples or acquisition columns")
    _freeze_arrays(metadata)
    return SampleMetadata(sample_count, columns, readonly_mapping(metadata))


def _initialize_group_metadata(metadata: dict[str, Any], path: Path) -> None:
    """Port MATLAB ``check_MS_metadata`` group-definition initialization."""
    comments = np.asarray(metadata.get("comment", ()), dtype=object)
    if comments.size == 0:
        raise LegacyProjectError(f"{path.name} requires MS_metadata.comment")
    if comments.ndim > 1:
        first = comments[:, 0]
        if not np.all(comments == first[:, None]):
            raise LegacyProjectError(
                f"{path.name} cannot map unequal per-acquisition comments"
            )
        comments = first
    else:
        comments = comments.reshape(-1)

    with path.open("r", encoding="latin-1", newline="") as handle:
        rows = [row for row in csv.reader(handle, delimiter="\t") if any(cell.strip() for cell in row)]
    if not rows or any(len(row) < 2 for row in rows):
        raise LegacyProjectError(f"{path.name} must contain group and display-label columns")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise LegacyProjectError(f"{path.name} has inconsistent column counts")

    group_labels = [row[0].strip() for row in rows]
    mapping = {label.casefold(): index + 1 for index, label in enumerate(group_labels)}
    if len(mapping) != len(group_labels):
        raise LegacyProjectError(f"{path.name} group identifiers are not unique")
    missing = sorted({str(value) for value in comments if str(value).strip().casefold() not in mapping})
    if missing:
        raise LegacyProjectError(
            f"{path.name} does not define MS_metadata comments: {', '.join(missing)}"
        )

    class_columns = list(zip(*(row[1:] for row in rows), strict=True))
    class_labels = []
    class_codes = np.zeros((len(rows), len(class_columns)), dtype=np.int64)
    for column_index, values in enumerate(class_columns):
        labels = list(dict.fromkeys(value.strip() for value in values))
        class_labels.append(labels)
        indices = {label: index + 1 for index, label in enumerate(labels)}
        class_codes[:, column_index] = [indices[value.strip()] for value in values]

    metadata["comment"] = comments
    metadata["group"] = np.asarray(
        [mapping[str(value).strip().casefold()] for value in comments], dtype=np.int64
    )
    metadata["group_label"] = np.asarray(group_labels, dtype=object)
    metadata["group_class"] = class_codes[:, 0] if class_codes.shape[1] == 1 else class_codes
    metadata["group_class_label"] = (
        np.asarray(class_labels[0], dtype=object)
        if len(class_labels) == 1
        else tuple(np.asarray(labels, dtype=object) for labels in class_labels)
    )


def _load_phase(phase: str, value: Any, compound_count: int, sample_count: int) -> PhaseData:
    if not isinstance(value, dict):
        raise LegacyProjectError(f"MS_data.{phase} is not a scalar struct")
    missing = sorted(set(PHASE_FIELDS) - value.keys())
    if missing:
        raise LegacyProjectError(f"MS_data.{phase} is missing fields: {', '.join(missing)}")
    arrays = {name: np.asarray(value[name]) for name in PHASE_FIELDS}
    peak_count = int(arrays["mz_grouped"].size)
    for name in ("RT_grouped", "x_grouped", "n_grouped", "RTp_grouped"):
        arrays[name] = arrays[name].reshape(-1)
        if arrays[name].shape[0] != peak_count:
            raise LegacyProjectError(f"MS_data.{phase}.{name} is not peak-aligned")
    arrays["mz_grouped"] = arrays["mz_grouped"].reshape(-1)
    annotation = arrays["annotation"]
    intensity = arrays["X_grouped"]
    if annotation.shape != (peak_count, 2):
        raise LegacyProjectError(f"MS_data.{phase}.annotation must be N x 2")
    if intensity.shape != (peak_count, sample_count):
        raise LegacyProjectError(
            f"MS_data.{phase}.X_grouped shape {intensity.shape} does not match "
            f"{peak_count} peaks x {sample_count} samples"
        )
    compound_ids = annotation[:, 0]
    if np.any((compound_ids < 0) | (compound_ids > compound_count)):
        raise LegacyProjectError(f"MS_data.{phase}.annotation has invalid compound ids")
    if np.any((compound_ids == 0) != (annotation[:, 1] == 0)):
        raise LegacyProjectError(f"MS_data.{phase}.annotation has half-empty rows")
    for array in arrays.values():
        _freeze_array(array)
    pattern_values = None
    if "X_pattern" in value:
        pattern_values = np.asarray(value["X_pattern"])
        if pattern_values.ndim == 1:
            pattern_values = pattern_values.reshape(-1, 1)
        if pattern_values.ndim != 2 or pattern_values.shape[0] != peak_count:
            raise LegacyProjectError(f"MS_data.{phase}.X_pattern is not peak-aligned")
        _freeze_array(pattern_values)
    return PhaseData(
        name=phase,
        mz_grouped=arrays["mz_grouped"],
        rt_grouped=arrays["RT_grouped"],
        representative_intensity=arrays["x_grouped"],
        grouped_peak_count=arrays["n_grouped"],
        predicted_rt=arrays["RTp_grouped"],
        annotation=annotation,
        sample_intensity=intensity,
        rt_axis=arrays["RT_range"].reshape(-1),
        pattern_values=pattern_values,
    )


def _validate_phase_state(compounds: CompoundCatalog, phases: tuple[str, ...]) -> None:
    for phase in phases:
        if phase not in compounds.priority:
            raise LegacyProjectError(f"MS lacks priority_{phase}")
        if phase not in compounds.accepted or phase not in compounds.interpolation:
            raise LegacyProjectError(f"MS lacks coordinated accepted/interpolation state for {phase}")


def _initialize_phase_state(ms: dict[str, Any], ms_data: dict[str, Any]) -> None:
    """Port the MATLAB viewer defaults for a pristine, unannotated project."""
    count = np.asarray(ms.get("class", ()), dtype=object).size
    for phase, value in ms_data.items():
        if not isinstance(value, dict):
            continue
        ms.setdefault(f"interpolation_{phase}", np.ones(count, dtype=np.uint8))
        if f"accepted_{phase}" not in ms:
            annotation = np.asarray(value.get("annotation", ()))
            accepted = np.zeros(count, dtype=np.uint8)
            if annotation.ndim == 2 and annotation.shape[1] >= 1:
                compound_ids = annotation[:, 0].astype(np.int64, copy=False)
                compound_ids = compound_ids[(compound_ids > 0) & (compound_ids <= count)]
                accepted[np.unique(compound_ids) - 1] = 1
            ms[f"accepted_{phase}"] = accepted


def _phase_vectors(ms: dict[str, Any], prefix: str, count: int) -> dict[str, np.ndarray[Any, Any]]:
    result = {}
    for name, value in ms.items():
        if not name.startswith(prefix):
            continue
        phase = name[len(prefix):]
        vector = np.asarray(value).reshape(-1)
        if vector.shape != (count,):
            raise LegacyProjectError(f"MS.{name} must contain {count} values")
        _freeze_array(vector)
        result[phase] = vector
    return result


def _string_tuple(value: Any) -> tuple[str, ...]:
    return tuple(_matlab_string(item) for item in np.asarray(value, dtype=object).reshape(-1))


def _matlab_string(value: Any) -> str:
    """Normalize MATLAB empty char/cell values without inventing a shared '[]' key."""
    array = np.asarray(value, dtype=object)
    if array.size == 0:
        return ""
    if array.size == 1:
        return str(array.reshape(-1)[0])
    return "".join(str(item) for item in array.reshape(-1))


def _fixed_tuple(value: tuple[Any, ...], count: int, label: str) -> tuple[Any, ...]:
    if len(value) != count:
        raise LegacyProjectError(f"{label} must contain {count} values")
    return value


def _object_tuple(value: Any, count: int, label: str) -> tuple[Any, ...]:
    array = np.asarray(value, dtype=object).reshape(-1)
    if array.size != count:
        raise LegacyProjectError(f"{label} must contain {count} values")
    return tuple(array.tolist())


def _membership_tuple(value: Any, count: int, label: str) -> tuple[list[list], ...]:
    """Normalize MATLAB cell rows without losing singleton table dimensions."""
    entries = _object_tuple(value, count, label)
    normalized = []
    for index, entry in enumerate(entries, start=1):
        array = np.asarray(entry, dtype=object)
        if array.size == 0:
            normalized.append([])
            continue
        if array.ndim == 1:
            if array.size != 4 or not isinstance(array[0], str):
                raise LegacyProjectError(
                    f"{label}{{{index}}} is not a four-column membership table"
                )
            array = array.reshape(1, 4)
        if array.ndim != 2 or array.shape[1] != 4:
            raise LegacyProjectError(
                f"{label}{{{index}}} is not a four-column membership table"
            )
        normalized.append(array.tolist())
    return tuple(normalized)


def _freeze_array(array: np.ndarray[Any, Any]) -> None:
    array.flags.writeable = False


def _freeze_arrays(value: Any) -> None:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            for item in value.flat:
                _freeze_arrays(item)
        _freeze_array(value)
    elif isinstance(value, dict):
        for item in value.values():
            _freeze_arrays(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _freeze_arrays(item)


def _array_bytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_array_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_array_bytes(item) for item in value)
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
