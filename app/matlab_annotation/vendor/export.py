"""Cross-platform, non-mutating peak and RT-database exports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .naming import compound_display_template, resolved_compound_name
from .project import LegacyProject, PhaseData


@dataclass(frozen=True)
class PeakExport:
    headers: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


def build_peak_export(
    project: LegacyProject,
    phase: str,
    mode: str,
    *,
    selected_rows: Iterable[int] = (),
    phase_override: PhaseData | None = None,
) -> PeakExport:
    """Build MATLAB-compatible row scopes with raw grouped intensities."""

    if phase not in project.phases:
        raise ValueError(f"project has no phase {phase!r}")
    if mode not in {"all", "selected", "annotated_raw"}:
        raise ValueError(f"unsupported export mode: {mode}")
    data = project.phases[phase] if phase_override is None else phase_override
    if data.name != phase:
        raise ValueError("phase_override does not match the requested phase")
    if mode == "all":
        indices = np.arange(data.peak_count, dtype=np.int64)
    elif mode == "selected":
        indices = np.asarray(tuple(dict.fromkeys(int(row) for row in selected_rows)), dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= data.peak_count):
            raise IndexError("selected export row is outside the phase")
    else:
        indices = np.flatnonzero(data.annotation[:, 0] != 0)

    sample_names = _sample_names(project, phase)
    headers = (
        "phase", "phase_row", "mz", "RT_min", "representative_intensity",
        "compound", "annotation_code", "annotation", *sample_names,
    )
    rows = []
    for row in indices:
        compound, code = (int(value) for value in data.annotation[row, :2])
        name = ""
        if compound:
            template = compound_display_template(project, compound - 1)
            name = resolved_compound_name(
                template, project.compounds.composition_attr[compound - 1], phase, code
            )
        rows.append((
            phase, int(row) + 1, float(data.mz_grouped[row]),
            float(data.rt_grouped[row]) / 60.0,
            float(data.representative_intensity[row]), compound, code, name,
            *(float(value) for value in data.sample_intensity[row]),
        ))
    return PeakExport(headers, tuple(rows))


def build_rtdb_export(
    project: LegacyProject, phase: str, *, phase_override: PhaseData | None = None
) -> PeakExport:
    """Average accepted annotation RTs into the repository RTDB schema."""

    data = project.phases[phase] if phase_override is None else phase_override
    if data.name != phase:
        raise ValueError("phase_override does not match the requested phase")
    grouped: dict[str, list[float]] = {}
    for row in np.flatnonzero((data.annotation[:, 0] != 0) & (data.annotation[:, 1] > 0)):
        compound, code = (int(value) for value in data.annotation[row, :2])
        template = compound_display_template(project, compound - 1)
        name = resolved_compound_name(
            template, project.compounds.composition_attr[compound - 1], phase, code
        ).split(" [", 1)[0]
        grouped.setdefault(name, []).append(float(data.rt_grouped[row]))
    rows = tuple(
        (name, float(np.mean(values)), phase) for name, values in sorted(grouped.items())
    )
    return PeakExport(("name", "RT", "column"), rows)


def write_tsv(table: PeakExport, destination: str | Path) -> Path:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(table.headers)
        writer.writerows(table.rows)
    return path


def write_xlsx(table: PeakExport, destination: str | Path) -> Path:
    """Write a portable Excel workbook without automating Microsoft Excel."""

    from openpyxl import Workbook

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet("peak data")
    sheet.append(table.headers)
    for row in table.rows:
        sheet.append(row)
    workbook.save(path)
    return path


def _sample_names(project: LegacyProject, phase: str) -> tuple[str, ...]:
    fields = project.metadata.fields
    names = np.asarray(fields.get("sample_name", []), dtype=object)
    columns = np.asarray(fields.get("column", []), dtype=object)
    if names.ndim == 2 and columns.shape == names.shape:
        matches = np.flatnonzero(columns[0] == phase)
        if len(matches):
            return tuple(str(value) for value in names[:, matches[0]])
    return tuple(f"sample_{index + 1}" for index in range(project.metadata.sample_count))
