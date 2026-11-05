"""Read-only loader for legacy TSV/JSON reference data and configuration."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from standard_lipidomics.internal_standards import read_internal_standards_csv

from .reference import (
    ProjectInternalStandard,
    ReferenceData,
    ReferenceTable,
    RetentionTimeObservation,
    StableIsotope,
    immutable_mapping,
)


DEFAULT_ADDUCT_TOKENS = (
    "0_pos", "0_neg", "0_pos2", "0_neg2", "0_pos3", "0_neg3",
    "0_pos4", "0_neg4", "NH3_pos", "NH3_pos2", "HCl_neg",
    "CH2O2_neg", "Na_pos", "K_pos", "0-H2O_pos", "0-H2O_neg",
    "Na_neg", "Li_pos", "Ca_pos2", "Mg_pos2",
)
R_GROUP_NAMES = ("FA", "FAfull", "ST", "isoprene")
DATABASE_FILES = (
    "internal_standards_all.tsv", "lipids.tsv", "lipids_info.tsv",
    "metabolites.tsv", "metabolites_info.tsv", "stable_isotopes.tsv",
)


class ReferenceDataError(ValueError):
    """Raised when legacy reference/config inputs violate their file contract."""


def load_reference_data(
    *,
    code_dir: str | Path,
    database_dir: str | Path,
    rtdb_dir: str | Path,
    project_dir: str | Path,
) -> ReferenceData:
    started = perf_counter()
    roots = {
        "code": Path(code_dir).expanduser().resolve(),
        "database": Path(database_dir).expanduser().resolve(),
        "rtdb": Path(rtdb_dir).expanduser().resolve(),
        "project": Path(project_dir).expanduser().resolve(),
    }
    for label, root in roots.items():
        if not root.is_dir():
            raise ReferenceDataError(f"{label} directory does not exist: {root}")
    files = [roots["code"] / "param.json"]
    files.extend(roots["database"] / name for name in DATABASE_FILES)
    files.extend(sorted(roots["rtdb"].glob("lipidomics_*.tsv")))
    csv_path = roots["project"] / "internal_standards.csv"
    files.append(csv_path if csv_path.is_file() else roots["project"] / "internal_standards.tsv")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ReferenceDataError(f"missing reference files: {', '.join(missing)}")
    before = {_relative_key(path, roots): _sha256(path) for path in files}

    parameters = _load_matlab_json(roots["code"] / "param.json")
    expression_paths = tuple(_find_expression_paths(parameters))
    tables = {name: _load_table(roots["database"] / name) for name in DATABASE_FILES}
    isotopes = _load_stable_isotopes(tables["stable_isotopes.tsv"])
    retention_times, studies = _load_retention_times(files[1 + len(DATABASE_FILES):-1])
    project_standards = _load_project_internal_standards(files[-1])
    _validate_internal_standard_names(project_standards, tables["internal_standards_all.tsv"])

    after = {_relative_key(path, roots): _sha256(path) for path in files}
    if after != before:
        raise ReferenceDataError("reference source changed while loading")
    return ReferenceData(
        code_dir=roots["code"],
        database_dir=roots["database"],
        rtdb_dir=roots["rtdb"],
        project_dir=roots["project"],
        parameters=immutable_mapping(parameters),
        parameter_expression_paths=expression_paths,
        database_tables=immutable_mapping(tables),
        stable_isotopes=isotopes,
        retention_times=retention_times,
        rtdb_studies=studies,
        internal_standards=project_standards,
        adduct_tokens=DEFAULT_ADDUCT_TOKENS,
        r_group_names=R_GROUP_NAMES,
        source_sha256=immutable_mapping(before),
        load_seconds=perf_counter() - started,
    )


def _load_table(path: Path) -> ReferenceTable:
    with path.open("r", encoding="latin-1", newline="") as handle:
        values = list(csv.reader(handle, delimiter="\t"))
    if not values:
        raise ReferenceDataError(f"empty table: {path}")
    headers = tuple(value.strip() for value in values[0])
    rows = []
    for line, row in enumerate(values[1:], start=2):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != len(headers):
            raise ReferenceDataError(
                f"{path.name}:{line} has {len(row)} columns; expected {len(headers)}"
            )
        rows.append(tuple(value.strip() for value in row))
    return ReferenceTable(path.name, headers, tuple(rows))


def _load_stable_isotopes(table: ReferenceTable) -> tuple[StableIsotope, ...]:
    if table.headers != ("atom", "mass", "abundance"):
        raise ReferenceDataError("stable_isotopes.tsv has unexpected headers")
    result = []
    for row in table.rows:
        try:
            result.append(StableIsotope(row[0], float(row[1]), float(row[2])))
        except ValueError as exc:
            raise ReferenceDataError(f"invalid stable isotope row: {row}") from exc
    if len({item.atom for item in result}) != len(result):
        raise ReferenceDataError("stable isotope atom labels are not unique")
    return tuple(result)


def _load_retention_times(paths: list[Path]) -> tuple[tuple[RetentionTimeObservation, ...], tuple[str, ...]]:
    observations = []
    studies = []
    for path in paths:
        table = _load_table(path)
        if table.headers not in (("name", "RT", "column"), ("trait", "RT", "column")):
            raise ReferenceDataError(f"{path.name} has unexpected RTDB headers")
        studies.append(path.stem.replace("_", " "))
        parsed = []
        for trait, retention_time, phase in table.rows:
            if phase not in {"NP", "RP", "HILIC"}:
                raise ReferenceDataError(f"{path.name} has unknown phase {phase!r}")
            try:
                value = float(retention_time)
            except ValueError as exc:
                raise ReferenceDataError(f"{path.name} has non-numeric RT {retention_time!r}") from exc
            if value == 0:
                continue
            parsed.append((trait.strip(), phase, value))
        # Legacy import averages duplicate trait+phase observations within each study.
        grouped: dict[tuple[str, str], list[float]] = {}
        for trait, phase, value in parsed:
            grouped.setdefault((trait, phase), []).append(value)
        maximum = max((value for _, _, value in parsed), default=math.nan)
        scale = 60.0 if maximum < 50 else 1.0
        observations.extend(
            RetentionTimeObservation(trait, sum(values) / len(values) * scale, phase, studies[-1])
            for (trait, phase), values in sorted(grouped.items())
        )
    return tuple(observations), tuple(studies)


def _load_project_internal_standards(path: Path) -> tuple[ProjectInternalStandard, ...]:
    if path.suffix.casefold() == ".csv":
        try:
            return tuple(ProjectInternalStandard(item.name, item.concentration, item.unit)
                         for item in read_internal_standards_csv(path))
        except ValueError as exc:
            raise ReferenceDataError(str(exc)) from exc
    table = _load_table(path)
    if len(table.headers) != 2 or not table.headers[1]:
        raise ReferenceDataError("internal_standards.tsv must have name and unit columns")
    unit = table.headers[1]
    result = []
    for name, quantity in table.rows:
        if not quantity:
            continue
        try:
            result.append(ProjectInternalStandard(name, float(quantity), unit))
        except ValueError as exc:
            raise ReferenceDataError(f"invalid internal-standard quantity for {name}") from exc
    if len({item.name.casefold() for item in result}) != len(result):
        raise ReferenceDataError("project internal-standard names are not unique")
    return tuple(result)


def _validate_internal_standard_names(
    project_standards: tuple[ProjectInternalStandard, ...],
    database: ReferenceTable,
) -> None:
    if database.headers[:2] != ("name", "trait") or "name_old" not in database.headers:
        raise ReferenceDataError("internal_standards_all.tsv has unexpected headers")
    old_index = database.headers.index("name_old")
    known = {
        value.casefold()
        for row in database.rows
        for value in (row[0], row[old_index])
        if value
    }
    missing = [item.name for item in project_standards if item.name.casefold() not in known]
    if missing:
        raise ReferenceDataError(
            f"project internal standards absent from database: {', '.join(missing)}"
        )


def _load_matlab_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="latin-1")
    normalized = _replace_inf_outside_strings(text)
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ReferenceDataError(f"invalid MATLAB JSON in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReferenceDataError("param.json root must be an object")
    return value


def _replace_inf_outside_strings(text: str) -> str:
    output = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
        elif text.startswith("-Inf", index):
            output.append("-Infinity")
            index += 4
        elif text.startswith("Inf", index):
            output.append("Infinity")
            index += 3
        else:
            output.append(char)
            index += 1
    return "".join(output)


def _find_expression_paths(value: Any, path: str = ""):
    if isinstance(value, dict):
        for name, child in value.items():
            child_path = f"{path}.{name}" if path else name
            yield from _find_expression_paths(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _find_expression_paths(child, f"{path}[{index}]")
    elif isinstance(value, str) and value.startswith("@"):
        yield path


def _relative_key(path: Path, roots: dict[str, Path]) -> str:
    for label, root in roots.items():
        try:
            return f"{label}/{path.relative_to(root)}"
        except ValueError:
            pass
    return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
