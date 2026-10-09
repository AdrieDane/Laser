"""Typed model for legacy reference tables and runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class ReferenceTable:
    name: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class StableIsotope:
    atom: str
    mass: float
    abundance: float


@dataclass(frozen=True)
class RetentionTimeObservation:
    trait: str
    retention_time_seconds: float
    phase: str
    study: str


@dataclass(frozen=True)
class ProjectInternalStandard:
    name: str
    quantity: float
    unit: str


@dataclass(frozen=True)
class ReferenceData:
    code_dir: Path
    database_dir: Path
    rtdb_dir: Path
    project_dir: Path
    parameters: Mapping[str, Any]
    parameter_expression_paths: tuple[str, ...]
    database_tables: Mapping[str, ReferenceTable]
    stable_isotopes: tuple[StableIsotope, ...]
    retention_times: tuple[RetentionTimeObservation, ...]
    rtdb_studies: tuple[str, ...]
    internal_standards: tuple[ProjectInternalStandard, ...]
    adduct_tokens: tuple[str, ...]
    r_group_names: tuple[str, ...]
    source_sha256: Mapping[str, str]
    load_seconds: float


def immutable_mapping(value: dict[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(value)
