"""Typed artifact descriptions shared by project writers and viewers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd

from extension_registry import ComponentRegistry
from errors import ArtifactContractError
from output_io import atomic_write_csv


def artifact_slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "table"


@dataclass(frozen=True)
class TableArtifactDefinition:
    """One stable top-level result table in a saved LASER project."""

    role: str
    result_key: str
    filename: str
    category: str = "table"
    required: bool = False

    def frame_from(self, result: Mapping[str, Any]) -> pd.DataFrame | None:
        value = result.get(self.result_key)
        if isinstance(value, pd.DataFrame):
            return value
        if self.required:
            raise ArtifactContractError(
                f"Required saved-project table {self.result_key!r} is absent."
            )
        return None


_STANDARD_CORE = (
    "identified_lipids",
    "equivalent_lipids",
    "automatic_annotations",
    "all_candidates",
    "policy_audit",
    "np_rt_audit",
    "grid_coherence_audit",
)


standard_table_artifacts: ComponentRegistry[TableArtifactDefinition] = (
    ComponentRegistry({
        key: TableArtifactDefinition(
            role=key,
            result_key=key,
            filename=f"{key}.csv",
            required=key in _STANDARD_CORE[:4],
        )
        for key in _STANDARD_CORE
    }, expected_type=TableArtifactDefinition, name_attribute="role")
)


def standard_result_table_definitions(
    result: Mapping[str, Any],
    registry: ComponentRegistry[TableArtifactDefinition] = (
        standard_table_artifacts
    ),
) -> tuple[TableArtifactDefinition, ...]:
    """Return registered tables plus automatically discovered extensions."""
    definitions = {item.result_key: item for _, item in registry.items()}
    for key, value in result.items():
        if (
            isinstance(value, pd.DataFrame)
            and not str(key).startswith("_")
            and key not in definitions
        ):
            slug = artifact_slug(key)
            definitions[str(key)] = TableArtifactDefinition(
                role=slug,
                result_key=str(key),
                filename=f"{slug}.csv",
            )
    ordered = tuple(definitions.values())
    roles = [definition.role for definition in ordered]
    if len(roles) != len(set(roles)):
        duplicates = sorted({role for role in roles if roles.count(role) > 1})
        raise ArtifactContractError(
            "Saved-project artifact roles must be unique: "
            + ", ".join(duplicates)
        )
    return ordered


def write_standard_result_tables(
    root: Path,
    result: Mapping[str, Any],
    registry: ComponentRegistry[TableArtifactDefinition] = (
        standard_table_artifacts
    ),
) -> dict[str, Path]:
    """Persist every declared/discovered result table atomically."""
    paths: dict[str, Path] = {}
    for definition in standard_result_table_definitions(result, registry):
        frame = definition.frame_from(result)
        if frame is None:
            continue
        paths[definition.role] = atomic_write_csv(
            root / definition.filename, frame,
        )
    return paths


__all__ = [
    "TableArtifactDefinition",
    "artifact_slug",
    "standard_result_table_definitions",
    "standard_table_artifacts",
    "write_standard_result_tables",
]
