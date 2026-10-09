"""Typed, read-only domain model for loaded legacy projects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CompoundCatalog:
    count: int
    classes: tuple[str, ...]
    class_names: tuple[str, ...]
    combined: tuple[str, ...]
    priority: Mapping[str, NDArray[np.floating[Any]]]
    accepted: Mapping[str, NDArray[np.floating[Any]]]
    interpolation: Mapping[str, NDArray[np.floating[Any]]]
    composition_attr: tuple[Any, ...]
    inliers: tuple[Any, ...]
    outliers: tuple[Any, ...]
    legacy_fields: Mapping[str, Any]


@dataclass(frozen=True)
class PhaseData:
    name: str
    mz_grouped: NDArray[np.floating[Any]]
    rt_grouped: NDArray[np.floating[Any]]
    representative_intensity: NDArray[np.floating[Any]]
    grouped_peak_count: NDArray[np.floating[Any]]
    predicted_rt: NDArray[np.floating[Any]]
    annotation: NDArray[np.floating[Any]]
    sample_intensity: NDArray[np.floating[Any]]
    rt_axis: NDArray[np.floating[Any]]
    pattern_values: NDArray[np.floating[Any]] | None = None

    @property
    def peak_count(self) -> int:
        return int(self.mz_grouped.shape[0])

    @property
    def annotated_peak_count(self) -> int:
        return int(np.count_nonzero(self.annotation[:, 0]))

    @property
    def internal_standard_peak_count(self) -> int:
        return int(np.count_nonzero(self.annotation[:, 1] < 0))


@dataclass(frozen=True)
class SampleMetadata:
    sample_count: int
    acquisition_columns: tuple[str, ...]
    fields: Mapping[str, Any]


@dataclass(frozen=True)
class LegacyProject:
    source: Path
    compounds: CompoundCatalog
    phases: Mapping[str, PhaseData]
    metadata: SampleMetadata
    source_sha256: Mapping[str, str]
    load_seconds: float
    array_bytes: int

    @property
    def phase_names(self) -> tuple[str, ...]:
        return tuple(self.phases)


def readonly_mapping(value: dict[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(value)
