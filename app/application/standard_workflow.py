"""Typed Standard Lipidomics application workflow and engine adapters.

Both Streamlit and configuration-driven execution call this module.  Engine
selection, settings translation, reference loading, and result validation are
therefore defined once and tested independently of either presentation layer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Protocol, cast, runtime_checkable

import pandas as pd

from errors import AnnotationStageError, WorkflowConfigurationError
from annotator.composition_ranges import (
    CompositionRangeSettings,
    composition_range_settings_from_dict,
)
from standard_lipidomics.workflow import discover_standard_streams

from .registry import ComponentRegistry


LASER_STANDARD_ENGINE = "laser"
MATLAB_STANDARD_ENGINE = "matlab"
STANDARD_RESULT_TABLES = (
    "all_candidates",
    "automatic_annotations",
    "identified_lipids",
    "equivalent_lipids",
    "policy_audit",
    "np_rt_audit",
    "grid_coherence_audit",
)

Progress = Callable[[float, str], None]
StandardSource = str | Path | bytes | BinaryIO


@dataclass(frozen=True)
class StandardAnnotationRequest:
    """Complete engine-neutral request for one conventional annotation."""

    engine: str
    project: Path
    settings: Mapping[str, object] = field(default_factory=dict)
    sources: Mapping[str, StandardSource] | None = None
    reference_path: Path | None = None
    matlab_reference_path: Path | None = None
    matlab_policy_path: Path | None = None
    np_rt_reference_path: Path | None = None
    progress: Progress | None = None

    def __post_init__(self) -> None:
        project = Path(self.project).expanduser().resolve()
        object.__setattr__(self, "project", project)
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))
        if self.sources is not None:
            object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))
        if not str(self.engine).strip():
            raise WorkflowConfigurationError(
                "standard annotation engine cannot be empty"
            )

    @property
    def composition_ranges(self) -> CompositionRangeSettings:
        return composition_range_settings_from_dict(
            cast(Mapping[str, object] | None, self.settings.get("composition_ranges"))
        )

    def resolved_sources(self) -> Mapping[str, StandardSource]:
        sources = self.sources or discover_standard_streams(self.project)
        if not sources:
            raise WorkflowConfigurationError(
                "No MATLAB/XCMS peakgrouplist streams were found."
            )
        return sources


@dataclass(frozen=True)
class StandardAnnotationResult:
    """Validated, engine-independent Standard Lipidomics result."""

    engine: str
    streams: Mapping[str, Mapping[str, object]]
    tables: Mapping[str, pd.DataFrame]
    metadata: Mapping[str, object]

    @classmethod
    def from_mapping(
        cls,
        engine: str,
        payload: Mapping[str, object],
    ) -> "StandardAnnotationResult":
        missing = [
            name
            for name in STANDARD_RESULT_TABLES[:6]
            if not isinstance(payload.get(name), pd.DataFrame)
        ]
        if missing:
            raise AnnotationStageError(
                f"Standard engine {engine!r} omitted required table(s): "
                + ", ".join(missing)
            )
        tables = {
            key: value
            for key, value in payload.items()
            if isinstance(value, pd.DataFrame)
        }
        tables.setdefault("grid_coherence_audit", pd.DataFrame())
        raw_streams = payload.get("streams", {})
        streams = raw_streams if isinstance(raw_streams, Mapping) else {}
        metadata = {
            key: value
            for key, value in payload.items()
            if key != "streams" and not isinstance(value, pd.DataFrame)
        }
        return cls(
            engine=str(engine),
            streams=MappingProxyType(dict(streams)),
            tables=MappingProxyType(tables),
            metadata=MappingProxyType(metadata),
        )

    @property
    def identified_lipids(self) -> pd.DataFrame:
        return self.tables["identified_lipids"]

    @property
    def all_candidates(self) -> pd.DataFrame:
        return self.tables["all_candidates"]

    def to_mapping(self) -> dict[str, object]:
        """Compatibility view for existing reports and saved projects."""
        return {
            "streams": dict(self.streams),
            **dict(self.tables),
            **dict(self.metadata),
            "standard_engine": self.engine,
        }


@runtime_checkable
class StandardEngine(Protocol):
    """Substitutable annotation engine boundary."""

    name: str

    def annotate(
        self,
        request: StandardAnnotationRequest,
    ) -> StandardAnnotationResult:
        """Return a validated engine-neutral annotation result."""


class StandardAnnotationService:
    """Engine-neutral Standard Lipidomics application service."""

    def __init__(
        self,
        engines: ComponentRegistry[StandardEngine],
    ) -> None:
        self._engines = engines

    @property
    def engine_names(self) -> tuple[str, ...]:
        return self._engines.names()

    def run(
        self,
        request: StandardAnnotationRequest,
    ) -> StandardAnnotationResult:
        try:
            engine = self._engines.require(request.engine)
        except ValueError as exc:
            raise WorkflowConfigurationError(str(exc)) from exc
        try:
            result = engine.annotate(request)
        except (AnnotationStageError, WorkflowConfigurationError):
            raise
        except Exception as exc:
            raise AnnotationStageError(
                f"Standard annotation engine {engine.name!r} failed: {exc}"
            ) from exc
        if result.engine != engine.name:
            raise AnnotationStageError(
                f"Engine {engine.name!r} returned result for {result.engine!r}."
            )
        return result
