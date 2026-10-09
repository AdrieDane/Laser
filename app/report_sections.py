"""Declarative, ordered report-section extension point.

Report builders should describe *which* sections a document contains and keep
each section renderer independently testable.  This module deliberately has
no knowledge of Streamlit or a concrete scientific workflow.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

from extension_registry import OrderedRegistry
from test_report import ReportTab


ContextT = TypeVar("ContextT")


@dataclass(frozen=True)
class ReportSection(Generic[ContextT]):
    """One independently renderable HTML-report section."""

    key: str
    title: str
    render: Callable[[ContextT], str]
    help_text: str = ""


@dataclass(frozen=True)
class ReportSectionRegistry(Generic[ContextT]):
    """Immutable ordered report-section registry.

    Order is part of report usability, so it is intentionally preserved while
    normalized keys provide stable lookup and collision detection.
    """

    _registry: OrderedRegistry[ReportSection[ContextT]]

    def __init__(self, sections: Iterable[ReportSection[ContextT]] = ()):
        object.__setattr__(self, "_registry", OrderedRegistry(sections))

    @property
    def sections(self) -> tuple[ReportSection[ContextT], ...]:
        return self._registry.items

    def with_section(
        self,
        section: ReportSection[ContextT],
        *,
        after: str | None = None,
    ) -> "ReportSectionRegistry[ContextT]":
        """Return a registry containing ``section`` at a stable location."""
        return ReportSectionRegistry(
            self._registry.with_item(section, after=after).items
        )

    def tabs(self, context: ContextT) -> list[ReportTab]:
        return [
            ReportTab(
                section.key,
                section.title,
                section.render(context),
                section.help_text,
            )
            for section in self.sections
        ]

    def __iter__(self) -> Iterator[ReportSection[ContextT]]:
        return iter(self.sections)

    def __contains__(self, key: object) -> bool:
        return key in self._registry


__all__ = ["ReportSection", "ReportSectionRegistry"]
