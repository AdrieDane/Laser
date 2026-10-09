"""Declarative Streamlit workspace sections."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

import streamlit as st

from extension_registry import OrderedRegistry


ContextT = TypeVar("ContextT")


@dataclass(frozen=True)
class WorkspaceSection(Generic[ContextT]):
    """One independently renderable native-project tab."""

    key: str
    title: str
    render: Callable[[ContextT], None]


@dataclass(frozen=True)
class WorkspaceSectionRegistry(Generic[ContextT]):
    """Immutable ordered registry for independently renderable workspace tabs."""

    _registry: OrderedRegistry[WorkspaceSection[ContextT]]

    def __init__(self, sections: Iterable[WorkspaceSection[ContextT]] = ()):
        object.__setattr__(self, "_registry", OrderedRegistry(sections))

    @property
    def sections(self) -> tuple[WorkspaceSection[ContextT], ...]:
        return self._registry.items

    def with_section(
        self,
        section: WorkspaceSection[ContextT],
        *,
        after: str | None = None,
    ) -> "WorkspaceSectionRegistry[ContextT]":
        return WorkspaceSectionRegistry(
            self._registry.with_item(section, after=after).items
        )

    def __contains__(self, key: object) -> bool:
        return key in self._registry

    def __iter__(self) -> Iterator[WorkspaceSection[ContextT]]:
        return iter(self.sections)


def render_workspace_sections(
    sections: Iterable[WorkspaceSection[ContextT]]
    | WorkspaceSectionRegistry[ContextT],
    context: ContextT,
) -> None:
    """Render sections without positional tab coupling."""
    ordered = WorkspaceSectionRegistry(sections).sections
    tabs = st.tabs([section.title for section in ordered])
    for section, tab in zip(ordered, tabs):
        with tab:
            section.render(context)


__all__ = [
    "WorkspaceSection", "WorkspaceSectionRegistry", "render_workspace_sections",
]
