"""Streamlit presentation components for standard lipidomics."""

from .workspace import (
    WorkspaceSection,
    WorkspaceSectionRegistry,
    render_workspace_sections,
)

__all__ = [
    "WorkspaceSection",
    "WorkspaceSectionRegistry",
    "render_workspace_sections",
]
