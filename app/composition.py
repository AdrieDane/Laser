"""Application composition root for the standard-lipidomics tool."""

from __future__ import annotations

from application.standard_workflow import StandardAnnotationService
from adapters.standard_engines import standard_engine_registry

standard_annotation_service = StandardAnnotationService(
    engines=standard_engine_registry,
)

__all__ = ["standard_annotation_service"]
