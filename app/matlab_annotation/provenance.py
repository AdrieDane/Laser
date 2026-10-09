"""Stable provenance projection for MATLAB-equivalent annotation results."""

from __future__ import annotations

from typing import Any, Mapping


def matlab_run_provenance(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return reproducibility fields without exposing mutable result tables."""

    return {
        "engine": str(result.get("engine") or "matlab"),
        "phases": list(result.get("phases") or ()),
        "reference_source": str(result.get("reference_source") or ""),
        "anchor_settings": dict(result.get("anchor_settings") or {}),
        "composition_range_settings": dict(
            result.get("composition_range_settings") or {}
        ),
        "grid_coherence_enabled": bool(result.get("grid_coherence_enabled", False)),
        "grid_coherence_threshold": result.get("grid_coherence_threshold"),
    }


__all__ = ["matlab_run_provenance"]
