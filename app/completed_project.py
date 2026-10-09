"""Read completed LASER Standard Lipidomics projects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from saved_project import SAVED_PROJECT_FILENAME
from saved_project_contract import normalize_saved_project_manifest


def _absolute_artifacts(root: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for role, raw in artifacts.items():
        entry = dict(raw or {})
        for field in ("path", "columnar_path"):
            value = entry.get(field)
            if not value:
                continue
            path = Path(str(value).replace("\\", "/"))
            if not path.is_absolute():
                path = root / path
            entry[field] = str(path.resolve())
        resolved[str(role)] = entry
    return resolved


def load_completed_project(directory: str | Path) -> dict[str, Any]:
    """Load and normalize one saved standard-lipidomics project."""
    root = Path(directory).expanduser().resolve()
    if root.is_file() and root.name == SAVED_PROJECT_FILENAME:
        root = root.parent
    manifest_path = root / SAVED_PROJECT_FILENAME
    if not manifest_path.is_file():
        raise ValueError(
            f"No {SAVED_PROJECT_FILENAME} was found in the selected folder: {root}"
        )
    try:
        manifest = normalize_saved_project_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"The LASER project manifest is invalid: {exc}") from exc
    if str(manifest.get("workflow", "")).casefold() != "standard":
        raise ValueError("The selected project is not a standard-lipidomics project.")
    artifacts = _absolute_artifacts(root, dict(manifest.get("artifacts") or {}))
    return {
        "root": str(root),
        "name": str(manifest.get("name") or root.name),
        "workflow": "standard",
        "report": (artifacts.get("standard_lipidomics_report") or {}).get("path"),
        "project_manifest": manifest,
        "artifacts": artifacts,
        "settings": dict(manifest.get("settings") or {}),
        "inputs": dict(manifest.get("inputs") or {}),
        "viewer": dict(manifest.get("viewer") or {}),
        "polarity": str(manifest.get("polarity") or "both").casefold(),
        "counts": dict(manifest.get("counts") or {}),
    }


__all__ = ["load_completed_project"]
