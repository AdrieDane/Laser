"""Typed, migratable contract for durable LASER project manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


CURRENT_SAVED_PROJECT_VERSION = 2
ARTIFACT_CONTRACT_VERSION = 1


def _mapping(value: object, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Saved-project field '{field_name}' must be an object.")
    return {str(key): item for key, item in value.items()}


def _text(value: object, field_name: str, *, default: str = "") -> str:
    text = str(value if value is not None else default).strip()
    if not text:
        raise ValueError(f"Saved-project field '{field_name}' must not be empty.")
    return text


@dataclass(frozen=True)
class SavedProjectArtifact:
    """Portable artifact reference with room for format-specific extensions."""

    path: str
    label: str
    category: str
    media_type: str
    extensions: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, role: str
    ) -> "SavedProjectArtifact":
        entry = _mapping(value, f"artifacts.{role}")
        known = {"path", "label", "category", "media_type"}
        return cls(
            path=_text(entry.get("path"), f"artifacts.{role}.path"),
            label=_text(entry.get("label"), f"artifacts.{role}.label", default=role),
            category=_text(
                entry.get("category"),
                f"artifacts.{role}.category",
                default="provenance",
            ),
            media_type=_text(
                entry.get("media_type"),
                f"artifacts.{role}.media_type",
                default="application/octet-stream",
            ),
            extensions={key: item for key, item in entry.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "label": self.label,
            "category": self.category,
            "media_type": self.media_type,
            **self.extensions,
        }


@dataclass(frozen=True)
class SavedProjectManifest:
    """Current in-memory representation of any supported manifest version."""

    name: str
    workflow: str
    polarity: str
    annotation_manifest: str | None
    counts: dict[str, Any]
    settings: dict[str, Any]
    inputs: dict[str, Any]
    artifacts: dict[str, SavedProjectArtifact]
    viewer: dict[str, Any]
    extensions: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SavedProjectManifest":
        manifest = _mapping(value, "manifest")
        if manifest.get("project_type") != "LASER_saved_project":
            raise ValueError("The selected manifest is not a LASER saved project.")
        if manifest.get("status", "complete") != "complete":
            raise ValueError("The selected LASER saved project is not complete.")
        artifacts = {
            role: SavedProjectArtifact.from_mapping(entry, role=role)
            for role, entry in _mapping(manifest.get("artifacts"), "artifacts").items()
        }
        known = {
            "format_version",
            "artifact_contract_version",
            "project_type",
            "name",
            "workflow",
            "status",
            "polarity",
            "annotation_manifest",
            "counts",
            "settings",
            "inputs",
            "artifacts",
            "viewer",
        }
        annotation_manifest = manifest.get("annotation_manifest")
        return cls(
            name=_text(manifest.get("name"), "name", default="LASER project"),
            workflow=_text(manifest.get("workflow"), "workflow").casefold(),
            polarity=str(manifest.get("polarity", "unknown")).casefold(),
            annotation_manifest=(
                str(annotation_manifest) if annotation_manifest else None
            ),
            counts=_mapping(manifest.get("counts"), "counts"),
            settings=_mapping(manifest.get("settings"), "settings"),
            inputs=_mapping(manifest.get("inputs"), "inputs"),
            artifacts=artifacts,
            viewer=_mapping(manifest.get("viewer"), "viewer"),
            extensions={
                key: item for key, item in manifest.items() if key not in known
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": CURRENT_SAVED_PROJECT_VERSION,
            "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
            "project_type": "LASER_saved_project",
            "name": self.name,
            "workflow": self.workflow,
            "status": "complete",
            "polarity": self.polarity,
            "annotation_manifest": self.annotation_manifest,
            "counts": self.counts,
            "settings": self.settings,
            "inputs": self.inputs,
            "artifacts": {
                role: artifact.to_dict() for role, artifact in self.artifacts.items()
            },
            "viewer": self.viewer,
            **self.extensions,
        }


def _migrate_v1_to_v2(manifest: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(manifest)
    migrated["format_version"] = 2
    migrated.setdefault("artifact_contract_version", ARTIFACT_CONTRACT_VERSION)
    migrated.setdefault("status", "complete")
    migrated.setdefault("settings", {})
    migrated.setdefault("inputs", {})
    migrated.setdefault("viewer", {})
    return migrated


def normalize_saved_project_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade a supported manifest and validate it through the typed contract."""
    manifest = _mapping(value, "manifest")
    try:
        version = int(manifest.get("format_version", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Saved-project format_version must be an integer.") from exc
    if version < 1 or version > CURRENT_SAVED_PROJECT_VERSION:
        raise ValueError("This LASER project format is not supported by this version.")
    if version == 1:
        manifest = _migrate_v1_to_v2(manifest)
    return SavedProjectManifest.from_mapping(manifest).to_dict()


__all__ = [
    "ARTIFACT_CONTRACT_VERSION",
    "CURRENT_SAVED_PROJECT_VERSION",
    "SavedProjectArtifact",
    "SavedProjectManifest",
    "normalize_saved_project_manifest",
]
