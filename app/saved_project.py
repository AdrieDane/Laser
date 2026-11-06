"""Versioned, workflow-neutral persistence for completed LASER projects.

The saved project is the durable analytical result.  HTML reports are optional
exports referenced by the project, never the object used to reopen it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from output_io import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    sha256_file,
    with_runtime_environment,
)
from project_artifacts import write_standard_result_tables
from saved_project_contract import (
    ARTIFACT_CONTRACT_VERSION,
    CURRENT_SAVED_PROJECT_VERSION,
    normalize_saved_project_manifest,
)


SAVED_PROJECT_FORMAT_VERSION = CURRENT_SAVED_PROJECT_VERSION
SAVED_PROJECT_FILENAME = "LASER_PROJECT.json"


def _plain(value: Any) -> Any:
    """Return JSON-safe stable metadata, omitting live Python objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    raise TypeError(type(value).__name__)


def _plain_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in attrs.items():
        try:
            output[str(key)] = _plain(value)
        except TypeError:
            continue
    return output


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "stream"


def _relative(
    root: Path,
    value: str | Path | None,
    *,
    allow_parent_paths: bool = False,
) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError:
        if allow_parent_paths:
            try:
                return os.path.relpath(path, root.resolve())
            except ValueError:
                pass
        return str(path)


def _artifact(
    root: Path,
    path: str | Path,
    *,
    label: str,
    category: str,
    media_type: str = "text/csv",
    allow_parent_paths: bool = False,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(_relative(root, path, allow_parent_paths=allow_parent_paths)),
        "label": str(label),
        "category": str(category),
        "media_type": str(media_type),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_saved_project_manifest(
    project_directory: str | Path,
    *,
    workflow: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    counts: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
    annotation_manifest: str | Path | None = None,
    polarity: str | None = None,
    name: str | None = None,
    allow_parent_paths: bool = False,
) -> Path:
    """Write the canonical manifest last, after all referenced artifacts."""
    root = Path(project_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    normalized_artifacts: dict[str, dict[str, Any]] = {}
    for role, item in artifacts.items():
        entry = dict(item)
        path = entry.get("path")
        if not path:
            continue
        entry["path"] = _relative(root, path, allow_parent_paths=allow_parent_paths)
        normalized_artifacts[str(role)] = entry
    workflow_name = str(workflow).casefold()
    viewer_roles = {
        "4d": ("all_candidates", "retained_features"),
        "standard": ("all_candidates", "identified_lipids"),
        "consolidator": ("consolidated_results", "consolidated_results"),
    }
    candidate_role, retained_role = viewer_roles.get(
        workflow_name, ("all_candidates", "identified_lipids")
    )
    manifest = normalize_saved_project_manifest(
        {
            "format_version": SAVED_PROJECT_FORMAT_VERSION,
            "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
            "project_type": "LASER_saved_project",
            "name": str(name or root.name),
            "workflow": workflow_name,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "polarity": str(polarity or "unknown").casefold(),
            "annotation_manifest": _relative(root, annotation_manifest),
            "counts": dict(counts or {}),
            "settings": dict(settings or {}),
            "inputs": dict(inputs or {}),
            "artifacts": normalized_artifacts,
            "viewer": {
                "candidate_role": candidate_role,
                "retained_role": retained_role,
            },
        }
    )
    manifest = with_runtime_environment(manifest)
    return atomic_write_json(root / SAVED_PROJECT_FILENAME, manifest)


def create_saved_project_from_run(
    project_directory: str | Path,
    *,
    workflow: str,
    paths: Mapping[str, str | Path | None],
    counts: Mapping[str, Any],
    settings: Mapping[str, Any],
    inputs: Mapping[str, Any],
    annotation_manifest: str | Path | None,
    polarity: str | None = None,
    allow_parent_paths: bool = False,
    table_frames: Mapping[str, pd.DataFrame] | None = None,
) -> Path:
    """Register machine-readable run outputs as one reopenable LASER project."""
    root = Path(project_directory).expanduser().resolve()
    exclusions = {
        key for key in paths if "excluded" in key or "failure" in key or "audit" in key
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for role, value in paths.items():
        if value is None:
            continue
        path = Path(value)
        if not path.is_file():
            continue
        suffix = path.suffix.casefold()
        category = (
            "exclusion"
            if role in exclusions
            else "optional_report"
            if suffix == ".html"
            else "table"
            if suffix in {".csv", ".tsv", ".txt"}
            else "provenance"
        )
        media_type = {
            ".html": "text/html",
            ".json": "application/json",
            ".csv": "text/csv",
            ".tsv": "text/tab-separated-values",
            ".txt": "text/plain",
            ".sqlite": "application/vnd.sqlite3",
        }.get(suffix, "application/octet-stream")
        artifact = _artifact(
            root,
            path,
            label=str(role).replace("_", " ").title(),
            category=category,
            media_type=media_type,
            allow_parent_paths=allow_parent_paths,
        )
        if suffix in {".csv", ".tsv"}:
            try:
                table = table_frames.get(role) if table_frames is not None else None
                if table is None:
                    table = pd.read_csv(
                        path,
                        sep="\t" if suffix == ".tsv" else ",",
                        low_memory=False,
                    )
                columnar = atomic_write_parquet(
                    path.with_suffix(path.suffix + ".parquet"),
                    table,
                )
                artifact.update(
                    {
                        "columnar_path": str(
                            _relative(
                                root,
                                columnar,
                                allow_parent_paths=allow_parent_paths,
                            )
                        ),
                        "columnar_media_type": "application/vnd.apache.parquet",
                        "columnar_size_bytes": columnar.stat().st_size,
                        "columnar_sha256": sha256_file(columnar),
                    }
                )
            except Exception:
                # The portable CSV is authoritative. Unsupported mixed object
                # columns must never make project creation fail.
                pass
        artifacts[str(role)] = artifact
    return write_saved_project_manifest(
        root,
        workflow=workflow,
        artifacts=artifacts,
        counts=counts,
        settings=settings,
        inputs=inputs,
        annotation_manifest=annotation_manifest,
        polarity=polarity,
        allow_parent_paths=allow_parent_paths,
    )


def save_standard_project(
    project_directory: str | Path,
    result: Mapping[str, Any],
    *,
    settings: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
    annotated_inputs: Mapping[str, bytes] | None = None,
    html_report: str | None = None,
) -> dict[str, Path]:
    """Persist an in-memory standard annotation without rerunning it."""
    root = Path(project_directory).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            f"LASER project folder is not empty: {root}. Choose a new folder."
        )
    tables_dir = root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths = write_standard_result_tables(tables_dir, result)
    stream_metadata: dict[str, dict[str, Any]] = {}
    for stream, stream_result in dict(result.get("streams") or {}).items():
        slug = _slug(stream)
        metadata: dict[str, Any] = {}
        for key, value in dict(stream_result or {}).items():
            if isinstance(value, pd.DataFrame):
                role = f"stream__{slug}__{_slug(key)}"
                paths[role] = atomic_write_csv(
                    tables_dir / "streams" / slug / f"{_slug(key)}.csv", value
                )
                metadata[f"{key}__attrs"] = _plain_attrs(value.attrs)
            else:
                try:
                    metadata[str(key)] = _plain(value)
                except TypeError:
                    continue
        stream_metadata[str(stream)] = {"slug": slug, "values": metadata}
    if stream_metadata:
        result_values: dict[str, Any] = {}
        for key, value in result.items():
            if key == "streams" or isinstance(value, pd.DataFrame):
                continue
            try:
                result_values[str(key)] = _plain(value)
            except TypeError:
                continue
        paths["standard_stream_metadata"] = atomic_write_json(
            root / "standard_stream_metadata.json",
            {
                "engine": str(result.get("engine", "unknown")),
                "anchor_settings": _plain(result.get("anchor_settings", {})),
                "result_values": result_values,
                "streams": stream_metadata,
            },
        )
    if annotated_inputs:
        input_dir = root / "annotated_peaklists"
        for stream, content in annotated_inputs.items():
            path = input_dir / f"peakgrouplist_{stream}_annotated.txt"
            paths[f"annotated_input_{stream}"] = atomic_write_bytes(path, content)
    if html_report is not None:
        paths["standard_lipidomics_report"] = atomic_write_bytes(
            root / "exports" / "standard_lipidomics_report.html",
            html_report.encode("utf-8"),
        )
    counts = {
        "input_streams": len(result.get("streams", {})),
        "identified_matches": len(result.get("identified_lipids", ())),
        "equivalent_matches": len(result.get("equivalent_lipids", ())),
        "candidate_matches": len(result.get("all_candidates", ())),
    }
    run_manifest = atomic_write_json(
        root / "run_manifest.json",
        {
            "format_version": 1,
            "workflow": "standard",
            "engine": str(result.get("engine", "unknown")),
            "counts": counts,
            "settings": dict(settings or {}),
            "inputs": dict(inputs or {}),
            "outputs": {key: str(value.relative_to(root)) for key, value in paths.items()},
        },
    )
    paths["project_manifest"] = create_saved_project_from_run(
        root,
        workflow="standard",
        paths=paths,
        counts=counts,
        settings=dict(settings or {}),
        inputs=dict(inputs or {}),
        annotation_manifest=run_manifest,
        polarity="both",
    )
    return paths


def resolve_artifact_path(project: Mapping[str, Any], role: str) -> Path | None:
    entry = dict(project.get("artifacts", {}).get(role) or {})
    value = entry.get("path")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(project["root"]) / path
    return path.resolve()


@lru_cache(maxsize=128)
def _read_project_table_cached(
    path_value: str,
    modified_ns: int,
    size_bytes: int,
    kind: str,
) -> pd.DataFrame:
    """Read an immutable project table once per on-disk file revision."""
    del modified_ns, size_bytes  # Values intentionally participate in the key.
    path = Path(path_value)
    if kind == "parquet":
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_csv(
        path,
        sep="\t" if kind == "tsv" else ",",
        low_memory=False,
    )


def _cached_table(path: Path, kind: str) -> pd.DataFrame:
    stat = path.stat()
    # Callers that need to modify a project table already make an explicit
    # copy.  A shallow copy protects the cached DataFrame object itself while
    # avoiding a second full allocation for large annotation tables.
    return _read_project_table_cached(
        str(path), stat.st_mtime_ns, stat.st_size, kind
    ).copy(deep=False)


def load_project_table(project: Mapping[str, Any], role: str) -> pd.DataFrame:
    entry = dict(project.get("artifacts", {}).get(role) or {})
    columnar_value = entry.get("columnar_path")
    columnar = None
    if columnar_value:
        columnar = Path(columnar_value)
        if not columnar.is_absolute():
            columnar = Path(project["root"]) / columnar
        columnar = columnar.resolve()
    if columnar is not None and columnar.is_file():
        try:
            return _cached_table(columnar, "parquet")
        except (OSError, ValueError, ImportError):
            pass
    path = resolve_artifact_path(project, role)
    if path is None or not path.is_file():
        return pd.DataFrame()
    try:
        kind = "tsv" if path.suffix.casefold() == ".tsv" else "csv"
        return _cached_table(path, kind)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
