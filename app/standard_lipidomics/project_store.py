"""Save and reopen the two LASER workflows inside an XCMS project."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

import pandas as pd
import numpy as np

from completed_project import load_completed_project
from output_io import atomic_write_csv, atomic_write_json
from saved_project import (
    create_saved_project_from_run,
    load_project_table,
    resolve_artifact_path,
    save_standard_project,
)


def _json_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, datetime):
            return value.isoformat()
        if value is pd.NA:
            return None
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    output: dict[str, Any] = {}
    for key, value in attrs.items():
        try:
            output[str(key)] = json.loads(json.dumps(value, default=convert))
        except (TypeError, ValueError):
            continue
    return output


def xcms_project_root(path: str | Path) -> Path:
    """Use the project folder, even when its old Processing/Annotation is selected."""
    root = Path(path).expanduser().resolve()
    if root.is_file():
        root = root.parent
    if root.name.casefold() == "annotation" and root.parent.name.casefold() == "processing":
        return root.parent.parent
    if root.name.casefold() in {"metadata", "laser_calibrated_xcms"}:
        return root.parent
    return root


def workflow_folder(project: str | Path, workflow: str) -> Path:
    if workflow not in {"Annotation", "Untargeted"}:
        raise ValueError(f"Unknown LASER workflow: {workflow}")
    return xcms_project_root(project) / "LASER" / workflow


def stage_project_metadata(
    project: str | Path, sources: Mapping[str, str | Path | bytes],
) -> dict[str, Path]:
    """Keep portable copies of the XCMS lists and LASER sample metadata."""
    root = xcms_project_root(project)
    destination = root / "metadata"
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    source_directories: set[Path] = {root, root / "Processing" / "Annotation"}
    for stream, value in sources.items():
        if isinstance(value, bytes):
            target = destination / f"peakgrouplist_{stream}.txt"
            target.write_bytes(value)
        else:
            source = Path(value).expanduser().resolve()
            if not source.is_file():
                raise ValueError(f"XCMS peak-group list is missing: {source}")
            source_directories.add(source.parent)
            target = destination / source.name
            if source != target:
                shutil.copy2(source, target)
        copied[str(stream)] = target
    for folder in source_directories:
        if not folder.is_dir() or folder == destination:
            continue
        metadata_files = (
            list(folder.glob("*.xlsx")) + list(folder.glob("*.xls"))
            + list(folder.glob("internal_standards.tsv"))
        )
        for source in metadata_files:
            if source.name.startswith("~$"):
                continue
            target = destination / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
    return copied


def _save_with_history(
    destination: Path, writer: Callable[[Path], dict[str, Path]], *,
    existing: str,
) -> tuple[dict[str, Path], Path | None]:
    """Move the previous version aside and restore it if writing fails."""
    if existing not in {"archive", "overwrite"}:
        raise ValueError("Choose archive or overwrite for an existing export.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    moved: Path | None = None
    archive: Path | None = None
    if destination.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        parent = destination.parent / "Archive" / destination.name
        parent.mkdir(parents=True, exist_ok=True)
        moved = parent / stamp
        destination.rename(moved)
        if existing == "archive":
            archive = moved
    try:
        paths = writer(destination)
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        if moved is not None:
            moved.rename(destination)
        raise
    if moved is not None and existing == "overwrite":
        shutil.rmtree(moved)
    return paths, archive


def save_annotation_workflow(
    project: str | Path, result: Mapping[str, Any], *,
    settings: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
    annotated_inputs: Mapping[str, bytes] | None = None,
    html_report: str | None = None,
    existing: str = "archive",
) -> tuple[dict[str, Path], Path | None]:
    destination = workflow_folder(project, "Annotation")
    return _save_with_history(
        destination,
        lambda root: save_standard_project(
            root, result, settings=settings, inputs=inputs,
            annotated_inputs=annotated_inputs, html_report=html_report,
        ),
        existing=existing,
    )


def _read_metadata(project: Mapping[str, Any]) -> dict[str, Any]:
    path = resolve_artifact_path(project, "standard_stream_metadata")
    if path is None or not path.is_file():
        raise ValueError("The saved project is missing its stream metadata.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_annotation_workflow(project: str | Path) -> tuple[dict, dict]:
    """Reconstruct the analysis result from saved tables, without annotation."""
    saved = load_completed_project(workflow_folder(project, "Annotation"))
    metadata = _read_metadata(saved)
    result: dict[str, Any] = dict(metadata.get("result_values") or {})
    result["engine"] = metadata.get("engine", "matlab")
    result["anchor_settings"] = metadata.get("anchor_settings", {})
    result["streams"] = {}
    stream_roles: set[str] = set()
    for name, item in dict(metadata.get("streams") or {}).items():
        slug = str(item.get("slug", name))
        values = dict(item.get("values") or {})
        stream_result: dict[str, Any] = {
            key: value for key, value in values.items()
            if not key.endswith("__attrs")
        }
        prefix = f"stream__{slug}__"
        for role in saved["artifacts"]:
            if role.startswith(prefix):
                key = role[len(prefix):]
                frame = load_project_table(saved, role)
                frame.attrs.update(values.get(f"{key}__attrs", {}))
                stream_result[key] = frame
                stream_roles.add(role)
        result["streams"][name] = stream_result
    for role, item in saved["artifacts"].items():
        if role in stream_roles or role in {"standard_stream_metadata", "run_manifest"}:
            continue
        if role.startswith("annotated_input_") or role == "standard_lipidomics_report":
            continue
        if str(item.get("path", "")).endswith(".csv"):
            result[role] = load_project_table(saved, role)
    result.setdefault("xcms_input_dir", saved["inputs"].get("project"))
    return result, saved


def _write_untargeted_project(
    root: Path, result: Mapping[str, Any], analysis: Mapping[str, Any], *,
    inputs: Mapping[str, Any] | None,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    tables = root / "tables"
    paths: dict[str, Path] = {}
    streams: dict[str, Any] = {}
    for name, stream in dict(result.get("streams") or {}).items():
        frame = stream.get("features")
        if not isinstance(frame, pd.DataFrame):
            continue
        slug = str(name).replace("/", "_")
        role = f"stream__{slug}__features"
        paths[role] = atomic_write_csv(tables / "streams" / slug / "features.csv", frame)
        streams[str(name)] = {"slug": slug, "features_attrs": _json_attrs(frame.attrs)}
    for key in ("features", "significant_features", "group_catalog", "skipped_streams"):
        frame = analysis.get(key)
        if isinstance(frame, pd.DataFrame):
            role = f"untargeted_{key}"
            paths[role] = atomic_write_csv(tables / f"{role}.csv", frame)
    for key in ("untargeted_review_basket", "untargeted_reacquisition_list",
                "untargeted_sample_metadata_audit"):
        frame = result.get(key)
        if isinstance(frame, pd.DataFrame):
            paths[key] = atomic_write_csv(tables / f"{key}.csv", frame)
    source_annotations = result.get("automatic_annotations")
    if isinstance(source_annotations, pd.DataFrame):
        paths["automatic_annotations"] = atomic_write_csv(
            tables / "automatic_annotations.csv", source_annotations
        )
    paths["standard_stream_metadata"] = atomic_write_json(
        root / "standard_stream_metadata.json",
        {"engine": "untargeted", "streams": streams},
    )
    manifest = atomic_write_json(root / "run_manifest.json", {
        "format_version": 1,
        "workflow": "untargeted",
        "criteria": dict(analysis.get("criteria") or {}),
        "method": str(analysis.get("method", "")),
        "interpretation_boundary": str(analysis.get("interpretation_boundary", "")),
        "inputs": dict(inputs or {}),
    })
    paths["project_manifest"] = create_saved_project_from_run(
        root, workflow="untargeted", paths=paths,
        counts={"features": len(analysis.get("features", ())),
                "significant_features": len(analysis.get("significant_features", ()))},
        settings=dict(analysis.get("criteria") or {}),
        inputs=dict(inputs or {}), annotation_manifest=manifest,
        polarity="both",
    )
    return paths


def save_untargeted_workflow(
    project: str | Path, result: Mapping[str, Any], analysis: Mapping[str, Any], *,
    inputs: Mapping[str, Any] | None = None,
    existing: str = "archive",
) -> tuple[dict[str, Path], Path | None]:
    if not analysis or not isinstance(analysis.get("features"), pd.DataFrame):
        raise ValueError("Run untargeted analysis before saving it.")
    return _save_with_history(
        workflow_folder(project, "Untargeted"),
        lambda root: _write_untargeted_project(root, result, analysis, inputs=inputs),
        existing=existing,
    )


def load_untargeted_workflow(project: str | Path) -> tuple[dict, dict]:
    """Open saved untargeted tables directly, without rerunning statistics."""
    root = workflow_folder(project, "Untargeted")
    manifest_path = root / "LASER_PROJECT.json"
    if not manifest_path.is_file():
        raise ValueError(f"No saved untargeted analysis was found in {root}.")
    from saved_project_contract import normalize_saved_project_manifest

    manifest = normalize_saved_project_manifest(json.loads(manifest_path.read_text()))
    if manifest["workflow"] != "untargeted":
        raise ValueError("The selected folder is not an untargeted LASER result.")
    saved = {"root": str(root), "artifacts": manifest["artifacts"],
             "settings": manifest["settings"], "inputs": manifest["inputs"]}
    metadata = _read_metadata(saved)
    run = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    streams = {}
    for name, item in dict(metadata.get("streams") or {}).items():
        role = f"stream__{item['slug']}__features"
        features = load_project_table(saved, role)
        features.attrs.update(item.get("features_attrs") or {})
        streams[name] = {"features": features}
    analysis: dict[str, Any] = {
        "criteria": run.get("criteria", {}),
        "method": run.get("method", ""),
        "interpretation_boundary": run.get("interpretation_boundary", ""),
    }
    for key in ("features", "significant_features", "group_catalog", "skipped_streams"):
        analysis[key] = load_project_table(saved, f"untargeted_{key}")
    by_stream = {}
    features = analysis["features"]
    if "stream" in features:
        for name, frame in features.groupby("stream", sort=False):
            by_stream[str(name)] = frame.copy()
    analysis["by_stream"] = by_stream
    result = {"engine": "untargeted", "streams": streams,
              "automatic_annotations": load_project_table(saved, "automatic_annotations"),
              "untargeted_analysis": analysis,
              "xcms_input_dir": saved["inputs"].get("project")}
    for key in ("untargeted_review_basket", "untargeted_reacquisition_list",
                "untargeted_sample_metadata_audit"):
        result[key] = load_project_table(saved, key)
    return result, saved
