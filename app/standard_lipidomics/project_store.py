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
from standard_lipidomics.project_layout import workflow_folder, xcms_project_root
from standard_lipidomics.acquisition_method import normalize_acquisition_method, method_id
from standard_lipidomics.internal_standards import find_project_is_file, project_is_definitions, project_is_signature
from standard_lipidomics.mass_calibration import write_calibrated_xcms_files
from standard_lipidomics.rt_calibration import calibrate_standard_rt_bytes, freeze_rt_calibration
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
            + list(folder.glob("internal_standards.tsv")) + list(folder.glob("internal_standards.csv"))
        )
        for source in metadata_files:
            if source.name.startswith("~$"):
                continue
            target = destination / source.name
            if source.name == "internal_standards.csv" and target.is_file():
                continue
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
        legacy = moved / "Untargeted" if moved is not None and destination.name == "LASER" else None
        if legacy is not None and legacy.is_dir():
            shutil.copytree(legacy, destination / "Untargeted")
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
    selected = Path(project).expanduser().resolve()
    root = selected if (selected / "LASER_PROJECT.json").is_file() else workflow_folder(project, "Annotation")
    if not (root / "LASER_PROJECT.json").is_file():
        root = root / "Annotation"  # Projects saved by earlier LASER versions.
    saved = load_completed_project(root)
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
    method = normalize_acquisition_method(analysis.get("acquisition_method", result.get("acquisition_method")))
    tables = root / "tables"
    paths: dict[str, Path] = {}
    # Experimental evidence includes text IDs, empty labels and full-precision
    # costs. Build its typed artifacts from the actual frames, rather than
    # inferring their types again from portable CSV exports.
    frames: dict[str, pd.DataFrame] = {}
    streams: dict[str, Any] = {}
    for name, stream in dict(result.get("streams") or {}).items():
        frame = stream.get("features")
        if not isinstance(frame, pd.DataFrame):
            continue
        slug = str(name).replace("/", "_")
        role = f"stream__{slug}__features"
        paths[role] = atomic_write_csv(tables / "streams" / slug / "features.csv", frame)
        frames[role] = frame
        streams[str(name)] = {"slug": slug, "features_attrs": _json_attrs(frame.attrs)}
    for key in ("features", "significant_features", "raw_p_features", "compound_features", "cluster_members", "adduct_matches", "homologous_matches", "lipid_grid_matches", "group_catalog", "skipped_streams", "ion_feature_ledger", "ion_relationships", "ion_isotope_candidates", "isotope_candidate_scores", "isotope_peak_evidence", "formula_search_audit", "joint_hypotheses", "joint_feature_ledger", "joint_compound_features", "joint_relationships", "joint_support", "joint_associations", "joint_selected_links", "joint_component_audit", "joint_search_audit", "msbuddy_candidates", "msbuddy_cases", "trace_pair_evidence", "trace_pair_samples", "trace_sample_quality", "trace_points"):
        frame = analysis.get(key)
        if isinstance(frame, pd.DataFrame):
            role = f"untargeted_{key}"
            paths[role] = atomic_write_csv(tables / f"{role}.csv", frame)
            frames[role] = frame
    for key in ("untargeted_review_basket", "untargeted_reacquisition_list",
                "untargeted_sample_metadata_audit"):
        frame = result.get(key)
        if isinstance(frame, pd.DataFrame):
            paths[key] = atomic_write_csv(tables / f"{key}.csv", frame)
            frames[key] = frame
    source_annotations = result.get("automatic_annotations")
    if isinstance(source_annotations, pd.DataFrame):
        paths["automatic_annotations"] = atomic_write_csv(
            tables / "automatic_annotations.csv", source_annotations
        )
        frames["automatic_annotations"] = source_annotations
    paths["standard_stream_metadata"] = atomic_write_json(
        root / "standard_stream_metadata.json",
        {"engine": "untargeted", "streams": streams},
    )
    paths["acquisition_method"] = atomic_write_json(root / "acquisition_method.json", method)
    manifest = atomic_write_json(root / "run_manifest.json", {
        "format_version": 1,
        "workflow": "untargeted",
        "criteria": dict(analysis.get("criteria") or {}),
        "method": str(analysis.get("method", "")),
        "evidence_version": analysis.get("evidence_version", 0),
        "compound_version": analysis.get("compound_version", 0),
        "ion_group_version": analysis.get("ion_group_version", 0),
        "ion_grouping_mode": analysis.get("ion_grouping_mode", "current"),
        "isotope_model_provenance": analysis.get("isotope_model_provenance"),
        "formula_model_provenance": analysis.get("formula_model_provenance"),
        "joint_provenance": analysis.get("joint_provenance"),
        "msbuddy_provenance": analysis.get("msbuddy_provenance"),
        "trace_provenance": analysis.get("trace_provenance"),
        "ion_group_settings": analysis.get("ion_group_settings"),
        "ion_group_source_signature": analysis.get("ion_group_source_signature"),
        "lipid_grid_version": analysis.get("lipid_grid_version", 0),
        "lipid_grid_method_id": analysis.get("lipid_grid_method_id"),
        "acquisition_method": method,
        "acquisition_method_id": method_id(method),
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
        table_frames=frames if any((analysis.get("criteria") or {}).get(k, "current") != "current" for k in ("joint_model", "trace_model")) else None,
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
    selected = Path(project).expanduser().resolve()
    root = selected if (selected / "LASER_PROJECT.json").is_file() else workflow_folder(project, "Untargeted")
    if not (root / "LASER_PROJECT.json").is_file():
        root = xcms_project_root(project) / "LASER" / "Untargeted"
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
        "evidence_version": run.get("evidence_version", 0),
        "compound_version": run.get("compound_version", 0),
        "ion_group_version": run.get("ion_group_version", 0),
        "ion_grouping_mode": run.get("ion_grouping_mode", "current"),
        "isotope_model_provenance": run.get("isotope_model_provenance"),
        "formula_model_provenance": run.get("formula_model_provenance"),
        "joint_provenance": run.get("joint_provenance"),
        "msbuddy_provenance": run.get("msbuddy_provenance"),
        "trace_provenance": run.get("trace_provenance"),
        "ion_group_settings": run.get("ion_group_settings"),
        "ion_group_source_signature": run.get("ion_group_source_signature"),
        "lipid_grid_version": run.get("lipid_grid_version", 0),
        "lipid_grid_method_id": run.get("lipid_grid_method_id"),
        "acquisition_method": normalize_acquisition_method(run.get("acquisition_method")),
        "interpretation_boundary": run.get("interpretation_boundary", ""),
    }
    for key in ("features", "significant_features", "raw_p_features", "compound_features", "cluster_members", "adduct_matches", "homologous_matches", "lipid_grid_matches", "group_catalog", "skipped_streams", "ion_feature_ledger", "ion_relationships", "ion_isotope_candidates", "isotope_candidate_scores", "isotope_peak_evidence", "formula_search_audit", "joint_hypotheses", "joint_feature_ledger", "joint_compound_features", "joint_relationships", "joint_support", "joint_associations", "joint_selected_links", "joint_component_audit", "joint_search_audit", "msbuddy_candidates", "msbuddy_cases", "trace_pair_evidence", "trace_pair_samples", "trace_sample_quality", "trace_points"):
        analysis[key] = load_project_table(saved, f"untargeted_{key}")
        if key in {"ion_feature_ledger", "ion_relationships", "ion_isotope_candidates"}:
            text = set(analysis[key].select_dtypes(include=['object', 'str', 'string']))
            text.update(c for c in analysis[key] if c.endswith(('_id', '_adduct', '_stream', '_class_hint', '_ion_state')))
            text.update({'adduct', 'association_status', 'detected_modes', 'adduct_alternatives', 'pairing_method', 'relationship', 'selection_status'} & set(analysis[key]))
            for column in text:
                analysis[key][column] = analysis[key][column].fillna('').astype(str)
    by_stream = {}
    features = analysis["features"]
    if "raw_p_significant" not in features:
        analysis["raw_p_features"] = analysis["significant_features"].copy()
    if "stream" in features:
        for name, frame in features.groupby("stream", sort=False):
            by_stream[str(name)] = frame.copy()
    analysis["by_stream"] = by_stream
    result = {"engine": "untargeted", "streams": streams,
              "acquisition_method": analysis["acquisition_method"],
              "automatic_annotations": load_project_table(saved, "automatic_annotations"),
              "untargeted_analysis": analysis,
              "xcms_input_dir": saved["inputs"].get("project")}
    for key in ("untargeted_review_basket", "untargeted_reacquisition_list",
                "untargeted_sample_metadata_audit"):
        result[key] = load_project_table(saved, key)
    return result, saved


def save_calibration_workflow(
    project: str | Path, sources: Mapping[str, str | Path | bytes],
    calibrations: Mapping[str, object], *, existing: str = "archive",
    rt_calibrations: Mapping[str, dict] | None = None,
) -> tuple[dict[str, Path], Path | None]:
    """Save reviewed calibration files, models and metadata with version history."""
    # Read sources before any directory can be moved into history.
    payloads = {name: value if isinstance(value, bytes) else Path(value).read_bytes()
                for name, value in sources.items()}
    metadata: dict[str, bytes] = {}
    root = xcms_project_root(project)
    folders = [root, root / "metadata"]
    folders.extend(Path(value).parent for value in sources.values() if not isinstance(value, bytes))
    for folder in folders:
        for pattern in ("*.xlsx", "*.xls", "internal_standards.tsv", "internal_standards.csv"):
            for path in folder.glob(pattern):
                if not path.name.startswith("~$"):
                    metadata[path.name] = path.read_bytes()

    selected_is = find_project_is_file(project)
    if selected_is is not None:
        metadata["internal_standards.csv"] = selected_is.read_bytes()
    is_path, is_definitions = project_is_definitions(project)

    def write(destination: Path) -> dict[str, Path]:
        paths = write_calibrated_xcms_files(payloads, calibrations, destination)
        manifest_path = destination / "LASER_standard_mass_calibration.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["project"] = str(root)
        manifest["saved_at"] = datetime.now(timezone.utc).isoformat()
        manifest["rt_calibration_source"] = "LASER-4D a188f72"
        manifest["internal_standard_source"] = str(is_path) if is_path else None
        manifest["internal_standard_signature"] = project_is_signature(project)
        manifest["internal_standards_expected"] = [
            {"name": item.name, "concentration": item.concentration, "unit": item.unit}
            for item in is_definitions]
        for name, path in paths.items():
            rt_model = (rt_calibrations or {}).get(name)
            if rt_model is not None:
                path.write_bytes(calibrate_standard_rt_bytes(path, rt_model))
            manifest["streams"][name]["rt_calibration"] = freeze_rt_calibration(rt_model)
            manifest["streams"][name]["rt_status"] = "applied" if rt_model else "not_applied"
        for name, value in sources.items():
            manifest["streams"][name]["source"] = (
                "uploaded_bytes" if isinstance(value, bytes) else str(Path(value).resolve()))
        atomic_write_json(manifest_path, manifest)
        for name, payload in metadata.items():
            (destination / name).write_bytes(payload)
        return paths

    return _save_with_history(workflow_folder(project, "Calibration"), write, existing=existing)
