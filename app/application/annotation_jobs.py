"""Detached annotation jobs with durable inputs, progress, logs and GUI saves."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from uuid import uuid4

from output_io import atomic_write_json
from application.job_monitor import process_activity
from standard_lipidomics.internal_standards import find_project_is_file, project_is_definitions
from standard_lipidomics.project_layout import xcms_project_root


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_status(path: Path) -> dict:
    """Read a status file, tolerating the worker's brief atomic replace on Windows."""
    attempts = 40 if os.name == "nt" else 1
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05)
    raise AssertionError("unreachable")


def read_annotation_job(directory: str | Path) -> dict:
    root = Path(directory)
    state = _read_status(root / "status.json")
    pid_path = root / "process.json"
    if state["status"] in {"queued", "running"} and pid_path.is_file():
        pid = json.loads(pid_path.read_text(encoding="utf-8"))["pid"]
        activity = process_activity(pid)
        if not activity["alive"]:
            state = _read_status(root / "status.json")
            if state["status"] in {"queued", "running"}:
                state = {**state, "status": "failed", "message": "The process stopped before completion."}
        state = {**state, "activity": {**activity, "pid": pid}}
    return state


def active_annotation_job(project: str | Path) -> Path | None:
    pointer = xcms_project_root(project) / ".laser" / "jobs" / "annotation" / "current.json"
    if not pointer.is_file():
        return None
    directory = Path(json.loads(pointer.read_text(encoding="utf-8"))["directory"])
    return directory if read_annotation_job(directory)["status"] in {"queued", "running"} else None


def launch_worker(directory: Path, module: str, log_name: str):
    """Start a worker without inheriting a browser/server terminal session."""
    env = dict(os.environ)
    app = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(app) + os.pathsep + env.get("PYTHONPATH", "")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(name, "1")
    # Logs are files: without this, Windows encodes worker output as cp1252
    # and progress messages containing e.g. "↔" crash the job.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # A hidden console (instead of DETACHED_PROCESS) keeps the worker
    # independent of the server terminal, while console programs it starts
    # inherit that hidden console rather than opening visible windows.
    options = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with (directory / log_name).open("ab") as log:
        return subprocess.Popen([sys.executable, "-u", "-m", module, str(directory)],
            cwd=app.parent, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            close_fds=True, **options)


def start_annotation_job(config: dict, sources: dict | None = None) -> Path:
    from standard_lipidomics.workflow import discover_standard_streams

    project = xcms_project_root(config["inputs"]["project"])
    jobs = project / ".laser" / "jobs" / "annotation"
    jobs.mkdir(parents=True, exist_ok=True)
    lock = jobs / "launch.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("An annotation is already being started for this project.") from exc
    os.close(descriptor)
    directory = None
    try:
        running = active_annotation_job(project)
        if running is not None:
            return running
        source_project = config["inputs"].get("xcms_source") or str(project)
        project_is_definitions(source_project)  # Never fall back after an invalid CSV.
        selected = sources if sources is not None else discover_standard_streams(source_project)
        if config["engine"] == "matlab":
            phases = config["annotation"].get("phases", ["RP", "NP"])
            selected = {name: value for name, value in selected.items() if name.split("_")[0] in phases}
        if not selected:
            raise ValueError("No selected XCMS peak-group lists were found.")
        directory = jobs / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:10])
        inputs = directory / "inputs"
        inputs.mkdir(parents=True)
        for name, value in selected.items():
            destination = inputs / f"peakgrouplist_{name}.txt"
            if isinstance(value, bytes):
                destination.write_bytes(value)
            else:
                shutil.copy2(value, destination)
        is_file = find_project_is_file(source_project)
        if is_file is not None:
            shutil.copy2(is_file, inputs / "internal_standards.csv")
        else:
            # Preserve legacy project declarations for existing projects.
            from matlab_annotation.engine import _internal_standard_project_dir
            from matlab_annotation.configuration import DEFAULT_MATLAB_PIPELINE_REFERENCE
            legacy = _internal_standard_project_dir(source_project, Path(DEFAULT_MATLAB_PIPELINE_REFERENCE)) / "internal_standards.tsv"
            if legacy.is_file():
                shutil.copy2(legacy, inputs / legacy.name)
        request = {**config, "project_directory": str(project), "created_at": _now()}
        atomic_write_json(directory / "request.json", request)
        atomic_write_json(directory / "status.json", {"status": "queued", "progress": 0.0,
            "message": "Starting annotation", "updated_at": _now()})
        process = launch_worker(directory, "application.annotation_jobs", "annotation.log")
        atomic_write_json(directory / "process.json", {"pid": process.pid})
        atomic_write_json(jobs / "current.json", {"directory": str(directory)})
        return directory
    except Exception:
        if directory is not None and not (directory / "process.json").is_file():
            shutil.rmtree(directory)
        raise
    finally:
        lock.unlink(missing_ok=True)


def run_annotation_job(directory: str | Path) -> int:
    from application.standard_workflow import StandardAnnotationRequest
    from annotator.standard_annotated_inputs import append_standard_annotations
    from composition import standard_annotation_service
    from standard_lipidomics.project_store import save_annotation_workflow, stage_project_metadata
    from standard_lipidomics.workflow import discover_standard_streams

    directory = Path(directory).resolve()
    config = json.loads((directory / "request.json").read_text())
    inputs = directory / "inputs"
    project = Path(config["project_directory"])

    def progress(fraction: float, message: str) -> None:
        print(f"[{_now()}] {message}", flush=True)
        atomic_write_json(directory / "status.json", {"status": "running",
            "progress": min(max(float(fraction), 0.0), 1.0), "message": message,
            "updated_at": _now(), "pid": os.getpid()})

    try:
        progress(0.0, "Starting annotation")
        paths = config["inputs"]
        request = StandardAnnotationRequest(engine=config["engine"], project=inputs,
            reference_path=Path(paths["reference"]) if paths.get("reference") else None,
            matlab_reference_path=Path(paths["matlab_reference"]) if paths.get("matlab_reference") else None,
            matlab_policy_path=Path(paths["matlab_policy"]) if paths.get("matlab_policy") else None,
            np_rt_reference_path=Path(paths["np_rt_reference"]) if paths.get("np_rt_reference") else None,
            settings=config["annotation"], progress=progress)
        result = standard_annotation_service.run(request).to_mapping()
        from standard_lipidomics.untargeted import attach_excel_sample_metadata
        result["untargeted_sample_metadata_audit"] = attach_excel_sample_metadata(result, project)
        progress(0.98, "Saving annotation")
        sources = discover_standard_streams(inputs)
        annotated = {name: append_standard_annotations(path, result["automatic_annotations"], stream=name)
                     for name, path in sources.items()}
        stage_project_metadata(project, discover_standard_streams(project))
        saved, archive = save_annotation_workflow(project, result, settings=config["annotation"],
            inputs={**paths, "input_snapshot": str(inputs)}, annotated_inputs=annotated,
            existing=config.get("existing", "archive"))
        atomic_write_json(directory / "status.json", {"status": "complete", "progress": 1.0,
            "message": "Annotation complete", "project_manifest": str(saved["project_manifest"]),
            "archive": str(archive) if archive else None, "updated_at": _now()})
        print("Annotation complete", flush=True)
        return 0
    except Exception as exc:
        traceback.print_exc()
        atomic_write_json(directory / "status.json", {"status": "failed", "progress": 0.0,
            "message": str(exc), "updated_at": _now()})
        return 1


if __name__ == "__main__":
    raise SystemExit(run_annotation_job(sys.argv[1]))
