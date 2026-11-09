"""Durable output primitives for local and network-drive LASER runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Iterable

import pandas as pd

from errors import OutputWriteError


RUNTIME_PACKAGES = (
    "streamlit", "plotly", "pandas", "numpy", "scipy", "openpyxl", "pyarrow",
)
EXCEL_SAFE_CELL_CHARACTERS = 30_000


def excel_safe_json_array(
    value: Any,
    *,
    max_characters: int = EXCEL_SAFE_CELL_CHARACTERS,
) -> str:
    """Keep a valid JSON array safely below Excel's per-cell text limit."""
    if isinstance(value, list):
        items = value
    else:
        try:
            items = json.loads(str(value).strip() or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            return "[]"
    if not isinstance(items, list):
        return "[]"
    compact: list[Any] = []
    for item in items:
        candidate = json.dumps(
            [*compact, item],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(candidate) > max(int(max_characters), 2):
            break
        compact.append(item)
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in RUNTIME_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def with_runtime_environment(manifest: dict[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    result.setdefault("runtime_environment", runtime_environment())
    return result


def _replace_with_retry(source: Path, target: Path) -> None:
    """``os.replace`` that tolerates brief reader locks on Windows.

    Windows refuses to replace a file while another process (for example the
    GUI polling a job's ``status.json``) has it open.  Such locks last
    milliseconds, so retry briefly before reporting the failure.  Other
    platforms replace open files atomically and take the first attempt.
    """
    attempts = 40 if os.name == "nt" else 1
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05)


def _atomic_replace(path: Path, writer) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        # Windows' ``_commit`` (used by ``os.fsync``) requires a descriptor
        # opened for writing.  Reopen read/write so the durability guarantee is
        # identical on Windows, macOS and Linux.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
        return path
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise OutputWriteError(f"Could not write {path} atomically: {exc}") from exc


def atomic_write_bytes(path: str | Path, content: bytes) -> Path:
    return _atomic_replace(Path(path), lambda temp: temp.write_bytes(content))


def atomic_write_text(
    path: str | Path, content: str, *, encoding: str = "utf-8",
) -> Path:
    return _atomic_replace(
        Path(path), lambda temp: temp.write_text(content, encoding=encoding),
    )


def atomic_append_text(
    path: str | Path, content: str, *, encoding: str = "utf-8",
) -> Path:
    """Append through atomic replacement for interruption-safe local logs.

    LASER uses this for modest run logs on network drives.  It deliberately
    favours a complete previous-or-next file over an in-place partial append.
    """
    destination = Path(path)
    previous = (
        destination.read_text(encoding=encoding)
        if destination.is_file()
        else ""
    )
    return atomic_write_text(
        destination, previous + content, encoding=encoding,
    )


def atomic_write_json(
    path: str | Path,
    data: Any,
    *,
    allow_nan: bool = True,
) -> Path:
    try:
        payload = json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=allow_nan,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise OutputWriteError(
            f"Could not serialize {Path(path)} as JSON: {exc}"
        ) from exc
    return atomic_write_text(path, payload)


def atomic_write_csv(
    path: str | Path,
    frame: pd.DataFrame,
    *,
    index: bool = False,
    **to_csv_options: Any,
) -> Path:
    """Write a delimited table atomically using pandas CSV options."""
    return _atomic_replace(
        Path(path),
        lambda temp: frame.to_csv(
            temp, index=index, **to_csv_options,
        ),
    )


def atomic_write_parquet(
    path: str | Path,
    frame: pd.DataFrame,
    *,
    index: bool = False,
    compression: str = "zstd",
) -> Path:
    """Write an internal typed table atomically while CSV remains portable."""
    return _atomic_replace(
        Path(path),
        lambda temp: frame.to_parquet(
            temp, index=index, compression=compression, engine="pyarrow",
        ),
    )


class StructuredStageLogger:
    """Small append-like JSONL log implemented as atomic rewrites."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._events: list[dict[str, Any]] = []
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self._events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def record(
        self, stage: str, status: str, *, message: str = "", **details: Any,
    ) -> dict[str, Any]:
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": str(stage),
            "status": str(status),
            "message": str(message),
            "details": details,
        }
        self._events.append(event)
        payload = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in self._events)
        atomic_write_text(self.path, payload)
        return event


def write_completion_marker(
    output_directory: str | Path,
    important_outputs: Iterable[str | Path],
    *,
    details: dict[str, Any] | None = None,
) -> Path:
    """Write the completion marker last, with checksums of durable outputs."""
    output = Path(output_directory)
    checksums = {}
    for item in important_outputs:
        if item is None:
            continue
        path = Path(item)
        if path.is_file():
            checksums[str(path.resolve())] = sha256_file(path)
    marker = {
        "format_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs_sha256": checksums,
        "details": dict(details or {}),
    }
    return atomic_write_json(output / "RUN_COMPLETE.json", marker)
