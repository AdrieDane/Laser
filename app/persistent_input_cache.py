"""Discardable, revision-keyed disk caches for large local GUI inputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

import pandas as pd


def input_cache_directory() -> Path | None:
    """Return LASER's platform-appropriate cache folder when it is writable."""
    override = os.environ.get("LASER_INPUT_CACHE_DIR")
    if override:
        root = Path(override).expanduser()
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        root = (
            Path(base) / "LASER" / "Cache"
            if base
            else Path.home() / "AppData" / "Local" / "LASER" / "Cache"
        )
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches" / "LASER"
    else:
        base = os.environ.get("XDG_CACHE_HOME")
        root = (
            Path(base) / "LASER"
            if base
            else Path.home() / ".cache" / "LASER"
        )
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return root.resolve()


def revision_cache_path(
    *,
    kind: str,
    suffix: str,
    source_path: str,
    size: int,
    modified_ns: int,
    sha256: str,
    variant: str = "",
) -> Path | None:
    """Name a cache from the exact source revision and cache schema variant."""
    root = input_cache_directory()
    if root is None:
        return None
    identity = {
        "source": str(Path(source_path).expanduser().resolve()) if not sha256 else "",
        "size": int(size),
        "modified_ns": int(modified_ns) if not sha256 else 0,
        "sha256": str(sha256),
        "variant": str(variant),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:32]
    return root / f"{kind}_{digest}{suffix}"


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Publish a complete cache file without exposing a partial write."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_text(path: Path, content: str) -> None:
    """Publish a UTF-8 cache document atomically."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


__all__ = [
    "atomic_write_parquet",
    "atomic_write_text",
    "input_cache_directory",
    "revision_cache_path",
]
