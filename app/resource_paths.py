"""Portable locations for LASER's bundled scientific resources."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _resource_directory(
    name: str,
    *,
    environment_variable: str,
    checkout_directory: Path,
) -> Path:
    configured = os.environ.get(environment_variable, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = (
        checkout_directory,
        _APP_DIRECTORY / name,
        Path(sys.prefix) / name,
        Path(sys.base_prefix) / name,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return checkout_directory.resolve()


_APP_DIRECTORY = Path(__file__).resolve().parent

REFERENCE_DATA_DIRECTORY = _resource_directory(
    "reference_data",
    environment_variable="LASER_REFERENCE_DATA",
    checkout_directory=_APP_DIRECTORY.parent / "reference_data",
)
RULES_DIRECTORY = _resource_directory(
    "rules",
    environment_variable="LASER_RULES_DIRECTORY",
    checkout_directory=_APP_DIRECTORY / "rules",
)
METHODS_DIRECTORY = _resource_directory(
    "methods",
    environment_variable="LASER_METHODS_DIRECTORY",
    checkout_directory=_APP_DIRECTORY.parent / "methods",
)
STANDARD_LIPIDOMICS_METHODS_DIRECTORY = METHODS_DIRECTORY / "standard_lipidomics"
STANDARD_CALIBRATION_LIBRARY_DIRECTORY = (
    STANDARD_LIPIDOMICS_METHODS_DIRECTORY / "calibration_libraries"
)


__all__ = [
    "METHODS_DIRECTORY",
    "REFERENCE_DATA_DIRECTORY",
    "RULES_DIRECTORY",
    "STANDARD_CALIBRATION_LIBRARY_DIRECTORY",
    "STANDARD_LIPIDOMICS_METHODS_DIRECTORY",
]
