"""Research snapshots must NEVER silently use installed scientific defaults."""

import hashlib
import os
from pathlib import Path
import sys


def verify_frozen_paths(snapshot, reference_path, methods_path):
    root = Path(snapshot).resolve()
    reference = (root / "reference_data").resolve()
    methods = (root / "methods").resolve()
    if not reference.is_dir() or not methods.is_dir():
        raise RuntimeError(
            "Complete frozen reference_data AND methods directories required"
        )
    if (
        Path(reference_path).resolve() != reference
        or Path(methods_path).resolve() != methods
    ):
        raise RuntimeError("Scientific resource fallback escaped the frozen snapshot")
    table = reference / "standard_lipid_composition_ranges.tsv"
    with table.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return dict(
        reference_data=str(reference),
        methods=str(methods),
        composition_ranges_sha256=digest,
    )


def configure_frozen_resources(snapshot):
    """Call BEFORE importing annotation modules, once per worker interpreter."""
    root = Path(snapshot).resolve()
    if not (root / "reference_data").is_dir() or not (root / "methods").is_dir():
        raise RuntimeError(
            "Complete frozen scientific resources missing; refusing installed fallback"
        )
    os.environ["LASER_REFERENCE_DATA"] = str(root / "reference_data")
    os.environ["LASER_METHODS_DIRECTORY"] = str(root / "methods")
    sys.path.insert(0, str(root / "app"))
    from resource_paths import REFERENCE_DATA_DIRECTORY, METHODS_DIRECTORY

    return verify_frozen_paths(root, REFERENCE_DATA_DIRECTORY, METHODS_DIRECTORY)
