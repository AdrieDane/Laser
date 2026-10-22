"""Explicit, external paths for reproducible local research (never app inputs)."""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if not os.environ.get("LASER_OPTIMIZATION_WORKSPACE") or not os.environ.get(
    "LASER_QE_DATASETS"
):
    raise RuntimeError(
        "Set LASER_OPTIMIZATION_WORKSPACE and LASER_QE_DATASETS; see tools/qe_optimization/README.txt"
    )
WORKSPACE = Path(os.environ["LASER_OPTIMIZATION_WORKSPACE"]).expanduser().resolve()
SOURCE = Path(os.environ["LASER_QE_DATASETS"]).expanduser().resolve()
if not SOURCE.is_dir() or not WORKSPACE.is_dir():
    raise RuntimeError(
        "The declared research workspace and source collection must already exist"
    )
