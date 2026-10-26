"""Actual049 complete-source/runtime/preflight permission gate; no fits."""

import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn

from context import SOURCE
import matrix_experiment_data as data

STAGE = data.QE / "matrix_context_stage_a_v1"

SOURCES = (
    "matrix_experiment_bindings.py",
    "matrix_experiment_data.py",
    "matrix_mode_model.py",
    "matrix_pipeline_assembly.py",
    "matrix_operating_points.py",
    "run_qe_matrix_context_experiment.py",
    "preflight_qe_matrix_context_models.py",
    "verify_qe_matrix_context_experiment.py",
    "report_qe_matrix_context_decision.py",
    "postprocess_qe_matrix_context_experiment.py",
    "test_matrix_mode_model.py",
    "test_matrix_pipeline_assembly.py",
    "test_matrix_operating_points.py",
    "test_matrix_experiment_controls.py",
    "evaluate_qe_baseline.py",
    "report_qe_experiment.py",
    "report_qe_target_scope.py",
    "report_qe_priority_classes.py",
    "report_qe_endogenous_companion.py",
    "report_qe_catboost_decision.py",
    "frozen_resources.py",
    "context.py",
)
CHECKER = Path(__file__).with_name("verify_qe_matrix_context_experiment.py")


def environment():
    return dict(
        python=sys.version,
        numpy=np.__version__,
        pandas=pd.__version__,
        scipy=scipy.__version__,
        sklearn=sklearn.__version__,
        machine=platform.machine(),
        thread_environment={
            k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    )


def source_inputs():
    ready = json.loads((STAGE / "receipt.json").read_text())
    assert (
        ready["all_passed"]
        and ready["classifier_fits"] == 0
        and ready["actual049_fit_permission"] is False
    )
    # Validate the frozen ledger without importing scientific assembly before
    # the actual experiment has created/configured its final resource root.
    base = dict(ready["source_input_sha256"])
    assert all(data.nested.pilot.sha(Path(p)) == h for p, h in base.items())
    for name in ("app", "methods", "reference_data"):
        current = {
            str(p)
            for p in (data.PLAN.parents[2] / name).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
        }
        assert current <= set(base)
    paths = {Path(__file__).with_name(n) for n in SOURCES}
    paths.update(
        (
            STAGE / "receipt.json",
            STAGE / "training_matrices.json",
            STAGE / "old_controls.json",
            data.nested.CHECK / "receipt.json",
            data.PLAN,
        )
    )
    datasets, _ = data.inventory()
    paths.update(
        data.PREVIOUS / "evaluation_v2" / d / "manual_comparison.pkl" for d in datasets
    )
    paths.update(
        p
        for p in (data.QE / "manual_reference_v2").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
    )
    for dataset in datasets:
        paths.add(
            data.QE.parent
            / "qe_dataset_audit_20261022/checkpoints"
            / (dataset + ".json")
        )
        paths.update(
            SOURCE / dataset / name
            for name in (
                "MS_data.xlsx",
                "MS_metadata.mat",
                *[f"peakgrouplist_{s}.txt" for s in data.MODES],
            )
        )
    base.update({str(p): data.nested.pilot.sha(p) for p in sorted(paths)})
    return base


def require_fit_permission():
    assert (
        os.environ.get("OMP_NUM_THREADS")
        == os.environ.get("OPENBLAS_NUM_THREADS")
        == "1"
    )
    receipt = json.loads((data.PREFLIGHT / "receipt.json").read_text())
    assert receipt["all_passed"] and receipt["actual049_fit_permission"] is True
    assert receipt["source_input_sha256"] == source_inputs()
    assert receipt["numeric_runtime"] == environment()
    assert receipt["planned_actual_verifier_sha256"] == data.nested.pilot.sha(CHECKER)
    return receipt
