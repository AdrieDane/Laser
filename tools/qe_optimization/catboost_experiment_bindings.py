"""Immutable048 input/source ledger; no scientific app import or fitting."""

import json
from pathlib import Path

from context import REPO
import catboost_experiment_data as data

CONSTRUCTION = data.QE / "catboost_input_compatibility_v1"
SOURCES = (
    "catboost_evidence_model.py",
    "catboost_experiment_data.py",
    "catboost_operating_points.py",
    "catboost_pipeline_assembly.py",
    "catboost_experiment_bindings.py",
    "run_qe_catboost_model_experiment.py",
    "preflight_qe_catboost_models.py",
    "verify_qe_catboost_experiment.py",
    "postprocess_qe_catboost_experiment.py",
    "report_qe_catboost_decision.py",
    "test_catboost_evidence_model.py",
    "test_catboost_operating_points.py",
    "test_catboost_experiment_controls.py",
    "requirements-catboost.txt",
    "frozen_resources.py",
    "context.py",
    "run_qe_atlas_experiment.py",
    "evaluate_qe_baseline.py",
    "report_qe_experiment.py",
    "report_qe_priority_classes.py",
    "report_qe_target_scope.py",
    "report_qe_endogenous_companion.py",
    "verify_immutable_exports.py",
)


def source_inputs():
    construction_path = CONSTRUCTION / "receipt.json"
    receipt = json.loads(construction_path.read_text())
    assert receipt["all_passed"] and receipt["unique_nested_tables"] == 363
    assert (
        receipt["parent_preparations"] == 25 and receipt["training_query_uses"] == 528
    )
    assert (
        receipt["classifier_fits"] == 0
        and receipt["actual_training_fit_authorized"] is False
    )
    paths = set(map(Path, receipt["source_input_sha256"]))
    for p, h in receipt["source_input_sha256"].items():
        assert data.sha(p) == h, p
    paths.update(Path(__file__).with_name(name) for name in SOURCES)
    paths.update(
        (
            construction_path,
            CONSTRUCTION / "all363_blind_receipt.json",
            CONSTRUCTION / "all363_blind_tables.csv",
            CONSTRUCTION / "all25_parent_mode_matrices.csv",
            CONSTRUCTION / "all165_prediction_contexts.csv",
            data.PLAN,
            data.PREVIOUS / "protocol.json",
            data.PREVIOUS / "integrity_receipt.json",
            data.CORE / "runner_source.py",
            data.STD / "protocol.json",
            data.MASS / "protocol.json",
            data.MASS / "integrity_receipt.json",
            data.QE / "manual_reference_v2/mapping_corrections.csv",
        )
    )
    assert receipt["matrix_table_sha256"] == data.sha(
        CONSTRUCTION / "all25_parent_mode_matrices.csv"
    )
    assert json.loads((data.PREVIOUS / "integrity_receipt.json").read_text())[
        "all_passed"
    ]
    assert json.loads((data.MASS / "integrity_receipt.json").read_text())["all_passed"]
    datasets, _ = data.inventory()
    for d in datasets:
        paths.update(
            (
                data.BASE / "runs" / d / "full_annotation.pkl",
                data.MASS / "runs" / d / "private_mass_candidates.pkl",
                data.CORE / "runs" / d / "automatic_annotations.csv",
                data.CORE / "runs" / d / "candidates.pkl",
                data.CORE / "runs" / d / "scoring_audit.pkl",
                data.PREVIOUS / "runs" / d / "automatic_annotations.csv",
                data.PREVIOUS / "runs" / d / "candidates.pkl",
                data.QE / "manual_reference_v2" / d / "curated_feature_crosswalk.pkl",
            )
        )
    for o in range(5):
        paths.add(data.STD / "models" / f"fold_{o}_standard_models.json")
        folder = data.CORE / "calibration" / f"fold_{o}"
        paths.add(folder / "operating_points.json")
        paths.update(folder.glob("*_blind_scores.pkl"))
    paths.update(
        p
        for root in ("app", "methods", "reference_data")
        for p in (REPO / root).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
    )
    return {str(p): data.sha(p) for p in sorted(paths)}
