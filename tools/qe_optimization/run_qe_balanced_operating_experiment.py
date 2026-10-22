"""Experiment009: balanced pooled-budget operating points, frozen007 scores.

No held-out annotation is used to choose the threshold. Physical candidates,
final score models, ambiguity margins and005 standard handling stay fixed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from context import WORKSPACE as ROOT, REPO

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
PREVIOUS = ROOT / "outputs/qe_optimization_20261022/experiment_007"
STD = ROOT / "outputs/qe_optimization_20261022/experiment_005"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_009"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qe_dual_experiment import model_matrix


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def initialize():
    if not (OUT / "protocol.json").exists():
        OUT.mkdir(parents=True, exist_ok=False)
        snapshot = OUT / "source_snapshot"
        snapshot.mkdir()
        for name in ("app", "reference_data", "methods"):
            shutil.copytree(
                REPO / name,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        hashes = {
            p.relative_to(snapshot).as_posix(): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in snapshot.rglob("*")
            if p.is_file()
        }
        previous = json.loads((PREVIOUS / "protocol.json").read_text())
        dump(
            OUT / "protocol.json",
            dict(
                name="009-balanced-pooled-budget-operating-points",
                baseline=str(BASE),
                previous=str(PREVIOUS),
                folds=5,
                inner_folds=3,
                ambiguity_margin=0.05,
                production_enabled=False,
                candidate_universe="EXACT007 candidates/evidence and frozen final score models; no physical admission changes.",
                classifier=previous["classifier"],
                curation_runtime_input=False,
                threshold_grid=[round(v, 2) for v in np.arange(0.2, 1.001, 0.05)],
                extras_budget_per_endogenous_manual=0.095,
                threshold_policy="Joint per-stream cutoffs from the SAME008 independent INNER training-group predictions. Maximize weakest-mode correct/manual coverage up to90%, then pooled correct matches, then fewer wrong-peak choices/extras and higher cutoffs, within POOLED9.5% endogenous-extras/endogenous-manual training budget. No outer evaluated labels choose thresholds. Standard extras separate; actual held-out pooled extras guard remains original baseline9697, not a guaranteed bound or FDR.",
                validation=previous["validation"],
                model_fit_note="Inner score-model fitting excludes each calibration project group AND all outer evaluation groups. Evidence is inherited from the OUTER-training RT atlas (not nested inner RT cross-fitting), so this is conditional grouped threshold validation, NOT pristine nested end-to-end cross-validation.",
                run_kind="Saved007 original-feature candidate/score replay; joint weakest-mode operating cutoffs learned from frozen008 INNER project-group training predictions with pooled extras budget. Same final models, physical candidates, ambiguity margin and EXACT005 standards.",
                source_snapshot_sha256=hashes,
            ),
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        for name in ("run_qe_dual_experiment.py", "run_qe_atlas_experiment.py"):
            shutil.copy2(Path(__file__).with_name(name), OUT / (name + ".dependency"))
        manifest = json.loads((BASE / "run_manifest.json").read_text())
        protocol = json.loads((OUT / "protocol.json").read_text())
        dump(
            OUT / "run_manifest.json",
            dict(
                datasets=manifest["datasets"],
                fresh_raw_input_runs=False,
                evaluation_note=protocol["run_kind"]
                + " "
                + protocol["validation"]
                + " "
                + protocol["model_fit_note"],
            ),
        )
    protocol = json.loads((OUT / "protocol.json").read_text())
    for name, expected in protocol["source_snapshot_sha256"].items():
        assert (
            hashlib.sha256((OUT / "source_snapshot" / name).read_bytes()).hexdigest()
            == expected
        ), name
    for name in ("run_qe_dual_experiment.py", "run_qe_atlas_experiment.py"):
        assert (OUT / (name + ".dependency")).read_bytes() == (
            Path(__file__).with_name(name)
        ).read_bytes()
    for name in (
        "standard_completion.py",
        "standard_rt_library.py",
        "standard_rt_transfer.py",
        "declared_standards.py",
    ):
        rel = Path("source_snapshot/app/matlab_annotation") / name
        assert (OUT / rel).read_bytes() == (STD / rel).read_bytes(), name
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    return protocol


def calibrate(protocol, projects, folds, fold):
    from matlab_annotation.operating_point import choose_balanced_training_points

    source = (
        ROOT
        / "outputs/qe_optimization_20261022/experiment_008/calibration"
        / f"fold_{fold}"
    )
    previous = json.loads((source / "operating_points.json").read_text())
    train_groups = {d[:7] for d in projects if folds[d] != fold}
    test_groups = {d[:7] for d in projects if folds[d] == fold}
    assert set(previous["training_groups"]) == train_groups
    assert set(previous["outer_excluded"]) == test_groups
    assert not train_groups & test_groups
    table_path = source / "training_operating_points.csv"
    table = pd.read_csv(table_path)
    assert set(table.stream) == {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}
    assert set(table.threshold) == set(protocol["threshold_grid"])
    thresholds = choose_balanced_training_points(
        table,
        extras_per_manual=protocol["extras_budget_per_endogenous_manual"],
        target=0.9,
    )
    assert thresholds is not None, (fold, "No feasible pooled training point")
    selected = table.loc[
        [
            float(t) == thresholds[s]
            for s, t in table[["stream", "threshold"]].itertuples(
                index=False, name=None
            )
        ]
    ]
    assert len(selected) == 4
    assert (
        selected.extras.sum()
        <= protocol["extras_budget_per_endogenous_manual"] * selected.manual.sum()
    )
    dest = OUT / "calibration" / f"fold_{fold}"
    dest.mkdir(parents=True, exist_ok=True)
    selected.to_csv(dest / "selected_training_points.csv", index=False)
    digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    dump(
        dest / "operating_points.json",
        dict(
            thresholds=thresholds,
            training_groups=sorted(train_groups),
            outer_excluded=sorted(test_groups),
            training_operating_points_sha256=digest,
            extras_budget_per_endogenous_manual=protocol[
                "extras_budget_per_endogenous_manual"
            ],
            selected_training_correct=int(selected.correct.sum()),
            selected_training_extras=int(selected.extras.sum()),
            selected_training_wrong_peak=int(selected.different_peak.sum()),
            minimum_training_mode_coverage=float(
                (selected.correct / selected.manual).min()
            ),
            note=protocol["model_fit_note"],
        ),
    )
    print("JOINT TRAINING-ONLY POINTS", fold, thresholds, flush=True)
    return thresholds


def main():
    protocol = initialize()
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards

    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    for fold in range(5):
        thresholds = calibrate(protocol, projects, folds, fold)
        model = pd.read_pickle(PREVIOUS / f"fold_{fold}_nonlinear_model.pkl")
        std_model = json.loads(
            (STD / "models" / f"fold_{fold}_standard_models.json").read_text()
        )
        heldout = [d for d in projects if folds[d] == fold]
        assert not (
            set(model["training_cohorts"]) | set(std_model["training_groups"])
        ) & {d[:7] for d in heldout}
        for dataset in heldout:
            dest = OUT / "runs" / dataset
            if (dest / "complete.json").exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            data = pd.read_pickle(
                PREVIOUS / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
            )
            raw, evidence = data["pool"], data["evidence"]
            scores = pd.Series(
                model["estimator"].predict_proba(model_matrix(evidence, model))[:, 1],
                index=raw.index,
            )
            result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
            catalog = declared_standard_targets(
                OUT / "source_snapshot/reference_data/matlab_pipeline",
                result["internal_standards_expected"],
            )
            parts, candidates, audits = (
                [],
                [raw.drop(columns=["training_positive", "training_label_known"])],
                [],
            )
            for stream, payload in result["streams"].items():
                pool = raw[raw.stream.eq(stream)]
                selected = select_operating_candidates(
                    pool,
                    scores.loc[pool.index],
                    threshold=thresholds[stream],
                    ambiguity_margin=0.05,
                )
                selected = selected.drop(
                    columns=["training_positive", "training_label_known"]
                )
                selected["auto_annotation_status"] = "balanced_training_operating_point"
                selected["operating_threshold"] = thresholds[stream]
                protected = (
                    payload["annotations"][
                        payload["annotations"].is_is.fillna(False).astype(bool)
                    ]
                    .copy()
                    .assign(stream=stream)
                )
                combined = pd.concat([selected, protected], ignore_index=True)
                assert not combined.feat_feature_id.astype(str).duplicated().any(), (
                    dataset,
                    stream,
                )
                corrected, extra, audit = complete_declared_standards(
                    combined,
                    payload["features"],
                    catalog[catalog.stream.eq(stream)],
                    std_model["library"],
                    std_model["atlas"],
                    std_model["transfer"],
                    std_model["channels"],
                    stream=stream,
                    alignment_annotations=payload["annotations"],
                )
                parts.append(corrected.assign(stream=stream))
                candidates.extend(
                    [
                        protected,
                        extra.assign(stream=stream),
                        corrected[corrected.is_is.fillna(False).astype(bool)].assign(
                            stream=stream
                        ),
                    ]
                )
                audits.append(audit.assign(dataset=dataset, stream=stream))
            annotations = pd.concat(parts, ignore_index=True)
            all_candidates = pd.concat(candidates, ignore_index=True)
            keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
            all_candidates = all_candidates.drop_duplicates(
                keys, keep="last"
            ).reset_index(drop=True)
            chosen = set(annotations[keys].itertuples(index=False, name=None))
            all_candidates["automatic_accepted"] = [
                k in chosen
                for k in all_candidates[keys].itertuples(index=False, name=None)
            ]
            annotations.to_csv(dest / "automatic_annotations.csv", index=False)
            all_candidates.to_pickle(dest / "candidates.pkl")
            scores.to_pickle(dest / "scoring_audit.pkl")
            pd.concat(audits, ignore_index=True).to_csv(
                dest / "standard_audit.csv", index=False
            )
            dump(
                dest / "complete.json",
                dict(
                    dataset=dataset,
                    fold=fold,
                    status="complete",
                    production_enabled=False,
                    thresholds=thresholds,
                    counts=annotations.groupby("stream").size().to_dict(),
                ),
            )
            print(
                "SELECTED BALANCED-BUDGET HELD-OUT",
                dataset,
                len(annotations),
                flush=True,
            )
    print(
        "EXPERIMENT009 COMPLETE; FULL132-ROW REPORT REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
