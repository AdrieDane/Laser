"""Experiment006: fixed class/mode specialists, unchanged004 candidates +005 standards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

from context import WORKSPACE as ROOT, REPO

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
POOL = ROOT / "outputs/qe_optimization_20261022/experiment_004"
PREVIOUS = ROOT / "outputs/qe_optimization_20261022/experiment_005"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_006"
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
                name="006-class-mode-specialists",
                baseline=str(BASE),
                previous=str(PREVIOUS),
                folds=5,
                threshold=0.5,
                ambiguity_margin=0.05,
                production_enabled=False,
                candidate_universe="EXACT experiment004 pools/evidence; no additional mass, RT or composition admissions.",
                support_policy="TRAINING only: >=100 positive, >=100 negative pairs, >=6 positive project groups; otherwise global004 fallback. Zero positives with >=6 groups of known negative candidates: withhold full calls, retain candidates. Unseen classes retain global fallback.",
                weighting="Equal project-group total weight WITHIN each class/mode specialist, mean row weight one.",
                classifier=dict(
                    max_iter=180,
                    max_leaf_nodes=15,
                    min_samples_leaf=10,
                    l2_regularization=2,
                    learning_rate=0.08,
                    early_stopping=False,
                    random_state=20261022,
                ),
                validation=previous["validation"],
                model_fit_note="Outer-training RT atlases generated outer-training evidence, not nested inner cross-fitting. No held-out labels choose specialists, hyperparameters or thresholds.",
                run_kind="Saved experiment004 candidate/evidence replay with new held-out class/mode classifiers, followed by EXACT005 declared-standard adjudication/models using original baseline automatic calibration anchors.",
                standard_policy=previous["standard_policy"],
                curation_runtime_input=False,
                source_snapshot_sha256=hashes,
            ),
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        shutil.copy2(
            Path(__file__).with_name("run_qe_dual_experiment.py"),
            OUT / "dual_runner_dependency.py",
        )
        manifest = json.loads((BASE / "run_manifest.json").read_text())
        protocol = json.loads((OUT / "protocol.json").read_text())
        dump(
            OUT / "run_manifest.json",
            dict(
                datasets=manifest["datasets"],
                fresh_raw_input_runs=False,
                evaluation_note=protocol["run_kind"] + " " + protocol["validation"],
            ),
        )
    protocol = json.loads((OUT / "protocol.json").read_text())
    for name, expected in protocol["source_snapshot_sha256"].items():
        assert (
            hashlib.sha256((OUT / "source_snapshot" / name).read_bytes()).hexdigest()
            == expected
        ), name
    assert (OUT / "dual_runner_dependency.py").read_bytes() == (
        Path(__file__).with_name("run_qe_dual_experiment.py")
    ).read_bytes()
    for name in (
        "standard_completion.py",
        "standard_rt_library.py",
        "standard_rt_transfer.py",
        "declared_standards.py",
    ):
        rel = Path("source_snapshot/app/matlab_annotation") / name
        assert (OUT / rel).read_bytes() == (PREVIOUS / rel).read_bytes(), name
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    return protocol


def main():
    protocol = initialize()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from matlab_annotation.class_specialists import (
        cohort_balanced_weights,
        specialist_training_plan,
    )
    from matlab_annotation.class_candidates import select_class_candidates
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards

    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    for fold in range(5):
        model_path = OUT / f"fold_{fold}_specialist_models.pkl"
        global_model = pd.read_pickle(POOL / f"fold_{fold}_nonlinear_model.pkl")
        heldout = [d for d in projects if folds[d] == fold]
        assert not set(global_model["training_cohorts"]) & {d[:7] for d in heldout}
        if not model_path.exists():
            xs, ys, groups = [], [], []
            for dataset in projects:
                if folds[dataset] == fold:
                    continue
                data = pd.read_pickle(
                    POOL / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
                )
                known = data["pool"].training_label_known.astype(bool)
                xs.append(data["evidence"].loc[known])
                ys.extend(data["pool"].loc[known, "training_positive"].astype(int))
                groups.extend([dataset[:7]] * int(known.sum()))
            x = pd.concat(xs, ignore_index=True)
            y, cohort = pd.Series(ys), pd.Series(groups)
            assert set(cohort) == set(global_model["training_cohorts"])
            plan = specialist_training_plan(x.stream_class, y, cohort)
            models = {}
            for key, support in plan.items():
                if support["action"] != "specialist":
                    continue
                idx = x.stream_class.eq(key)
                estimator = HistGradientBoostingClassifier(
                    **protocol["classifier"],
                    categorical_features=[
                        c in global_model["categories"] for c in global_model["columns"]
                    ],
                )
                estimator.fit(
                    model_matrix(x.loc[idx], global_model),
                    y.loc[idx].to_numpy(),
                    sample_weight=cohort_balanced_weights(cohort.loc[idx]),
                )
                models[key] = estimator
                print("FIT SPECIALIST", fold, key, support, flush=True)
            saved = dict(
                plan=plan,
                estimators=models,
                training_groups=sorted(set(groups)),
                heldout=heldout,
            )
            pd.to_pickle(saved, model_path)
            dump(
                OUT / f"fold_{fold}_specialist_support.json",
                {k: v for k, v in saved.items() if k != "estimators"},
            )
        saved = pd.read_pickle(model_path)
        assert not set(saved["training_groups"]) & {d[:7] for d in heldout}
        std_model = json.loads(
            (PREVIOUS / "models" / f"fold_{fold}_standard_models.json").read_text()
        )
        assert not set(std_model["training_groups"]) & {d[:7] for d in heldout}
        for dataset in heldout:
            dest = OUT / "runs" / dataset
            if (dest / "complete.json").exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            data = pd.read_pickle(
                POOL / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
            )
            raw, evidence = data["pool"], data["evidence"]
            scores = pd.Series(
                global_model["estimator"].predict_proba(
                    model_matrix(evidence, global_model)
                )[:, 1],
                index=raw.index,
            )
            source = pd.Series("global004", index=raw.index)
            for key, support in saved["plan"].items():
                idx = evidence.stream_class.eq(key)
                if not idx.any():
                    continue
                if support["action"] == "specialist":
                    scores.loc[idx] = saved["estimators"][key].predict_proba(
                        model_matrix(evidence.loc[idx], global_model)
                    )[:, 1]
                    source.loc[idx] = "class_specialist"
                elif support["action"] == "withhold_zero_positive_support":
                    scores.loc[idx] = 0.0
                    source.loc[idx] = "no_training_positives"
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
                selected = select_class_candidates(
                    pool, scores.loc[pool.index], threshold=0.5, ambiguity_margin=0.05
                )
                selected = selected.drop(
                    columns=["training_positive", "training_label_known"]
                )
                selected["auto_annotation_status"] = "class_specialist_experimental"
                protected = payload["annotations"][
                    payload["annotations"].is_is.fillna(False).astype(bool)
                ].assign(stream=stream)
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
            pd.DataFrame({"score": scores, "model_source": source}).to_pickle(
                dest / "scoring_audit.pkl"
            )
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
                    counts=annotations.groupby("stream").size().to_dict(),
                    model_sources=source.value_counts().to_dict(),
                ),
            )
            print(
                "SELECTED SPECIALISTS HELD-OUT", dataset, len(annotations), flush=True
            )
    print(
        "EXPERIMENT006 COMPLETE; FULL132-ROW REPORT REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
