"""Experiment011: conditional original-peak choice, fixed009 accepted identities."""

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
POOL = ROOT / "outputs/qe_optimization_20261022/experiment_007"
PREVIOUS = ROOT / "outputs/qe_optimization_20261022/experiment_009"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_011"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qe_dual_experiment import model_matrix

REMOVE = {
    "carbon",
    "db",
    "carbon_odd",
    "db_per_carbon",
    "composition_training_cohorts",
    "composition_joint_frequency",
    "composition_carbon_frequency",
    "composition_db_frequency",
    "q_context_carbon_median",
    "q_context_carbon_p90",
    "q_context_db_median",
    "q_context_db_p90",
    "q_carbon_minus_context_median",
    "q_db_minus_context_median",
}


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
                name="011-conditional-original-peak-choice",
                baseline=str(BASE),
                previous=str(PREVIOUS),
                folds=5,
                production_enabled=False,
                curation_runtime_input=False,
                candidate_universe="EXACT007 candidate/evidence pools. EXACT009 accepted identity/adduct/mode/IS sets and annotation counts. No new identities, mass/RT admissions or raw-grid source.",
                training="Only resolved identities with exactly ONE true original feature and >=1 wrong alternative. Equal true-peak/combined-alternative weight per identity; equal total project-group weight. Constant composition prevalence descriptors explicitly removed.",
                support="Runtime class/mode requires >=50 conditional training identities and >=6 positive project groups; otherwise no peak replacement.",
                min_score=0.75,
                ambiguity_margin=0.15,
                removed_features=sorted(REMOVE),
                classifier=dict(
                    max_iter=180,
                    max_leaf_nodes=31,
                    min_samples_leaf=25,
                    l2_regularization=2,
                    learning_rate=0.08,
                    early_stopping=False,
                    random_state=20261022,
                ),
                replacement_policy="Only unoccupied original features; no borrowing standard/manual/other accepted peaks, no iterative cascades. Conflicting proposals ALL withheld. Declared-spike masses protected, including unselected standards. Old joint confidence cleared for replaced peaks, separate conditional score retained.",
                validation=previous["validation"],
                model_fit_note=previous["model_fit_note"]
                + " Conditional peak model also excludes each evaluated project group; inherited009 identity decisions remain fixed.",
                run_kind="Saved009 accepted identity-list replay with a new grouped-held-out conditional peak-choice model on existing007 raw-feature evidence. Standards/identity sets and extra identity counts must remain identical.",
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
    assert (OUT / "dual_runner_dependency.py").read_bytes() == (
        Path(__file__).with_name("run_qe_dual_experiment.py")
    ).read_bytes()
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    return protocol


def main():
    protocol = initialize()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from matlab_annotation.conditional_peak_choice import (
        conditional_training_weights,
        rerank_accepted_peaks,
    )
    from matlab_annotation.declared_standards import declared_standard_targets

    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    for fold in range(5):
        path = OUT / f"fold_{fold}_conditional_peak_model.pkl"
        heldout = [d for d in projects if folds[d] == fold]
        if not path.exists():
            pools, xs, datasets, cohorts = [], [], [], []
            for dataset in projects:
                if folds[dataset] == fold:
                    continue
                data = pd.read_pickle(
                    POOL / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
                )
                pools.append(data["pool"])
                xs.append(data["evidence"].drop(columns=list(REMOVE), errors="ignore"))
                datasets.extend([dataset] * len(data["pool"]))
                cohorts.extend([dataset[:7]] * len(data["pool"]))
            pool, x = (
                pd.concat(pools, ignore_index=True),
                pd.concat(xs, ignore_index=True),
            )
            dataset_series, cohort = pd.Series(datasets), pd.Series(cohorts)
            weights = conditional_training_weights(pool, dataset_series, cohort)
            keep = weights.gt(0)
            x, y, weights = (
                x.loc[keep],
                pool.loc[keep, "training_positive"].astype(int),
                weights.loc[keep],
            )
            assert not set(cohort[keep]) & {d[:7] for d in heldout}
            support_rows = []
            positive = pool[keep & pool.training_positive].copy()
            positive["cohort"] = cohort.loc[positive.index]
            for (stream, cls), part in positive.groupby(["stream", "class"]):
                support_rows.append(
                    dict(
                        stream=stream,
                        lipid_class=cls,
                        training_identities=len(part),
                        training_groups=int(part.cohort.nunique()),
                        enabled=len(part) >= 50 and part.cohort.nunique() >= 6,
                    )
                )
            allowed = [
                (r["stream"], r["lipid_class"]) for r in support_rows if r["enabled"]
            ]
            categories = {
                c: {v: i for i, v in enumerate(sorted(x[c].astype(str).unique()))}
                for c in x
                if not pd.api.types.is_numeric_dtype(x[c])
            }
            assert all(len(v) <= 255 for v in categories.values())
            model = dict(
                columns=list(x),
                categories=categories,
                training_cohorts=sorted(set(cohort[keep])),
                held_out_datasets=heldout,
                fold=fold,
                training_pairs=len(x),
                positives=int(y.sum()),
                allowed_classes=allowed,
            )
            estimator = HistGradientBoostingClassifier(
                **protocol["classifier"],
                categorical_features=[c in categories for c in x],
            )
            estimator.fit(
                model_matrix(x, model), y.to_numpy(), sample_weight=weights.to_numpy()
            )
            model["estimator"] = estimator
            pd.to_pickle(model, path)
            dump(
                OUT / f"fold_{fold}_support.json",
                dict(
                    training_groups=model["training_cohorts"],
                    held_out_groups=sorted({d[:7] for d in heldout}),
                    class_support=support_rows,
                    training_pairs=len(x),
                ),
            )
            print(
                "FIT CONDITIONAL PEAK MODEL",
                fold,
                len(x),
                "rows",
                len(allowed),
                "supported class/modes",
                flush=True,
            )
        model = pd.read_pickle(path)
        assert not set(model["training_cohorts"]) & {d[:7] for d in heldout}
        for dataset in heldout:
            dest = OUT / "runs" / dataset
            if (dest / "complete.json").exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            data = pd.read_pickle(
                POOL / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
            )
            pool = data["pool"].drop(
                columns=["training_positive", "training_label_known"]
            )
            scores = pd.Series(
                model["estimator"].predict_proba(model_matrix(data["evidence"], model))[
                    :, 1
                ],
                index=pool.index,
            )
            before = pd.read_csv(
                PREVIOUS / "runs" / dataset / "automatic_annotations.csv",
                dtype={"feat_feature_id": str},
                low_memory=False,
            )
            result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
            catalog = declared_standard_targets(
                OUT / "source_snapshot/reference_data/matlab_pipeline",
                result["internal_standards_expected"],
            )
            forbidden = {}
            for stream, payload in result["streams"].items():
                features = payload["features"]
                masses = catalog.loc[catalog.stream.eq(stream), "ref_mz"].to_numpy(
                    float
                )
                if len(masses):
                    ppm = (
                        np.abs(features.mz.to_numpy(float)[:, None] - masses[None, :])
                        / masses[None, :]
                        * 1e6
                    )
                    forbidden[stream] = set(
                        features.loc[(ppm < 3).any(axis=1), "feature_id"].astype(str)
                    )
            after, audit = rerank_accepted_peaks(
                before,
                pool,
                scores,
                min_score=0.75,
                margin=0.15,
                allowed_classes=set(map(tuple, model["allowed_classes"])),
                forbidden_features=forbidden,
            )
            assert not any(c.startswith("training_") for c in after)
            std = before.is_is.fillna(False).astype(bool)
            pd.testing.assert_frame_equal(
                before.loc[std].reset_index(drop=True),
                after.loc[std, before.columns].reset_index(drop=True),
                check_dtype=False,
            )
            candidates = pd.read_pickle(PREVIOUS / "runs" / dataset / "candidates.pkl")
            keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
            chosen = set(after[keys].itertuples(index=False, name=None))
            candidates["automatic_accepted"] = [
                k in chosen for k in candidates[keys].itertuples(index=False, name=None)
            ]
            after.to_csv(dest / "automatic_annotations.csv", index=False)
            candidates.to_pickle(dest / "candidates.pkl")
            audit.to_csv(dest / "peak_choice_audit.csv", index=False)
            dump(
                dest / "complete.json",
                dict(
                    dataset=dataset,
                    fold=fold,
                    status="complete",
                    production_enabled=False,
                    annotations_before=len(before),
                    annotations_after=len(after),
                    actions=audit.action.value_counts().to_dict() if len(audit) else {},
                ),
            )
            print(
                "RERANKED CONDITIONAL HELD-OUT",
                dataset,
                len(audit),
                "proposals/changes",
                flush=True,
            )
    print(
        "EXPERIMENT011 COMPLETE; FULL132-ROW REPORT REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
