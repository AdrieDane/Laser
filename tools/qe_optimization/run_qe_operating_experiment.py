"""Experiment008: training-group-calibrated operating points, frozen007 scores.

No held-out annotation is used to choose the threshold. Physical candidates,
final score models, ambiguity margins and005 standard handling stay fixed.
"""

from __future__ import annotations

from collections import Counter
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
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_008"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qe_dual_experiment import model_matrix
from run_qe_atlas_experiment import species, adduct


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
                name="008-training-group-calibrated-operating-points",
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
                extras_budget_per_endogenous_manual=0.09,
                threshold_policy="Per measurement stream: highest correct same-original-feature count on THREE INDEPENDENT TRAINING-group score folds within fixed9% unconfirmed-endogenous-extras/endogenous-manual budget. Ties prefer fewer extras then higher threshold. No evaluated outer-group labels, class names, matrix names or feature identities choose a threshold. Standards handled separately;9% endogenous budget leaves approximately1 percentage point for standard extras, NOT a guaranteed test-set bound or FDR.",
                validation=previous["validation"],
                model_fit_note="Inner score-model fitting excludes each calibration project group AND all outer evaluation groups. Evidence is inherited from the OUTER-training RT atlas (not nested inner RT cross-fitting), so this is conditional grouped threshold validation, NOT pristine nested end-to-end cross-validation.",
                run_kind="Saved007 original-feature candidate/score replay; operating cutoffs learned from three project-group-held-out predictions within each outer training set; same ambiguity margin and EXACT005 standard adjudication/models.",
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
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from matlab_annotation.operating_point import (
        select_operating_candidates,
        choose_training_operating_point,
    )

    folder = OUT / "calibration" / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "operating_points.json"
    train_projects = [d for d in projects if folds[d] != fold]
    test_groups = {d[:7] for d in projects if folds[d] == fold}
    if path.exists():
        saved = json.loads(path.read_text())
        assert set(saved["training_groups"]) == {d[:7] for d in train_projects}
        assert not set(saved["training_groups"]) & test_groups
        return saved["thresholds"]
    data = {
        d: pd.read_pickle(PREVIOUS / "prepared" / f"fold_{fold}" / (d + ".pkl"))
        for d in train_projects
    }
    cohort = [d[:7] for d in train_projects]
    predictions = {}
    for inner, (train_idx, val_idx) in enumerate(
        GroupKFold(n_splits=3).split(train_projects, groups=cohort)
    ):
        train = [train_projects[i] for i in train_idx]
        val = [train_projects[i] for i in val_idx]
        assert not {d[:7] for d in train} & ({d[:7] for d in val} | test_groups)
        model_path = folder / f"inner_{inner}_score_model.pkl"
        if not model_path.exists():
            xs, ys, groups = [], [], []
            for d in train:
                known = data[d]["pool"].training_label_known.astype(bool)
                xs.append(data[d]["evidence"].loc[known])
                ys.extend(data[d]["pool"].loc[known, "training_positive"].astype(int))
                groups.extend([d[:7]] * int(known.sum()))
            x = pd.concat(xs, ignore_index=True)
            categories = {
                c: {v: i for i, v in enumerate(sorted(x[c].astype(str).unique()))}
                for c in x
                if not pd.api.types.is_numeric_dtype(x[c])
            }
            assert all(len(v) <= 255 for v in categories.values())
            model = dict(
                columns=list(x),
                categories=categories,
                training_cohorts=sorted(set(groups)),
                calibration_groups=sorted({d[:7] for d in val}),
                outer_excluded=sorted(test_groups),
            )
            counts = Counter(groups)
            weights = np.asarray(
                [len(groups) / len(counts) / counts[g] for g in groups]
            )
            params = {k: v for k, v in protocol["classifier"].items() if k != "kind"}
            estimator = HistGradientBoostingClassifier(
                **params, categorical_features=[c in categories for c in x]
            )
            estimator.fit(model_matrix(x, model), np.asarray(ys), sample_weight=weights)
            model["estimator"] = estimator
            pd.to_pickle(model, model_path)
            print("FIT INNER CALIBRATION", fold, inner, len(x), flush=True)
        model = pd.read_pickle(model_path)
        assert set(model["training_cohorts"]) == {d[:7] for d in train}
        assert not set(model["training_cohorts"]) & ({d[:7] for d in val} | test_groups)
        for d in val:
            predictions[d] = pd.Series(
                model["estimator"].predict_proba(
                    model_matrix(data[d]["evidence"], model)
                )[:, 1],
                index=data[d]["pool"].index,
            )
    assert set(predictions) == set(train_projects)
    manual_counts, cases = Counter(), {}
    for d in train_projects:
        # Calibration references are loaded ONLY for outer TRAINING datasets.
        manual = pd.read_pickle(BASE / "evaluation_v2" / d / "manual_comparison.pkl")
        manual = manual[~manual.internal_standard]
        identities = set(
            manual[["stream", "species", "adduct"]].itertuples(index=False, name=None)
        )
        manual_counts.update(manual.stream.value_counts().to_dict())
        pool = data[d]["pool"].copy()
        pool["calibration_identity_in_manual"] = [
            (s, species(n), adduct(a)) in identities
            for s, n, a in pool[["stream", "name", "adduct"]].itertuples(
                index=False, name=None
            )
        ]
        cases[d] = pool
    records = []
    for threshold in protocol["threshold_grid"]:
        counts = {s: Counter() for s in manual_counts}
        for d, pool in cases.items():
            for stream, part in pool.groupby("stream", sort=False):
                selected = select_operating_candidates(
                    part,
                    predictions[d].loc[part.index],
                    threshold=threshold,
                    ambiguity_margin=0.05,
                )
                counts[stream]["correct"] += int(selected.training_positive.sum())
                counts[stream]["extras"] += int(
                    (~selected.calibration_identity_in_manual).sum()
                )
                counts[stream]["different_peak"] += int(
                    (
                        selected.calibration_identity_in_manual
                        & selected.training_label_known
                        & ~selected.training_positive
                    ).sum()
                )
        for stream, n in manual_counts.items():
            records.append(
                dict(
                    stream=stream, threshold=threshold, manual=n, **dict(counts[stream])
                )
            )
    table = pd.DataFrame(records).fillna(0)
    table.to_csv(folder / "training_operating_points.csv", index=False)
    thresholds = {}
    for stream, part in table.groupby("stream"):
        threshold = choose_training_operating_point(
            part, extras_per_manual=protocol["extras_budget_per_endogenous_manual"]
        )
        assert threshold is not None, (
            fold,
            stream,
            "No feasible training point; inspect rather than silently exceed budget",
        )
        thresholds[stream] = threshold
    dump(
        path,
        dict(
            thresholds=thresholds,
            training_groups=sorted(set(cohort)),
            outer_excluded=sorted(test_groups),
            note=protocol["model_fit_note"],
            extras_budget_per_endogenous_manual=0.09,
        ),
    )
    print("TRAINING-ONLY OPERATING POINTS", fold, thresholds, flush=True)
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
                selected["auto_annotation_status"] = (
                    "training_calibrated_operating_point"
                )
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
                "SELECTED TRAINING-CALIBRATED HELD-OUT",
                dataset,
                len(annotations),
                flush=True,
            )
    print(
        "EXPERIMENT008 COMPLETE; FULL132-ROW REPORT REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
