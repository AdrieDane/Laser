"""Experiment012: conservative NP series completion, every009 call preserved."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

from context import WORKSPACE as ROOT, REPO

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
PREVIOUS = ROOT / "outputs/qe_optimization_20261022/experiment_009"
GRID = ROOT / "outputs/qe_optimization_20261022/raw_grid_development_v1"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_012"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def initialize():
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
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
            p.relative_to(snapshot).as_posix(): sha(p)
            for p in snapshot.rglob("*")
            if p.is_file()
        }
        dependencies = {str(GRID / "protocol.json"): sha(GRID / "protocol.json")}
        for d in projects:
            for path in (
                GRID / "predictions" / (d + ".pkl"),
                PREVIOUS / "runs" / d / "automatic_annotations.csv",
                PREVIOUS / "runs" / d / "candidates.pkl",
            ):
                dependencies[str(path)] = sha(path)
        models = OUT / "models"
        models.mkdir()
        for fold in range(5):
            path = (
                ROOT
                / f"outputs/qe_optimization_20261022/experiment_001/fold_{fold}_selection_model.json"
            )
            data = json.loads(path.read_text())
            dump(
                models / f"fold_{fold}_composition_support.json",
                dict(
                    priors=data["composition_priors"],
                    training_groups=data["training_cohorts"],
                    source_sha256=sha(path),
                ),
            )
        previous = json.loads((PREVIOUS / "protocol.json").read_text())
        protocol = dict(
            name="012-conservative-independent-NP-series-completion",
            baseline=str(BASE),
            previous=str(PREVIOUS),
            folds=5,
            production_enabled=False,
            curation_runtime_input=False,
            candidate_policy="NP ONLY. Frozen independently inferred raw mass-series RT from other-group atlas/prevalence and original raw mass/RT/intensity. Class must lack >=4 agreeing old automatic local anchors. Unique original hit within<1.5ppm/<=5s BEFORE occupied/standard filtering. Target composition in>=25% and>=3 OTHER training groups. Existing identities/occupied features/declared spike masses protected; conflicting identity claims ALL withheld.",
            old_calls_policy="EVERY009 annotation/coordinate/identity is preserved. No reranking, removals, classifier refit, score threshold change or RP additions.",
            RT_source="raw_grid_development_v1/predictions ONLY: blind target/RT hypotheses saved BEFORE loading that dataset's evaluation labels. NOT the root per-feature diagnostic pickles/CSVs. Original frozen pre-prefilter RT function used in those cached predictions; its protocol/runner are preserved as dependencies.",
            validation=previous["validation"]
            + " Other-group RT/composition models also exclude this evaluated project. No new classifier fitted.",
            model_fit_note="Conditional grouped replay, inherited baseline prior exposure, not untouched external or pristine end-to-end validation. New additions have deterministic physical evidence, NOT a fabricated confidence/FDR.",
            run_kind="Saved009 annotation-list replay plus conservative unique-hit NP completion using blind held-out raw RT-grid predictions and original features. RP/old calls unchanged; no new classifier.",
            source_snapshot_sha256=hashes,
            dependencies_sha256=dependencies,
        )
        dump(OUT / "protocol.json", protocol)
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        shutil.copy2(GRID / "protocol.json", OUT / "raw_grid_protocol_dependency.json")
        shutil.copy2(
            GRID / "runner_source.py", OUT / "raw_grid_prediction_runner_dependency.py"
        )
        dump(
            OUT / "run_manifest.json",
            dict(
                datasets=projects,
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
        assert sha(OUT / "source_snapshot" / name) == expected, name
    for name, expected in protocol["dependencies_sha256"].items():
        assert sha(Path(name)) == expected, name
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    return projects


def main():
    projects = initialize()
    from matlab_annotation.np_series_completion import complete_np_series
    from matlab_annotation.declared_standards import declared_standard_targets

    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    for dataset in projects:
        dest = OUT / "runs" / dataset
        if (dest / "complete.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        before = pd.read_csv(
            PREVIOUS / "runs" / dataset / "automatic_annotations.csv",
            dtype={"feat_feature_id": str},
            low_memory=False,
        )
        candidates = [pd.read_pickle(PREVIOUS / "runs" / dataset / "candidates.pkl")]
        predictions = pd.read_pickle(GRID / "predictions" / (dataset + ".pkl"))
        assert not {
            "same_manual_peak",
            "training_positive",
            "mapped",
            "newly_inside_window",
            "manual_feature_id",
        } & set(predictions)
        model = json.loads(
            (
                OUT / "models" / f"fold_{folds[dataset]}_composition_support.json"
            ).read_text()
        )
        assert dataset[:7] not in model["training_groups"]
        atlas = json.loads((ATLAS / f"fold_{folds[dataset]}_atlas.json").read_text())
        assert dataset[:7] not in atlas["training_groups"]
        result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        catalog = declared_standard_targets(
            OUT / "source_snapshot/reference_data/matlab_pipeline",
            result["internal_standards_expected"],
        )
        additions, audits = [], []
        for stream in ("NP_Pos", "NP_Neg"):
            old = before[before.stream.eq(stream)].copy()
            targets = predictions[predictions.stream.eq(stream)].copy()
            completed, raw, audit = complete_np_series(
                old,
                result["streams"][stream]["features"],
                targets,
                model["priors"],
                stream=stream,
                declared_standard_masses=catalog.loc[
                    catalog.stream.eq(stream), "ref_mz"
                ],
                min_prevalence=0.25,
                min_training_groups=3,
            )
            if len(completed) > len(old):
                additions.append(completed.iloc[len(old) :].assign(stream=stream))
            if len(raw):
                candidates.append(raw.assign(stream=stream))
            if len(audit):
                audits.append(audit.assign(stream=stream))
        after = pd.concat([before, *additions], ignore_index=True)
        pd.testing.assert_frame_equal(
            after.iloc[: len(before)][before.columns], before, check_dtype=False
        )
        assert (
            not after.assign(feat_feature_id=after.feat_feature_id.astype(str))
            .duplicated(["stream", "feat_feature_id"])
            .any()
        )
        combined = pd.concat(candidates, ignore_index=True)
        keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
        combined = combined.drop_duplicates(keys, keep="first").reset_index(drop=True)
        chosen = set(after[keys].itertuples(index=False, name=None))
        combined["automatic_accepted"] = [
            k in chosen for k in combined[keys].itertuples(index=False, name=None)
        ]
        after.to_csv(dest / "automatic_annotations.csv", index=False)
        combined.to_pickle(dest / "candidates.pkl")
        audit = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
        audit.to_csv(dest / "np_series_audit.csv", index=False)
        dump(
            dest / "complete.json",
            dict(
                dataset=dataset,
                fold=folds[dataset],
                status="complete",
                production_enabled=False,
                annotations_before=len(before),
                annotations_after=len(after),
                added=len(after) - len(before),
                actions=audit.action.value_counts().to_dict() if len(audit) else {},
            ),
        )
        print(
            "NP SERIES COMPLETED",
            dataset,
            "added",
            len(after) - len(before),
            flush=True,
        )
    print(
        "EXPERIMENT012 COMPLETE; FULL132-ROW REPORT REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
