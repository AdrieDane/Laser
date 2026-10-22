"""Experiment015: one-pass strong automatic-anchor RT completion, all modes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

from context import REPO, WORKSPACE as ROOT

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
PREVIOUS = ROOT / "outputs/qe_optimization_20261022/experiment_009"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_015"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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
        dependencies = {}
        for d in projects:
            for p in (
                PREVIOUS / "runs" / d / "automatic_annotations.csv",
                PREVIOUS / "runs" / d / "candidates.pkl",
                BASE / "runs" / d / "full_annotation.pkl",
            ):
                dependencies[str(p)] = sha(p)
        for fold in range(5):
            for p in (
                ATLAS / f"fold_{fold}_atlas.json",
                ROOT
                / f"outputs/qe_optimization_20261022/experiment_001/fold_{fold}_selection_model.json",
                ROOT
                / f"outputs/qe_optimization_20261022/experiment_003/fold_{fold}_composition_envelope.json",
            ):
                dependencies[str(p)] = sha(p)
        for p in (
            ATLAS / "observations.pkl",
            ATLAS / "fold_assignment.json",
            PREVIOUS / "protocol.json",
        ):
            dependencies[str(p)] = sha(p)
        previous = json.loads((PREVIOUS / "protocol.json").read_text())
        protocol = dict(
            name="015-strong-automatic-anchor-RT-completion",
            baseline=str(BASE),
            previous=str(PREVIOUS),
            folds=5,
            production_enabled=False,
            curation_runtime_input=False,
            candidate_policy="All modes. Same default+003 other-group chemical envelope. Frozen009 automatic endogenous score>=.95 and<1.5ppm anchors. Robust other-class affine warp; target excluded from local class residual. >=6 OTHER compositions,MAD<=2s,p90<=5s,abs shift<=60s,Cwithinanchor extrema+/-2,DB+/-1. Query never self-calibrates. Unique original raw hit<1.5ppm/<=5s BEFORE occupied/standard filtering. Composition>=25%/>=3 OTHER groups and class/adduct>=50%/>=3 OTHER groups. Existing009 calls and all declared spike masses protected; conflicting new identities on a feature ALL withheld. No newly completed call may become an anchor in this one pass.",
            validation=previous["validation"]
            + " RT/chemical/composition/channel models exclude evaluated groups; strong automatic anchors are frozen009 held-out predictions, not manual labels.",
            model_fit_note=previous["model_fit_note"]
            + " No classifier refit or test-label cutoff selection in015. Fixed deterministic completion criteria, not invented confidence/FDR.",
            run_kind="Saved009 selection/raw-feature replay with one-pass strong automatic-anchor RT completion across all four modes. Every old009 call preserved, no reranking or mass recalibration.",
            source_snapshot_sha256=hashes,
            dependencies_sha256=dependencies,
        )
        dump(OUT / "protocol.json", protocol)
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
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
    from matlab_annotation.anchored_series import (
        fit_class_channels,
        predict_strong_anchor_rt,
        complete_strong_anchor_series,
    )
    from matlab_annotation.chemical_targets import compile_additional_targets
    from matlab_annotation.declared_standards import declared_standard_targets

    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    observations = pd.read_pickle(ATLAS / "observations.pkl")
    models = {}
    (OUT / "models").mkdir(exist_ok=True)
    for fold in range(5):
        path = OUT / "models" / f"fold_{fold}_support.json"
        heldout = {d[:7] for d in projects if folds[d] == fold}
        if not path.exists():
            training = observations[
                ~observations.cohort.isin(heldout) & observations.mapped
            ].copy()
            assert not set(training.cohort) & heldout
            priors = json.loads(
                (
                    ROOT
                    / f"outputs/qe_optimization_20261022/experiment_001/fold_{fold}_selection_model.json"
                ).read_text()
            )
            assert not set(priors["training_cohorts"]) & heldout
            dump(
                path,
                dict(
                    priors=priors["composition_priors"],
                    channels=fit_class_channels(training),
                    training_groups=sorted(set(training.cohort)),
                    held_out_groups=sorted(heldout),
                ),
            )
        models[fold] = json.loads(path.read_text())
        assert not set(models[fold]["training_groups"]) & heldout
    # Training reference is not passed to dataset discovery/prediction.
    del observations
    for dataset in projects:
        dest = OUT / "runs" / dataset
        if (dest / "complete.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        fold = folds[dataset]
        model = models[fold]
        assert dataset[:7] not in model["training_groups"]
        atlas = json.loads((ATLAS / f"fold_{fold}_atlas.json").read_text())
        assert dataset[:7] not in atlas["training_groups"]
        envelope = json.loads(
            (
                ROOT
                / f"outputs/qe_optimization_20261022/experiment_003/fold_{fold}_composition_envelope.json"
            ).read_text()
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
        additions, audits = [], []
        candidates = [pd.read_pickle(PREVIOUS / "runs" / dataset / "candidates.pkl")]
        for stream, payload in result["streams"].items():
            old = before[before.stream.eq(stream)].copy()
            generated, _, _ = compile_additional_targets(
                payload["reference"],
                envelope["envelope"],
                stream=stream,
                declared_standards=catalog[catalog.stream.eq(stream)],
            )
            targets = (
                pd.concat([payload["reference"], generated], ignore_index=True)
                .drop_duplicates(["name", "adduct"])
                .reset_index(drop=True)
            )
            predicted = predict_strong_anchor_rt(
                targets, old, atlas["atlas"], stream=stream
            )
            targets = targets.drop(columns=list(predicted), errors="ignore").join(
                predicted
            )
            targets.assign(stream=stream).to_pickle(
                dest / (stream + "_blind_RT_targets.pkl")
            )
            completed, raw, audit = complete_strong_anchor_series(
                old,
                payload["features"],
                targets,
                model["priors"],
                model["channels"],
                stream=stream,
                declared_standard_masses=catalog.loc[
                    catalog.stream.eq(stream), "ref_mz"
                ],
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
        audit.to_csv(dest / "strong_anchor_audit.csv", index=False)
        dump(
            dest / "complete.json",
            dict(
                dataset=dataset,
                fold=fold,
                status="complete",
                production_enabled=False,
                annotations_before=len(before),
                annotations_after=len(after),
                added=len(after) - len(before),
                actions=audit.action.value_counts().to_dict() if len(audit) else {},
            ),
        )
        print(
            "STRONG ANCHOR COMPLETED",
            dataset,
            "added",
            len(after) - len(before),
            flush=True,
        )
    print(
        "EXPERIMENT015 COMPLETE; FULL132 TABLE REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
