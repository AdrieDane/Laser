"""Experiment017: independent native catalogue recovery, every009 call retained."""

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
SUPPORT = ROOT / "outputs/qe_optimization_20261022/experiment_015/models"
CHEMICAL = ROOT / "outputs/qe_optimization_20261022/experiment_003"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_017"


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
                BASE / "runs" / d / "full_annotation.pkl",
                PREVIOUS / "runs" / d / "automatic_annotations.csv",
                PREVIOUS / "runs" / d / "candidates.pkl",
            ):
                dependencies[str(p)] = sha(p)
        (OUT / "models").mkdir()
        for fold in range(5):
            p = SUPPORT / f"fold_{fold}_support.json"
            shutil.copy2(p, OUT / "models" / p.name)
            for dependency in (
                p,
                CHEMICAL / f"fold_{fold}_composition_envelope.json",
                ATLAS / f"fold_{fold}_atlas.json",
            ):
                dependencies[str(dependency)] = sha(dependency)
        for p in (ATLAS / "fold_assignment.json", PREVIOUS / "protocol.json"):
            dependencies[str(p)] = sha(p)
        previous = json.loads((PREVIOUS / "protocol.json").read_text())
        protocol = dict(
            name="017-independent-native-catalogue-recovery",
            baseline=str(BASE),
            previous=str(PREVIOUS),
            folds=5,
            production_enabled=False,
            curation_runtime_input=False,
            candidate_policy="ONLY native chemical targets absent from baseline+003 chemical reference,within SAME003 other-group envelope/native CDB axes/adduct channels. Native CodeDecoder/exact signed-charge masses,no inferred two-node lattice or guessed formulas. Actual positive-quantity declared-spike masses protected. NO added targets solely from rejected016 mass scoring.",
            acceptance="Other-group RTatlas>=3cohorts/trainingMAD<=5s and reliable blind other-class run alignment(automatic_anchor_alignment or identity_warp_no_reliable_improvement). Common composition>=25% and>=3OTHERgroups; class/adduct>=50% and>=3OTHERgroups. Unique original raw hit<1.5ppm/<=5s BEFORE occupancy/spike filtering. Existing009 identities/features protected,ALL conflicting new identities withheld. No new score/FDR fabricated.",
            prior_source="Frozen015 per-outer-fold composition/channel SUPPORT MODELS ONLY,not rejected015 selections. Same group exclusions. RT/chemical envelopes from original corresponding outer folds. No new classifier or evaluated-label tuning.",
            preservation="Every009 annotation/identity/coordinate exactly preserved. No mass recalibration,old removal/repeak,score cutoff change or iterative new-call anchors. New hypotheses saved BEFORE manual comparison.",
            validation=previous["validation"]
            + " Native RT/composition/channel/envelope models exclude evaluated groups. No new classifier fit.",
            model_fit_note=previous["model_fit_note"]
            + " Conditional grouped replay; baseline prior exposure remains. Native formula enumeration/parity is not annotation accuracy.",
            run_kind="Saved009 original-feature annotation-list replay plus conservative restoration of targets lost from the old theoretical reference. All modes/all old calls retained. No new full production run.",
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
    for fold in range(5):
        name = f"fold_{fold}_support.json"
        assert sha(OUT / "models" / name) == sha(SUPPORT / name), name
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    return projects


def main():
    projects = initialize()
    from matlab_annotation.catalogue_targets import catalogue_missing_targets
    from matlab_annotation.catalogue_completion import complete_catalogue_targets
    from matlab_annotation.chemical_targets import compile_additional_targets
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.retention_atlas import predict_aligned_retention_atlas

    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    for dataset in projects:
        dest = OUT / "runs" / dataset
        if (dest / "complete.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        fold = folds[dataset]
        support = json.loads((OUT / "models" / f"fold_{fold}_support.json").read_text())
        atlas = json.loads((ATLAS / f"fold_{fold}_atlas.json").read_text())
        chemical = json.loads(
            (CHEMICAL / f"fold_{fold}_composition_envelope.json").read_text()
        )
        assert dataset[:7] not in (
            set(support["training_groups"])
            | set(atlas["training_groups"])
            | set(chemical["training_groups"])
        )
        before = pd.read_csv(
            PREVIOUS / "runs" / dataset / "automatic_annotations.csv",
            dtype={"feat_feature_id": str},
            low_memory=False,
        )
        result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        catalogue = declared_standard_targets(
            OUT / "source_snapshot/reference_data/matlab_pipeline",
            result["internal_standards_expected"],
        )
        candidates = [pd.read_pickle(PREVIOUS / "runs" / dataset / "candidates.pkl")]
        additions, audits, theory_audits = [], [], []
        for stream, payload in result["streams"].items():
            std = catalogue[catalogue.stream.eq(stream)]
            generated, _, _ = compile_additional_targets(
                payload["reference"],
                chemical["envelope"],
                stream=stream,
                declared_standards=std,
            )
            old_reference = pd.concat(
                [payload["reference"], generated], ignore_index=True
            )
            targets, ta = catalogue_missing_targets(
                old_reference,
                OUT / "source_snapshot/reference_data/matlab_pipeline",
                chemical["envelope"],
                stream=stream,
                declared_standards=std,
            )
            old_keys = set(
                old_reference[["name", "adduct"]].itertuples(index=False, name=None)
            )
            assert not old_keys & set(
                targets[["name", "adduct"]].itertuples(index=False, name=None)
            )
            predictions = predict_aligned_retention_atlas(
                targets, payload["annotations"], atlas["atlas"], stream=stream
            )
            targets = targets.join(predictions)
            targets.assign(stream=stream).to_pickle(
                dest / (stream + "_blind_catalogue_targets.pkl")
            )
            if len(ta):
                theory_audits.append(ta)
            old = before[before.stream.eq(stream)].copy()
            completed, raw, audit = complete_catalogue_targets(
                old,
                payload["features"],
                targets,
                support["priors"],
                support["channels"],
                stream=stream,
                declared_standard_masses=std.ref_mz,
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
        audit.to_csv(dest / "catalogue_completion_audit.csv", index=False)
        theory = (
            pd.concat(theory_audits, ignore_index=True)
            if theory_audits
            else pd.DataFrame()
        )
        theory.to_csv(dest / "chemical_enumeration_audit.csv", index=False)
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
                recovered_theoretical_targets=int(theory.added.sum())
                if len(theory)
                else 0,
                actions=audit.action.value_counts().to_dict() if len(audit) else {},
            ),
        )
        print(
            "CATALOGUE RECOVERY COMPLETED",
            dataset,
            "added",
            len(after) - len(before),
            flush=True,
        )
    print(
        "EXPERIMENT017 COMPLETE; FULL132 TABLE REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
