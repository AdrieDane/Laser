"""Experiment018: bounded RP residual plane, FIXED009 identity/adduct counts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
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
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_018"


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
        for p in [
            ATLAS / "fold_assignment.json",
            PREVIOUS / "protocol.json",
            *(ATLAS / f"fold_{f}_atlas.json" for f in range(5)),
        ]:
            dependencies[str(p)] = sha(p)
        previous = json.loads((PREVIOUS / "protocol.json").read_text())
        protocol = dict(
            name="018-bounded-RP-residual-plane-original-peak-choice",
            baseline=str(BASE),
            previous=str(PREVIOUS),
            folds=5,
            production_enabled=False,
            curation_runtime_input=False,
            candidate_policy="FIXED009 identity/adduct/mode/IS multisets and counts. RP ONLY. New ORIGINAL features may be considered at same theoretical identity mass under independent bounded within-run RT residual plane; no new identities, score thresholds or mass calibration.",
            anchors="Frozen009 automatic endogenous class score>=.95,absppm<1.5. No manual/standard rows. Distinct CDB compositions once,highest frozen score over repeated channels. OTHER-group RTatlas; other-class robust global warp. Each target composition excluded from own local fit AND every leave-one-anchor-out validation fit.",
            plane=">=12 OTHER compositions,Cspan>=6/DBspan>=2,full rank/condition<=100. Query inside OTHER-anchor CDB convex hull(NO extrapolation). Median/p90 leave-one-anchor-out errors<=1.5/3s. Residual carbon slope<=3sec/C,DBslope<=5sec/DB; abs residual shift<=60s at anchors/query. Total RP RTsurface carbon derivative>0 andDB derivative<0 at all used anchors/query.",
            replacement="Old peak>=7s from prediction; UNIQUE RAW alternative<1.5ppm/<=3s before occupancy/standard filtering. Original occupied peaks and all declared-positive-quantity standard masses protected. Old manual/standard rows untouched; no cascades. Conflicting proposals ALL withheld. Replacement coordinates from original raw feature; old feature-dependent scores/quality cleared rather than transferred. Separate physical provenance,NOT chemical confidence/FDR.",
            preservation="NP EXACTLY unchanged. ALL132 counts/extras/accepted identity multisets must equal009. No classifier refit. No catalogue017 additions reused/cherry-picked.",
            validation=previous["validation"]
            + " RTatlas and frozen009 anchor score models exclude evaluated project groups; no held-out manual RT enters the plane.",
            model_fit_note=previous["model_fit_note"]
            + " Conditional grouped physical peak-choice replay,inherited baseline prior exposure,not untouched external/pristine end-to-end validation.",
            run_kind="Saved009 fixed-identity replay with bounded blind RP RTshape inference and conservative original-peak replacement. NP/counts/extras unchanged; not a new production/raw-input run.",
            numerical_workers=3,
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
    return projects


def run_dataset(dataset):
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    from matlab_annotation.rp_residual_plane import predict_rp_residual_plane
    from matlab_annotation.rp_plane_peak_choice import repeak_with_rp_plane
    from matlab_annotation.declared_standards import declared_standard_targets

    dest = OUT / "runs" / dataset
    if (dest / "complete.json").exists():
        return dataset, "already complete"
    dest.mkdir(parents=True, exist_ok=True)
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    fold = folds[dataset]
    atlas = json.loads((ATLAS / f"fold_{fold}_atlas.json").read_text())
    assert dataset[:7] not in atlas["training_groups"]
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
    parts, audits, statuses = [], [], {}
    for stream, payload in result["streams"].items():
        old = before[before.stream.eq(stream)].copy()
        if stream.startswith("NP_"):
            parts.append(old)
            continue
        predictions = predict_rp_residual_plane(old, old, atlas["atlas"], stream=stream)
        predictions.to_pickle(dest / (stream + "_blind_plane_predictions.pkl"))
        statuses[stream] = predictions.rp_plane_status.value_counts().to_dict()
        completed, raw, audit = repeak_with_rp_plane(
            old,
            payload["features"],
            predictions,
            stream=stream,
            declared_standard_masses=catalogue.loc[
                catalogue.stream.eq(stream), "ref_mz"
            ],
        )
        parts.append(completed)
        if len(raw):
            candidates.append(raw)
        if len(audit):
            audits.append(audit.assign(stream=stream))
    after = pd.concat(parts, axis=0).sort_index()
    assert after.index.equals(before.index)
    pd.testing.assert_frame_equal(
        before[before.stream.str.startswith("NP_")],
        after.loc[after.stream.str.startswith("NP_"), before.columns],
        check_dtype=False,
    )
    keys_identity = ["stream", "name", "adduct", "is_is"]
    from collections import Counter

    assert Counter(before[keys_identity].itertuples(index=False, name=None)) == Counter(
        after[keys_identity].itertuples(index=False, name=None)
    )
    assert (
        not after.assign(feat_feature_id=after.feat_feature_id.astype(str))
        .duplicated(["stream", "feat_feature_id"])
        .any()
    )
    combined = pd.concat(candidates, ignore_index=True)
    keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
    combined = combined.drop_duplicates(keys, keep="last").reset_index(drop=True)
    chosen = set(after[keys].itertuples(index=False, name=None))
    combined["automatic_accepted"] = [
        k in chosen for k in combined[keys].itertuples(index=False, name=None)
    ]
    after.to_csv(dest / "automatic_annotations.csv", index=False)
    combined.to_pickle(dest / "candidates.pkl")
    audit = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    audit.to_csv(dest / "rp_plane_peak_audit.csv", index=False)
    actions = audit.action.value_counts().to_dict() if len(audit) else {}
    dump(
        dest / "complete.json",
        dict(
            dataset=dataset,
            fold=fold,
            status="complete",
            production_enabled=False,
            annotations_before=len(before),
            annotations_after=len(after),
            actions=actions,
            plane_statuses=statuses,
        ),
    )
    return dataset, actions


def main():
    projects = initialize()
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_dataset, d): d for d in projects}
        for future in as_completed(futures):
            dataset, actions = future.result()
            print("RP PLANE COMPLETED", dataset, actions, flush=True)
    print(
        "EXPERIMENT018 COMPLETE; FULL132 TABLE REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
