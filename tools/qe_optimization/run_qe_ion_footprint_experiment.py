"""Experiment013: neutral-ion ambiguity guard on012 NEW NP-positive calls only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

from context import REPO, WORKSPACE as ROOT

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
OLD = ROOT / "outputs/qe_optimization_20261022/experiment_009"
PREVIOUS = ROOT / "outputs/qe_optimization_20261022/experiment_012"
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_013"


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
        dependencies = {}
        for parent in (OLD, PREVIOUS):
            dependencies[str(parent / "protocol.json")] = sha(parent / "protocol.json")
            for d in projects:
                p = parent / "runs" / d / "automatic_annotations.csv"
                dependencies[str(p)] = sha(p)
        for d in projects:
            p = PREVIOUS / "runs" / d / "candidates.pkl"
            dependencies[str(p)] = sha(p)
        previous = json.loads((PREVIOUS / "protocol.json").read_text())
        protocol = dict(
            name="013-new-NP-positive-neutral-ion-ambiguity-guard",
            baseline=str(BASE),
            previous=str(PREVIOUS),
            old_preserved=str(OLD),
            production_enabled=False,
            curation_runtime_input=False,
            exclusion_rule="NEW NP_Pos calls ONLY: withhold unknown neutral coordinates or different canonical classes within5ppm neutral mass and2s RT of an existing009 call or one another. All conflicting NEW classes withheld, never greedy first winner. Exact native adduct/signed-charge neutral inversion. No same-class channel filtering. Exclusion-only window, not expanded admission or proof of false identity.",
            preservation="Every009 call and every012 RP/NP-negative call exactly preserved. Candidate universe unchanged; rejected new calls stay review candidates. No refit, threshold adjustment or held-out label input.",
            validation=previous["validation"],
            model_fit_note=previous["model_fit_note"],
            run_kind="Saved012 annotation-list replay with a fixed chemical ambiguity filter only on new positive-NP additions; all original009 calls and other012 modes unchanged.",
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
    from matlab_annotation.ion_footprints import filter_new_np_positive_isobars

    for dataset in projects:
        dest = OUT / "runs" / dataset
        if (dest / "complete.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        kwargs = dict(dtype={"feat_feature_id": str}, low_memory=False)
        old = pd.read_csv(
            OLD / "runs" / dataset / "automatic_annotations.csv", **kwargs
        )
        previous = pd.read_csv(
            PREVIOUS / "runs" / dataset / "automatic_annotations.csv", **kwargs
        )
        pd.testing.assert_frame_equal(
            previous.iloc[: len(old)][old.columns], old, check_dtype=False
        )
        additions = previous.iloc[len(old) :].copy().reset_index(drop=True)
        kept, audit = filter_new_np_positive_isobars(old, additions)
        after = pd.concat([previous.iloc[: len(old)], kept], ignore_index=True)
        pd.testing.assert_frame_equal(
            after.iloc[: len(old)][old.columns], old, check_dtype=False
        )
        pd.testing.assert_frame_equal(
            after[~after.stream.eq("NP_Pos")].reset_index(drop=True),
            previous[~previous.stream.eq("NP_Pos")].reset_index(drop=True),
        )
        assert not after.duplicated(["stream", "feat_feature_id"]).any()
        candidates = pd.read_pickle(PREVIOUS / "runs" / dataset / "candidates.pkl")
        keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
        chosen = set(after[keys].itertuples(index=False, name=None))
        candidates["automatic_accepted"] = [
            k in chosen for k in candidates[keys].itertuples(index=False, name=None)
        ]
        after.to_csv(dest / "automatic_annotations.csv", index=False)
        candidates.to_pickle(dest / "candidates.pkl")
        audit.to_csv(dest / "neutral_ion_ambiguity_audit.csv", index=False)
        dump(
            dest / "complete.json",
            dict(
                dataset=dataset,
                status="complete",
                production_enabled=False,
                annotations_before=len(previous),
                annotations_after=len(after),
                withheld_new_calls=len(previous) - len(after),
                actions=audit.action.value_counts().to_dict() if len(audit) else {},
            ),
        )
        print(
            "NEUTRAL GUARD COMPLETED",
            dataset,
            "withheld",
            len(previous) - len(after),
            flush=True,
        )
    print(
        "EXPERIMENT013 COMPLETE; FULL132-ROW REPORT REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
