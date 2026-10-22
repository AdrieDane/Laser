"""Experiment 005: explicit declared-standard completion and channel handling.

Endogenous classifier/models/settings remain those of experiment004. No raw
source is edited, no benchmark isomer-equivalence rule is changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

from context import WORKSPACE as ROOT, REPO

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
PREVIOUS = ROOT / "outputs/qe_optimization_20261022/experiment_004"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
STD = ROOT / "outputs/qe_optimization_20261022/standard_rt_development_v3"
TRANSFER = ROOT / "outputs/qe_optimization_20261022/standard_transfer_development"
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_005"


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
        protocol = dict(
            name="005-declared-standard-handling",
            baseline=str(BASE),
            previous=str(PREVIOUS),
            folds=5,
            production_enabled=False,
            source_snapshot_sha256=hashes,
            validation="New standard RT, residual-transfer, reporting-label and channel models exclude the evaluated project group. Inherited baseline/endogenous models have known prior exposure; NOT untouched external or pristine-pipeline validation.",
            run_kind="Saved experiment004 annotation replay plus declared-standard adjudication on ORIGINAL raw features. Endogenous models, settings and candidate discovery otherwise unchanged.",
            standard_policy="Only explicitly declared positive-quantity standards; new channel >=3 training groups / >=50% prevalence; <3ppm, <=10s, one raw feature BEFORE occupancy filtering; no forced replacements of existing standard peak choices or other occupied non-alias features.",
            channel_policy="Existing automatic standard channels unobserved in >=3 independent TRAINING groups curating that standard become companion candidates; channels seen even once are retained. Unknown/unsupported standard panels are not silently deleted.",
            RT_policy="Other automatically detected standards, or independent class-atlas transfer with grouped standard-specific residuals; declared-spike masses excluded from endogenous calibration. Conflicting independent RT references >10s withhold new completion.",
            reporting_policy="Declared-spike phase-specific report metadata requires >=90% training-group consensus AND an independent mass/RT match. Never override an explicit other sn position. Benchmark equivalence rules unchanged.",
            curation_runtime_input=False,
            initial_alignment="Frozen original baseline automatic annotations; new calls cannot bootstrap their own calibration",
        )
        dump(OUT / "protocol.json", protocol)
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        manifest = json.loads((BASE / "run_manifest.json").read_text())
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
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    return protocol


def main():
    initialize()
    from matlab_annotation.standard_completion import (
        complete_declared_standards,
        fit_standard_channel_prior,
    )
    from matlab_annotation.declared_standards import declared_standard_targets

    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    manual = pd.read_pickle(STD / "manual_standard_entries.pkl")
    models = {}
    folder = OUT / "models"
    folder.mkdir(exist_ok=True)
    for fold in range(5):
        train = manual[manual.dataset.map(folds).ne(fold) & manual.mapped]
        test = {d[:7] for d, f in folds.items() if f == fold}
        assert not set(train.cohort) & test
        library = json.loads((STD / f"fold_{fold}_standard_library.json").read_text())[
            "library"
        ]
        transfer = json.loads(
            (TRANSFER / f"fold_{fold}_standard_transfer.json").read_text()
        )["transfer"]
        atlas = json.loads((ATLAS / f"fold_{fold}_atlas.json").read_text())["atlas"]
        channels = fit_standard_channel_prior(train)
        models[fold] = dict(
            library=library,
            transfer=transfer,
            atlas=atlas,
            channels=channels,
            training_groups=sorted(train.cohort.unique()),
            held_out_groups=sorted(test),
        )
        dump(folder / f"fold_{fold}_standard_models.json", models[fold])
    for dataset in projects:
        dest = OUT / "runs" / dataset
        if (dest / "complete.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        baseline = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        selected = pd.read_csv(
            PREVIOUS / "runs" / dataset / "automatic_annotations.csv",
            low_memory=False,
            dtype={"feat_feature_id": str},
        )
        old_candidates = pd.read_pickle(PREVIOUS / "runs" / dataset / "candidates.pkl")
        catalog = declared_standard_targets(
            OUT / "source_snapshot/reference_data/matlab_pipeline",
            baseline["internal_standards_expected"],
        )
        model = models[folds[dataset]]
        parts = []
        candidates = [old_candidates]
        audits = []
        for stream, payload in baseline["streams"].items():
            annotations = selected[selected.stream.eq(stream)].copy()
            declared = catalog[catalog.stream.eq(stream)].copy()
            corrected, extra, audit = complete_declared_standards(
                annotations,
                payload["features"],
                declared,
                model["library"],
                model["atlas"],
                model["transfer"],
                model["channels"],
                stream=stream,
                alignment_annotations=payload["annotations"],
            )
            parts.append(corrected.assign(stream=stream))
            candidates.extend(
                [
                    annotations[annotations.is_is.fillna(False).astype(bool)],
                    extra.assign(stream=stream),
                    corrected[corrected.is_is.fillna(False).astype(bool)],
                ]
            )
            audits.append(audit.assign(dataset=dataset, stream=stream))
        combined = pd.concat(parts, ignore_index=True)
        all_candidates = pd.concat(candidates, ignore_index=True)
        keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
        all_candidates = all_candidates.drop_duplicates(keys, keep="last").reset_index(
            drop=True
        )
        chosen = set(combined[keys].itertuples(index=False, name=None))
        all_candidates["automatic_accepted"] = [
            k in chosen for k in all_candidates[keys].itertuples(index=False, name=None)
        ]
        combined.to_csv(dest / "automatic_annotations.csv", index=False)
        all_candidates.to_pickle(dest / "candidates.pkl")
        audit = pd.concat(audits, ignore_index=True)
        audit.to_csv(dest / "standard_audit.csv", index=False)
        dump(
            dest / "complete.json",
            dict(
                dataset=dataset,
                fold=folds[dataset],
                status="complete",
                counts=combined.groupby("stream").size().to_dict(),
                actions=audit.action.value_counts().to_dict(),
                production_enabled=False,
            ),
        )
        print(
            "STANDARD HANDLING HELD-OUT",
            dataset,
            audit.action.value_counts().to_dict(),
            flush=True,
        )
    print(
        "EXPERIMENT 005 COMPLETE; FULL132-ROW REPORT REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
