"""Experiment 004: raw ion/profile evidence and label-free run context.

UNCHANGED experiment003 candidate universe, classifier settings and thresholds.
No old rejected PE selector is activated; only read-only evidence is reused.
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
PREVIOUS = ROOT / "outputs/qe_optimization_20261022/experiment_003"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_004"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_qe_dual_experiment as dual


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
        protocol = {
            **previous,
            "name": "004-raw-ion-and-run-context-evidence",
            "previous": str(PREVIOUS),
            "source_snapshot_sha256": hashes,
            "candidate_universe": "Identical to experiment003 in every fold; no extra candidate admissions, RT changes or wider mass windows.",
            "new_evidence": "Charge-aware raw isotope/ion neighbors within3ppm/2s and >=4 usable same-stream profiles; explicit/validated sample roles only. OTHER strong automatic class compositions, and response-normalized abundance from declared commensurate nM spikes. No filenames, matrix names or curator labels as predictors.",
            "old_PE_selector_enabled": False,
            "local_RT_surfaces_enabled": False,
            "run_kind": "Saved raw-feature-table replay; same experiment003 proposals and SAME nonlinear model settings/threshold/margin; new label-free scoring evidence only.",
            "baseline": str(BASE),
            "production_enabled": False,
            "validation": previous["validation"]
            + " New model group exclusion does not erase prior exposure in inherited baseline models.",
        }
        dump(OUT / "protocol.json", protocol)
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        shutil.copy2(
            Path(__file__).with_name("run_qe_dual_experiment.py"),
            OUT / "dual_runner_dependency.py",
        )
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
    assert (OUT / "dual_runner_dependency.py").read_bytes() == (
        Path(__file__).with_name("run_qe_dual_experiment.py")
    ).read_bytes()
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    return protocol


def prepare(projects):
    from annotator.peak_evidence import peak_evidence
    from matlab_annotation.chemical_targets import _charge
    from matlab_annotation.run_context_evidence import run_context_evidence
    from matlab_annotation.class_candidates import (
        candidate_evidence,
        attach_composition_prior_evidence,
    )

    priors = {
        f: json.loads(
            (
                ROOT
                / f"outputs/qe_optimization_20261022/experiment_001/fold_{f}_selection_model.json"
            ).read_text()
        )["composition_priors"]
        for f in range(5)
    }
    for dataset in projects:
        if all(
            (OUT / "prepared" / f"fold_{f}" / (dataset + ".pkl")).exists()
            for f in range(5)
        ):
            continue
        print("PREPARE RAW ION CONTEXT", dataset, flush=True)
        result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        existing = {
            f: pd.read_pickle(PREVIOUS / "prepared" / f"fold_{f}" / (dataset + ".pkl"))[
                "pool"
            ]
            for f in range(5)
        }
        rawfolder = OUT / "raw_evidence" / dataset
        rawfolder.mkdir(parents=True, exist_ok=True)
        quality = {}
        for stream, payload in result["streams"].items():
            queries = pd.concat(
                [
                    p[p.stream.eq(stream)][["feat_feature_id", "adduct"]]
                    for p in existing.values()
                ],
                ignore_index=True,
            )
            queries["charge"] = queries.adduct.map(_charge)
            for charge, group in queries.groupby("charge"):
                path = rawfolder / f"{stream}_z{charge}.pkl"
                if path.exists():
                    measured = pd.read_pickle(path)
                else:
                    measured = peak_evidence(
                        payload["features"],
                        group.feat_feature_id,
                        polarity=stream.split("_")[1],
                        charge=int(charge),
                    )
                    measured.to_pickle(path)
                measured = measured.set_index("feature_id")
                quality[(stream, int(charge))] = measured[
                    [c for c in measured if c.startswith("q_")]
                ]
        for fold in range(5):
            folder = OUT / "prepared" / f"fold_{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (dataset + ".pkl")
            if path.exists():
                continue
            parts = []
            evidences = []
            for stream, pool in existing[fold].groupby("stream", sort=False):
                pool = pool.copy()
                charges = pool.adduct.map(_charge)
                for charge, indices in charges.groupby(charges).groups.items():
                    table = quality[(stream, int(charge))]
                    aligned = table.loc[
                        pool.loc[indices, "feat_feature_id"].astype(str)
                    ]
                    for c in aligned:
                        pool.loc[indices, "q_raw_" + c.removeprefix("q_")] = aligned[
                            c
                        ].to_numpy()
                context = run_context_evidence(
                    pool, result["streams"][stream]["annotations"]
                )
                for c in context:
                    pool[c] = context[c]
                evidence = attach_composition_prior_evidence(
                    candidate_evidence(pool), priors[fold]
                )
                assert not {
                    "name",
                    "feat_feature_id",
                    "dataset",
                    "cohort",
                    "training_positive",
                    "training_label_known",
                } & set(evidence)
                original = existing[fold].loc[pool.index]
                pd.testing.assert_frame_equal(pool[original.columns], original)
                parts.append(pool)
                evidences.append(evidence)
            pd.to_pickle(
                dict(
                    pool=pd.concat(parts, ignore_index=True),
                    evidence=pd.concat(evidences, ignore_index=True),
                ),
                path,
            )
        print("PREPARED RAW CONTEXT ALL FOLDS", dataset, flush=True)


def main():
    protocol = initialize()
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    prepare(projects)
    dual.OUT = OUT
    dual.fit_select(protocol, projects, folds)
    print(
        "EXPERIMENT 004 COMPLETE; FULL132-ROW REPORT REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
