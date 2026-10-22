"""Experiment 003: group-trained chemical target coverage, same dual-RT scorer.

Fixed protocol. No production hook and no held-out threshold sweeps.
"""

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
PREVIOUS = ROOT / "outputs/qe_optimization_20261022/experiment_002"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_003"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_qe_dual_experiment as dual
from run_qe_atlas_experiment import manual_keys, species, adduct


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
        prior = json.loads((PREVIOUS / "protocol.json").read_text())
        protocol = {
            **prior,
            "name": "003-group-trained-chemical-target-coverage",
            "previous": str(PREVIOUS),
            "source_snapshot_sha256": hashes,
            "composition_limits": "Explicit experimental per-class/mode union of current effective defaults and OTHER-group manual C:DB extrema padded by 2 carbons and 1 double bond; >=3 training groups; DB <= carbon/2 for newly compiled targets.",
            "chemical_target_validation": "Charge-aware CH2/-H2 mass lattice, max residual 2e-5 Da; no non-C:DB extrapolation; identical-mass declared spikes protected, labelled-spike shadow repaired.",
            "run_kind": "Saved raw-feature-table replay, experiment002 strict dual-RT proposals plus newly compiled chemical targets; SAME nonlinear classifier settings, threshold and margin.",
            "baseline": str(BASE),
            "production_enabled": False,
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


def quality(pool, features):
    chosen = features.set_index(features.feature_id.astype(str)).loc[
        pool.feat_feature_id.astype(str)
    ]
    columns = [c for c in features.attrs.get("sample_columns", []) if c in features]
    if columns:
        matrix = chosen[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = np.isfinite(matrix) & (matrix >= 0)
        count = valid.sum(axis=1)
        pool["q_observed_detection_fraction"] = np.divide(
            ((matrix > 0) & valid).sum(axis=1),
            count,
            out=np.full(len(pool), np.nan),
            where=count > 0,
        )
        pool["q_log_observed_median"] = np.log1p(
            np.nanmedian(np.where(valid, matrix, np.nan), axis=1)
        )
    for a, b, label, factor in [
        ("rtmax", "rtmin", "q_peakgroup_RT_spread_seconds", 60.0),
        ("mzmax", "mzmin", "q_peakgroup_mass_spread_ppm", 1e6),
    ]:
        if a in chosen and b in chosen:
            span = (
                pd.to_numeric(chosen[a], errors="coerce")
                - pd.to_numeric(chosen[b], errors="coerce")
            ).to_numpy(float) * factor
            if "mass" in label:
                span /= pool.feat_mz.to_numpy(float)
            pool[label] = span
    return pool


def prepare(projects, folds, atlases, envelopes):
    from matlab_annotation.chemical_targets import compile_additional_targets
    from matlab_annotation.class_candidates import (
        prepare_class_candidates,
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
    audits = []
    for dataset in projects:
        if all(
            (OUT / "prepared" / f"fold_{f}" / (dataset + ".pkl")).exists()
            for f in range(5)
        ):
            continue
        print("PREPARE CHEMICAL TARGETS", dataset, flush=True)
        result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        exact, unresolved = manual_keys(dataset)
        for fold in range(5):
            folder = OUT / "prepared" / f"fold_{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (dataset + ".pkl")
            if path.exists():
                continue
            old = pd.read_pickle(
                PREVIOUS / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
            )["pool"]
            pools = []
            evidences = []
            audit_rows = []
            for stream, payload in result["streams"].items():
                targets, ranges, audit = compile_additional_targets(
                    payload["reference"], envelopes[fold], stream=stream
                )
                if len(audit):
                    audit_rows.extend(
                        audit.assign(dataset=dataset, fold=fold).to_dict("records")
                    )
                previous = (
                    old[old.stream.eq(stream)]
                    .drop(columns=["training_positive", "training_label_known"])
                    .copy()
                )
                previous["q_chemical_target_generated"] = 0.0
                extra = pd.DataFrame()
                if len(targets):
                    extra, discovery = prepare_class_candidates(
                        payload["annotations"],
                        targets,
                        payload["features"],
                        stream=stream,
                        atlas=atlases[fold],
                        ranges=ranges,
                    )
                    if len(extra):
                        extra = quality(extra, payload["features"])
                        extra["q_chemical_target_generated"] = 1.0
                        extra["q_atlas_source"] = 1.0
                        extra["q_fitted_source"] = 0.0
                        extra["q_previously_selected"] = 0.0
                        extra["q_atlas_RT_error_seconds"] = (
                            extra.feat_rt * 60 - extra.atlas_rt_seconds
                        )
                        extra["q_absolute_atlas_RT_error_seconds"] = (
                            extra.q_atlas_RT_error_seconds.abs()
                        )
                    audit_rows.append(
                        dict(
                            dataset=dataset,
                            fold=fold,
                            status="strict_additional_discovery",
                            **discovery,
                        )
                    )
                pool = (
                    pd.concat([previous, extra], ignore_index=True)
                    .drop_duplicates(["name", "adduct", "feat_feature_id"])
                    .reset_index(drop=True)
                )
                evidence = attach_composition_prior_evidence(
                    candidate_evidence(pool), priors[fold]
                )
                keys = [
                    (stream, species(n), adduct(a), False, str(fid))
                    for n, a, fid in pool[
                        ["name", "adduct", "feat_feature_id"]
                    ].itertuples(index=False, name=None)
                ]
                pool["training_positive"] = [key in exact for key in keys]
                pool["training_label_known"] = [
                    key[:-1] not in unresolved or key in exact for key in keys
                ]
                assert not {
                    "name",
                    "feat_feature_id",
                    "dataset",
                    "cohort",
                    "training_positive",
                    "training_label_known",
                } & set(evidence)
                pools.append(pool)
                evidences.append(evidence)
            pd.to_pickle(
                dict(
                    pool=pd.concat(pools, ignore_index=True),
                    evidence=pd.concat(evidences, ignore_index=True),
                    audit=audit_rows,
                ),
                path,
            )
            audits.extend(audit_rows)
        print("PREPARED CHEMICAL ALL FOLDS", dataset, flush=True)
    # Read audit from all checkpoints so resumed runs do not omit earlier rows.
    audits = []
    for fold in range(5):
        for dataset in projects:
            audits.extend(
                pd.read_pickle(OUT / "prepared" / f"fold_{fold}" / (dataset + ".pkl"))[
                    "audit"
                ]
            )
    pd.DataFrame(audits).to_csv(OUT / "chemical_target_audit.csv", index=False)


def main():
    protocol = initialize()
    from matlab_annotation.chemical_targets import fit_composition_envelopes

    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    atlases = {
        f: json.loads((ATLAS / f"fold_{f}_atlas.json").read_text())["atlas"]
        for f in range(5)
    }
    observations = pd.read_pickle(ATLAS / "observations.pkl")
    envelopes = {}
    for fold in range(5):
        train = observations[
            observations.dataset.map(folds).ne(fold) & observations.mapped
        ]
        train_groups = set(train.cohort)
        test_groups = {d[:7] for d in projects if folds[d] == fold}
        assert not train_groups & test_groups
        envelopes[fold] = fit_composition_envelopes(train)
        dump(
            OUT / f"fold_{fold}_composition_envelope.json",
            dict(
                envelope=envelopes[fold],
                training_groups=sorted(train_groups),
                held_out_groups=sorted(test_groups),
            ),
        )
    prepare(projects, folds, atlases, envelopes)
    dual.OUT = OUT
    dual.fit_select(protocol, projects, folds)
    print(
        "EXPERIMENT 003 COMPLETE; FULL132-ROW REPORT REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
