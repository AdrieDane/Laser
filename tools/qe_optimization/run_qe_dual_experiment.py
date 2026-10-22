"""Experiment 002: fixed dual-RT proposals and nonlinear evidence selection.

Reuses immutable original feature tables and the same grouped RT atlases.
Every outer held-out project group is excluded from supervised model fitting.
No threshold sweeps or production default changes are performed.
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
PREVIOUS = ROOT / "outputs/qe_optimization_20261022/experiment_001"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_002"
sys.path.insert(0, str(Path(__file__).resolve().parent))
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
        dump(
            OUT / "protocol.json",
            dict(
                name="002-dual-RT-nonlinear-evidence",
                baseline=str(BASE),
                previous=str(PREVIOUS),
                folds=5,
                threshold=0.5,
                ambiguity_margin=0.05,
                ppm=3,
                rt_seconds=10,
                composition_limits="unchanged default limits",
                classifier=dict(
                    kind="HistGradientBoostingClassifier",
                    max_iter=180,
                    max_leaf_nodes=31,
                    min_samples_leaf=50,
                    l2_regularization=2,
                    learning_rate=0.08,
                    early_stopping=False,
                    random_state=20261022,
                ),
                validation="Same five project-group folds as 001. Each evaluated group is excluded from RT, composition-prior and classifier training. Development has inspected example diagnostics; NOT untouched external validation.",
                model_fit_note="Outer-training RT atlases also generate outer-training features, not nested inner cross-fit features. Conditional grouped evaluation, not externally calibrated probabilities.",
                run_kind="Saved raw-feature-table replay; union of strict fitted-RT and atlas-RT proposals, nonlinear classifier, fixed threshold and margin.",
                curation_runtime_input=False,
                production_enabled=False,
                source_snapshot_sha256=hashes,
            ),
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        manifest = json.loads((BASE / "run_manifest.json").read_text())
        protocol = json.loads((OUT / "protocol.json").read_text())
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


def prepare(projects, atlases):
    from matlab_annotation.dual_candidates import combine_rt_sources
    from matlab_annotation.class_candidates import (
        candidate_evidence,
        attach_composition_prior_evidence,
    )

    priors = {
        fold: json.loads((PREVIOUS / f"fold_{fold}_selection_model.json").read_text())[
            "composition_priors"
        ]
        for fold in range(5)
    }
    for dataset in projects:
        if all(
            (OUT / "prepared" / f"fold_{fold}" / (dataset + ".pkl")).exists()
            for fold in range(5)
        ):
            continue
        print("PREPARE DUAL RT", dataset, flush=True)
        result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        exact, unresolved = manual_keys(dataset)
        for fold in range(5):
            dest = OUT / "prepared" / f"fold_{fold}"
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / (dataset + ".pkl")
            if path.exists():
                continue
            old = pd.read_pickle(
                PREVIOUS / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
            )
            pools = []
            evidence = []
            for stream, payload in result["streams"].items():
                learned = old["pool"][old["pool"].stream.eq(stream)]
                pool = combine_rt_sources(
                    learned,
                    payload["all_pairs"],
                    payload["annotations"],
                    atlases[fold],
                    stream=stream,
                )
                if pool.empty:
                    continue
                features = payload["features"]
                sample_columns = [
                    c for c in features.attrs.get("sample_columns", []) if c in features
                ]
                chosen = features.set_index(features.feature_id.astype(str)).loc[
                    pool.feat_feature_id.astype(str)
                ]
                if sample_columns:
                    matrix = (
                        chosen[sample_columns]
                        .apply(pd.to_numeric, errors="coerce")
                        .to_numpy(float)
                    )
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
                # Exported peak-group spreads, NOT chromatographic EIC peak widths.
                for source1, source2, label, factor in [
                    ("rtmax", "rtmin", "q_peakgroup_RT_spread_seconds", 60.0),
                    ("mzmax", "mzmin", "q_peakgroup_mass_spread_ppm", 1e6),
                ]:
                    if source1 in chosen and source2 in chosen:
                        spread = (
                            pd.to_numeric(chosen[source1], errors="coerce")
                            - pd.to_numeric(chosen[source2], errors="coerce")
                        ).to_numpy(float) * factor
                        if "mass" in label:
                            spread /= pool.feat_mz.to_numpy(float)
                        pool[label] = spread
                e = attach_composition_prior_evidence(
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
                } & set(e)
                pools.append(pool)
                evidence.append(e)
            pd.to_pickle(
                dict(
                    pool=pd.concat(pools, ignore_index=True),
                    evidence=pd.concat(evidence, ignore_index=True),
                ),
                path,
            )
        print("PREPARED DUAL ALL FOLDS", dataset, flush=True)


def model_matrix(evidence, model):
    columns = model["columns"]
    arrays = []
    for c in columns:
        if c in model["categories"]:
            arrays.append(
                evidence[c].astype(str).map(model["categories"][c]).to_numpy(float)
            )
        else:
            arrays.append(pd.to_numeric(evidence[c], errors="coerce").to_numpy(float))
    return np.column_stack(arrays)


def fit_select(protocol, projects, folds):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from matlab_annotation.class_candidates import select_class_candidates

    for fold in range(5):
        path = OUT / f"fold_{fold}_nonlinear_model.pkl"
        heldout = [d for d in projects if folds[d] == fold]
        if not path.exists():
            xs = []
            ys = []
            cohorts = []
            for dataset in projects:
                if folds[dataset] == fold:
                    continue
                data = pd.read_pickle(
                    OUT / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
                )
                known = data["pool"].training_label_known.astype(bool)
                xs.append(data["evidence"].loc[known])
                ys.extend(data["pool"].loc[known, "training_positive"].astype(int))
                cohorts.extend([dataset[:7]] * int(known.sum()))
            x = pd.concat(xs, ignore_index=True)
            categories = {
                c: {v: i for i, v in enumerate(sorted(x[c].astype(str).unique()))}
                for c in x
                if not pd.api.types.is_numeric_dtype(x[c])
            }
            assert all(len(v) <= 255 for v in categories.values()), (
                "Categorical cardinality exceeds fixed model support"
            )
            model = dict(
                columns=list(x),
                categories=categories,
                training_cohorts=sorted(set(cohorts)),
                held_out_datasets=heldout,
                fold=fold,
                training_pairs=len(x),
                positives=int(sum(ys)),
            )
            counts = Counter(cohorts)
            weights = np.asarray(
                [len(cohorts) / len(counts) / counts[c] for c in cohorts]
            )
            params = {k: v for k, v in protocol["classifier"].items() if k != "kind"}
            classifier = HistGradientBoostingClassifier(
                **params, categorical_features=[c in categories for c in x]
            )
            classifier.fit(
                model_matrix(x, model), np.asarray(ys), sample_weight=weights
            )
            model["estimator"] = classifier
            pd.to_pickle(model, path)
            print(
                "FIT NONLINEAR FOLD",
                fold,
                len(x),
                "pairs",
                model["positives"],
                "positive",
                flush=True,
            )
        model = pd.read_pickle(path)
        for dataset in heldout:
            dest = OUT / "runs" / dataset
            if (dest / "complete.json").exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            data = pd.read_pickle(
                OUT / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
            )
            raw = data["pool"]
            scores = pd.Series(
                model["estimator"].predict_proba(model_matrix(data["evidence"], model))[
                    :, 1
                ],
                index=raw.index,
            )
            parts = []
            baseline = pd.read_csv(
                BASE / "runs" / dataset / "automatic_annotations.csv",
                low_memory=False,
                dtype={"feat_feature_id": str},
            )
            for stream, pool in raw.groupby("stream"):
                selected = select_class_candidates(
                    pool,
                    scores.loc[pool.index],
                    threshold=protocol["threshold"],
                    ambiguity_margin=protocol["ambiguity_margin"],
                )
                selected = selected.drop(
                    columns=["training_positive", "training_label_known"]
                )
                selected["auto_annotation_status"] = "dual_RT_nonlinear_selected"
                protected = baseline[
                    baseline.stream.eq(stream)
                    & baseline.is_is.fillna(False).astype(bool)
                ]
                combined = pd.concat([selected, protected], ignore_index=True)
                assert not combined.feat_feature_id.astype(str).duplicated().any(), (
                    dataset,
                    stream,
                )
                parts.append(combined)
            selected = pd.concat(parts, ignore_index=True)
            selected.to_csv(dest / "automatic_annotations.csv", index=False)
            raw.drop(columns=["training_positive", "training_label_known"]).to_pickle(
                dest / "candidates.pkl"
            )
            dump(
                dest / "complete.json",
                dict(
                    dataset=dataset,
                    fold=fold,
                    status="complete",
                    counts=selected.groupby("stream").size().to_dict(),
                    production_enabled=False,
                ),
            )
            print("SELECTED DUAL HELD-OUT", dataset, len(selected), flush=True)


def main():
    protocol = initialize()
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    atlases = {
        fold: json.loads((ATLAS / f"fold_{fold}_atlas.json").read_text())["atlas"]
        for fold in range(5)
    }
    prepare(projects, atlases)
    fit_select(protocol, projects, folds)
    print(
        "EXPERIMENT 002 COMPLETE; REPORT ALL 132 ROWS BEFORE FURTHER EXPERIMENTS",
        flush=True,
    )


if __name__ == "__main__":
    main()
