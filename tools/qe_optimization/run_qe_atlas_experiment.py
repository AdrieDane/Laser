"""Experiment 001: grouped cross-fitted class RT atlas and evidence selection.

Finite local experiment, not a scheduled monitor. No production hook is enabled.
Frozen baseline feature tables are replayed; raw source datasets are never edited.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import pickle
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd

from context import WORKSPACE as ROOT, REPO

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_001"


def species(value):
    text = " ".join(str(value).split()).removeprefix("IS: ").removesuffix(" (IS)")
    return re.sub(r"\b([12]-acyl)\s+(LPC|LPE)\b", r"\1\2", text)


def adduct(value):
    text = re.sub(r"\]\((\d+)([+-])\)$", r"]\1\2", str(value).strip())
    return {"[M+HCOO]-": "[M+CHO2]-"}.get(text, text)


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
            name="001-class-atlas-evidence",
            baseline=str(BASE),
            folds=5,
            threshold=0.65,
            ambiguity_margin=0.05,
            C=1.0,
            classifier="cohort-balanced L2 logistic regression, liblinear, max_iter=500",
            ppm=3,
            rt_seconds=10,
            composition_limits="unchanged default limits",
            curation_runtime_input=False,
            production_enabled=False,
            validation="Grouped held-out model fits. Algorithm development has inspected example-data RT diagnostics; this is NOT untouched external validation.",
            interpretation="Score estimates agreement with manual curation, not analytical identification probability or FDR.",
            run_kind="Saved baseline-output replay with independent RT-target-to-original-feature discovery in frozen raw feature tables.",
            model_fit_note="Each outer fold uses only its training-group RT atlas and composition priors for both training and held-out feature generation. The held-out manual labels never enter any model fit.",
            source_snapshot_sha256=hashes,
        )
        dump(OUT / "protocol.json", protocol)
    protocol = json.loads((OUT / "protocol.json").read_text())
    if not (OUT / "run_manifest.json").exists():
        baseline_manifest = json.loads((BASE / "run_manifest.json").read_text())
        dump(
            OUT / "run_manifest.json",
            dict(
                datasets=baseline_manifest["datasets"],
                baseline=str(BASE),
                fresh_raw_input_runs=False,
                evaluation_note=protocol["run_kind"] + " " + protocol["validation"],
            ),
        )
    snapshot = OUT / "source_snapshot"
    for name, expected in protocol["source_snapshot_sha256"].items():
        assert hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == expected, (
            name
        )
    sys.path.insert(0, str(snapshot / "app"))
    return protocol


def manual_keys(dataset):
    # Frozen independent crosswalk, not provided to candidate discovery/scoring.
    path = ROOT / "outputs/qe_dataset_audit_20261022/checkpoints" / (dataset + ".json")
    proof = json.loads(path.read_text())["proofs"]
    exact = set()
    unresolved = set()
    for p in proof:
        standard = p["name"].endswith(" (IS)") or p["name"].startswith("IS: ")
        identity = (p["stream"], species(p["name"]), adduct(p["adduct"]), standard)
        if p["matched_original_feature"]:
            exact.add((*identity, str(p["original_feature_id"])))
        else:
            unresolved.add(identity)
    return exact, unresolved


def prepare_dataset(dataset, priors, atlases):
    from matlab_annotation.class_candidates import (
        prepare_class_candidates,
        candidate_evidence,
        attach_composition_prior_evidence,
    )

    if all(
        (OUT / "prepared" / f"fold_{fold}" / (dataset + ".pkl")).exists()
        for fold in range(5)
    ):
        return
    print("PREPARE", dataset, flush=True)
    with (BASE / "runs" / dataset / "full_annotation.pkl").open("rb") as handle:
        result = pickle.load(
            handle
        )  # Trusted local file generated by our baseline runner.
    exact, unresolved = manual_keys(dataset)
    for fold in range(5):
        dest = OUT / "prepared" / f"fold_{fold}"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / (dataset + ".pkl")
        if path.exists():
            continue
        all_pool = []
        all_evidence = []
        audits = []
        for stream, payload in result["streams"].items():
            pool, audit = prepare_class_candidates(
                payload["annotations"],
                payload["reference"],
                payload["features"],
                stream=stream,
                atlas=atlases[fold],
            )
            audits.append(audit)
            if pool.empty:
                continue
            # Descriptive, within-stream sample statistics. No blank/biological
            # role or cross-mode specimen pairing is inferred from filenames.
            features = payload["features"]
            sample_columns = [
                c for c in features.attrs.get("sample_columns", []) if c in features
            ]
            if sample_columns:
                chosen = features.set_index(features.feature_id.astype(str)).loc[
                    pool.feat_feature_id.astype(str)
                ]
                matrix = (
                    chosen[sample_columns]
                    .apply(pd.to_numeric, errors="coerce")
                    .to_numpy(float)
                )
                valid = np.isfinite(matrix) & (matrix >= 0)
                count = valid.sum(axis=1)
                fraction = np.divide(
                    ((matrix > 0) & valid).sum(axis=1),
                    count,
                    out=np.full(len(pool), np.nan),
                    where=count > 0,
                )
                pool["q_observed_detection_fraction"] = fraction
                pool["q_log_observed_median"] = np.log1p(
                    np.nanmedian(np.where(valid, matrix, np.nan), axis=1)
                )
            evidence = candidate_evidence(pool)
            evidence = attach_composition_prior_evidence(evidence, priors[fold])
            keys = [
                (stream, species(n), adduct(a), False, str(fid))
                for n, a, fid in pool[["name", "adduct", "feat_feature_id"]].itertuples(
                    index=False, name=None
                )
            ]
            pool["training_positive"] = [key in exact for key in keys]
            pool["training_label_known"] = [
                key[:-1] not in unresolved or key in exact for key in keys
            ]
            # These two supervised-label columns are explicitly NOT in evidence.
            assert not {
                "training_positive",
                "training_label_known",
                "dataset",
                "cohort",
                "feat_feature_id",
                "name",
            } & set(evidence)
            all_pool.append(pool)
            all_evidence.append(evidence)
        pool = pd.concat(all_pool, ignore_index=True) if all_pool else pd.DataFrame()
        evidence = (
            pd.concat(all_evidence, ignore_index=True)
            if all_evidence
            else pd.DataFrame()
        )
        pd.to_pickle(
            dict(
                pool=pool,
                evidence=evidence,
                audits=audits,
                dataset=dataset,
                cohort=dataset[:7],
            ),
            path,
        )
    print("PREPARED ALL FOLDS", dataset, flush=True)


def records_for_model(evidence, imputation=None):
    numeric = [c for c in evidence if pd.api.types.is_numeric_dtype(evidence[c])]
    if imputation is None:
        imputation = {
            c: float(evidence[c].median()) if evidence[c].notna().any() else 0.0
            for c in numeric
        }
    clean = evidence.copy()
    for c in numeric:
        clean[c + "_missing"] = clean[c].isna().astype(float)
        clean[c] = clean[c].fillna(imputation.get(c, 0.0)).astype(float)
    return clean.to_dict("records"), imputation


def fit_and_select(protocol, projects, folds, priors):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.linear_model import LogisticRegression
    from matlab_annotation.class_candidates import (
        score_linear_evidence,
        select_class_candidates,
    )

    for fold in range(5):
        model_path = OUT / f"fold_{fold}_selection_model.json"
        heldout = [d for d in projects if folds[d] == fold]
        if not model_path.exists():
            evidence = []
            labels = []
            cohorts = []
            for dataset in projects:
                if folds[dataset] == fold:
                    continue
                payload = pd.read_pickle(
                    OUT / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
                )
                known = payload["pool"].training_label_known.astype(bool)
                evidence.append(payload["evidence"].loc[known])
                labels.extend(
                    payload["pool"].loc[known, "training_positive"].astype(int)
                )
                cohorts.extend([dataset[:7]] * int(known.sum()))
            x = pd.concat(evidence, ignore_index=True)
            labels = np.asarray(labels, dtype=int)
            records, imputation = records_for_model(x)
            vectorizer = DictVectorizer(sparse=True)
            matrix = vectorizer.fit_transform(records)
            scaler = MaxAbsScaler()
            scaled = scaler.fit_transform(matrix)
            counts = Counter(cohorts)
            sample_weight = np.asarray(
                [len(cohorts) / len(counts) / counts[c] for c in cohorts]
            )
            estimator = LogisticRegression(
                C=protocol["C"], solver="liblinear", max_iter=500, tol=1e-5
            )
            estimator.fit(scaled, labels, sample_weight=sample_weight)
            coefficients = estimator.coef_[0] / scaler.scale_
            model = dict(
                format="laser-class-evidence-logistic-v1",
                intercept=float(estimator.intercept_[0]),
                coefficients=dict(
                    zip(vectorizer.get_feature_names_out(), coefficients.tolist())
                ),
                imputation=imputation,
                composition_priors=priors[fold],
                training_cohorts=sorted(counts),
                held_out_datasets=heldout,
                fold=fold,
                training_pairs=len(x),
                training_positives=int(labels.sum()),
                threshold=protocol["threshold"],
                ambiguity_margin=protocol["ambiguity_margin"],
            )
            probe = x.iloc[: min(1000, len(x))]
            records, _ = records_for_model(probe, imputation)
            expected = estimator.predict_proba(
                scaler.transform(vectorizer.transform(records))
            )[:, 1]
            actual = score_linear_evidence(probe, model).to_numpy()
            assert np.max(np.abs(expected - actual)) < 1e-9, (
                "Portable model parity failed"
            )
            dump(model_path, model)
            print(
                "FIT SELECTION FOLD",
                fold,
                "pairs",
                len(x),
                "positives",
                int(labels.sum()),
                flush=True,
            )
        model = json.loads(model_path.read_text())
        for dataset in heldout:
            dest = OUT / "runs" / dataset
            if (dest / "complete.json").exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            payload = pd.read_pickle(
                OUT / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
            )
            selected = []
            for stream, group in payload["pool"].groupby("stream"):
                scores = score_linear_evidence(
                    payload["evidence"].loc[group.index], model
                )
                chosen = select_class_candidates(
                    group,
                    scores,
                    threshold=protocol["threshold"],
                    ambiguity_margin=protocol["ambiguity_margin"],
                )
                chosen = chosen.drop(
                    columns=["training_positive", "training_label_known"]
                )
                baseline = pd.read_csv(
                    BASE / "runs" / dataset / "automatic_annotations.csv",
                    low_memory=False,
                )
                protected = baseline[
                    baseline.stream.eq(stream)
                    & baseline.is_is.fillna(False).astype(bool)
                ]
                combined = pd.concat([chosen, protected], ignore_index=True)
                assert not combined.feat_feature_id.astype(str).duplicated().any(), (
                    dataset,
                    stream,
                )
                selected.append(combined)
            combined = pd.concat(selected, ignore_index=True)
            combined.to_csv(dest / "automatic_annotations.csv", index=False)
            payload["pool"].drop(
                columns=["training_positive", "training_label_known"]
            ).to_pickle(dest / "candidates.pkl")
            dump(
                dest / "complete.json",
                dict(
                    dataset=dataset,
                    fold=fold,
                    status="complete",
                    counts=combined.groupby("stream").size().to_dict(),
                    run_kind=protocol["run_kind"],
                    production_enabled=False,
                ),
            )
            print("SELECTED HELD-OUT", dataset, len(combined), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-baseline", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    protocol = initialize()
    from matlab_annotation.class_candidates import fit_class_composition_priors

    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    observations = pd.read_pickle(ATLAS / "observations.pkl")
    atlases = {
        fold: json.loads((ATLAS / f"fold_{fold}_atlas.json").read_text())["atlas"]
        for fold in range(5)
    }
    priors = {
        fold: fit_class_composition_priors(
            observations[observations.dataset.map(folds).ne(fold) & observations.mapped]
        )
        for fold in range(5)
    }
    while True:
        complete = 0
        for dataset in projects:
            if not (BASE / "runs" / dataset / "complete.json").exists():
                continue
            prepare_dataset(dataset, priors, atlases)
            complete += 1
        dump(
            OUT / "preparation_progress.json",
            dict(
                prepared=complete,
                expected=len(projects),
                baseline_complete=(BASE / "run_summary.json").exists(),
                selection_started=False,
            ),
        )
        print("PREPARATION PROGRESS", complete, "/", len(projects), flush=True)
        if complete == len(projects) or not args.watch_baseline:
            break
        if (BASE / "run_summary.json").exists():
            raise RuntimeError(
                "Baseline ended without completing every dataset; inspect errors"
            )
        time.sleep(20)
    if complete != len(projects):
        print("Preparation incomplete; no annotation selections generated", flush=True)
        return
    if not args.prepare_only:
        fit_and_select(protocol, projects, folds, priors)
        print(
            "EXPERIMENT 001 SELECTIONS COMPLETE; FULL COMPARISON REPORT REQUIRED BEFORE ANOTHER EXPERIMENT",
            flush=True,
        )


if __name__ == "__main__":
    main()
