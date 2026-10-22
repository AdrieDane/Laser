"""Experiment007: cross-mode automatic evidence, same004 pools/005 standards.

Experiment006 specialists were rejected for lower overall correct coverage.
This returns to005's global-classifier architecture, adds independent evidence
from OTHER modes and refits the SAME fixed HGB configuration.
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
POOL = ROOT / "outputs/qe_optimization_20261022/experiment_004"
PREVIOUS = ROOT / "outputs/qe_optimization_20261022/experiment_005"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_007"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qe_dual_experiment import model_matrix


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
        global_protocol = json.loads((POOL / "protocol.json").read_text())
        dump(
            OUT / "protocol.json",
            dict(
                name="007-cross-mode-independent-automatic-evidence",
                baseline=str(BASE),
                previous=str(PREVIOUS),
                rejected_preceding_experiment="006: specialist models reduced overall coverage; NOT used.",
                folds=5,
                threshold=0.5,
                ambiguity_margin=0.05,
                production_enabled=False,
                candidate_universe="EXACT004 candidate/evidence rows plus six independent OTHER-mode descriptors; no mass/RT/composition admissions.",
                new_evidence="Frozen baseline automatic OTHER-stream exact-class/C/DB matches (confidence>=.9, absppm<1.5, disagreeing adduct RTs withheld). Same-phase RT offsets from >=8 OTHER compositions only. Abundance correlations require >=4 matching validated biological GROUP labels, not guessed individual-sample pairs. No merging ether or positional classes; no NP/RP RT comparisons.",
                classifier=global_protocol["classifier"],
                validation=previous["validation"],
                model_fit_note=global_protocol["model_fit_note"],
                run_kind="Saved004 original-feature proposal replay; refit global HGB with SAME004 hyperparameters/threshold/margin plus OTHER-mode evidence, followed by EXACT005 standard adjudication/models.",
                standard_policy=previous["standard_policy"],
                curation_runtime_input=False,
                source_snapshot_sha256=hashes,
            ),
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        shutil.copy2(
            Path(__file__).with_name("run_qe_dual_experiment.py"),
            OUT / "dual_runner_dependency.py",
        )
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
    assert (OUT / "dual_runner_dependency.py").read_bytes() == (
        Path(__file__).with_name("run_qe_dual_experiment.py")
    ).read_bytes()
    for name in (
        "standard_completion.py",
        "standard_rt_library.py",
        "standard_rt_transfer.py",
        "declared_standards.py",
    ):
        rel = Path("source_snapshot/app/matlab_annotation") / name
        assert (OUT / rel).read_bytes() == (PREVIOUS / rel).read_bytes(), name
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    return protocol


def prepare(projects):
    from matlab_annotation.cross_mode_evidence import cross_mode_evidence

    keys = ["stream", "class", "carbon", "db", "feat_feature_id", "feat_rt"]
    for dataset in projects:
        if all(
            (OUT / "prepared" / f"fold_{f}" / (dataset + ".pkl")).exists()
            for f in range(5)
        ):
            continue
        existing = {
            f: pd.read_pickle(POOL / "prepared" / f"fold_{f}" / (dataset + ".pkl"))
            for f in range(5)
        }
        result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        queries = (
            pd.concat([p["pool"][keys] for p in existing.values()], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # Only raw queries and AUTOMATIC original baseline calls enter this helper.
        automatic = pd.concat(
            [p["annotations"].assign(stream=s) for s, p in result["streams"].items()],
            ignore_index=True,
        )
        extra = cross_mode_evidence(
            queries, {s: p["features"] for s, p in result["streams"].items()}, automatic
        )
        extra.index = pd.MultiIndex.from_frame(queries[keys])
        assert extra.index.is_unique
        folder = OUT / "evidence_audits"
        folder.mkdir(exist_ok=True)
        dump(
            folder / (dataset + ".json"),
            dict(
                queries=len(queries),
                nonmissing=extra.notna().sum().to_dict(),
                supported_queries=int(
                    extra.q_cross_mode_supporting_streams.gt(0).sum()
                ),
                note="Group-level profiles only, not individual-sample pairing; no annotation selections in preparation.",
            ),
        )
        for fold, data in existing.items():
            dest = OUT / "prepared" / f"fold_{fold}"
            dest.mkdir(parents=True, exist_ok=True)
            pool, evidence = data["pool"].copy(), data["evidence"].copy()
            aligned = extra.reindex(pd.MultiIndex.from_frame(pool[keys]))
            for col in extra:
                assert col not in evidence and col not in pool
                pool[col] = aligned[col].to_numpy()
                evidence[col] = aligned[col].to_numpy()
            pd.testing.assert_frame_equal(pool[data["pool"].columns], data["pool"])
            pd.testing.assert_frame_equal(
                evidence[data["evidence"].columns], data["evidence"]
            )
            assert not {
                "name",
                "feat_feature_id",
                "dataset",
                "cohort",
                "training_positive",
                "training_label_known",
            } & set(evidence)
            pd.to_pickle(dict(pool=pool, evidence=evidence), dest / (dataset + ".pkl"))
        print("PREPARED CROSS MODE", dataset, len(queries), flush=True)


def main():
    protocol = initialize()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from matlab_annotation.class_candidates import select_class_candidates
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards

    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    prepare(projects)
    for fold in range(5):
        model_path = OUT / f"fold_{fold}_nonlinear_model.pkl"
        heldout = [d for d in projects if folds[d] == fold]
        if not model_path.exists():
            xs, ys, groups = [], [], []
            for dataset in projects:
                if folds[dataset] == fold:
                    continue
                data = pd.read_pickle(
                    OUT / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
                )
                known = data["pool"].training_label_known.astype(bool)
                xs.append(data["evidence"].loc[known])
                ys.extend(data["pool"].loc[known, "training_positive"].astype(int))
                groups.extend([dataset[:7]] * int(known.sum()))
            assert not set(groups) & {d[:7] for d in heldout}
            x = pd.concat(xs, ignore_index=True)
            categories = {
                c: {v: i for i, v in enumerate(sorted(x[c].astype(str).unique()))}
                for c in x
                if not pd.api.types.is_numeric_dtype(x[c])
            }
            assert all(len(v) <= 255 for v in categories.values())
            model = dict(
                columns=list(x),
                categories=categories,
                training_cohorts=sorted(set(groups)),
                held_out_datasets=heldout,
                fold=fold,
                training_pairs=len(x),
                positives=int(sum(ys)),
            )
            counts = Counter(groups)
            weights = np.asarray(
                [len(groups) / len(counts) / counts[g] for g in groups]
            )
            params = {k: v for k, v in protocol["classifier"].items() if k != "kind"}
            estimator = HistGradientBoostingClassifier(
                **params, categorical_features=[c in categories for c in x]
            )
            estimator.fit(model_matrix(x, model), np.asarray(ys), sample_weight=weights)
            model["estimator"] = estimator
            pd.to_pickle(model, model_path)
            print("FIT CROSS MODE FOLD", fold, len(x), flush=True)
        model = pd.read_pickle(model_path)
        std_model = json.loads(
            (PREVIOUS / "models" / f"fold_{fold}_standard_models.json").read_text()
        )
        assert not set(std_model["training_groups"]) & {d[:7] for d in heldout}
        for dataset in heldout:
            dest = OUT / "runs" / dataset
            if (dest / "complete.json").exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            data = pd.read_pickle(
                OUT / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
            )
            raw, evidence = data["pool"], data["evidence"]
            scores = pd.Series(
                model["estimator"].predict_proba(model_matrix(evidence, model))[:, 1],
                index=raw.index,
            )
            result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
            catalog = declared_standard_targets(
                OUT / "source_snapshot/reference_data/matlab_pipeline",
                result["internal_standards_expected"],
            )
            parts, candidates, audits = (
                [],
                [raw.drop(columns=["training_positive", "training_label_known"])],
                [],
            )
            for stream, payload in result["streams"].items():
                pool = raw[raw.stream.eq(stream)]
                selected = select_class_candidates(
                    pool, scores.loc[pool.index], threshold=0.5, ambiguity_margin=0.05
                )
                selected = selected.drop(
                    columns=["training_positive", "training_label_known"]
                )
                selected["auto_annotation_status"] = "cross_mode_evidence_experimental"
                protected = (
                    payload["annotations"][
                        payload["annotations"].is_is.fillna(False).astype(bool)
                    ]
                    .copy()
                    .assign(stream=stream)
                )
                combined = pd.concat([selected, protected], ignore_index=True)
                assert not combined.feat_feature_id.astype(str).duplicated().any(), (
                    dataset,
                    stream,
                )
                corrected, extra, audit = complete_declared_standards(
                    combined,
                    payload["features"],
                    catalog[catalog.stream.eq(stream)],
                    std_model["library"],
                    std_model["atlas"],
                    std_model["transfer"],
                    std_model["channels"],
                    stream=stream,
                    alignment_annotations=payload["annotations"],
                )
                parts.append(corrected.assign(stream=stream))
                candidates.extend(
                    [
                        protected,
                        extra.assign(stream=stream),
                        corrected[corrected.is_is.fillna(False).astype(bool)].assign(
                            stream=stream
                        ),
                    ]
                )
                audits.append(audit.assign(dataset=dataset, stream=stream))
            annotations = pd.concat(parts, ignore_index=True)
            all_candidates = pd.concat(candidates, ignore_index=True)
            keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
            all_candidates = all_candidates.drop_duplicates(
                keys, keep="last"
            ).reset_index(drop=True)
            chosen = set(annotations[keys].itertuples(index=False, name=None))
            all_candidates["automatic_accepted"] = [
                k in chosen
                for k in all_candidates[keys].itertuples(index=False, name=None)
            ]
            annotations.to_csv(dest / "automatic_annotations.csv", index=False)
            all_candidates.to_pickle(dest / "candidates.pkl")
            scores.to_pickle(dest / "scoring_audit.pkl")
            pd.concat(audits, ignore_index=True).to_csv(
                dest / "standard_audit.csv", index=False
            )
            dump(
                dest / "complete.json",
                dict(
                    dataset=dataset,
                    fold=fold,
                    status="complete",
                    production_enabled=False,
                    counts=annotations.groupby("stream").size().to_dict(),
                ),
            )
            print("SELECTED CROSS MODE HELD-OUT", dataset, len(annotations), flush=True)
    print(
        "EXPERIMENT007 COMPLETE; FULL132-ROW REPORT REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
