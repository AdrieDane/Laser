"""Experiment016: declared-standard mass CONTEXT, exact007 physical pools."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from context import REPO, WORKSPACE as ROOT

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
POOL = ROOT / "outputs/qe_optimization_20261022/experiment_007"
PREVIOUS = ROOT / "outputs/qe_optimization_20261022/experiment_009"
STD = ROOT / "outputs/qe_optimization_20261022/experiment_005"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
OUT = ROOT / "outputs/qe_optimization_20261022/experiment_016"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qe_dual_experiment import model_matrix


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
        for dataset in projects:
            p = BASE / "runs" / dataset / "full_annotation.pkl"
            dependencies[str(p)] = sha(p)
            for fold in range(5):
                p = POOL / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
                dependencies[str(p)] = sha(p)
        for fold in range(5):
            for p in (
                STD / "models" / f"fold_{fold}_standard_models.json",
                PREVIOUS / "calibration" / f"fold_{fold}" / "operating_points.json",
            ):
                dependencies[str(p)] = sha(p)
        for p in (
            POOL / "protocol.json",
            PREVIOUS / "protocol.json",
            ATLAS / "fold_assignment.json",
        ):
            dependencies[str(p)] = sha(p)
        previous = json.loads((PREVIOUS / "protocol.json").read_text())
        source = json.loads((POOL / "protocol.json").read_text())
        protocol = dict(
            name="016-validated-declared-standard-mass-context",
            baseline=str(BASE),
            previous=str(PREVIOUS),
            folds=5,
            production_enabled=False,
            curation_runtime_input=False,
            candidate_universe="EXACT007 physical candidate/evidence rows plus8 mass-context descriptors. No changed RT/mass/CDB admissions, original m/z/RT/feature IDs or candidate rows.",
            new_evidence="BASELINE automatic standard calls ONLY, validated name/adduct/reference mass within.05ppm against actual positive-quantity declared labelled chemical spike catalogue; observed standard<3ppm. Repeated features/adducts of one neutral chemical mass vote once. Median shift requires>=6 distinct neutral masses,MAD<=.6ppm,p90<=1.5ppm,abs center<=2ppm. Query excludes own feature AND all neutral groups with a reference channel within3ppm of query mass. Raw ppm remains; add center,MAD,p90,unique count,enabled,centered signed/absolute/squared ppm. Unknown staysNaN. No005/009 other-outer-fold standard predictions feed training mass evidence.",
            classifier=source["classifier"],
            ambiguity_margin=0.05,
            threshold_policy="EXACT009 frozen per-fold/per-stream thresholds, not recalibrated for the new evidence. Isolated evidence test. Threshold/model-score compatibility must be assessed; no evaluated labels choose new cutoffs.",
            standard_policy="EXACT005 standard adjudication/models with current outer fold exclusion, same as009. Mass CONTEXT does not change standard completion or recalibrate measured masses.",
            validation=previous["validation"],
            model_fit_note=previous["model_fit_note"]
            + " New mass-context classifier also excludes each evaluated project group. Automatic baseline has known prior exposure. No analytical FDR/probability-calibration claim.",
            run_kind="Saved007 physical proposal replay, refit same180-round global HGB plus8 declared-standard mass-context features; frozen009 cutoffs and005 standard handling. Not a new full production/raw-input run.",
            source_snapshot_sha256=hashes,
            dependencies_sha256=dependencies,
        )
        dump(OUT / "protocol.json", protocol)
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        shutil.copy2(
            ROOT / "work/run_qe_dual_experiment.py", OUT / "dual_runner_dependency.py"
        )
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
    assert (OUT / "dual_runner_dependency.py").read_bytes() == (
        ROOT / "work/run_qe_dual_experiment.py"
    ).read_bytes()
    for name in (
        "standard_completion.py",
        "standard_rt_library.py",
        "standard_rt_transfer.py",
        "declared_standards.py",
    ):
        p = Path("source_snapshot/app/matlab_annotation") / name
        assert (OUT / p).read_bytes() == (STD / p).read_bytes(), name
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    return protocol, projects


def prepare(projects):
    from matlab_annotation.standard_mass_context import (
        validated_mass_anchors,
        standard_mass_evidence,
        summarize_mass_anchors,
    )
    from matlab_annotation.declared_standards import declared_standard_targets

    keys = ["stream", "ref_mz", "ppm_error", "feat_feature_id"]
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
        catalogue = declared_standard_targets(
            OUT / "source_snapshot/reference_data/matlab_pipeline",
            result["internal_standards_expected"],
        )
        automatic = pd.concat(
            [p["annotations"].assign(stream=s) for s, p in result["streams"].items()],
            ignore_index=True,
        )
        anchors = validated_mass_anchors(automatic, catalogue)
        queries = (
            pd.concat([p["pool"][keys] for p in existing.values()], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        extra = standard_mass_evidence(queries, anchors)
        extra.index = pd.MultiIndex.from_frame(queries[keys])
        assert extra.index.is_unique
        folder = OUT / "mass_context_audits"
        folder.mkdir(exist_ok=True)
        anchors.to_csv(folder / (dataset + "_private_anchors.csv"), index=False)
        # JSON explicitly preserves missing calibration as null, not zero.
        summaries = summarize_mass_anchors(anchors)
        for values in summaries.values():
            for k, v in values.items():
                if isinstance(v, float) and not np.isfinite(v):
                    values[k] = None
        dump(
            folder / (dataset + ".json"),
            dict(
                queries=len(queries),
                enabled_queries=int(extra.q_standard_mass_enabled.sum()),
                unique_chemical_summary=summaries,
                source="Blind baseline automatic standards validated against actual declared positive-quantity spikes; no curated input",
            ),
        )
        for fold, data in existing.items():
            dest = OUT / "prepared" / f"fold_{fold}"
            dest.mkdir(parents=True, exist_ok=True)
            pool, evidence = data["pool"].copy(), data["evidence"].copy()
            aligned = extra.reindex(pd.MultiIndex.from_frame(pool[keys]))
            for col in extra:
                assert col not in pool and col not in evidence
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
        print(
            "PREPARED MASS CONTEXT",
            dataset,
            len(queries),
            "queries",
            int(extra.q_standard_mass_enabled.sum()),
            "enabled",
            flush=True,
        )


def main():
    protocol, projects = initialize()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards

    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    prepare(projects)
    for fold in range(5):
        path = OUT / f"fold_{fold}_nonlinear_model.pkl"
        heldout = [d for d in projects if folds[d] == fold]
        if not path.exists():
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
            print("FIT MASS CONTEXT START", fold, len(x), "rows", flush=True)
            estimator.fit(model_matrix(x, model), np.asarray(ys), sample_weight=weights)
            model["estimator"] = estimator
            pd.to_pickle(model, path)
            print("FIT MASS CONTEXT COMPLETE", fold, flush=True)
            del xs, x, ys, groups, weights
        model = pd.read_pickle(path)
        std_model = json.loads(
            (STD / "models" / f"fold_{fold}_standard_models.json").read_text()
        )
        cutoff = json.loads(
            (
                PREVIOUS / "calibration" / f"fold_{fold}" / "operating_points.json"
            ).read_text()
        )
        thresholds = cutoff["thresholds"]
        assert not (
            set(model["training_cohorts"])
            | set(std_model["training_groups"])
            | set(cutoff["training_groups"])
        ) & {d[:7] for d in heldout}
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
            catalogue = declared_standard_targets(
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
                selected = select_operating_candidates(
                    pool,
                    scores.loc[pool.index],
                    threshold=thresholds[stream],
                    ambiguity_margin=0.05,
                )
                selected = selected.drop(
                    columns=["training_positive", "training_label_known"]
                )
                selected["auto_annotation_status"] = (
                    "declared_standard_mass_context_experimental"
                )
                selected["operating_threshold"] = thresholds[stream]
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
                    catalogue[catalogue.stream.eq(stream)],
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
                    thresholds=thresholds,
                    counts=annotations.groupby("stream").size().to_dict(),
                ),
            )
            print(
                "SELECTED MASS CONTEXT HELD-OUT", dataset, len(annotations), flush=True
            )
    print(
        "EXPERIMENT016 COMPLETE; FULL132-ROW REPORT REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
