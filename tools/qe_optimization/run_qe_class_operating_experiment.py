"""Experiment021: class-targeted TRAINING-only cutoffs, exact007 scores/pools."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from context import WORKSPACE as ROOT, REPO

QE = ROOT / "outputs/qe_optimization_20261022"
BASE = QE / "baseline"
PREVIOUS = QE / "experiment_007"
REFERENCE = QE / "experiment_009"
STD = QE / "experiment_005"
ATLAS = QE / "rt_atlas_development"
OUT = QE / "experiment_021"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qe_dual_experiment import model_matrix
from run_qe_atlas_experiment import species, adduct


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
                BASE / "evaluation_v2" / d / "manual_comparison.pkl",
            ):
                dependencies[str(p)] = sha(p)
            for f in range(5):
                p = PREVIOUS / "prepared" / f"fold_{f}" / (d + ".pkl")
                dependencies[str(p)] = sha(p)
        for f in range(5):
            for p in (
                PREVIOUS / f"fold_{f}_nonlinear_model.pkl",
                STD / "models" / f"fold_{f}_standard_models.json",
                REFERENCE / "calibration" / f"fold_{f}" / "operating_points.json",
                REFERENCE
                / "calibration"
                / f"fold_{f}"
                / "selected_training_points.csv",
            ):
                dependencies[str(p)] = sha(p)
            for i in range(3):
                p = (
                    QE
                    / "experiment_008/calibration"
                    / f"fold_{f}"
                    / f"inner_{i}_score_model.pkl"
                )
                dependencies[str(p)] = sha(p)
        for p in (
            ATLAS / "fold_assignment.json",
            REFERENCE / "protocol.json",
            Path(__file__).with_name("run_qe_dual_experiment.py"),
            Path(__file__).with_name("run_qe_atlas_experiment.py"),
        ):
            dependencies[str(p)] = sha(p)
        old = json.loads((REFERENCE / "protocol.json").read_text())
        protocol = dict(
            name="021-class-targeted-training-operating-points",
            baseline=str(BASE),
            previous=str(REFERENCE),
            folds=5,
            inner_folds=3,
            production_enabled=False,
            curation_runtime_input=False,
            candidate_universe="EXACT007 physical candidates/evidence/final models and EXACT005 standards. No new candidates, score refit, mass/RT/feature-coordinate changes or rejected020 additions.",
            threshold_grid=[round(v, 2) for v in np.arange(0.05, 1.001, 0.05)],
            target=0.9,
            extras_budget_per_endogenous_manual=0.095,
            minimum_cohorts=3,
            minimum_manual=30,
            solver_time_limit_per_stage_seconds=60,
            threshold_policy="One TRAINING-only cutoff per canonical class/mode. Class/mode floors=min(reference009 correct,ceil(.9manual)),total correct>=reference009. Pooled9.5% endogenous TRAINING extras budget. Maximize SUMper-classcoverage capped90%,thenpooledcorrect,thenfewerwrongpeaks/extras. Under-supported(<3cohorts/<30manual) frozen009reference;unknownclass fallback009modecutoff. MILPmustconfirmoptimality,otherwiseSTOPwithout silently relaxing guards. No held-out labels choose thresholds.",
            competition="Global feature/identity winners remain fixed BEFORE class cutoffs. Withholding top label CANNOT promote a lower label. Independent class counts additive ONLY under this invariant; uniform-cutoff and exact009 training-table parity checked.",
            validation=old["validation"],
            model_fit_note=old["model_fit_note"]
            + " SAME008 inner OOF007 score fits and007 final fits exclude evaluated outer project groups. New per-class training tables count ALLendogenousmanual/unresolved,extra-onlyclasses,uniqueprojectgroups. Inherited outerRT/prior/oldcutoffs not fully innercrossfitted; baselineknownprior exposure,not pristine external/end-to-end validation or analyticalFDR.",
            run_kind="Saved007 candidate/score replay with new grouped INNER-training class/mode acceptance allocation; SAME final scoremodels and005standards. Not a fresh production/raw-input run.",
            source_snapshot_sha256=hashes,
            dependencies_sha256=dependencies,
        )
        dump(OUT / "protocol.json", protocol)
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        for name in ("run_qe_dual_experiment.py", "run_qe_atlas_experiment.py"):
            shutil.copy2(Path(__file__).with_name(name), OUT / (name + ".dependency"))
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
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    for name in (
        "standard_completion.py",
        "standard_rt_library.py",
        "standard_rt_transfer.py",
        "declared_standards.py",
    ):
        p = Path("source_snapshot/app/matlab_annotation") / name
        assert (OUT / p).read_bytes() == (STD / p).read_bytes(), name
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    return protocol


def calibrate(protocol, projects, folds, fold):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.class_operating_point import choose_class_training_points

    folder = OUT / "calibration" / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "operating_points.json"
    train = [d for d in projects if folds[d] != fold]
    groups = {d[:7] for d in train}
    excluded = {d[:7] for d in projects if folds[d] == fold}
    fallback = json.loads(
        (
            REFERENCE / "calibration" / f"fold_{fold}" / "operating_points.json"
        ).read_text()
    )["thresholds"]
    if path.exists():
        saved = json.loads(path.read_text())
        assert (
            set(saved["training_groups"]) == groups
            and set(saved["excluded_groups"]) == excluded
        )
        assert saved["fallback_thresholds"] == fallback
        return saved
    table_path = folder / "training_class_operating_points.csv"
    if not table_path.exists():
        predictions = {}
        data = {
            d: pd.read_pickle(PREVIOUS / "prepared" / f"fold_{fold}" / (d + ".pkl"))
            for d in train
        }
        for inner in range(3):
            model = pd.read_pickle(
                QE
                / "experiment_008/calibration"
                / f"fold_{fold}"
                / f"inner_{inner}_score_model.pkl"
            )
            assert set(model["outer_excluded"]) == excluded
            assert not set(model["training_cohorts"]) & (
                set(model["calibration_groups"]) | excluded
            )
            for d in train:
                if d[:7] not in model["calibration_groups"]:
                    continue
                assert d not in predictions
                predictions[d] = pd.Series(
                    model["estimator"].predict_proba(
                        model_matrix(data[d]["evidence"], model)
                    )[:, 1],
                    index=data[d]["pool"].index,
                )
        assert set(predictions) == set(train)
        manuals = Counter()
        cohorts = defaultdict(set)
        all_classes = set()
        winners = []
        for d in train:
            # ONLY outer-TRAINING references read for acceptance optimization.
            manual = pd.read_pickle(
                BASE / "evaluation_v2" / d / "manual_comparison.pkl"
            )
            manual = manual[~manual.internal_standard].copy()
            manual["_class"] = manual.lipid_class.map(canonical_lipid_class)
            manual_classes = manual.groupby(["stream", "_class"]).size().to_dict()
            manuals.update(manual_classes)
            for key in manual_classes:
                cohorts[key].add(d[:7])
            all_classes.update(manual_classes)
            identities = set(
                manual[["stream", "species", "adduct"]].itertuples(
                    index=False, name=None
                )
            )
            pool = data[d]["pool"].copy()
            pool["_class"] = pool["class"].map(canonical_lipid_class)
            all_classes.update(
                pool[["stream", "_class"]].itertuples(index=False, name=None)
            )
            pool["calibration_identity_in_manual"] = [
                (s, species(n), adduct(a)) in identities
                for s, n, a in pool[["stream", "name", "adduct"]].itertuples(
                    index=False, name=None
                )
            ]
            for _stream, part in pool.groupby("stream", sort=False):
                selected = select_operating_candidates(
                    part,
                    predictions[d].loc[part.index],
                    threshold=0.05,
                    ambiguity_margin=0.05,
                )
                winners.append(selected)
        winner = pd.concat(winners, ignore_index=True)
        records = []
        for threshold in sorted(
            set(protocol["threshold_grid"]) | set(fallback.values())
        ):
            selected = winner[winner.class_selection_score.ge(threshold)].copy()
            selected["_correct"] = selected.training_positive.astype(int)
            selected["_extra"] = (~selected.calibration_identity_in_manual).astype(int)
            selected["_wrong"] = (
                selected.calibration_identity_in_manual
                & selected.training_label_known
                & ~selected.training_positive
            ).astype(int)
            counts = (
                selected.groupby(["stream", "_class"])[["_correct", "_extra", "_wrong"]]
                .sum()
                .to_dict("index")
            )
            for stream, cls in sorted(all_classes):
                c = counts.get((stream, cls), {})
                records.append(
                    dict(
                        stream=stream,
                        lipid_class=cls,
                        threshold=float(threshold),
                        manual=int(manuals[(stream, cls)]),
                        correct=int(c.get("_correct", 0)),
                        extras=int(c.get("_extra", 0)),
                        wrong_peak=int(c.get("_wrong", 0)),
                        training_cohorts=len(cohorts[(stream, cls)]),
                        reference=float(threshold) == fallback[stream],
                    )
                )
        table = pd.DataFrame(records)
        refs = table[table.reference]
        old = pd.read_csv(
            REFERENCE / "calibration" / f"fold_{fold}" / "selected_training_points.csv"
        )
        for r in old.itertuples():
            x = refs[refs.stream.eq(r.stream)]
            assert (
                int(x.manual.sum()),
                int(x.correct.sum()),
                int(x.extras.sum()),
                int(x.wrong_peak.sum()),
            ) == (
                int(r.manual),
                int(r.correct),
                int(r.extras),
                int(r.different_peak),
            ), (fold, r.stream, "Reference parity failed")
        table.to_csv(table_path, index=False)
        print("CLASS TRAINING-TABLE PARITY VERIFIED", fold, len(table), flush=True)
    table = pd.read_csv(table_path)
    learned = choose_class_training_points(
        table,
        training_groups=groups,
        excluded_groups=excluded,
        extras_per_manual=protocol["extras_budget_per_endogenous_manual"],
        target=protocol["target"],
        minimum_cohorts=protocol["minimum_cohorts"],
        minimum_manual=protocol["minimum_manual"],
        time_limit=protocol["solver_time_limit_per_stage_seconds"],
    )
    assert learned is not None, (
        fold,
        "No verified optimal class cutoff solution; do not relax safeguards",
    )
    learned["fallback_thresholds"] = fallback
    learned["training_table_sha256"] = sha(table_path)
    dump(path, learned)
    pd.DataFrame(learned["selected"]).to_csv(
        folder / "selected_class_training_points.csv", index=False
    )
    print(
        "CLASS CUTOFFS LEARNED",
        fold,
        "correct",
        learned["reference_correct"],
        "->",
        learned["correct"],
        "extras",
        learned["extras"],
        "budget",
        learned["pooled_training_extra_budget"],
        flush=True,
    )
    return learned


def main():
    protocol = initialize()
    from matlab_annotation.class_operating_point import (
        select_class_operating_candidates,
    )
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards

    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    for fold in range(5):
        thresholds = calibrate(protocol, projects, folds, fold)
        model = pd.read_pickle(PREVIOUS / f"fold_{fold}_nonlinear_model.pkl")
        std_model = json.loads(
            (STD / "models" / f"fold_{fold}_standard_models.json").read_text()
        )
        heldout = [d for d in projects if folds[d] == fold]
        assert not (
            set(model["training_cohorts"]) | set(std_model["training_groups"])
        ) & {d[:7] for d in heldout}
        for dataset in heldout:
            dest = OUT / "runs" / dataset
            if (dest / "complete.json").exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            data = pd.read_pickle(
                PREVIOUS / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
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
                selected = select_class_operating_candidates(
                    pool,
                    scores.loc[pool.index],
                    thresholds["thresholds"],
                    fallback_threshold=thresholds["fallback_thresholds"][stream],
                    ambiguity_margin=0.05,
                )
                selected = selected.drop(
                    columns=["training_positive", "training_label_known"]
                )
                selected["auto_annotation_status"] = "class_training_operating_point"
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
                    thresholds=thresholds,
                    counts=annotations.groupby("stream").size().to_dict(),
                ),
            )
            print(
                "SELECTED CLASS-CUTOFF HELD-OUT", dataset, len(annotations), flush=True
            )
    print(
        "EXPERIMENT021 COMPLETE; FULL132-ROW REPORT REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
