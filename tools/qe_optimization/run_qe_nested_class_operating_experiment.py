"""Experiment023: class cutoffs on genuine nested022 INNER OOF predictions.

Same022 final scores/candidates and005 standards. No refit, new candidates or
held-out curation at inference. Class allocations use outer-TRAINING labels
ONLY, after independently scored query-excluded calibration features exist.
"""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from context import REPO, WORKSPACE
from frozen_resources import configure_frozen_resources
from run_qe_atlas_experiment import species, adduct

QE = WORKSPACE / "outputs/qe_optimization_20261022"
BASE = QE / "baseline"
PREVIOUS = QE / "experiment_022"
PREPARED = QE / "nested_candidate_preparation_v2"
STD = QE / "experiment_005"
OUT = QE / "experiment_023"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def prepared_path(dataset, mask):
    return (
        PREPARED
        / "prepared"
        / ("exclude_" + "_".join(map(str, sorted(mask))))
        / (dataset + ".pkl")
    )


def initialize():
    ready = json.loads((PREVIOUS / "reports_v2/summary.json").read_text())
    assert ready["complete"] and ready["datasets_evaluated"] == 33
    assert json.loads((PREVIOUS / "integrity_receipt.json").read_text())["all_passed"]
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    if not (OUT / "protocol.json").exists():
        OUT.mkdir(exist_ok=False)
        snapshot = OUT / "source_snapshot"
        snapshot.mkdir()
        for name in ("app", "reference_data", "methods"):
            shutil.copytree(
                REPO / name,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        inputs = [
            Path(__file__).with_name("frozen_resources.py"),
            Path(__file__).with_name("run_qe_atlas_experiment.py"),
            QE / "rt_atlas_development/fold_assignment.json",
            PREVIOUS / "protocol.json",
            PREVIOUS / "integrity_receipt.json",
            PREPARED / "protocol.json",
            PREPARED / "summary.json",
        ]
        for outer in range(5):
            folder = PREVIOUS / "calibration" / f"fold_{outer}"
            inputs.extend(
                [
                    folder / "operating_points.json",
                    folder / "selected_training_points.csv",
                    STD / "models" / f"fold_{outer}_standard_models.json",
                    PREVIOUS / f"fold_{outer}_nested_model.pkl",
                ]
            )
            for inner in range(5):
                if inner != outer:
                    inputs.append(folder / f"inner_block_{inner}_model.pkl")
            for dataset in projects:
                if folds[dataset] == outer:
                    inputs.extend(
                        [
                            prepared_path(dataset, {outer}),
                            PREVIOUS / "runs" / dataset / "scoring_audit.pkl",
                            BASE / "runs" / dataset / "full_annotation.pkl",
                        ]
                    )
                else:
                    inputs.extend(
                        [
                            prepared_path(dataset, {outer, folds[dataset]}),
                            folder / (dataset + "_blind_scores.pkl"),
                            BASE / "evaluation_v2" / dataset / "manual_comparison.pkl",
                        ]
                    )
        old = json.loads((PREVIOUS / "protocol.json").read_text())
        protocol = dict(
            name="023-class-operating-points-with-nested-training-predictions",
            baseline=str(BASE),
            previous=str(PREVIOUS),
            production_enabled=False,
            curation_runtime_input=False,
            folds=5,
            inner_folds=4,
            numerical_workers=3,
            threshold_grid=[round(v, 2) for v in np.arange(0.05, 1.001, 0.05)],
            target=0.9,
            extras_budget_per_endogenous_manual=0.095,
            minimum_cohorts=3,
            minimum_manual=30,
            solver_time_limit_per_stage_seconds=60,
            candidate_policy="EXACT022 outer-query candidates/evidence/FINAL SCORES and005standardmodels. No refit/admission/mass/RT/composition-window or margin changes. No post-hoc choice between009 and022 class models.",
            threshold_policy="Same021 class allocation algorithm, but now calibrated on genuine nested022 INNER OOF query scores/features. One canonical class/mode cutoff. Class and mode training floors=min(reference022correct,ceil(.9manual)),totalcorrect>=reference022. Pooled9.5% endogenous extras/manual TRAINING budget. Maximize summed class coverage capped90%,thenpooledcorrect,thenfewerwrongfeatures/extras. <3groups/<30manual frozen022cutoff;unknownclasses fallback022modecutoff. Verified optimalMILP required; no silent budget relaxation. .05..1grid is SAME021,not new mass/RT permission or evidence of structural certainty.",
            competition="Global feature/identity winners fixed before class cutoffs. Withholding top label CANNOT promote losing label. Exact022 INNER training reference totals and uniform-cutoff selector parity required.",
            validation=old["validation"],
            model_fit_note=old["model_fit_note"]
            + " Class cutoffs now use022 four-block truly nested endogenous score OOF predictions. Every calibration query reference excludes its own and outer project blocks; corresponding score model excludes query and outer blocks. Runtime query manual curation is never read. Training floors/budget do NOT guarantee held-out floors/extras or FDR.",
            run_kind="Saved022 original-feature/score replay; class cutoffs fit on frozen022 query-excluded INNER training predictions. Same022 final scoremodels and005standardadjudication; NOT fresh production raw-input runs.",
            source_snapshot_sha256={
                p.relative_to(snapshot).as_posix(): sha(p)
                for p in snapshot.rglob("*")
                if p.is_file()
            },
            resource_receipt=configure_frozen_resources(snapshot),
            input_sha256={str(p): sha(p) for p in inputs},
        )
        dump(OUT / "protocol.json", protocol)
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        dump(
            OUT / "run_manifest.json",
            dict(
                datasets=projects,
                fresh_raw_input_runs=False,
                evaluation_note=protocol["run_kind"] + " " + protocol["model_fit_note"],
            ),
        )
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    for path, digest in protocol["source_snapshot_sha256"].items():
        assert sha(OUT / "source_snapshot" / path) == digest, path
    for path, digest in protocol["input_sha256"].items():
        assert sha(Path(path)) == digest, path
    assert (
        configure_frozen_resources(OUT / "source_snapshot")
        == protocol["resource_receipt"]
    )
    for name in (
        "standard_completion.py",
        "standard_rt_library.py",
        "standard_rt_transfer.py",
        "declared_standards.py",
    ):
        p = Path("source_snapshot/app/matlab_annotation") / name
        assert (OUT / p).read_bytes() == (STD / p).read_bytes()
    assert (
        OUT / "source_snapshot/app/matlab_annotation/class_operating_point.py"
    ).read_bytes() == (
        PREVIOUS / "source_snapshot/app/matlab_annotation/class_operating_point.py"
    ).read_bytes()
    return protocol


def calibrate(protocol, projects, folds, outer):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.class_operating_point import choose_class_training_points
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.evidence_training import validate_feature_scope

    train = [d for d in projects if folds[d] != outer]
    groups, excluded = (
        {d[:7] for d in train},
        {d[:7] for d in projects if folds[d] == outer},
    )
    folder = OUT / "calibration" / f"fold_{outer}"
    folder.mkdir(parents=True, exist_ok=True)
    reference = PREVIOUS / "calibration" / f"fold_{outer}"
    fallback = json.loads((reference / "operating_points.json").read_text())[
        "thresholds"
    ]
    destination = folder / "operating_points.json"
    if destination.exists():
        result = json.loads(destination.read_text())
        assert (
            set(result["training_groups"]) == groups
            and set(result["excluded_groups"]) == excluded
        )
        assert result["fallback_thresholds"] == fallback
        assert result["training_table_sha256"] == sha(
            folder / "training_class_operating_points.csv"
        )
        return result
    manuals, cohorts, all_classes, winners = Counter(), defaultdict(set), set(), []
    for dataset in train:
        inner = folds[dataset]
        model = pd.read_pickle(reference / f"inner_block_{inner}_model.pkl")
        data = pd.read_pickle(prepared_path(dataset, {outer, inner}))
        validate_feature_scope(
            data,
            model["training_cohorts"],
            model["excluded_groups"],
            query_is_training=False,
        )
        assert set(model["excluded_groups"]) == excluded | {
            d[:7] for d in projects if folds[d] == inner
        }
        scores = pd.read_pickle(reference / (dataset + "_blind_scores.pkl"))
        assert scores.index.equals(data["pool"].index)
        # Check split BEFORE reading calibration truth. This function is never
        # invoked with outer-evaluated dataset identities.
        assert dataset[:7] in groups and dataset[:7] not in excluded
        manual = pd.read_pickle(
            BASE / "evaluation_v2" / dataset / "manual_comparison.pkl"
        )
        manual = manual[~manual.internal_standard].copy()
        manual["_class"] = manual.lipid_class.map(canonical_lipid_class)
        class_counts = manual.groupby(["stream", "_class"]).size().to_dict()
        manuals.update(class_counts)
        for key in class_counts:
            cohorts[key].add(dataset[:7])
        all_classes.update(class_counts)
        identity = set(
            manual[["stream", "species", "adduct"]].itertuples(index=False, name=None)
        )
        exact = set(
            manual.loc[
                manual.mapped, ["stream", "species", "adduct", "feature_id"]
            ].itertuples(index=False, name=None)
        )
        unresolved = set(
            manual.loc[~manual.mapped, ["stream", "species", "adduct"]].itertuples(
                index=False, name=None
            )
        )
        pool = data["pool"].copy()
        assert not {"training_positive", "training_label_known"} & set(pool)
        pool["_class"] = pool["class"].map(canonical_lipid_class)
        keys = [
            (s, species(n), adduct(a), str(fid))
            for s, n, a, fid in pool[
                ["stream", "name", "adduct", "feat_feature_id"]
            ].itertuples(index=False, name=None)
        ]
        pool["training_positive"] = [k in exact for k in keys]
        pool["training_label_known"] = [
            k[:3] not in unresolved or k in exact for k in keys
        ]
        pool["calibration_identity_in_manual"] = [k[:3] in identity for k in keys]
        all_classes.update(
            pool[["stream", "_class"]].itertuples(index=False, name=None)
        )
        for _stream, part in pool.groupby("stream", sort=False):
            winners.append(
                select_operating_candidates(
                    part, scores.loc[part.index], threshold=0.05, ambiguity_margin=0.05
                )
            )
    winner = pd.concat(winners, ignore_index=True)
    records = []
    for threshold in sorted(set(protocol["threshold_grid"]) | set(fallback.values())):
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
    old = pd.read_csv(reference / "selected_training_points.csv")
    for r in old.itertuples():
        part = refs[refs.stream.eq(r.stream)]
        assert (
            int(part.manual.sum()),
            int(part.correct.sum()),
            int(part.extras.sum()),
            int(part.wrong_peak.sum()),
        ) == (int(r.manual), int(r.correct), int(r.extras), int(r.different_peak)), (
            outer,
            r.stream,
            "022reference parity failed",
        )
    table_path = folder / "training_class_operating_points.csv"
    table.to_csv(table_path, index=False)
    print("NESTED CLASS TRAINING REFERENCE PARITY PASS", outer, flush=True)
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
        "No verified optimal cutoff allocation; do not relax guards"
    )
    learned["fallback_thresholds"] = fallback
    learned["training_table_sha256"] = sha(table_path)
    learned["reference_parity_passed"] = True
    dump(destination, learned)
    pd.DataFrame(learned["selected"]).to_csv(
        folder / "selected_class_training_points.csv", index=False
    )
    print(
        "NESTED CLASS CUTOFFS",
        outer,
        learned["reference_correct"],
        "->",
        learned["correct"],
        "trainingcorrect",
        learned["extras"],
        "trainingextras",
        flush=True,
    )
    return learned


def run_fold(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.class_operating_point import (
        select_class_operating_candidates,
    )
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards

    protocol = json.loads((OUT / "protocol.json").read_text())
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    thresholds = calibrate(protocol, projects, folds, outer)
    std = json.loads(
        (STD / "models" / f"fold_{outer}_standard_models.json").read_text()
    )
    heldout = [d for d in projects if folds[d] == outer]
    assert not set(std["training_groups"]) & {d[:7] for d in heldout}
    for dataset in heldout:
        dest = OUT / "runs" / dataset
        if (dest / "complete.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        data = pd.read_pickle(prepared_path(dataset, {outer}))
        raw = data["pool"]
        assert not {"training_positive", "training_label_known"} & set(raw)
        scores = pd.read_pickle(PREVIOUS / "runs" / dataset / "scoring_audit.pkl")
        assert scores.index.equals(raw.index)
        result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        catalogue = declared_standard_targets(
            OUT / "source_snapshot/reference_data/matlab_pipeline",
            result["internal_standards_expected"],
        )
        parts, candidates, audits = [], [raw], []
        for stream, payload in result["streams"].items():
            pool = raw[raw.stream.eq(stream)]
            reference = select_operating_candidates(
                pool,
                scores.loc[pool.index],
                threshold=thresholds["fallback_thresholds"][stream],
                ambiguity_margin=0.05,
            )
            uniform = select_class_operating_candidates(
                pool,
                scores.loc[pool.index],
                {},
                fallback_threshold=thresholds["fallback_thresholds"][stream],
                ambiguity_margin=0.05,
            )
            assert reference.index.equals(uniform.index), (
                "Uniform class selector changed022 competition"
            )
            selected = select_class_operating_candidates(
                pool,
                scores.loc[pool.index],
                thresholds["thresholds"],
                fallback_threshold=thresholds["fallback_thresholds"][stream],
                ambiguity_margin=0.05,
            )
            selected["auto_annotation_status"] = "nested_class_operating_point"
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
                std["library"],
                std["atlas"],
                std["transfer"],
                std["channels"],
                stream=stream,
                alignment_annotations=payload["annotations"],
            )
            parts.append(corrected.assign(stream=stream))
            audits.append(audit.assign(dataset=dataset, stream=stream))
            candidates.extend(
                [
                    protected,
                    extra.assign(stream=stream),
                    corrected[corrected.is_is.fillna(False).astype(bool)].assign(
                        stream=stream
                    ),
                ]
            )
        annotations = pd.concat(parts, ignore_index=True)
        candidate = pd.concat(candidates, ignore_index=True)
        keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
        candidate = candidate.drop_duplicates(keys, keep="last").reset_index(drop=True)
        chosen = set(annotations[keys].itertuples(index=False, name=None))
        candidate["automatic_accepted"] = [
            k in chosen for k in candidate[keys].itertuples(index=False, name=None)
        ]
        annotations.to_csv(dest / "automatic_annotations.csv", index=False)
        candidate.to_pickle(dest / "candidates.pkl")
        scores.to_pickle(dest / "scoring_audit.pkl")
        pd.concat(audits, ignore_index=True).to_csv(
            dest / "standard_audit.csv", index=False
        )
        dump(
            dest / "complete.json",
            dict(
                dataset=dataset,
                fold=outer,
                status="complete",
                production_enabled=False,
                class_operating_points_file=str(
                    OUT / "calibration" / f"fold_{outer}/operating_points.json"
                ),
                uniform_cutoff_parity_passed=True,
                counts=annotations.groupby("stream").size().to_dict(),
            ),
        )
        print("NESTED CLASS HELDOUT SELECTED", dataset, len(annotations), flush=True)
    return outer


def main():
    protocol = initialize()
    with ProcessPoolExecutor(max_workers=protocol["numerical_workers"]) as executor:
        for future in as_completed([executor.submit(run_fold, f) for f in range(5)]):
            print("NESTED CLASS FOLD COMPLETE", future.result(), flush=True)
    print("EXPERIMENT023 COMPLETE; FULL132TABLE REQUIRED BEFORE024", flush=True)


if __name__ == "__main__":
    main()
