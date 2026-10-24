"""Experiment025: independent fully nested declared-standard scoring context."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from context import WORKSPACE as ROOT, REPO
from frozen_resources import configure_frozen_resources

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qe_atlas_experiment import species, adduct

QE = ROOT / "outputs/qe_optimization_20261022"
BASE = QE / "baseline"
PREPARED = QE / "standard_context_preparation_v1"
REFERENCES = QE / "nested_reference_development_v2"
STD = QE / "experiment_005"
OLD = QE / "experiment_022"
OUT = QE / "experiment_025"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def initialize():
    ready = json.loads((PREPARED / "summary.json").read_text())
    assert (
        ready["all_complete"]
        and ready["dataset_mask_tables"] == 363
        and not ready["supervised_labels_used_for_predictors"]
    )
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    if not (OUT / "protocol.json").exists():
        OUT.mkdir(exist_ok=False)
        snapshot = OUT / "source_snapshot"
        snapshot.mkdir()
        for name in ("app", "methods", "reference_data"):
            shutil.copytree(
                REPO / name,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        sources = {
            p.relative_to(snapshot).as_posix(): sha(p)
            for p in snapshot.rglob("*")
            if p.is_file()
        }
        inputs = [
            Path(__file__).with_name("frozen_resources.py"),
            Path(__file__).with_name("run_qe_atlas_experiment.py"),
            PREPARED / "summary.json",
            PREPARED / "protocol.json",
            REFERENCES / "summary.json",
            QE / "rt_atlas_development/fold_assignment.json",
        ]
        for dataset in projects:
            receipt = json.loads(
                (PREPARED / "checkpoints" / (dataset + ".json")).read_text()
            )
            for path, digest in receipt["prepared_sha256"].items():
                assert sha(Path(path)) == digest, path
                inputs.append(Path(path))
            inputs.extend(
                [
                    BASE / "runs" / dataset / "full_annotation.pkl",
                    BASE / "evaluation_v2" / dataset / "manual_comparison.pkl",
                ]
            )
        for fold in range(5):
            inputs.append(STD / "models" / f"fold_{fold}_standard_models.json")
        for name in (
            "candidate_preparation.py",
            "evidence_training.py",
            "nested_feature_folds.py",
            "declared_standard_context.py",
        ):
            assert (snapshot / "app/matlab_annotation" / name).read_bytes() == (
                PREPARED / "source_snapshot/app/matlab_annotation" / name
            ).read_bytes(), name
        inputs.extend((QE / "nested_standard_development_v1").glob("exclude_*.json"))
        inputs.append(QE / "nested_standard_development_v1/summary.json")
        old_models = list(OLD.glob("fold_*_nested_model.pkl")) + list(
            OLD.glob("calibration/fold_*/inner_block_*_model.pkl")
        )
        assert len(old_models) == 25
        inputs.extend(old_models)
        assert ready["new_numeric_features"] == 15
        assert ready["original_pool_exact_parity_checks"] == 363
        assert ready["original_predictor_exact_parity_checks"] == 363
        classifier = json.loads((QE / "experiment_007/protocol.json").read_text())[
            "classifier"
        ]
        protocol = dict(
            name="025-independent-nested-declared-standard-context",
            baseline=str(BASE),
            previous=str(OLD),
            production_enabled=False,
            curation_runtime_input=False,
            partial_curation_labels_used=False,
            new_standard_context_used=True,
            new_numeric_features=15,
            folds=5,
            inner_folds=4,
            numerical_workers=3,
            classifier=classifier,
            threshold_grid=[round(v, 2) for v in np.arange(0.2, 1.001, 0.05)],
            extras_budget_per_endogenous_manual=0.095,
            ambiguity_margin=0.05,
            candidate_policy="EXACT022 all363 original physical pools and old predictors, plus15numeric independent declared-standard features. Pool and old-column exact parity proved for every query/mask. Same3ppm/10s chemistry/discovery/ranges. No024partiallabels, new calls, widened mass/RTwindows or class cutoff sweeps.",
            nested_policy="Existing5project-groupblocks. Outerf:finalfitqueryk usesreferencesexcluding{f,k},heldoutquery{f}. Fourinner validationblocksj:fitqueryk referencesexclude{f,j,k},validationquery{f,j}. Everyreference group is a subset of score-traininggroups; ownquery,outertest andinnervalidation excluded. 25numerical referencebundles,363label-free querytables. Manual labels attachedONLYafterfeaturegeneration andONLYforpermittedtraining/calibrationgroups.",
            threshold_policy="NEW four INNER group-held-out score predictions. EXACT009 joint weakest-mode capped90% objective/pooled9.5%endogenous-training extras budget and.2..1grid. No outerheldout labels choose cutoffs. Same180roundHGB/cohortweights/margin. No FDR or test-extras guarantee.",
            standard_policy="EXACT005 adjudication/models from matchingouterfold; standards excluded fromendogenous thresholdcalibration. BlindBASELINE automaticanchors/rawquality remain fixed. No retuning standards againstouterlabels.",
            validation="Grouped held-out evaluation across33datasets/30projectgroups. Newly learned endogenous feature references,scoremodels andcutoffs exclude relevantquery/evaluation/calibrationgroups. Reference/preparation parity is engineering evidence,not annotationaccuracy. Prior algorithm development has inspected these examples.",
            model_fit_note="Endogenous AND new standard-context reference features exclude relevant query/outer/inner groups.25newly nested standard reference bundles; independent unique raw spike detection with EMPTY endogenous proposals, no self/companion anchoring, exact classes and actual isotope masses. Training labels and row counts unchanged022. Inherited blind automatic/vendor models retain prior development exposure. Final005standard calls unchanged method/not newly nested. NOT untouched external/pristine whole-pipeline validation or analytical FDR.",
            run_kind="Saved EXACT022 physical-feature/old-predictor replay, adding15nested independently measured standard-context predictors.25fixed180-round score refits and new four-INNER-block calibrated mode cutoffs. Same022supervision and005final standard handling. Not fresh production raw-input runs.",
            source_snapshot_sha256=sources,
            resource_receipt=configure_frozen_resources(snapshot),
            input_sha256={str(p): sha(p) for p in inputs},
        )
        dump(OUT / "protocol.json", protocol)
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        shutil.copy2(
            Path(__file__).with_name("run_qe_atlas_experiment.py"),
            OUT / "atlas_runner_dependency.py",
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
    for path, digest in protocol["source_snapshot_sha256"].items():
        assert sha(OUT / "source_snapshot" / path) == digest, path
    for path, digest in protocol["input_sha256"].items():
        assert sha(Path(path)) == digest, path
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    assert (OUT / "atlas_runner_dependency.py").read_bytes() == Path(
        __file__
    ).with_name("run_qe_atlas_experiment.py").read_bytes()
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
        rel = Path("source_snapshot/app/matlab_annotation") / name
        assert (OUT / rel).read_bytes() == (STD / rel).read_bytes(), name
    return protocol, projects


def load_predictors(dataset, mask):
    path = (
        PREPARED
        / "prepared"
        / ("exclude_" + "_".join(map(str, sorted(mask))))
        / (dataset + ".pkl")
    )
    data = pd.read_pickle(path)
    assert data["query_dataset"] == dataset and data["query_group"] == dataset[:7]
    assert set(data["excluded_folds"]) == set(mask)
    assert not {
        "training_positive",
        "training_label_known",
        "same_manual_peak",
        "manual_feature_id",
    } & set(data["pool"])
    assert {
        "standard_reference_training_groups",
        "standard_reference_excluded_groups",
        "standard_reference_sha256",
    }.issubset(data)
    assert set(data["standard_reference_training_groups"]) == set(
        data["reference_training_groups"]
    )
    assert set(data["standard_reference_excluded_groups"]) == set(
        data["reference_excluded_groups"]
    )
    assert (
        sum(str(c).startswith("q_declared_standard_") for c in data["evidence"]) == 15
    )
    return data


def with_training_labels(data, *, permitted_groups, excluded_groups):
    # Check authorization/split BEFORE opening a manual reference. This routine
    # is NEVER called on the final outer-evaluated dataset.
    group = data["query_group"]
    assert group in permitted_groups and group not in excluded_groups
    dataset = data["query_dataset"]
    manual = pd.read_pickle(BASE / "evaluation_v2" / dataset / "manual_comparison.pkl")
    manual = manual[~manual.internal_standard]
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
    identities = set(
        manual[["stream", "species", "adduct"]].itertuples(index=False, name=None)
    )
    result = dict(data)
    pool = data["pool"].copy()
    keys = [
        (s, species(n), adduct(a), str(fid))
        for s, n, a, fid in pool[
            ["stream", "name", "adduct", "feat_feature_id"]
        ].itertuples(index=False, name=None)
    ]
    pool["training_positive"] = [k in exact for k in keys]
    pool["training_label_known"] = [k[:3] not in unresolved or k in exact for k in keys]
    result["pool"] = pool
    return result, manual, identities


def fit_score(datasets, training_groups, excluded_groups, parameters, path):
    from matlab_annotation.evidence_training import fit_grouped_evidence_model

    if path.exists():
        model = pd.read_pickle(path)
        assert set(model["training_cohorts"]) == set(training_groups)
        assert set(model["excluded_groups"]) == set(excluded_groups)
        return model
    model = fit_grouped_evidence_model(
        datasets,
        training_groups=training_groups,
        excluded_groups=excluded_groups,
        parameters=parameters,
    )
    model["training_feature_masks"] = {
        d: data["excluded_folds"] for d, data in datasets.items()
    }
    old_path = OLD / path.relative_to(OUT)
    old_model = pd.read_pickle(old_path)
    assert model["training_pairs"] == old_model["training_pairs"]
    assert model["positives"] == old_model["positives"]
    assert set(model["columns"]) - set(old_model["columns"]) == {
        c for c in model["columns"] if str(c).startswith("q_declared_standard_")
    }
    assert len(model["columns"]) == len(old_model["columns"]) + 15
    model["old_model_sha256"] = sha(old_path)
    model["standard_training_feature_scopes"] = {
        d: dict(
            training=data["standard_reference_training_groups"],
            excluded=data["standard_reference_excluded_groups"],
            reference_sha256=data["standard_reference_sha256"],
        )
        for d, data in datasets.items()
    }
    pd.to_pickle(model, path)
    print(
        "FIT STANDARD-CONTEXT SCORE",
        path.name,
        model["training_pairs"],
        "rows; positives/negatives unchanged",
        flush=True,
    )
    return model


def calibrate(protocol, projects, folds, outer):
    from matlab_annotation.evidence_training import predict_grouped_evidence
    from matlab_annotation.operating_point import (
        select_operating_candidates,
        choose_balanced_training_points,
    )

    folder = OUT / "calibration" / f"fold_{outer}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "operating_points.json"
    outer_groups = {d[:7] for d in projects if folds[d] == outer}
    allowed = {d[:7] for d in projects if folds[d] != outer}
    if path.exists():
        result = json.loads(path.read_text())
        assert (
            set(result["training_groups"]) == allowed
            and set(result["outer_excluded"]) == outer_groups
        )
        return result["thresholds"]
    params = {k: v for k, v in protocol["classifier"].items() if k != "kind"}
    cases, predictions, manual_counts = {}, {}, Counter()
    for inner in sorted(set(folds.values()) - {outer}):
        fitting = [d for d in projects if folds[d] not in {outer, inner}]
        validation = [d for d in projects if folds[d] == inner]
        training_groups = {d[:7] for d in fitting}
        excluded = outer_groups | {d[:7] for d in validation}
        data = {}
        for dataset in fitting:
            raw = load_predictors(dataset, {outer, inner, folds[dataset]})
            data[dataset], _, _ = with_training_labels(
                raw, permitted_groups=training_groups, excluded_groups=excluded
            )
        model = fit_score(
            data,
            training_groups,
            excluded,
            params,
            folder / f"inner_block_{inner}_model.pkl",
        )
        del data
        for dataset in validation:
            raw = load_predictors(dataset, {outer, inner})
            scores = predict_grouped_evidence(raw, model)
            scores.to_pickle(folder / (dataset + "_blind_scores.pkl"))
            labeled, manual, identities = with_training_labels(
                raw, permitted_groups=allowed, excluded_groups=outer_groups
            )
            pool = labeled["pool"]
            pool["calibration_identity_in_manual"] = [
                (s, species(n), adduct(a)) in identities
                for s, n, a in pool[["stream", "name", "adduct"]].itertuples(
                    index=False, name=None
                )
            ]
            cases[dataset] = pool
            predictions[dataset] = scores
            manual_counts.update(manual.stream.value_counts().to_dict())
    assert set(cases) == {d for d in projects if folds[d] != outer}
    records = []
    for threshold in protocol["threshold_grid"]:
        counts = {s: Counter() for s in manual_counts}
        for dataset, pool in cases.items():
            for stream, part in pool.groupby("stream", sort=False):
                chosen = select_operating_candidates(
                    part,
                    predictions[dataset].loc[part.index],
                    threshold=threshold,
                    ambiguity_margin=0.05,
                )
                counts[stream]["correct"] += int(chosen.training_positive.sum())
                counts[stream]["extras"] += int(
                    (~chosen.calibration_identity_in_manual).sum()
                )
                counts[stream]["different_peak"] += int(
                    (
                        chosen.calibration_identity_in_manual
                        & chosen.training_label_known
                        & ~chosen.training_positive
                    ).sum()
                )
        records.extend(
            dict(stream=s, threshold=threshold, manual=n, **dict(counts[s]))
            for s, n in manual_counts.items()
        )
    table = pd.DataFrame(records).fillna(0)
    table.to_csv(folder / "training_operating_points.csv", index=False)
    thresholds = choose_balanced_training_points(
        table,
        extras_per_manual=protocol["extras_budget_per_endogenous_manual"],
        target=0.9,
    )
    assert thresholds is not None, (
        "No feasible nested training operating point; do not relax budget"
    )
    chosen = table.loc[
        [
            float(t) == thresholds[s]
            for s, t in table[["stream", "threshold"]].itertuples(
                index=False, name=None
            )
        ]
    ]
    chosen.to_csv(folder / "selected_training_points.csv", index=False)
    dump(
        path,
        dict(
            thresholds=thresholds,
            training_groups=sorted(allowed),
            outer_excluded=sorted(outer_groups),
            selected_training_correct=int(chosen.correct.sum()),
            selected_training_extras=int(chosen.extras.sum()),
            training_table_sha256=sha(folder / "training_operating_points.csv"),
            note=protocol["nested_policy"],
        ),
    )
    print("NESTED TRAINING CUTOFFS", outer, thresholds, flush=True)
    return thresholds


def run_fold(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.evidence_training import predict_grouped_evidence
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards

    protocol = json.loads((OUT / "protocol.json").read_text())
    projects = json.loads((OUT / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    heldout = [d for d in projects if folds[d] == outer]
    if all((OUT / "runs" / d / "complete.json").exists() for d in heldout):
        return outer, "already complete"
    thresholds = calibrate(protocol, projects, folds, outer)
    train_groups = {d[:7] for d in projects if folds[d] != outer}
    excluded = {d[:7] for d in heldout}
    training = {}
    for dataset in projects:
        if folds[dataset] == outer:
            continue
        data = load_predictors(dataset, {outer, folds[dataset]})
        training[dataset], _, _ = with_training_labels(
            data, permitted_groups=train_groups, excluded_groups=excluded
        )
    params = {k: v for k, v in protocol["classifier"].items() if k != "kind"}
    model = fit_score(
        training, train_groups, excluded, params, OUT / f"fold_{outer}_nested_model.pkl"
    )
    del training
    standards = json.loads(
        (STD / "models" / f"fold_{outer}_standard_models.json").read_text()
    )
    assert not set(standards["training_groups"]) & excluded
    for dataset in heldout:
        dest = OUT / "runs" / dataset
        if (dest / "complete.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        data = load_predictors(dataset, {outer})
        scores = predict_grouped_evidence(data, model)
        raw = data["pool"]
        assert not {"training_positive", "training_label_known"} & set(raw)
        baseline = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        catalogue = declared_standard_targets(
            OUT / "source_snapshot/reference_data/matlab_pipeline",
            baseline["internal_standards_expected"],
        )
        parts, candidates, audits = [], [raw], []
        for stream, payload in baseline["streams"].items():
            pool = raw[raw.stream.eq(stream)]
            selected = select_operating_candidates(
                pool,
                scores.loc[pool.index],
                threshold=thresholds[stream],
                ambiguity_margin=0.05,
            )
            selected["auto_annotation_status"] = "independent_standard_context_selected"
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
                standards["library"],
                standards["atlas"],
                standards["transfer"],
                standards["channels"],
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
        accepted = set(annotations[keys].itertuples(index=False, name=None))
        candidate["automatic_accepted"] = [
            k in accepted for k in candidate[keys].itertuples(index=False, name=None)
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
                thresholds=thresholds,
                counts=annotations.groupby("stream").size().to_dict(),
            ),
        )
        print("NESTED HELDOUT SELECTED", dataset, len(annotations), flush=True)
    return outer, "complete"


def main():
    protocol, _ = initialize()
    with ProcessPoolExecutor(max_workers=protocol["numerical_workers"]) as executor:
        for future in as_completed([executor.submit(run_fold, f) for f in range(5)]):
            print("NESTED OUTER FOLD", *future.result(), flush=True)
    print(
        "EXPERIMENT025 COMPLETE; FULL132-ROW TABLE REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
