"""035: fixed family/mode experts; unchanged029physical inputs and labels."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

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
OLD = QE / "experiment_029"
MASS = QE / "experiment_027"
OUT = QE / "experiment_035"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def initialize():
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    assert len(projects) == 33
    assert json.loads((OLD / "integrity_receipt.json").read_text())["all_passed"]
    ready = json.loads((PREPARED / "summary.json").read_text())
    assert ready["all_complete"] and ready["dataset_mask_tables"] == 363
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
        inputs = [
            OLD / "protocol.json",
            OLD / "integrity_receipt.json",
            OLD / "runner_source.py",
            PREPARED / "protocol.json",
            PREPARED / "summary.json",
            QE / "rt_atlas_development/fold_assignment.json",
            MASS / "protocol.json",
            MASS / "integrity_receipt.json",
            BASE / "run_manifest.json",
        ]
        inputs.extend(
            Path(__file__).with_name(n)
            for n in ("frozen_resources.py", "run_qe_atlas_experiment.py")
        )
        for dataset in projects:
            receipt = json.loads(
                (PREPARED / "checkpoints" / (dataset + ".json")).read_text()
            )
            for path, digest in receipt["prepared_sha256"].items():
                assert sha(Path(path)) == digest
                inputs.append(Path(path))
            inputs.extend(
                [
                    BASE / "runs" / dataset / "full_annotation.pkl",
                    BASE / "evaluation_v2" / dataset / "manual_comparison.pkl",
                    MASS / "runs" / dataset / "private_mass_candidates.pkl",
                ]
            )
        models = list(OLD.glob("fold_*_nested_model.pkl")) + list(
            OLD.glob("calibration/fold_*/inner_block_*_model.pkl")
        )
        assert len(models) == 25
        inputs.extend(models)
        inputs.extend(
            STD / "models" / f"fold_{f}_standard_models.json" for f in range(5)
        )
        old = json.loads((OLD / "protocol.json").read_text())
        protocol = dict(
            name="035-predeclared-family-mode-experts",
            baseline=str(BASE),
            previous=str(OLD),
            production_enabled=False,
            curation_runtime_input=False,
            numerical_workers=2,
            classifier=old["classifier"],
            folds=5,
            inner_folds=4,
            family_minimum_rows=200,
            family_minimum_positive=20,
            family_minimum_negative=20,
            family_minimum_positive_projects=3,
            family_routing="Fixed class taxonomy, separately in ALL4modes: glycerophospholipid,sphingolipid,glycerolipid,fatty_acid_ester. Unknown classes fallback. Pure NEWfamilyexpert if TRAININGsupport passes,else unchanged explicit029modeexpert. No outer-performance class choice; all classes compete GLOBALLY downstream.",
            supervision_policy="EXACT029 all363physical/evidence tables and known labels. Same180rounds per eligible family expert; equal projectweight WITHINfamily. Full allowed reference scope and actual supervised contributors recorded SEPARATELY; no invented rows. No new predictors,candidates,matrix IDs,chemistry/windows/ranges.",
            threshold_grid=old["threshold_grid"],
            ambiguity_margin=0.05,
            extras_budget_per_endogenous_manual=0.095,
            threshold_policy="Fresh genuine4INNERblock scores. SAME009weakest-mode capped90objective/pooled9.5%endogenous extrasbudget/.2..1grid/.05margin. NOT034classcutoffs. BudgetCOREonly; outer-only027masscache never entersINNERcalibration.",
            candidate_policy="EXACT029standard_context_preparation_v1 pools/predictors/labels,005STDcleanup,fixed027massrule. No rejected030-034scores/features/cutoffs.",
            nested_policy=old["nested_policy"],
            validation=old["validation"],
            model_fit_note=old["model_fit_note"]
            + "035freshfamilyexperts use SAME029preparedrows/labels/180rounds. Frozen029modeexpert fallback ONLY unsupported trainingfamilies; explicit hashbinding. Actual supervised contributors distinct from full allowed reference groups. No outer-tested winner splicing.",
            run_kind="Cached original-feature research replay,NOTfresh production annotation. Newly fitted predefined family/mode experts and newINNERcalibrated MODEcutoffs.029frozenmodeexpert fallback ONLY unsupported trainingfamilies. Same005STD/fixed027massstage. Compare WHOLEalgorithm versus029.",
            source_snapshot_sha256={
                p.relative_to(snapshot).as_posix(): sha(p)
                for p in snapshot.rglob("*")
                if p.is_file()
            },
            input_sha256={str(p): sha(p) for p in sorted(set(inputs))},
            resource_receipt=configure_frozen_resources(snapshot),
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
        "candidate_preparation.py",
        "nested_feature_folds.py",
        "declared_standard_context.py",
        "mode_evidence_experts.py",
        "operating_point.py",
        "standard_completion.py",
        "calibrated_mass_discovery.py",
        "cross_mode_evidence.py",
    ):
        rel = Path("source_snapshot/app/matlab_annotation") / name
        assert (OUT / rel).read_bytes() == (OLD / rel).read_bytes(), name
    return protocol, projects


def predict_family(data, model):
    from matlab_annotation.family_evidence_experts import (
        predict_family_evidence_experts,
    )

    path = Path(model["fallback_model_path"])
    assert sha(path) == model["fallback_model_sha256"]
    return predict_family_evidence_experts(data, model, pd.read_pickle(path))


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
    from matlab_annotation.family_evidence_experts import fit_family_evidence_experts

    fallback_path = OLD / path.relative_to(OUT)
    fallback = pd.read_pickle(fallback_path)
    assert set(fallback["training_cohorts"]) == set(training_groups)
    assert set(fallback["excluded_groups"]) == set(excluded_groups)
    if path.exists():
        model = pd.read_pickle(path)
        assert set(model["training_cohorts"]) == set(training_groups)
        assert set(model["excluded_groups"]) == set(excluded_groups)
        assert model["fallback_model_sha256"] == sha(fallback_path)
        return model
    p = json.loads((OUT / "protocol.json").read_text())
    model = fit_family_evidence_experts(
        datasets,
        training_groups=training_groups,
        excluded_groups=excluded_groups,
        parameters=parameters,
        fallback_model=fallback,
        minimum_rows=p["family_minimum_rows"],
        minimum_positive=p["family_minimum_positive"],
        minimum_negative=p["family_minimum_negative"],
        minimum_positive_projects=p["family_minimum_positive_projects"],
    )
    assert (
        model["training_pairs"] == fallback["training_pairs"]
        and model["positives"] == fallback["positives"]
    )
    assert model["columns"] == fallback["columns"]
    model["fallback_model_path"] = str(fallback_path)
    model["fallback_model_sha256"] = sha(fallback_path)
    model["training_feature_masks"] = {
        d: data["excluded_folds"] for d, data in datasets.items()
    }
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
        "FIT035FAMILYEXPERTS",
        path.name,
        len(model["experts"]),
        "newexperts;",
        model["training_pairs"],
        "original knownlabels",
        flush=True,
    )
    return model


def calibrate(protocol, projects, folds, outer):
    from run_qe_family_experiment import predict_family as predict_grouped_evidence
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
    from run_qe_family_experiment import predict_family as predict_grouped_evidence
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards
    from matlab_annotation.calibrated_mass_discovery import select_calibrated_additions

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
        mass_pool = pd.read_pickle(
            MASS / "runs" / dataset / "private_mass_candidates.pkl"
        )
        assert not {
            "training_positive",
            "training_label_known",
            "same_manual_peak",
        } & set(mass_pool)
        automatic = pd.concat(
            [
                p["annotations"].copy().assign(stream=s)
                for s, p in baseline["streams"].items()
            ],
            ignore_index=True,
        )
        mass_audits = []
        parts, candidates, audits = [], [raw, mass_pool], []
        for stream, payload in baseline["streams"].items():
            pool = raw[raw.stream.eq(stream)]
            selected = select_operating_candidates(
                pool,
                scores.loc[pool.index],
                threshold=thresholds[stream],
                ambiguity_margin=0.05,
            )
            selected["auto_annotation_status"] = "family_mode_expert_selected"
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
            local_mass = (
                mass_pool[mass_pool.stream.eq(stream)] if len(mass_pool) else mass_pool
            )
            added = select_calibrated_additions(
                local_mass, automatic, corrected, stream=stream
            )
            mass_audits.append(
                dict(
                    dataset=dataset,
                    stream=stream,
                    added_annotations=len(added),
                    current_core_annotations=len(corrected),
                    fixed_027_rule=True,
                )
            )
            combined = pd.concat([corrected, added], ignore_index=True)
            assert not combined.feat_feature_id.astype(str).duplicated().any()
            parts.append(combined.assign(stream=stream))
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
        pd.DataFrame(mass_audits).to_csv(dest / "mass_discovery_audit.csv", index=False)
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
        "EXPERIMENT035 COMPLETE; FULL132-ROW TABLE REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
