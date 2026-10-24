"""Experiment028: four fixed mode experts, retaining unchanged027mass discovery."""

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
PREPARED = QE / "nested_candidate_preparation_v2"
REFERENCES = QE / "nested_reference_development_v2"
STD = QE / "experiment_005"
CORE = QE / "experiment_022"
OLD = QE / "experiment_027"
OUT = QE / "experiment_028"


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
            "nested_feature_folds.py",
        ):
            assert (snapshot / "app/matlab_annotation" / name).read_bytes() == (
                PREPARED / "source_snapshot/app/matlab_annotation" / name
            ).read_bytes(), name
        mass_receipt = json.loads((OLD / "integrity_receipt.json").read_text())
        assert mass_receipt["all_passed"] and mass_receipt["datasets"] == 33
        inputs.extend(
            [
                OLD / "protocol.json",
                OLD / "integrity_receipt.json",
                OLD / "reports_v2/summary.json",
            ]
        )
        inputs.extend(
            OLD / "runs" / d / "private_mass_candidates.pkl" for d in projects
        )
        core_models = list(CORE.glob("fold_*_nested_model.pkl")) + list(
            CORE.glob("calibration/fold_*/inner_block_*_model.pkl")
        )
        assert len(core_models) == 25
        inputs.extend(core_models)
        for name in ("calibrated_mass_discovery.py", "cross_mode_evidence.py"):
            assert (snapshot / "app/matlab_annotation" / name).read_bytes() == (
                OLD / "source_snapshot/app/matlab_annotation" / name
            ).read_bytes()
        classifier = json.loads((QE / "experiment_007/protocol.json").read_text())[
            "classifier"
        ]
        protocol = dict(
            name="028-fixed-four-mode-score-experts-with-027mass-discovery",
            baseline=str(BASE),
            previous=str(OLD),
            production_enabled=False,
            curation_runtime_input=False,
            score_architecture="ALL four fixed measurement-mode experts;180rounds EACH,100classifiers in25nested parent fits. Not a posthoc choice of old phase/class winners.",
            partial_curation_labels_used=False,
            new_standard_context_used=False,
            mass_discovery_policy="EXACT027fixed label-free physical proposal pools and same selection helper,applied AFTER newcore/005STD calls. Mass scope excludes query/outer. Novelty is relative to unchanged022physical pool. Existing current core calls immutable within mass stage. Changed core occupancy may change mass additions; full algorithm evaluated versus027.",
            folds=5,
            inner_folds=4,
            numerical_workers=3,
            classifier=classifier,
            threshold_grid=[round(v, 2) for v in np.arange(0.2, 1.001, 0.05)],
            extras_budget_per_endogenous_manual=0.095,
            ambiguity_margin=0.05,
            candidate_policy="Same strict001-007 chemistry/default+group-trained ranges and3ppm/10s dual-RT discovery. All165 original-preparation parity checks passed; every outerheldout physical/query-order/predictor table matches007. Endogenous TRAINING pools regenerated with query-excluded RT/priors/envelopes,not forced tooldself-informedtrainingpools. No020 additions/021cutoffs or new mass calibration.",
            nested_policy="Existing5project-groupblocks. Outerf:finalfitqueryk usesreferencesexcluding{f,k},heldoutquery{f}. Fourinner validationblocksj:fitqueryk referencesexclude{f,j,k},validationquery{f,j}. Everyreference group is a subset of score-traininggroups; ownquery,outertest andinnervalidation excluded. 25numerical referencebundles,363label-free querytables. Manual labels attachedONLYafterfeaturegeneration andONLYforpermittedtraining/calibrationgroups.",
            threshold_policy="NEW four INNER group-held-out score predictions. EXACT009 joint weakest-mode capped90% objective/pooled9.5%endogenous-training extras budget and.2..1grid. No outerheldout labels choose cutoffs. Same180roundHGB/cohortweights/margin. No FDR or test-extras guarantee.",
            standard_policy="EXACT005 adjudication/models from matchingouterfold; standards excluded fromendogenous thresholdcalibration. BlindBASELINE automaticanchors/rawquality remain fixed. No retuning standards againstouterlabels.",
            validation="Grouped held-out evaluation across33datasets/30projectgroups. Newly learned endogenous feature references,scoremodels andcutoffs exclude relevantquery/evaluation/calibrationgroups. Reference/preparation parity is engineering evidence,not annotationaccuracy. Prior algorithm development has inspected these examples.",
            model_fit_note="Same022pools/predictors/knownpositive-negative rows. New independent180roundexpert for EACHof4modes,cohort-balanced WITHIN mode. No024unknownlabels/025features/026classweights.100classifiers in25nested parentfits. Same005standardhandling+fixed027massstage. OOFcutoff budget covers core endogenous calls ONLY as in027;fixed later mass additions not used to fit cutoffs,so no outer-only mass-reference information enters inner calibration. Finalextras not testbounded. Priorbaseline/vendor exposure remains;not untouchedexternal validation/FDR.",
            run_kind="Cached original-feature replay;fourmodeexperts refitted together/properly nested,NEWfour-INNER-corecutoffs. Same022physical scorefeatures and supervision,005STDandunchanged027fixedphysicalmass additions. Compare entire pipeline against027,not fresh production raw-input runs.",
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
    from matlab_annotation.mode_evidence_experts import (
        fit_mode_evidence_experts as fit_grouped_evidence_model,
    )

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
    old_path = CORE / path.relative_to(OUT)
    core_model = pd.read_pickle(old_path)
    assert model["training_pairs"] == core_model["training_pairs"]
    assert model["positives"] == core_model["positives"]
    assert model["columns"] == core_model["columns"]
    assert len(model["experts"]) == 4
    assert all(
        m["estimator"].n_iter_ == 180 and "sample_weight_policy" not in m
        for m in model["experts"].values()
    )
    model["core_model_sha256"] = sha(old_path)
    pd.to_pickle(model, path)
    print(
        "FIT FOUR MODE EXPERTS",
        path.name,
        model["training_pairs"],
        "rows; all4modes refit; all labels unchanged",
        flush=True,
    )
    return model


def calibrate(protocol, projects, folds, outer):
    from matlab_annotation.mode_evidence_experts import (
        predict_mode_evidence_experts as predict_grouped_evidence,
    )
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
    from matlab_annotation.mode_evidence_experts import (
        predict_mode_evidence_experts as predict_grouped_evidence,
    )
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
            OLD / "runs" / dataset / "private_mass_candidates.pkl"
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
            selected["auto_annotation_status"] = "four_mode_expert_selected"
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
        "EXPERIMENT028 COMPLETE; FULL132-ROW TABLE REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
