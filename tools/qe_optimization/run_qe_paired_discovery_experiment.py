"""Experiment020: additive paired-profile-supported physical discovery."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from context import WORKSPACE as ROOT, REPO, SOURCE

QE = ROOT / "outputs/qe_optimization_20261022"
BASE = QE / "baseline"
POOL = QE / "experiment_007"
PREVIOUS = QE / "experiment_009"
ATLAS = QE / "rt_atlas_development"
OUT = QE / "experiment_020"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qe_dual_experiment import model_matrix
from run_qe_atlas_experiment import species, adduct
from run_qe_paired_sample_experiment import fit_score_model


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
                SOURCE / d / "MS_metadata.mat",
                PREVIOUS / "runs" / d / "automatic_annotations.csv",
                PREVIOUS / "runs" / d / "candidates.pkl",
            ):
                dependencies[str(p)] = sha(p)
            for f in range(5):
                p = POOL / "prepared" / f"fold_{f}" / (d + ".pkl")
                dependencies[str(p)] = sha(p)
        for f in range(5):
            for p in (
                ATLAS / f"fold_{f}_atlas.json",
                QE / "experiment_001" / f"fold_{f}_selection_model.json",
                QE / "experiment_003" / f"fold_{f}_composition_envelope.json",
                PREVIOUS / "calibration" / f"fold_{f}" / "operating_points.json",
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
            POOL / "protocol.json",
            PREVIOUS / "protocol.json",
            Path(__file__).with_name("run_qe_dual_experiment.py"),
            Path(__file__).with_name("run_qe_atlas_experiment.py"),
            Path(__file__).with_name("run_qe_paired_sample_experiment.py"),
        ):
            dependencies[str(p)] = sha(p)
        old = json.loads((PREVIOUS / "protocol.json").read_text())
        source = json.loads((POOL / "protocol.json").read_text())
        protocol = dict(
            name="020-additive-paired-profile-supported-discovery",
            baseline=str(BASE),
            previous=str(PREVIOUS),
            folds=5,
            inner_folds=3,
            numerical_workers=3,
            production_enabled=False,
            curation_runtime_input=False,
            candidate_universe="Native monoisotopic CDB targets inside SAME003 OTHER-group envelope(>=3groups),same chemical adduct/charge definitions. OTHER-stream BASELINE automatic exact class/CDB support .9confidence/<1.5ppm. OTHER-group atlas>=3cohorts/MAD<=10s; ORIGINAL raw<3ppm/<=30s. Keep ALL alternatives for identities having at least one query absentfrom007. No measured coordinate edits or NP/RP RT transfer. Actual declared-spike masses exclude entire target within3ppm.",
            evidence="Same frozen BASELINE automatic anchors,validated explicit specimen pairs and8 paired descriptors as019; raw charge-aware peak_evidence,sample roles,run_context,composition priors and007 group-profile/phase-RT descriptors. No novel-to007/old-acceptance/feature/dataset IDs as predictors. Blind proposals/evidence frozen BEFORE dataset manual labels are attached.",
            completion_policy="Every009 call immutable. Winner must be best BOTH identity and feature withscore margin>=.1 against ALL raw alternatives BEFORE novelty/profile/occupancy filtering. New winner absentfrom007,>=8 positive paired biological samples and rawlog correlation>=.8,score>=training cutoff. Existing identity or occupied winning feature withholds; no weaker-free-peak fallback or cascades. No standard/manual edits.",
            classifier=source["classifier"],
            threshold_grid=[round(v, 2) for v in np.arange(0.5, 1.001, 0.05)],
            margin=0.1,
            extra_budget_per_endogenous_manual=0.003,
            minimum_training_correct_fraction=0.8,
            threshold_policy="Single cutoff chosen from NEW three INNER group-excluded discovery score fits per outer fold. Maximize additional SAME-ORIGINAL-feature training corrects within .3% endogenous-manual additional extras budget and >=80% correct/new-call fraction; ties fewer wrong peaks/extras,higher cutoff. None feasible=>withhold at1. No evaluated outer labels choose acceptance.",
            calibration_old_policy="Training occupancy/identities reconstructed from008 INNER OOF007 score models matching SAMEouterfold and009 outer-training cutoffs,plus BASELINE protected standards. Never use published training-dataset009 outputs whose different outer model could contain evaluated-project labels. Actual spike-mass gate protects all declared standards. Test uses complete published009 including005 standard adjudication. Conditional inherited old-policy thresholds/RT features,not fully nested entire pipeline.",
            validation=old["validation"],
            model_fit_note=old["model_fit_note"]
            + " NEW discovery final/inner fits group-excluded. Inner old-policy scores also exclude validation/outer groups. Inherited outer RT/envelope/prior/old-policy thresholds are NOT inner cross-fitted. Baseline known prior exposure; not external validation,probability calibration or FDR.",
            run_kind="Saved009 IMMUTABLE calls plus newly learned paired-profile-supported ORIGINAL-feature discovery inside native3ppm/30s bounds. Conditional grouped saved-input replay,not fresh production/raw-input annotation.",
            source_snapshot_sha256=hashes,
            dependencies_sha256=dependencies,
        )
        dump(OUT / "protocol.json", protocol)
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        for name in (
            "run_qe_dual_experiment.py",
            "run_qe_atlas_experiment.py",
            "run_qe_paired_sample_experiment.py",
        ):
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
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    from matlab_annotation.catalogue_targets import catalogue_missing_targets

    for f in range(5):
        chemical = json.loads(
            (QE / "experiment_003" / f"fold_{f}_composition_envelope.json").read_text()
        )
        folder = OUT / "theory" / f"fold_{f}"
        folder.mkdir(parents=True, exist_ok=True)
        for stream in ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"):
            dest = folder / (stream + ".pkl")
            if dest.exists():
                continue
            targets, audit = catalogue_missing_targets(
                pd.DataFrame(columns=["name", "adduct"]),
                OUT / "source_snapshot/reference_data/matlab_pipeline",
                chemical["envelope"],
                stream=stream,
                declared_standards=pd.DataFrame(
                    columns=["adduct", "ref_mz", "internal_standard_quantity"]
                ),
            )
            pd.to_pickle(targets, dest)
            audit.to_csv(folder / (stream + "_enumeration_audit.csv"), index=False)
        print("FROZEN NATIVE THEORY", f, flush=True)
    return protocol, projects


def prepare_dataset(dataset):
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    from scipy.io import loadmat
    from annotator.peak_evidence import peak_evidence
    from matlab_annotation.chemical_targets import _charge
    from matlab_annotation.retention_atlas import predict_aligned_retention_atlas
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.specimen_pairs import explicit_specimen_pairs
    from matlab_annotation.cross_mode_evidence import cross_mode_evidence, _strong, _key
    from matlab_annotation.paired_sample_evidence import paired_sample_evidence
    from matlab_annotation.paired_discovery import paired_discovery_candidates
    from matlab_annotation.run_context_evidence import run_context_evidence
    from matlab_annotation.class_candidates import (
        candidate_evidence,
        attach_composition_prior_evidence,
    )

    if all(
        (OUT / "prepared" / f"fold_{f}" / (dataset + ".pkl")).exists() for f in range(5)
    ):
        return dataset, "already prepared"
    result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
    features = {s: p["features"] for s, p in result["streams"].items()}
    automatic = pd.concat(
        [
            p["annotations"].copy().assign(stream=s)
            for s, p in result["streams"].items()
        ],
        ignore_index=True,
    )
    strong = _strong(automatic)
    catalogue = declared_standard_targets(
        OUT / "source_snapshot/reference_data/matlab_pipeline",
        result["internal_standards_expected"],
    )
    metadata_path = SOURCE / dataset / "MS_metadata.mat"
    digest = sha(metadata_path)
    metadata = loadmat(
        metadata_path, variable_names=["MS_metadata"], simplify_cells=True
    )["MS_metadata"]
    try:
        mapping, pairing_audit = explicit_specimen_pairs(metadata, features)
        error = ""
    except ValueError as exc:
        mapping, pairing_audit = {}, {}
        error = str(exc)
    assert sha(metadata_path) == digest
    all_pools = {}
    for fold in range(5):
        atlas = json.loads((ATLAS / f"fold_{fold}_atlas.json").read_text())["atlas"]
        old = pd.read_pickle(POOL / "prepared" / f"fold_{fold}" / (dataset + ".pkl"))[
            "pool"
        ]
        parts = []
        for stream, payload in result["streams"].items():
            targets = pd.read_pickle(
                OUT / "theory" / f"fold_{fold}" / (stream + ".pkl")
            )
            supported = (
                set(_key(strong[strong.stream.ne(stream)])) if len(strong) else set()
            )
            targets = targets.loc[_key(targets).isin(supported)].reset_index(drop=True)
            predictions = predict_aligned_retention_atlas(
                targets, payload["annotations"], atlas, stream=stream
            )
            targets = targets.join(predictions)
            proposed = paired_discovery_candidates(
                targets,
                payload["features"],
                automatic,
                old[old.stream.eq(stream)],
                stream=stream,
                declared_standard_masses=catalogue.loc[
                    catalogue.stream.eq(stream), "ref_mz"
                ],
            )
            if len(proposed):
                parts.append(proposed)
        assert parts, (dataset, fold, "No physical queries; inspect explicitly")
        all_pools[fold] = pd.concat(parts, ignore_index=True)
    keys = [
        "stream",
        "class",
        "carbon",
        "db",
        "feat_feature_id",
        "feat_rt",
        "feat_intensity",
        "adduct",
    ]
    queries = (
        pd.concat([p[keys] for p in all_pools.values()], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    extra = paired_sample_evidence(queries, features, automatic, mapping)
    cross = cross_mode_evidence(queries, features, automatic)
    extra = pd.concat([extra, cross], axis=1)
    for stream, part in queries.groupby("stream", sort=False):
        context = run_context_evidence(part, result["streams"][stream]["annotations"])
        for c in context:
            extra.loc[part.index, c] = context[c]
        charges = part.adduct.map(_charge)
        for charge, indices in charges.groupby(charges).groups.items():
            measured = peak_evidence(
                features[stream],
                part.loc[indices, "feat_feature_id"],
                polarity=stream.split("_")[1],
                charge=int(charge),
            ).set_index("feature_id")
            aligned = measured.reindex(part.loc[indices, "feat_feature_id"].astype(str))
            for c in [x for x in measured if x.startswith("q_")]:
                extra.loc[indices, "q_raw_" + c.removeprefix("q_")] = aligned[
                    c
                ].to_numpy()
    extra.index = pd.MultiIndex.from_frame(queries[keys])
    assert extra.index.is_unique
    blind = {}
    for fold, pool in all_pools.items():
        aligned = extra.reindex(pd.MultiIndex.from_frame(pool[keys]))
        for c in extra:
            pool[c] = aligned[c].to_numpy()
        priors = json.loads(
            (QE / "experiment_001" / f"fold_{fold}_selection_model.json").read_text()
        )["composition_priors"]
        evidence = attach_composition_prior_evidence(candidate_evidence(pool), priors)
        assert not {
            "name",
            "feat_feature_id",
            "dataset",
            "cohort",
            "training_positive",
            "training_label_known",
            "discovery_novel_to007",
        } & set(evidence)
        folder = OUT / "blind" / f"fold_{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (dataset + ".pkl")
        pd.to_pickle(dict(pool=pool, evidence=evidence), path)
        blind[str(path)] = sha(path)
    audit = OUT / "discovery_audits"
    audit.mkdir(exist_ok=True)
    dump(
        audit / (dataset + ".json"),
        dict(
            dataset=dataset,
            metadata_sha256=digest,
            pairing_counts=pairing_audit,
            pairing_error=error,
            unique_queries=len(queries),
            paired_queries=int(extra.q_paired_best_log_correlation.notna().sum()),
            novel_by_fold={
                str(f): int(p.discovery_novel_to007.sum()) for f, p in all_pools.items()
            },
            blind_sha256=blind,
            note="Physical proposals/evidence saved BEFORE the following manual-label read. No curation enters discovery/profile evidence.",
        ),
    )
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
    for fold in range(5):
        path = OUT / "blind" / f"fold_{fold}" / (dataset + ".pkl")
        assert sha(path) == blind[str(path)]
        data = pd.read_pickle(path)
        pool = data["pool"]
        labels = [
            (s, species(n), adduct(a), str(fid))
            for s, n, a, fid in pool[
                ["stream", "name", "adduct", "feat_feature_id"]
            ].itertuples(index=False, name=None)
        ]
        pool["training_positive"] = [k in exact for k in labels]
        pool["training_label_known"] = [
            k[:3] not in unresolved or k in exact for k in labels
        ]
        folder = OUT / "prepared" / f"fold_{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(data, folder / (dataset + ".pkl"))
    return dataset, dict(
        queries=len(queries),
        paired=int(extra.q_paired_best_log_correlation.notna().sum()),
        novel=int(all_pools[0].discovery_novel_to007.sum()),
    )


def calibration_old_calls(dataset, fold, inner_model):
    from matlab_annotation.operating_point import select_operating_candidates

    old = pd.read_pickle(POOL / "prepared" / f"fold_{fold}" / (dataset + ".pkl"))
    scores = pd.Series(
        inner_model["estimator"].predict_proba(
            model_matrix(old["evidence"], inner_model)
        )[:, 1],
        index=old["pool"].index,
    )
    thresholds = json.loads(
        (
            PREVIOUS / "calibration" / f"fold_{fold}" / "operating_points.json"
        ).read_text()
    )["thresholds"]
    parts = []
    for stream, part in old["pool"].groupby("stream", sort=False):
        parts.append(
            select_operating_candidates(
                part,
                scores.loc[part.index],
                threshold=thresholds[stream],
                ambiguity_margin=0.05,
            )
        )
    result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
    parts.extend(
        p["annotations"][p["annotations"].is_is.fillna(False).astype(bool)]
        .copy()
        .assign(stream=s)
        for s, p in result["streams"].items()
    )
    return pd.concat(parts, ignore_index=True)


def calibrate(protocol, projects, folds, fold, data):
    from sklearn.model_selection import GroupKFold
    from matlab_annotation.paired_discovery import select_paired_additions

    folder = OUT / "calibration" / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "operating_points.json"
    train = [d for d in projects if folds[d] != fold]
    excluded = {d[:7] for d in projects if folds[d] == fold}
    if path.exists():
        saved = json.loads(path.read_text())
        assert set(saved["outer_excluded"]) == excluded
        assert set(saved["training_groups"]) == {d[:7] for d in train}
        return saved["threshold"]
    predictions, old_calls = {}, {}
    for inner, (ti, vi) in enumerate(
        GroupKFold(n_splits=3).split(train, groups=[d[:7] for d in train])
    ):
        fitting = [train[i] for i in ti]
        validation = [train[i] for i in vi]
        model = fit_score_model(
            data,
            fitting,
            excluded | {d[:7] for d in validation},
            protocol,
            folder / f"inner_{inner}_discovery_model.pkl",
        )
        old_model = pd.read_pickle(
            QE
            / "experiment_008/calibration"
            / f"fold_{fold}"
            / f"inner_{inner}_score_model.pkl"
        )
        assert set(old_model["training_cohorts"]) == {d[:7] for d in fitting}
        assert set(old_model["calibration_groups"]) == {d[:7] for d in validation}
        assert not set(old_model["training_cohorts"]) & (
            excluded | {d[:7] for d in validation}
        )
        for d in validation:
            predictions[d] = pd.Series(
                model["estimator"].predict_proba(
                    model_matrix(data[d]["evidence"], model)
                )[:, 1],
                index=data[d]["pool"].index,
            )
            old_calls[d] = calibration_old_calls(d, fold, old_model)
    assert set(predictions) == set(train)
    manual_count = 0
    cases = {}
    for d in train:
        manual = pd.read_pickle(BASE / "evaluation_v2" / d / "manual_comparison.pkl")
        manual = manual[~manual.internal_standard]
        manual_count += len(manual)
        identities = set(
            manual[["stream", "species", "adduct"]].itertuples(index=False, name=None)
        )
        pool = data[d]["pool"].copy()
        pool["calibration_identity_in_manual"] = [
            (s, species(n), adduct(a)) in identities
            for s, n, a in pool[["stream", "name", "adduct"]].itertuples(
                index=False, name=None
            )
        ]
        cases[d] = pool
    records = []
    for threshold in protocol["threshold_grid"]:
        counts = Counter()
        for d, pool in cases.items():
            for stream, part in pool.groupby("stream", sort=False):
                chosen = select_paired_additions(
                    part,
                    predictions[d].loc[part.index],
                    old_calls[d][old_calls[d].stream.eq(stream)],
                    threshold=threshold,
                    margin=protocol["margin"],
                )
                counts["correct"] += int(chosen.training_positive.sum())
                counts["calls"] += len(chosen)
                counts["extras"] += int((~chosen.calibration_identity_in_manual).sum())
                counts["wrong_peak"] += int(
                    (
                        chosen.calibration_identity_in_manual
                        & chosen.training_label_known
                        & ~chosen.training_positive
                    ).sum()
                )
        records.append(dict(threshold=threshold, manual=manual_count, **dict(counts)))
    table = pd.DataFrame(records).fillna(0)
    table.to_csv(folder / "training_operating_points.csv", index=False)
    feasible = table[
        table.extras.le(protocol["extra_budget_per_endogenous_manual"] * table.manual)
        & table.correct.ge(protocol["minimum_training_correct_fraction"] * table.calls)
    ]
    assert len(feasible), "No feasible cutoff; inspect rather than exceeding budget"
    selected = feasible.sort_values(
        ["correct", "wrong_peak", "extras", "threshold"],
        ascending=[False, True, True, False],
    ).iloc[0]
    threshold = float(selected.threshold)
    dump(
        path,
        dict(
            threshold=threshold,
            training_groups=sorted({d[:7] for d in train}),
            outer_excluded=sorted(excluded),
            selected_counts=selected.to_dict(),
            note=protocol["calibration_old_policy"] + " " + protocol["model_fit_note"],
        ),
    )
    print(
        "DISCOVERY TRAINING-ONLY CUTOFF",
        fold,
        threshold,
        selected.to_dict(),
        flush=True,
    )
    return threshold


def run_fold(fold):
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    from matlab_annotation.paired_discovery import select_paired_additions

    protocol = json.loads((OUT / "protocol.json").read_text())
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    heldout = [d for d in projects if folds[d] == fold]
    if all((OUT / "runs" / d / "complete.json").exists() for d in heldout):
        return fold, "already complete"
    data = {
        d: pd.read_pickle(OUT / "prepared" / f"fold_{fold}" / (d + ".pkl"))
        for d in projects
    }
    threshold = calibrate(protocol, projects, folds, fold, data)
    model = fit_score_model(
        data,
        [d for d in projects if folds[d] != fold],
        {d[:7] for d in heldout},
        protocol,
        OUT / f"fold_{fold}_discovery_model.pkl",
    )
    for dataset in heldout:
        dest = OUT / "runs" / dataset
        if (dest / "complete.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        raw, evidence = data[dataset]["pool"], data[dataset]["evidence"]
        scores = pd.Series(
            model["estimator"].predict_proba(model_matrix(evidence, model))[:, 1],
            index=raw.index,
        )
        before = pd.read_csv(
            PREVIOUS / "runs" / dataset / "automatic_annotations.csv",
            dtype={"feat_feature_id": str},
            low_memory=False,
        )
        additions = []
        for stream, part in raw.groupby("stream", sort=False):
            chosen = select_paired_additions(
                part,
                scores.loc[part.index],
                before[before.stream.eq(stream)],
                threshold=threshold,
                margin=protocol["margin"],
            )
            chosen = chosen.drop(columns=["training_positive", "training_label_known"])
            if len(chosen):
                additions.append(chosen)
        after = pd.concat([before, *additions], ignore_index=True)
        pd.testing.assert_frame_equal(
            after.iloc[: len(before)][before.columns], before, check_dtype=False
        )
        assert (
            not after.assign(feat_feature_id=after.feat_feature_id.astype(str))
            .duplicated(["stream", "feat_feature_id"])
            .any()
        )
        old = pd.read_pickle(PREVIOUS / "runs" / dataset / "candidates.pkl")
        candidate = raw.drop(
            columns=["training_positive", "training_label_known"]
        ).assign(class_selection_score=scores)
        combined = pd.concat([old, candidate], ignore_index=True)
        keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
        combined = combined.drop_duplicates(keys, keep="first").reset_index(drop=True)
        accepted = set(after[keys].itertuples(index=False, name=None))
        combined["automatic_accepted"] = [
            k in accepted for k in combined[keys].itertuples(index=False, name=None)
        ]
        after.to_csv(dest / "automatic_annotations.csv", index=False)
        combined.to_pickle(dest / "candidates.pkl")
        scores.to_pickle(dest / "discovery_scoring_audit.pkl")
        dump(
            dest / "complete.json",
            dict(
                dataset=dataset,
                fold=fold,
                status="complete",
                production_enabled=False,
                threshold=threshold,
                annotations_before=len(before),
                annotations_after=len(after),
                added=len(after) - len(before),
                counts=after.groupby("stream").size().to_dict(),
            ),
        )
        print(
            "COMPLETED ADDITIVE DISCOVERY",
            dataset,
            "added",
            len(after) - len(before),
            flush=True,
        )
    return fold, "complete"


def main():
    protocol, projects = initialize()
    with ProcessPoolExecutor(max_workers=protocol["numerical_workers"]) as executor:
        for future in as_completed(
            [executor.submit(prepare_dataset, d) for d in projects]
        ):
            print("PREPARED DISCOVERY", *future.result(), flush=True)
    with ProcessPoolExecutor(max_workers=protocol["numerical_workers"]) as executor:
        for future in as_completed([executor.submit(run_fold, f) for f in range(5)]):
            print("DISCOVERY FOLD", *future.result(), flush=True)
    print(
        "EXPERIMENT020 COMPLETE; FULL132-ROW TABLE REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
