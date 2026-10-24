"""Experiment030: freshly nested, additive NP BMP specialist. Core029 immutable."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from context import WORKSPACE, REPO
from frozen_resources import configure_frozen_resources
from run_qe_atlas_experiment import species, adduct

QE = WORKSPACE / "outputs/qe_optimization_20261022"
BASE = QE / "baseline"
PREVIOUS = QE / "experiment_029"
PREPARED = QE / "standard_context_preparation_v1"
OUT = QE / "experiment_030_scoped_v2"


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


def load_predictors(dataset, mask):
    data = pd.read_pickle(prepared_path(dataset, mask))
    assert data["query_dataset"] == dataset and data["query_group"] == dataset[:7]
    assert set(data["excluded_folds"]) == set(mask)
    assert not {
        "training_positive",
        "training_label_known",
        "same_manual_peak",
        "manual_feature_id",
    } & set(data["pool"])
    assert set(data["standard_reference_training_groups"]) == set(
        data["reference_training_groups"]
    )
    assert set(data["standard_reference_excluded_groups"]) == set(
        data["reference_excluded_groups"]
    )
    return data


def label_allowed_query(data, training, excluded, *, partial):
    """Authorization checked BEFORE reading curation; NEVER used on final query."""
    from matlab_annotation.partial_curation_training import (
        mask_uncurated_opposite_np_bmp,
    )

    assert data["query_group"] in training and data["query_group"] not in excluded
    manual = pd.read_pickle(
        BASE / "evaluation_v2" / data["query_dataset"] / "manual_comparison.pkl"
    )
    manual = manual.loc[~manual.internal_standard].copy()
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
    pool = data["pool"].copy()
    keys = [
        (s, species(n), adduct(a), str(fid))
        for s, n, a, fid in pool[
            ["stream", "name", "adduct", "feat_feature_id"]
        ].itertuples(index=False, name=None)
    ]
    pool["training_positive"] = [k in exact for k in keys]
    pool["training_label_known"] = [k[:3] not in unresolved or k in exact for k in keys]
    pool["calibration_identity_in_manual"] = [k[:3] in identities for k in keys]
    audit = None
    if partial:
        pool, audit = mask_uncurated_opposite_np_bmp(
            pool,
            manual,
            query_group=data["query_group"],
            permitted_groups=training,
            excluded_groups=excluded,
        )
    return dict(data, pool=pool), manual, audit


def initialize():
    assert json.loads((PREVIOUS / "integrity_receipt.json").read_text())["all_passed"]
    assert (
        json.loads((PREVIOUS / "reports_v2/summary.json").read_text())[
            "datasets_evaluated"
        ]
        == 33
    )
    projects = json.loads((PREVIOUS / "run_manifest.json").read_text())["datasets"]
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
            PREVIOUS / "protocol.json",
            PREVIOUS / "integrity_receipt.json",
            PREPARED / "protocol.json",
            PREPARED / "summary.json",
            QE / "rt_atlas_development/fold_assignment.json",
            Path(__file__).with_name("run_qe_atlas_experiment.py"),
            Path(__file__).with_name("frozen_resources.py"),
        ]
        for dataset in projects:
            receipt = PREPARED / "checkpoints" / (dataset + ".json")
            inputs.append(receipt)
            for p, digest in json.loads(receipt.read_text())["prepared_sha256"].items():
                assert sha(Path(p)) == digest
                inputs.append(Path(p))
            inputs.extend(
                [
                    BASE / "evaluation_v2" / dataset / "manual_comparison.pkl",
                    BASE / "runs" / dataset / "full_annotation.pkl",
                    PREVIOUS / "runs" / dataset / "automatic_annotations.csv",
                    PREVIOUS / "runs" / dataset / "candidates.pkl",
                ]
            )
        for f in range(5):
            folder = PREVIOUS / "calibration" / f"fold_{f}"
            inputs.append(folder / "operating_points.json")
            for d in projects:
                p = folder / (d + "_blind_scores.pkl")
                if p.exists():
                    inputs.append(p)
        dump(
            OUT / "protocol.json",
            dict(
                name="030-additive-newly-nested-NP-BMP-specialist-explicit-supervision-scope",
                baseline=str(BASE),
                previous=str(PREVIOUS),
                production_enabled=False,
                curation_runtime_input=False,
                folds=5,
                inner_folds=4,
                numerical_workers=3,
                classifier=dict(
                    max_iter=180,
                    max_leaf_nodes=15,
                    min_samples_leaf=10,
                    l2_regularization=2,
                    learning_rate=0.08,
                    early_stopping=False,
                    random_state=20261024,
                ),
                threshold_grid=[round(v, 2) for v in np.arange(0.5, 1.001, 0.05)],
                additional_extras_per_endogenous_np_bmp_manual=0.10,
                additional_different_features_per_np_mode_manual=0.02,
                candidate_policy="EXACT029 strict physical NP endogenous BMP candidates; no new "
                "discovery/mass/RT/range windows. 029core calls ALL immutable including standards "
                "and fixed027mass additions. Unique feature/identity winners fixed BEFORE "
                "ambiguity and occupancy; no alternative promotion or replacements.",
                supervision="Fresh specialist fits, NOT copied024predictions. Only permitted "
                "training-group absent-opposite-NP BMP negatives become unknown, never positive. "
                "No manual-presence/polarity flags enter predictors or final query. All other "
                "labels immutable. Same15nested independent STD context and original predictors. "
                "One shared NP BMP model per nested fit, two fixed acquisition categories.",
                threshold_policy="Four proper INNER project-held-out specialist predictions. "
                "NPmode cutoffs .5..1 jointly maximize minimum total core+rescue BMP coverage "
                "capped90, then total correct; fewer added wrongfeatures/extras as tiebreaks. "
                "Additional extras pooled <=10% of training NP BMP manual rows; added wrong "
                "features <=2% in EACH NP mode. Calibration truth/extras unchanged, including "
                "uncurated oppositeNP calls. Threshold1 is a no-addition fallback, no relaxation.",
                inner_core_occupancy="Unchanged029 INNER OOF core scores/mode cutoffs plus blind "
                "baseline automatic standards. Do NOT use outer-only005cleanup or027mass "
                "references in inner calibration. Final immutable029full pipeline occupancy "
                "may differ after standard cleanup/massstage. Training budgets NOT testbounds.",
                validation="Same33datasets/30projectgroups and same5outer/fourinner blocks. "
                "References exclude ownquery,outer andinnervalidation groups. Cached original "
                "feature replay,not fresh app runs. Priorbaseline/vendor/development exposure "
                "remains. Extras unconfirmed,not FDR; all92102manual rows incl28unresolved.",
                run_kind="Cached029immutable annotations plus freshly nested/OOFcalibrated "
                "NP BMP specialist additions. All classes/modes fully evaluated.",
                scope_implementation_amendment="Initial030stopped at the strict generic "
                "every-project-has-known-rows guard (13/33dataset outputs,16fits). Preserved "
                "unchanged outsideGit. Restart ALL25fits in this new directory. Same scientific "
                "rescue strategy/labels/predictors/budgets;explicitly distinguish full allowed "
                "reference projects from projects contributing known BMP rows. ALL allowed "
                "queries represented/validated,zero-label projects recorded,not silently "
                "dropped;no invented labels or weaker query/test exclusions.",
                model_fit_note="25new180round pooledNP BMP fits,not phase/class cherry-picking "
                "among outer-tested oldmodels. minleaf10/maxleaf15 fixed before030evaluation. "
                "Conditional inherited core calibration limitations are explicit.",
                resource_receipt=configure_frozen_resources(snapshot),
                source_snapshot_sha256={
                    p.relative_to(snapshot).as_posix(): sha(p)
                    for p in snapshot.rglob("*")
                    if p.is_file()
                },
                input_sha256={str(p): sha(p) for p in inputs},
            ),
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        dump(
            OUT / "run_manifest.json",
            dict(
                datasets=projects,
                fresh_raw_input_runs=False,
                evaluation_note="Cached immutable029core plus new nested/OOF NP BMP rescue; "
                "same92102manual reference. Known development/vendor exposure,not untouched externalvalidation/FDR.",
            ),
        )
    protocol = json.loads((OUT / "protocol.json").read_text())
    for p, digest in protocol["source_snapshot_sha256"].items():
        assert sha(OUT / "source_snapshot" / p) == digest
    for p, digest in protocol["input_sha256"].items():
        assert sha(Path(p)) == digest
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    assert (
        configure_frozen_resources(OUT / "source_snapshot")
        == protocol["resource_receipt"]
    )
    return protocol, projects


def fit_score(datasets, training, excluded, parameters, path):
    from matlab_annotation.bmp_rescue import bmp_queries
    from matlab_annotation.class_specialist_training import (
        fit_class_specialist as fit_grouped_evidence_model,
    )

    if path.exists():
        model = pd.read_pickle(path)
        assert set(model["training_cohorts"]) == training
        assert set(model["excluded_groups"]) == excluded
        return model
    fitting, audits = {}, {}
    for d, raw in datasets.items():
        # Unknown labels are restricted to training, never calibration truth.
        labeled, _, audit = label_allowed_query(raw, training, excluded, partial=True)
        fitting[d] = bmp_queries(labeled)
        audits[d] = audit
    model = fit_grouped_evidence_model(
        fitting,
        training_groups=training,
        excluded_groups=excluded,
        parameters=parameters,
    )
    model["training_feature_masks"] = {
        d: data["excluded_folds"] for d, data in fitting.items()
    }
    model["partial_supervision_audit"] = audits
    model["training_feature_scopes"] = {
        d: dict(
            training=data["reference_training_groups"],
            excluded=data["reference_excluded_groups"],
            standard_training=data["standard_reference_training_groups"],
            standard_excluded=data["standard_reference_excluded_groups"],
        )
        for d, data in fitting.items()
    }
    model["fit_class"] = "BMP"
    assert set(model["categories"]["stream"]) <= {"NP_Pos", "NP_Neg"}
    pd.to_pickle(model, path)
    print(
        "FIT NEW BMP SPECIALIST",
        path.name,
        model["training_pairs"],
        model["positives"],
        flush=True,
    )
    return model


def calibrate(protocol, projects, folds, outer):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.bmp_rescue import bmp_queries, select_bmp_rescue, NP_MODES
    from matlab_annotation.evidence_training import predict_grouped_evidence
    from matlab_annotation.operating_point import (
        select_operating_candidates,
        choose_balanced_training_points,
    )

    folder = OUT / "calibration" / f"fold_{outer}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "operating_points.json"
    training = {d[:7] for d in projects if folds[d] != outer}
    excluded = {d[:7] for d in projects if folds[d] == outer}
    if path.exists():
        result = json.loads(path.read_text())
        assert (
            set(result["training_groups"]) == training
            and set(result["outer_excluded"]) == excluded
        )
        return result["thresholds"]
    cases, manual_counts, base_correct = [], Counter(), Counter()
    core_folder = PREVIOUS / "calibration" / f"fold_{outer}"
    core_cutoffs = json.loads((core_folder / "operating_points.json").read_text())[
        "thresholds"
    ]
    for inner in sorted(set(folds.values()) - {outer}):
        groups = {d[:7] for d in projects if folds[d] not in {outer, inner}}
        blocked = {d[:7] for d in projects if folds[d] in {outer, inner}}
        data = {
            d: load_predictors(d, {outer, inner, folds[d]})
            for d in projects
            if folds[d] not in {outer, inner}
        }
        model = fit_score(
            data,
            groups,
            blocked,
            protocol["classifier"],
            folder / f"inner_block_{inner}_model.pkl",
        )
        del data
        for d in projects:
            if folds[d] != inner:
                continue
            raw = load_predictors(d, {outer, inner})
            query = bmp_queries(raw)
            scores = predict_grouped_evidence(query, model)
            scores.to_pickle(folder / (d + "_blind_scores.pkl"))
            labeled, manual, _ = label_allowed_query(
                raw, training, excluded, partial=False
            )
            pool = bmp_queries(labeled)["pool"]
            manual_counts.update(
                manual.loc[
                    manual.lipid_class.map(canonical_lipid_class).eq("BMP")
                    & manual.stream.isin(NP_MODES),
                    "stream",
                ]
                .value_counts()
                .to_dict()
            )
            core_scores = pd.read_pickle(core_folder / (d + "_blind_scores.pkl"))
            assert core_scores.index.equals(raw["pool"].index)
            baseline = pd.read_pickle(BASE / "runs" / d / "full_annotation.pkl")
            for stream in NP_MODES:
                all_core = labeled["pool"].loc[labeled["pool"].stream.eq(stream)]
                core = select_operating_candidates(
                    all_core,
                    core_scores.loc[all_core.index],
                    threshold=core_cutoffs[stream],
                    ambiguity_margin=0.05,
                )
                base_correct[stream] += int(
                    core.loc[
                        core["class"].map(canonical_lipid_class).eq("BMP"),
                        "training_positive",
                    ].sum()
                )
                protected = baseline["streams"][stream]["annotations"]
                protected = protected.loc[
                    protected.is_is.fillna(False).astype(bool)
                ].assign(stream=stream)
                occupancy = pd.concat([core, protected], ignore_index=True)
                part = pool.loc[pool.stream.eq(stream)]
                cases.append((stream, part, scores.loc[part.index], occupancy))
    records = []
    for threshold in protocol["threshold_grid"]:
        counts = {s: Counter() for s in NP_MODES}
        for stream, part, scores, occupancy in cases:
            chosen = select_bmp_rescue(
                part, scores, occupancy, stream=stream, threshold=threshold
            )
            counts[stream]["added_correct"] += int(chosen.training_positive.sum())
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
        for stream in NP_MODES:
            records.append(
                dict(
                    stream=stream,
                    threshold=threshold,
                    manual=manual_counts[stream],
                    core_correct=base_correct[stream],
                    correct=base_correct[stream] + counts[stream]["added_correct"],
                    **dict(counts[stream]),
                )
            )
    table = pd.DataFrame(records)
    table.to_csv(folder / "training_operating_points.csv", index=False)
    feasible = table.loc[
        table.different_peak.le(
            protocol["additional_different_features_per_np_mode_manual"] * table.manual
        )
    ]
    thresholds = choose_balanced_training_points(
        feasible,
        extras_per_manual=protocol["additional_extras_per_endogenous_np_bmp_manual"],
        target=0.9,
    )
    assert thresholds is not None, "No feasible point; do not relax budget"
    chosen = table.loc[
        [
            t == thresholds[s]
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
            training_groups=sorted(training),
            outer_excluded=sorted(excluded),
            training_table_sha256=sha(folder / "training_operating_points.csv"),
            note=protocol["threshold_policy"],
        ),
    )
    print("BMP INNER CUTOFFS", outer, thresholds, flush=True)
    return thresholds


def run_fold(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.bmp_rescue import bmp_queries, select_bmp_rescue, NP_MODES
    from matlab_annotation.evidence_training import predict_grouped_evidence

    protocol = json.loads((OUT / "protocol.json").read_text())
    projects = json.loads((OUT / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    heldout = [d for d in projects if folds[d] == outer]
    if all((OUT / "runs" / d / "complete.json").exists() for d in heldout):
        return outer
    thresholds = calibrate(protocol, projects, folds, outer)
    training = {d[:7] for d in projects if folds[d] != outer}
    excluded = {d[:7] for d in heldout}
    data = {
        d: load_predictors(d, {outer, folds[d]}) for d in projects if folds[d] != outer
    }
    model = fit_score(
        data,
        training,
        excluded,
        protocol["classifier"],
        OUT / f"fold_{outer}_nested_model.pkl",
    )
    del data
    for d in heldout:
        dest = OUT / "runs" / d
        if (dest / "complete.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        raw = bmp_queries(load_predictors(d, {outer}))
        scores = predict_grouped_evidence(raw, model)
        # NO curation file is opened for the final query.
        previous = pd.read_csv(
            PREVIOUS / "runs" / d / "automatic_annotations.csv",
            dtype={"feat_feature_id": str},
            low_memory=False,
        )
        additions = []
        for stream in NP_MODES:
            part = raw["pool"].loc[raw["pool"].stream.eq(stream)]
            additions.append(
                select_bmp_rescue(
                    part,
                    scores.loc[part.index],
                    previous.loc[previous.stream.eq(stream)],
                    stream=stream,
                    threshold=thresholds[stream],
                )
            )
        added = pd.concat(additions, ignore_index=True)
        annotations = pd.concat([previous, added], ignore_index=True)
        assert not annotations.duplicated(["stream", "feat_feature_id"]).any()
        candidate = pd.read_pickle(PREVIOUS / "runs" / d / "candidates.pkl")
        keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
        accepted = set(annotations[keys].itertuples(index=False, name=None))
        candidate["automatic_accepted"] = [
            k in accepted for k in candidate[keys].itertuples(index=False, name=None)
        ]
        annotations.to_csv(dest / "automatic_annotations.csv", index=False)
        candidate.to_pickle(dest / "candidates.pkl")
        scores.to_pickle(dest / "scoring_audit.pkl")
        dump(
            dest / "complete.json",
            dict(
                dataset=d,
                fold=outer,
                status="complete",
                production_enabled=False,
                previous_calls_immutable=True,
                added_annotations=len(added),
                thresholds=thresholds,
                counts=annotations.groupby("stream").size().to_dict(),
            ),
        )
        print("HELDOUT BMP ADDITIONS", d, len(added), flush=True)
    return outer


def main():
    protocol, _ = initialize()
    with ProcessPoolExecutor(max_workers=protocol["numerical_workers"]) as executor:
        for f in executor.map(run_fold, range(5)):
            print("BMP FOLD COMPLETE", f, flush=True)
    print("030 COMPLETE; FULL132-ROW TABLE REQUIRED BEFORE NEXT EXPERIMENT", flush=True)


if __name__ == "__main__":
    main()
