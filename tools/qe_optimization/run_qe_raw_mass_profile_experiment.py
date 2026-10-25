"""043 fresh four-mode binary scoring with23 raw measurements, immutable041.

Cannot initialize without independentALL363construction, complete model/export
preflight, and a pre-existing full verifier. No production/default changes.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
from immutable_exports import assemble_immutable
import prepare_qe_raw_mass_profile_tables as preparation
import run_qe_add_only_class_experiment as inherited
import run_qe_feature_label_experiment as occupancy

QE, BASE, CORE = inherited.QE, inherited.BASE, inherited.OLD
PREVIOUS, OUT = QE / "experiment_041", QE / "experiment_043"
PREFLIGHT = QE / "raw_mass_profile_model_preflight_v1"
CONSTRUCTION = QE / "raw_mass_profile_independent_check_v1"
MODES = inherited.MODES
sha, dump, inventory = inherited.sha, inherited.dump, inherited.inventory
METHOD = "raw_mass_profile_conservative_completion"


def load_predictors(dataset, mask):
    raw = pd.read_pickle(preparation.prepared_path(dataset, mask))
    assert raw["query_dataset"] == dataset and set(raw["excluded_folds"]) == set(mask)
    assert len(raw["evidence"].columns) == 131
    assert raw["pool"].index.equals(raw["evidence"].index)
    assert not {
        "training_positive",
        "training_label_known",
        "same_manual_peak",
        "calibration_identity_in_manual",
    } & set(raw["pool"])
    provenance = raw["raw_mass_profile_provenance"]
    assert provenance["mask"] == sorted(mask)
    assert provenance["fitted_RT_links_used"] is False
    assert provenance["curation_runtime_input"] is False
    assert provenance["original_prepared_sha256"] == sha(
        inherited.common.prepared_path(dataset, mask)
    )
    return raw


def authorize_labels(raw, training, excluded):
    from matlab_annotation.evidence_training import validate_feature_scope

    validate_feature_scope(raw, training, excluded, query_is_training=True)
    return inherited.common.label_allowed_query(raw, training, excluded, partial=False)


def training_data(outer, inner, datasets, folds):
    blocked = {outer} if inner is None else {outer, inner}
    training = {d[:7] for d in datasets if folds[d] not in blocked}
    excluded = {d[:7] for d in datasets if folds[d] in blocked}
    fitting = {}
    for d in datasets:
        if folds[d] not in blocked:
            raw = load_predictors(d, blocked | {folds[d]})
            fitting[d], _, _ = authorize_labels(raw, training, excluded)
    return fitting, training, excluded


def parent_path(outer, inner):
    return OUT / (
        f"fold_{outer}_nested_model.pkl"
        if inner is None
        else f"calibration/fold_{outer}/inner_block_{inner}_model.pkl"
    )


def previous_annotations(dataset):
    return pd.read_csv(
        PREVIOUS / "runs" / dataset / "automatic_annotations.csv",
        dtype=str,
        keep_default_na=False,
    )


def assemble(dataset, previous, selected):
    filled = inherited.common.fill_original_features(dataset, selected)
    if len(filled):
        filled.rename(
            columns={"context_completion_score": "raw_mass_profile_agreement_rank"},
            inplace=True,
        )
        filled["auto_annotation_status"] = METHOD
    candidates = pd.read_pickle(PREVIOUS / "runs" / dataset / "candidates.pkl")
    annotations, candidates = assemble_immutable(
        previous, candidates, filled, method=METHOD
    )
    return annotations, candidates, filled


def proposals(dataset, mask, raw, scores, previous):
    assert scores.index.equals(raw["pool"].index)
    return inherited.proposals_for(dataset, mask, raw, scores, previous)


def inner_previous(dataset, outer, inner, labeled, old_scores):
    # Reuse ONLY the independently verified041INNER occupancy reconstruction.
    # No042 ranks, factorized models or annotation outputs enter this method.
    assert occupancy.CORE == CORE and occupancy.PREVIOUS == PREVIOUS
    return occupancy.inner_previous(dataset, outer, inner, labeled, old_scores)


def source_inputs():
    datasets, folds = inventory()
    construction = json.loads((CONSTRUCTION / "receipt.json").read_text())
    assert construction["all_passed"] and construction["unique_nested_tables"] == 363
    assert construction["total_predictors"] == 131
    assert construction["preparation_receipt_sha256"] == sha(
        preparation.OUT / "receipt.json"
    )
    built = json.loads((preparation.OUT / "receipt.json").read_text())
    assert built["all_passed"] and built["source_input_sha256"] == preparation.bindings(
        datasets, folds
    )
    assert all(sha(Path(p)) == h for p, h in built["artifact_sha256"].items())
    paths = set(map(Path, built["source_input_sha256"]))
    paths.update(map(Path, built["artifact_sha256"]))
    paths.update(
        [
            Path(__file__),
            Path(inherited.__file__),
            Path(inherited.common.__file__),
            Path(occupancy.__file__),
            Path(preparation.__file__),
            Path(__file__).with_name("immutable_exports.py"),
            Path(__file__).with_name("verify_immutable_exports.py"),
            Path(__file__).with_name("verify_qe_raw_mass_profile_tables.py"),
            Path(__file__).with_name("verify_qe_raw_mass_profile_experiment.py"),
            Path(__file__).with_name("preflight_qe_raw_mass_profile_models.py"),
            Path(__file__).with_name("postprocess_qe_raw_mass_profile.py"),
            Path(__file__).with_name("report_qe_endogenous_companion.py"),
            Path(__file__).with_name("evaluate_qe_baseline.py"),
            Path(__file__).with_name("report_qe_experiment.py"),
            Path(__file__).with_name("report_qe_priority_classes.py"),
            Path(__file__).with_name("report_qe_target_scope.py"),
            Path(__file__).with_name("frozen_resources.py"),
            Path(__file__).with_name("context.py"),
            CONSTRUCTION / "receipt.json",
            preparation.OUT / "receipt.json",
            PREVIOUS / "integrity_receipt.json",
            PREVIOUS / "protocol.json",
            PREVIOUS / "run_manifest.json",
            CORE / "protocol.json",
            CORE / "integrity_receipt.json",
            REPO / "docs/validation/next_raw_counterpart_mass_profile_completion.txt",
        ]
    )
    for d in datasets:
        paths.update(
            [
                PREVIOUS / "runs" / d / "automatic_annotations.csv",
                PREVIOUS / "runs" / d / "candidates.pkl",
                CORE / "runs" / d / "scoring_audit.pkl",
                BASE / "evaluation_v2" / d / "manual_comparison.pkl",
            ]
        )
    for root in (
        QE / "manual_reference_v2",
        PREVIOUS / "reports_v2",
        BASE / "reports_v2",
    ):
        paths.update(
            p
            for p in root.rglob("*")
            if p.is_file() and p.name != ".DS_Store" and "__pycache__" not in p.parts
        )
    for outer in range(5):
        folder = CORE / "calibration" / f"fold_{outer}"
        paths.update(
            [
                CORE / f"fold_{outer}_nested_model.pkl",
                folder / "operating_points.json",
                PREVIOUS / "calibration" / f"fold_{outer}/operating_points.json",
            ]
        )
        paths.update(
            folder / f"inner_block_{i}_model.pkl" for i in range(5) if i != outer
        )
        paths.update(folder.glob("*_blind_scores.pkl"))
    return {str(p): sha(p) for p in sorted(paths)}


def initialize():
    gate = json.loads((PREFLIGHT / "receipt.json").read_text())
    assert gate["all_passed"] and gate["classifier_fits"] == 0
    assert gate["parent_preparations"] == 25 and gate["training_query_uses"] == 528
    assert gate["disabled_full_041_datasets"] == 33
    assert gate["source_input_sha256"] == source_inputs()
    assert gate["parent_support_sha256"] == sha(PREFLIGHT / "parent_support.json")
    verifier = Path(__file__).with_name("verify_qe_raw_mass_profile_experiment.py")
    assert verifier.is_file()
    datasets, _ = inventory()
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
        dump(
            OUT / "protocol.json",
            dict(
                name="043-raw-mass-profile-conservative-completion",
                previous=str(PREVIOUS),
                production_enabled=False,
                curation_runtime_input=False,
                classifier=json.loads((CORE / "protocol.json").read_text())[
                    "classifier"
                ],
                maximum_fresh_classifiers=100,
                planned_verifier_sha256=sha(verifier),
                preflight_sha256=sha(PREFLIGHT / "receipt.json"),
                preflight_support_sha256=sha(PREFLIGHT / "parent_support.json"),
                source_input_sha256=gate["source_input_sha256"],
                source_snapshot_sha256={
                    p.relative_to(snapshot).as_posix(): sha(p)
                    for p in snapshot.rglob("*")
                    if p.is_file()
                },
                resource_receipt=configure_frozen_resources(snapshot),
                plan=(
                    REPO
                    / "docs/validation/next_raw_counterpart_mass_profile_completion.txt"
                ).read_text(),
                run_kind="Previously exposed saved-input grouped development. Fresh25four-modeparents/100classifiers,131inputs,INNERonlyclasscuts,ALL041callsimmutable. Primary92100includes1456STDs;endogenous90644companionseparate. Notdeployment/externalvalidation/FDR.",
            ),
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        dump(
            OUT / "run_manifest.json",
            dict(
                datasets=datasets,
                fresh_raw_input_runs=False,
                evaluation_note="Freshnestedbinarymodeexperts with23rawfields+immutable041calls;notfreshappdeployment.",
            ),
        )
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["source_input_sha256"] == source_inputs()
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    assert sha(verifier) == protocol["planned_verifier_sha256"]
    for p, h in protocol["source_snapshot_sha256"].items():
        assert sha(OUT / "source_snapshot" / p) == h
    return protocol, datasets


def fit_parent(outer, inner, protocol, datasets, folds):
    from matlab_annotation.mode_evidence_experts import fit_mode_evidence_experts

    path = parent_path(outer, inner)
    blocked = {outer} if inner is None else {outer, inner}
    training = {d[:7] for d in datasets if folds[d] not in blocked}
    excluded = {d[:7] for d in datasets if folds[d] in blocked}
    complete = path.with_suffix(".complete.json")
    if path.exists():
        receipt = json.loads(complete.read_text())
        assert receipt["model_sha256"] == sha(path)
        model = pd.read_pickle(path)
        assert (
            set(model["training_cohorts"]) == training
            and set(model["excluded_groups"]) == excluded
        )
        return model
    fitting, training, excluded = training_data(outer, inner, datasets, folds)
    model = fit_mode_evidence_experts(
        fitting,
        training_groups=training,
        excluded_groups=excluded,
        parameters=inherited.common.estimator_parameters(protocol["classifier"]),
    )
    assert len(model["columns"]) == 131 and set(model["experts"]) == set(MODES)
    model["training_feature_masks"] = {
        d: raw["excluded_folds"] for d, raw in fitting.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(model, path)
    dump(complete, dict(model_sha256=sha(path), fresh_classifiers=4))
    print("043 FRESH FOUR-MODE BINARY PARENT", outer, inner, len(fitting), flush=True)
    return model


def calibrate(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts
    from matlab_annotation.add_only_class_completion import (
        class_completion_curves,
        choose_completion_thresholds,
    )

    protocol = json.loads((OUT / "protocol.json").read_text())
    datasets, folds = inventory()
    destination = OUT / "calibration" / f"fold_{outer}"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "operating_points.json").exists():
        return outer
    training = {d[:7] for d in datasets if folds[d] != outer}
    excluded = {d[:7] for d in datasets if folds[d] == outer}
    pieces, manuals, base_counts, audits = [], [], Counter(), []
    for inner in range(5):
        if inner == outer:
            continue
        model = fit_parent(outer, inner, protocol, datasets, folds)
        oldmodel = pd.read_pickle(
            CORE / "calibration" / f"fold_{outer}/inner_block_{inner}_model.pkl"
        )
        for d in datasets:
            if folds[d] != inner:
                continue
            raw = load_predictors(d, {outer, inner})
            scores = predict_mode_evidence_experts(raw, model)
            scores.to_pickle(destination / (d + "_blind_scores.pkl"))
            oldraw, old_scores = inherited.reproduce(
                d,
                {outer, inner},
                oldmodel,
                CORE / "calibration" / f"fold_{outer}" / (d + "_blind_scores.pkl"),
            )
            pd.testing.assert_frame_equal(raw["pool"], oldraw["pool"], check_exact=True)
            labeled, manual, _ = authorize_labels(raw, training, excluded)
            occupied, old_core = inner_previous(d, outer, inner, labeled, old_scores)
            base_counts.update(
                {
                    (s, c): int(n)
                    for (s, c), n in old_core.assign(
                        _cls=old_core["class"].map(canonical_lipid_class)
                    )
                    .groupby(["stream", "_cls"])
                    .training_positive.sum()
                    .items()
                }
            )
            pieces.append(
                proposals(d, {outer, inner}, labeled, scores, occupied).assign(
                    project=d[:7]
                )
            )
            manuals.append(manual.assign(project=d[:7]))
            audits.append(
                dict(
                    dataset=d,
                    outer=outer,
                    inner=inner,
                    new_model_sha256=sha(parent_path(outer, inner)),
                )
            )
    table = class_completion_curves(
        pd.concat(pieces, ignore_index=True),
        pd.concat(manuals, ignore_index=True),
        base_counts,
        training_groups=training,
        excluded_groups=excluded,
    )
    thresholds, chosen = choose_completion_thresholds(table)
    table.to_csv(destination / "training_class_operating_points.csv", index=False)
    chosen.to_csv(destination / "selected_training_points.csv", index=False)
    pd.DataFrame(audits).to_csv(destination / "scope_audit.csv", index=False)
    dump(
        destination / "operating_points.json",
        dict(
            thresholds=thresholds,
            training_groups=sorted(training),
            outer_excluded=sorted(excluded),
        ),
    )
    print(
        "043 CALIBRATION COMPLETE",
        outer,
        "enabled",
        int(chosen.threshold.lt(1).sum()),
        "addedcorrect",
        int(chosen.added_correct.sum()),
        "extras",
        int(chosen.extras.sum()),
        flush=True,
    )
    return outer


def run_fold(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts
    from matlab_annotation.add_only_class_completion import select_completion

    protocol = json.loads((OUT / "protocol.json").read_text())
    datasets, folds = inventory()
    model = fit_parent(outer, None, protocol, datasets, folds)
    cuts = json.loads(
        (OUT / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )["thresholds"]
    for d in datasets:
        if folds[d] != outer:
            continue
        destination = OUT / "runs" / d
        destination.mkdir(parents=True, exist_ok=False)
        raw = load_predictors(d, {outer})
        scores = predict_mode_evidence_experts(raw, model)
        previous = previous_annotations(d)
        free = proposals(d, {outer}, raw, scores, previous)
        selected = select_completion(free, cuts)
        annotations, candidates, filled = assemble(d, previous, selected)
        annotations.to_csv(destination / "automatic_annotations.csv", index=False)
        candidates.to_pickle(destination / "candidates.pkl")
        filled.to_pickle(destination / "private_additions.pkl")
        scores.to_pickle(destination / "scoring_audit.pkl")
        dump(
            destination / "complete.json",
            dict(
                dataset=d,
                fold=outer,
                status="complete",
                production_enabled=False,
                previous_calls_immutable=True,
                added_annotations=len(filled),
                counts=annotations.groupby("stream").size().to_dict(),
            ),
        )
        print("043 OUTER RAW MASS/PROFILE ADDITIONS", d, len(filled), flush=True)


def main():
    initialize()
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(calibrate, range(5)))
    # No outer annotation or evaluation before ALLfive calibrations finish.
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(run_fold, range(5)))
    print(
        "043 COMPLETE;FULL132chat/integrity/endogenouscompanion/wholedecision BEFORE044",
        flush=True,
    )


if __name__ == "__main__":
    main()
