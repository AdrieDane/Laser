"""042 fresh nested feature detectors/conditional label models; immutable041."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
from immutable_exports import assemble_immutable
import run_qe_add_only_class_experiment as inherited

QE, BASE, CORE = inherited.QE, inherited.BASE, inherited.OLD
PREVIOUS, OUT = QE / "experiment_041", QE / "experiment_042"
PREFLIGHT = QE / "feature_label_model_preflight_v1"
CONSTRUCTION = QE / "feature_label_target_audit_v2"
MODES = inherited.MODES
sha, dump = inherited.sha, inherited.dump


def inventory():
    return inherited.inventory()


def load_predictors(dataset, mask):
    return inherited.common.load_predictors(dataset, mask)


def authorize_labels(raw, training, excluded):
    from matlab_annotation.evidence_training import validate_feature_scope

    validate_feature_scope(raw, training, excluded, query_is_training=True)
    return inherited.common.label_allowed_query(raw, training, excluded, partial=False)


def training_data(outer, inner, datasets, folds):
    blocked = {outer} if inner is None else {outer, inner}
    allowed = {d[:7] for d in datasets if folds[d] not in blocked}
    excluded = {d[:7] for d in datasets if folds[d] in blocked}
    fitting = {}
    for d in datasets:
        if folds[d] in blocked:
            continue
        raw = load_predictors(d, blocked | {folds[d]})
        labeled, _, _ = authorize_labels(raw, allowed, excluded)
        fitting[d] = labeled
    return fitting, allowed, excluded


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
            columns={"context_completion_score": "factorized_agreement_rank"},
            inplace=True,
        )
        filled["auto_annotation_status"] = "feature_label_factorized_completion"
    old_candidates = pd.read_pickle(PREVIOUS / "runs" / dataset / "candidates.pkl")
    annotations, candidates = assemble_immutable(
        previous, old_candidates, filled, method="feature_label_factorized_completion"
    )
    return annotations, candidates, filled


def proposals(dataset, mask, raw, ranks, previous):
    assert ranks.index.equals(raw["pool"].index)
    augmented = dict(raw, pool=raw["pool"].assign(**{c: ranks[c] for c in ranks}))
    free = inherited.proposals_for(
        dataset, mask, augmented, ranks.joint_agreement_rank, previous
    )
    # Apply this AFTER global competition/identity winner selection. Never drop
    # an unsupported competing label then promote a weaker supported label.
    return free.loc[free.supported_cluster.astype(bool)].copy()


def inner_previous(dataset, outer, inner, labeled, scores):
    """Reconstruct041INNER occupancy, never outer-only005/027 references."""
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.add_only_class_completion import select_completion

    core_cuts = json.loads(
        (CORE / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )["thresholds"]
    old_cuts = json.loads(
        (PREVIOUS / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )["thresholds"]
    blind = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
    core, standards = [], []
    for stream in MODES:
        part = labeled["pool"].loc[labeled["pool"].stream.eq(stream)]
        core.append(
            select_operating_candidates(
                part,
                scores.loc[part.index],
                threshold=core_cuts[stream],
                ambiguity_margin=0.05,
            )
        )
        auto = blind["streams"][stream]["annotations"]
        standards.append(
            auto.loc[auto.is_is.fillna(False).astype(bool)].assign(stream=stream)
        )
    base_core = pd.concat(core, ignore_index=True)
    occupied = pd.concat([base_core, *standards], ignore_index=True)
    free = inherited.proposals_for(dataset, {outer, inner}, labeled, scores, occupied)
    completion = select_completion(free, old_cuts)
    return pd.concat([occupied, completion], ignore_index=True), pd.concat(
        [base_core, completion], ignore_index=True
    )


def source_inputs():
    datasets, folds = inventory()
    paths = {
        Path(__file__),
        Path(inherited.__file__),
        Path(inherited.common.__file__),
        Path(__file__).with_name("immutable_exports.py"),
        Path(__file__).with_name("preflight_qe_feature_label_models.py"),
        Path(__file__).with_name("run_qe_atlas_experiment.py"),
        Path(__file__).with_name("frozen_resources.py"),
        Path(__file__).with_name("context.py"),
        REPO / "docs/validation/next_feature_label_factorization.txt",
        CONSTRUCTION / "receipt.json",
        CONSTRUCTION / "by_dataset_mask_mode.csv",
        PREVIOUS / "protocol.json",
        PREVIOUS / "integrity_receipt.json",
        PREVIOUS / "run_manifest.json",
        CORE / "protocol.json",
        CORE / "integrity_receipt.json",
        QE / "rt_atlas_development/fold_assignment.json",
    }
    audit = json.loads((CONSTRUCTION / "receipt.json").read_text())
    assert audit["all_passed"] and audit["unique_training_tables"] == 330
    # The earlier construction gate does not cover the later-added MODEL file.
    # Preserve/check all sources it DID bind, then bind the full current app too.
    for p, h in audit["source_input_sha256"].items():
        assert sha(Path(p)) == h, p
        paths.add(Path(p))
    paths.update(
        p
        for name in ("app", "reference_data", "methods")
        for p in (REPO / name).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
    )
    for d in datasets:
        paths.update(
            [
                BASE / "runs" / d / "full_annotation.pkl",
                PREVIOUS / "runs" / d / "automatic_annotations.csv",
                PREVIOUS / "runs" / d / "candidates.pkl",
                CORE / "runs" / d / "scoring_audit.pkl",
                QE / "nested_standard_development_v1/automatic_inputs" / (d + ".pkl"),
            ]
        )
        own = folds[d]
        # All330 training masks already bound above; outer-only query table
        # plus all matching independent protection measurements added here.
        masks = [{own}]
        masks.extend({own, o} for o in range(5) if o != own)
        for mask in masks:
            name = "exclude_" + "_".join(map(str, sorted(mask)))
            paths.update(
                [
                    inherited.common.prepared_path(d, mask),
                    inherited.common.PREPARED / "measurements" / name / (d + ".pkl"),
                    QE / "nested_standard_development_v1" / (name + ".json"),
                ]
            )
    for o in range(5):
        folder = CORE / "calibration" / f"fold_{o}"
        paths.add(CORE / f"fold_{o}_nested_model.pkl")
        paths.update(folder / f"inner_block_{i}_model.pkl" for i in range(5) if i != o)
        paths.update(
            [
                folder / "operating_points.json",
                PREVIOUS / "calibration" / f"fold_{o}/operating_points.json",
            ]
        )
        paths.update(p for p in folder.glob("*_blind_scores.pkl"))
    return {str(p): sha(p) for p in sorted(paths)}


def initialize():
    gate = json.loads((PREFLIGHT / "receipt.json").read_text())
    assert gate["all_passed"] and gate["classifier_fits"] == 0
    assert (
        gate["parent_preparations"] == 25
        and gate["training_query_uses"] == 528
        and gate["disabled_full_041_datasets"] == 33
    )
    assert gate["source_input_sha256"] == source_inputs()
    assert gate["parent_support_sha256"] == sha(PREFLIGHT / "parent_support.json")
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
        verifier = Path(__file__).with_name("verify_qe_feature_label_experiment.py")
        assert verifier.is_file()
        dump(
            OUT / "protocol.json",
            dict(
                name="042-feature-label-factorized-conservative-completion",
                previous=str(PREVIOUS),
                production_enabled=False,
                curation_runtime_input=False,
                classifier=json.loads((CORE / "protocol.json").read_text())[
                    "classifier"
                ],
                maximum_fresh_classifiers=200,
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
                    REPO / "docs/validation/next_feature_label_factorization.txt"
                ).read_text(),
                run_kind="Previously exposed saved-input grouped development replay. Fresh25two-stageparents/up to200classifiers,INNER-only class thresholds,ALL041calls immutable. Not deployment/externalvalidation/FDR.",
            ),
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        dump(
            OUT / "run_manifest.json",
            dict(
                datasets=datasets,
                fresh_raw_input_runs=False,
                evaluation_note="Freshnestedfeature/conditional-label models + immutable041calls, saved-input groupeddevelopment, not fresh app deployment.",
            ),
        )
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["source_input_sha256"] == source_inputs()
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    assert sha(PREFLIGHT / "receipt.json") == protocol["preflight_sha256"]
    for p, h in protocol["source_snapshot_sha256"].items():
        assert sha(OUT / "source_snapshot" / p) == h, p
    return protocol, datasets


def fit_parent(outer, inner, protocol, datasets, folds):
    from matlab_annotation.feature_label_model import fit_feature_label_model

    path = parent_path(outer, inner)
    complete = path.with_suffix(".complete.json")
    blocked = {outer} if inner is None else {outer, inner}
    expected = {d[:7] for d in datasets if folds[d] not in blocked}
    excluded = {d[:7] for d in datasets if folds[d] in blocked}
    if path.exists():
        receipt = json.loads(complete.read_text())
        assert receipt["model_sha256"] == sha(path)
        model = pd.read_pickle(path)
        assert (
            set(model["training_reference_permissions"]) == expected
            and set(model["excluded_groups"]) == excluded
        )
        return model
    fitting, training, excluded = training_data(outer, inner, datasets, folds)
    model = fit_feature_label_model(
        fitting,
        training_groups=training,
        excluded_groups=excluded,
        parameters=inherited.common.estimator_parameters(protocol["classifier"]),
    )
    model["training_feature_masks"] = {
        d: data["excluded_folds"] for d, data in fitting.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(model, path)
    dump(
        complete,
        dict(model_sha256=sha(path), fresh_classifiers=model["fresh_classifier_fits"]),
    )
    print(
        "042 FRESHFEATURE/LABEL PARENT",
        outer,
        inner,
        model["fresh_classifier_fits"],
        flush=True,
    )
    return model


def calibrate(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.feature_label_model import predict_feature_label_model
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
        for d in datasets:
            if folds[d] != inner:
                continue
            raw = load_predictors(d, {outer, inner})
            ranks = predict_feature_label_model(raw, model)
            ranks.to_pickle(destination / (d + "_blind_scores.pkl"))
            # Final query prediction first; only permitted calibration truth now.
            labeled, manual, _ = authorize_labels(raw, training, excluded)
            old_scores = pd.read_pickle(
                CORE / "calibration" / f"fold_{outer}" / (d + "_blind_scores.pkl")
            )
            assert old_scores.index.equals(raw["pool"].index)
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
                proposals(d, {outer, inner}, labeled, ranks, occupied).assign(
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
        "042 CALIBRATION COMPLETE",
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
    from matlab_annotation.feature_label_model import predict_feature_label_model
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
        ranks = predict_feature_label_model(raw, model)
        previous = previous_annotations(d)
        free = proposals(d, {outer}, raw, ranks, previous)
        selected = select_completion(free, cuts)
        annotations, candidates, filled = assemble(d, previous, selected)
        annotations.to_csv(destination / "automatic_annotations.csv", index=False)
        candidates.to_pickle(destination / "candidates.pkl")
        filled.to_pickle(destination / "private_additions.pkl")
        ranks.to_pickle(destination / "scoring_audit.pkl")
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
        print("042 OUTER FACTORIZED ADDITIONS", d, len(filled), flush=True)


def main():
    initialize()
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(calibrate, range(5)))
    # All calibrations finish BEFORE any outer annotation or evaluation.
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(run_fold, range(5)))
    print("042 COMPLETE;FULL132chat/integrity/wholedecision BEFORE043", flush=True)


if __name__ == "__main__":
    main()
