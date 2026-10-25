"""044: fresh NP physical-context BMP completion; all041 calls immutable.

No example fits before complete363 construction and25-parent no-fit gates.
Five INNER joint grids complete before ANY outer annotation output.
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
import prepare_qe_bmp_context_tables as preparation
import bmp_context_completion as completion

original = preparation.original
QE, BASE, CORE, PREVIOUS = original.QE, original.BASE, original.CORE, original.PREVIOUS
OUT = QE / "experiment_044"
CONSTRUCTION = QE / "BMP_candidate_context_construction_check_v1"
PREFLIGHT = QE / "BMP_candidate_context_model_preflight_v2"
MODES, NP_MODES = original.MODES, completion.MODES
sha, dump, inventory = original.sha, original.dump, original.inventory
METHOD = "BMP_candidate_context_completion"


def load_predictors(d, mask):
    return preparation.load_augmented(d, mask)


def authorize_labels(raw, allowed, excluded, *, partial):
    preparation.validate_scope(raw, allowed, excluded, query_is_training=True)
    labeled, manual, audit = original.inherited.common.label_allowed_query(
        raw, allowed, excluded, partial=partial
    )
    if partial:
        labeled["bmp_partial_supervision_audit"] = audit
    return labeled, manual, audit


def training_data(outer, inner, datasets, folds):
    blocked = {outer} if inner is None else {outer, inner}
    allowed = {d[:7] for d in datasets if folds[d] not in blocked}
    excluded = {d[:7] for d in datasets if folds[d] in blocked}
    fitting = {}
    for d in datasets:
        if folds[d] not in blocked:
            raw = load_predictors(d, blocked | {folds[d]})
            fitting[d], _, _ = authorize_labels(raw, allowed, excluded, partial=True)
    return fitting, allowed, excluded


def parent_path(outer, inner):
    return OUT / (
        f"fold_{outer}_nested_model.pkl"
        if inner is None
        else f"calibration/fold_{outer}/inner_block_{inner}_model.pkl"
    )


def previous_annotations(d):
    return original.previous_annotations(d)


def assemble(d, previous, selected):
    filled = original.inherited.common.fill_original_features(d, selected)
    if len(filled):
        filled.rename(
            columns={"context_completion_score": "BMP_context_agreement_rank"},
            inplace=True,
        )
        filled["auto_annotation_status"] = METHOD
    candidates = pd.read_pickle(PREVIOUS / "runs" / d / "candidates.pkl")
    annotations, candidates = assemble_immutable(
        previous, candidates, filled, method=METHOD
    )
    return annotations, candidates, filled


def proposals(d, mask, raw, scores, previous, thresholds, supported):
    pieces = []
    for stream in NP_MODES:
        part = raw["pool"].loc[raw["pool"].stream.eq(stream)]
        if not supported[stream]:
            pieces.append(part.iloc[:0].copy())
            continue
        protection = original.inherited.common.physical_protection(d, mask, stream, raw)
        pieces.append(
            completion.select_additions(
                part,
                scores.loc[part.index],
                raw["evidence"].loc[part.index],
                previous.loc[previous.stream.eq(stream)],
                stream=stream,
                threshold=thresholds[stream],
                **protection,
            )
        )
    return pd.concat(pieces, axis=0)


def full_training_manual(raw, allowed, excluded):
    # Authorization precedes the read, including standard entries used ONLY
    # in calibration denominators/old-checkpoint match counts, never training.
    preparation.validate_scope(raw, allowed, excluded, query_is_training=True)
    return pd.read_pickle(
        BASE / "evaluation_v2" / raw["query_dataset"] / "manual_comparison.pkl"
    )


def prior_counts(occupied, manual):
    common = original.inherited.common
    keys = set(common.accepted_keys(occupied))
    manual = manual.loc[manual.lipid_class.eq("BMP") & manual.stream.isin(NP_MODES)]
    result = {}
    for s in NP_MODES:
        part = manual.loc[manual.stream.eq(s)]
        correct = sum(
            bool(r.mapped)
            and (
                r.stream,
                common.species(r.species),
                common.adduct(r.adduct),
                bool(r.internal_standard),
                str(r.feature_id),
            )
            in keys
            for r in part.itertuples()
        )
        result[s] = dict(
            manual=len(part),
            endogenous_manual=int((~part.internal_standard).sum()),
            previous_correct=correct,
        )
    return result


def addition_counts(chosen, full_manual):
    common = original.inherited.common
    other = {
        (r.stream, common.species(r.species))
        for r in full_manual.loc[
            ~full_manual.internal_standard & full_manual.lipid_class.eq("BMP")
        ].itertuples()
    }
    extra = ~chosen.calibration_identity_in_manual.astype(bool)
    without = []
    for r in chosen.itertuples():
        opposite = "NP_Neg" if r.stream == "NP_Pos" else "NP_Pos"
        without.append((opposite, common.species(r.name)) not in other)
    unsupported = pd.Series(without, index=chosen.index, dtype=bool)
    return dict(
        added_correct=int(chosen.training_positive.sum()),
        extras=int(extra.sum()),
        extras_without_opposite_composition=int((extra & unsupported).sum()),
        different_peak=int(
            (
                chosen.calibration_identity_in_manual
                & chosen.training_label_known
                & ~chosen.training_positive
            ).sum()
        ),
        unresolved_calls=int(
            (chosen.calibration_identity_in_manual & ~chosen.training_label_known).sum()
        ),
    )


def calibration_tables(cases):
    # Pure case aggregation is replayed after actual fitting by the independent
    # master checker, which also separately verifies per-case counts.
    rows = []
    totals = {s: Counter() for s in NP_MODES}
    potentials = []
    for case in cases:
        for s in NP_MODES:
            totals[s].update(prior_counts(case["occupied"], case["manual"])[s])
        potentials.append(
            proposals(
                case["dataset"],
                case["mask"],
                case["raw"],
                case["scores"],
                case["occupied"],
                dict.fromkeys(NP_MODES, min(completion.LEVELS)),
                case["supported"],
            )
        )
    for t in completion.LEVELS:
        counts = {s: Counter() for s in NP_MODES}
        projects = {s: set() for s in NP_MODES}
        for case, potential in zip(cases, potentials):
            # Fixed global/identity winners and physical gates do not depend
            # on cutoffs; filtering cannot promote weaker alternatives. The
            # full verifier compares EVERY threshold to direct selection.
            selected = threshold_subset(potential, t)
            for s in NP_MODES:
                chosen = selected.loc[selected.stream.eq(s)]
                values = addition_counts(chosen, case["manual"])
                counts[s].update(values)
                if values["added_correct"]:
                    projects[s].add(case["dataset"][:7])
        for s in NP_MODES:
            rows.append(
                dict(
                    stream=s,
                    threshold=t,
                    **dict(totals[s]),
                    **dict(counts[s]),
                    positive_projects=len(projects[s]),
                )
            )
    table = pd.DataFrame(rows)
    thresholds, joint = completion.choose_joint_grid(table)
    return thresholds, table, joint


def threshold_subset(potential, threshold):
    if threshold not in completion.LEVELS:
        raise ValueError("Declared cutoff required")
    if potential.empty or threshold == 1:
        return potential.iloc[:0].copy()
    return potential.loc[potential.class_selection_score.ge(threshold)].copy()


def source_inputs():
    receipt = json.loads((CONSTRUCTION / "receipt.json").read_text())
    assert (
        receipt["all_passed"]
        and receipt["unique_nested_tables"] == 363
        and receipt["total_predictors"] == 163
    )
    assert receipt["preparation_receipt_sha256"] == sha(
        preparation.OUT / "receipt.json"
    )
    assert all(sha(Path(p)) == h for p, h in receipt["source_input_sha256"].items())
    paths = set(map(Path, receipt["source_input_sha256"]))
    paths.update(
        {
            CONSTRUCTION / "receipt.json",
            preparation.OUT / "receipt.json",
            Path(__file__),
            Path(completion.__file__),
            Path(original.__file__),
            Path(original.preparation.__file__),
            Path(original.occupancy.__file__),
            Path(original.inherited.__file__),
            Path(original.inherited.common.__file__),
            Path(__file__).with_name("immutable_exports.py"),
            Path(__file__).with_name("verify_immutable_exports.py"),
            Path(__file__).with_name("preflight_qe_bmp_context_models.py"),
            Path(__file__).with_name("verify_qe_bmp_context_experiment.py"),
            Path(__file__).with_name("postprocess_qe_bmp_context.py"),
            Path(__file__).with_name("report_qe_endogenous_companion.py"),
            Path(__file__).with_name("evaluate_qe_baseline.py"),
            Path(__file__).with_name("report_qe_experiment.py"),
            Path(__file__).with_name("report_qe_priority_classes.py"),
            Path(__file__).with_name("report_qe_target_scope.py"),
            CORE / "protocol.json",
            CORE / "integrity_receipt.json",
            PREVIOUS / "protocol.json",
            PREVIOUS / "integrity_receipt.json",
        }
    )
    datasets, folds = inventory()
    for d in datasets:
        paths.update(
            {
                PREVIOUS / "runs" / d / "automatic_annotations.csv",
                PREVIOUS / "runs" / d / "candidates.pkl",
                BASE / "evaluation_v2" / d / "manual_comparison.pkl",
            }
        )
        for o in range(5):
            if o != folds[d]:
                paths.add(
                    CORE / "calibration" / f"fold_{o}" / (d + "_blind_scores.pkl")
                )
    for o in range(5):
        paths.update(
            {
                CORE / "calibration" / f"fold_{o}/operating_points.json",
                PREVIOUS / "calibration" / f"fold_{o}/operating_points.json",
            }
        )
        for i in [None, *sorted(set(range(5)) - {o})]:
            paths.add(
                CORE
                / (
                    f"fold_{o}_nested_model.pkl"
                    if i is None
                    else f"calibration/fold_{o}/inner_block_{i}_model.pkl"
                )
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
    return {str(p): sha(p) for p in sorted(paths)}


def initialize():
    gate = json.loads((PREFLIGHT / "receipt.json").read_text())
    assert (
        gate["all_passed"]
        and gate["parent_models_prepared"] == 25
        and gate["training_query_uses"] == 528
        and gate["classifier_fits"] == 0
    )
    assert gate["source_input_sha256"] == source_inputs()
    assert gate["parent_support_sha256"] == sha(PREFLIGHT / "parent_support.json")
    verifier = Path(__file__).with_name("verify_qe_bmp_context_experiment.py")
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
                name="044-BMP-candidate-context-completion",
                previous=str(PREVIOUS),
                production_enabled=False,
                curation_runtime_input=False,
                classifier=json.loads((CORE / "protocol.json").read_text())[
                    "classifier"
                ],
                maximum_fresh_classifiers=50,
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
                plan=preparation.PLAN.read_text(),
                threshold_grid=list(completion.LEVELS),
                run_kind="Previously exposed saved-input grouped development. Fresh25NPcontextparents/upto50classifiers,163inputs,INNERjointcuts,ALL041callsimmutable. Primary92100inclSTDs and all extras;notdeployment/externalvalidation/FDR.",
            ),
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        dump(
            OUT / "run_manifest.json",
            dict(
                datasets=datasets,
                fresh_raw_input_runs=False,
                evaluation_note="Research nested NP BMP completion, all041calls immutable;notfreshappdeployment.",
            ),
        )
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["source_input_sha256"] == source_inputs()
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    assert sha(verifier) == protocol["planned_verifier_sha256"]
    assert all(
        sha(OUT / "source_snapshot" / p) == h
        for p, h in protocol["source_snapshot_sha256"].items()
    )
    return protocol, datasets


def fit_parent(outer, inner, protocol, datasets, folds):
    path = parent_path(outer, inner)
    blocked = {outer} if inner is None else {outer, inner}
    allowed = {d[:7] for d in datasets if folds[d] not in blocked}
    excluded = {d[:7] for d in datasets if folds[d] in blocked}
    if path.exists():
        receipt = json.loads(path.with_suffix(".complete.json").read_text())
        assert receipt["model_sha256"] == sha(path)
        model = pd.read_pickle(path)
        assert (
            set(model["training_cohorts"]) == allowed
            and set(model["excluded_groups"]) == excluded
        )
        return model
    data, allowed, excluded = training_data(outer, inner, datasets, folds)
    model = completion.fit_context(
        data,
        training_groups=allowed,
        excluded_groups=excluded,
        parameters=original.inherited.common.estimator_parameters(
            protocol["classifier"]
        ),
    )
    model["training_feature_masks"] = {
        d: raw["excluded_folds"] for d, raw in data.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(model, path)
    dump(
        path.with_suffix(".complete.json"),
        dict(
            model_sha256=sha(path),
            fresh_classifiers=model["fresh_mode_classifier_fits"],
        ),
    )
    print(
        "044 FRESH NP CONTEXT PARENT",
        outer,
        inner,
        model["fresh_mode_classifier_fits"],
        flush=True,
    )
    return model


def calibration_cases(outer, protocol, datasets, folds, *, fit):
    allowed = {d[:7] for d in datasets if folds[d] != outer}
    excluded = {d[:7] for d in datasets if folds[d] == outer}
    cases = []
    for inner in range(5):
        if inner == outer:
            continue
        model = (
            fit_parent(outer, inner, protocol, datasets, folds)
            if fit
            else pd.read_pickle(parent_path(outer, inner))
        )
        oldmodel = pd.read_pickle(
            CORE / "calibration" / f"fold_{outer}/inner_block_{inner}_model.pkl"
        )
        for d in datasets:
            if folds[d] != inner:
                continue
            raw = load_predictors(d, {outer, inner})
            scores = completion.predict_context(raw, model)
            _, oldscores = original.inherited.reproduce(
                d,
                {outer, inner},
                oldmodel,
                CORE / "calibration" / f"fold_{outer}" / (d + "_blind_scores.pkl"),
            )
            labeled, _, _ = authorize_labels(raw, allowed, excluded, partial=False)
            occupied, _ = original.inner_previous(d, outer, inner, labeled, oldscores)
            manual = full_training_manual(raw, allowed, excluded)
            cases.append(
                dict(
                    dataset=d,
                    mask={outer, inner},
                    raw=labeled,
                    scores=scores,
                    occupied=occupied,
                    manual=manual,
                    supported={s: model["experts"][s] is not None for s in NP_MODES},
                )
            )
    return cases


def calibrate(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    protocol = json.loads((OUT / "protocol.json").read_text())
    datasets, folds = inventory()
    destination = OUT / "calibration" / f"fold_{outer}"
    destination.mkdir(parents=True, exist_ok=True)
    assert not (destination / "operating_points.json").exists()
    cases = calibration_cases(outer, protocol, datasets, folds, fit=True)
    for case in cases:
        case["scores"].to_pickle(destination / (case["dataset"] + "_blind_scores.pkl"))
    thresholds, table, joint = calibration_tables(cases)
    table.to_csv(destination / "training_operating_points.csv", index=False)
    joint.to_csv(destination / "complete_joint_grid.csv", index=False)
    dump(
        destination / "operating_points.json",
        dict(
            thresholds=thresholds,
            training_groups=sorted({d[:7] for d in datasets if folds[d] != outer}),
            outer_excluded=sorted({d[:7] for d in datasets if folds[d] == outer}),
            training_table_sha256=sha(destination / "training_operating_points.csv"),
            complete_grid_sha256=sha(destination / "complete_joint_grid.csv"),
        ),
    )
    print("044 FULL INNER JOINT CALIBRATION COMPLETE", outer, thresholds, flush=True)
    return outer


def run_fold(outer):
    configure_frozen_resources(OUT / "source_snapshot")
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
        scores = completion.predict_context(raw, model)
        previous = previous_annotations(d)
        selected = proposals(
            d,
            {outer},
            raw,
            scores,
            previous,
            cuts,
            {s: model["experts"][s] is not None for s in NP_MODES},
        )
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
        print("044 OUTER BMP CONTEXT ADDITIONS", d, len(filled), flush=True)


def main():
    initialize()
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(calibrate, range(5)))
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(run_fold, range(5)))
    print(
        "044 COMPLETE;FULL132chat/integrity/companions/WHOLEdecision BEFORE045",
        flush=True,
    )


if __name__ == "__main__":
    main()
