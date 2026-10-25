"""041: frozen029 scoring, INNER-only conservative additions, oldcalls immutable."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import run_qe_np_context_bmp_experiment as common

QE, BASE, OLD = common.QE, common.BASE, common.PREVIOUS
OUT = QE / "experiment_041"
PREFLIGHT = QE / "add_only_class_preflight_v1"
MODES = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")
sha, dump = common.sha, common.dump


def inventory():
    datasets = json.loads((OLD / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    assert len(datasets) == 33 and len({d[:7] for d in datasets}) == 30
    assert all(
        len({folds[d] for d in datasets if d[:7] == g}) == 1
        for g in {d[:7] for d in datasets}
    )
    return datasets, folds


def binding_files():
    datasets, folds = inventory()
    files = {
        Path(__file__),
        Path(__file__).with_name("frozen_resources.py"),
        Path(__file__).with_name("context.py"),
        Path(__file__).with_name("verify_qe_np_context_bmp_experiment.py"),
        Path(common.__file__),
        Path(__file__).with_name("run_qe_atlas_experiment.py"),
        REPO / "docs/validation/next_add_only_class_completion.txt",
        QE / "rt_atlas_development/fold_assignment.json",
        OLD / "run_manifest.json",
        OLD / "protocol.json",
        OLD / "integrity_receipt.json",
        QE / "experiment_040_v2/integrity_receipt.json",
    }
    for o in range(5):
        folder = OLD / "calibration" / f"fold_{o}"
        files.add(OLD / f"fold_{o}_nested_model.pkl")
        files.add(folder / "operating_points.json")
        files.update(folder / f"inner_block_{i}_model.pkl" for i in range(5) if i != o)
        for d in datasets:
            mask = {o, folds[d]}
            files.add(common.prepared_path(d, mask))
            files.add(
                OLD / "runs" / d / "scoring_audit.pkl"
                if folds[d] == o
                else folder / (d + "_blind_scores.pkl")
            )
            name = "exclude_" + "_".join(map(str, sorted(mask)))
            files.add(QE / "nested_standard_development_v1" / (name + ".json"))
            files.add(common.PREPARED / "measurements" / name / (d + ".pkl"))
    for d in datasets:
        files.update(
            [
                BASE / "runs" / d / "full_annotation.pkl",
                BASE / "evaluation_v2" / d / "manual_comparison.pkl",
                OLD / "runs" / d / "automatic_annotations.csv",
                OLD / "runs" / d / "candidates.pkl",
                QE / "nested_standard_development_v1/automatic_inputs" / (d + ".pkl"),
            ]
        )
    files.update(
        p
        for root in ("app", "methods", "reference_data")
        for p in (REPO / root).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
    )
    return {str(p): sha(p) for p in sorted(files)}


def reproduce(dataset, mask, model, score_path):
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts

    raw = common.load_predictors(dataset, mask)
    assert not {
        "training_positive",
        "training_label_known",
        "calibration_identity_in_manual",
    } & set(raw["pool"])
    predicted = predict_mode_evidence_experts(raw, model)
    pd.testing.assert_series_equal(
        predicted, pd.read_pickle(score_path), check_exact=True
    )
    return raw, predicted


def proposals_for(dataset, mask, raw, scores, previous):
    from matlab_annotation.add_only_class_completion import free_completion_winners

    parts = []
    for stream in MODES:
        pool = raw["pool"].loc[raw["pool"].stream.eq(stream)]
        parts.append(
            free_completion_winners(
                pool,
                scores.loc[pool.index],
                previous.loc[previous.stream.eq(stream)],
                stream=stream,
                **common.physical_protection(dataset, mask, stream, raw),
            )
        )
    return pd.concat(parts, ignore_index=True)


def calibration_data(outer):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.operating_point import select_operating_candidates

    datasets, folds = inventory()
    allowed = {d[:7] for d in datasets if folds[d] != outer}
    excluded = {d[:7] for d in datasets if folds[d] == outer}
    folder = OLD / "calibration" / f"fold_{outer}"
    calibration = json.loads((folder / "operating_points.json").read_text())
    assert set(calibration["training_groups"]) == allowed
    assert set(calibration["outer_excluded"]) == excluded
    rows, manuals, audits, base_counts = [], [], [], Counter()
    for inner in range(5):
        if inner == outer:
            continue
        model_path = folder / f"inner_block_{inner}_model.pkl"
        model = pd.read_pickle(model_path)
        assert set(model["training_cohorts"]) == {
            d[:7] for d in datasets if folds[d] not in {outer, inner}
        }
        assert set(model["excluded_groups"]) == {
            d[:7] for d in datasets if folds[d] in {outer, inner}
        }
        for d in datasets:
            if folds[d] != inner:
                continue
            raw, scores = reproduce(
                d, {outer, inner}, model, folder / (d + "_blind_scores.pkl")
            )
            # Only now, after scopes and prediction checks, open authorized truth.
            labeled, manual, _ = common.label_allowed_query(
                raw, allowed, excluded, partial=False
            )
            pool = labeled["pool"]
            occupancy = []
            blind = pd.read_pickle(BASE / "runs" / d / "full_annotation.pkl")
            for stream in MODES:
                part = pool.loc[pool.stream.eq(stream)]
                core = select_operating_candidates(
                    part,
                    scores.loc[part.index],
                    threshold=calibration["thresholds"][stream],
                    ambiguity_margin=0.05,
                )
                base_counts.update(
                    {
                        (stream, c): int(n)
                        for c, n in core.assign(
                            _cls=core["class"].map(canonical_lipid_class)
                        )
                        .groupby("_cls")
                        .training_positive.sum()
                        .items()
                    }
                )
                auto = blind["streams"][stream]["annotations"]
                occupancy.extend(
                    [
                        core,
                        auto.loc[auto.is_is.fillna(False).astype(bool)].assign(
                            stream=stream
                        ),
                    ]
                )
            # Labels stay attached for counts but physical selector never reads them.
            free = proposals_for(
                d,
                {outer, inner},
                dict(raw, pool=pool),
                scores,
                pd.concat(occupancy, ignore_index=True),
            )
            rows.append(free.assign(project=d[:7]))
            manuals.append(manual.assign(project=d[:7]))
            audits.append(
                dict(
                    dataset=d,
                    outer=outer,
                    inner=inner,
                    scores_bitexact=True,
                    prepared_sha256=sha(common.prepared_path(d, {outer, inner})),
                    model_sha256=sha(model_path),
                )
            )
    return (
        pd.concat(rows, ignore_index=True),
        pd.concat(manuals, ignore_index=True),
        base_counts,
        allowed,
        excluded,
        audits,
    )


def assemble(dataset, previous, added):
    # Reuse the already audited all-cell/raw-export helper, not040models/scores.
    filled = common.fill_original_features(dataset, added)
    if len(filled):
        filled["auto_annotation_status"] = "conservative_add_only_class_completion"
    annotations, candidates = common.assemble_exports(dataset, previous, filled)
    new_keys = set(common.accepted_keys(filled))
    mask = pd.Series(
        [k in new_keys for k in common.accepted_keys(candidates)],
        index=candidates.index,
    )
    if mask.any():
        candidates.loc[mask, "auto_annotation_status"] = (
            "conservative_add_only_class_completion"
        )
    # The shared helper's newly introduced score column is metadata, not a BMP fit.
    for frame in (annotations, candidates, filled):
        if "context_completion_score" in frame:
            frame.rename(
                columns={"context_completion_score": "class_completion_score"},
                inplace=True,
            )
    return annotations, candidates, filled


def preflight():
    configure_frozen_resources(REPO)
    before = binding_files()
    assert json.loads((QE / "experiment_040_v2/integrity_receipt.json").read_text())[
        "all_passed"
    ]
    datasets, folds = inventory()
    uses = 0
    for o in range(5):
        p, m, counts, allowed, excluded, audit = calibration_data(o)
        assert set(p.project).issubset(allowed) and not set(p.project) & excluded
        assert set(m.project).issubset(allowed)
        assert len(audit) == sum(folds[d] != o for d in datasets)
        uses += len(audit)
        print("041 PREFLIGHT INNER", o, len(audit), len(p), flush=True)
    for d in datasets:
        old = common.read_previous(d)
        pool = pd.read_pickle(OLD / "runs" / d / "candidates.pkl")
        annotations, candidates, _ = assemble(d, old, pool.iloc[:0])
        pd.testing.assert_frame_equal(annotations, old, check_exact=True)
        pd.testing.assert_frame_equal(candidates, pool, check_exact=True)
    assert uses == 132 and before == binding_files()
    PREFLIGHT.mkdir(exist_ok=False)
    dump(
        PREFLIGHT / "receipt.json",
        dict(
            all_passed=True,
            classifier_fits=0,
            inner_score_reproductions=132,
            disabled_full_029_datasets=33,
            source_input_sha256=before,
            note="No thresholds chosen or outer annotations changed. Full132 nested score/scope/authorizedlabel checks and33 disabled entire exports exact029.",
        ),
    )


def initialize():
    pre = json.loads((PREFLIGHT / "receipt.json").read_text())
    assert (
        pre["all_passed"]
        and pre["inner_score_reproductions"] == 132
        and pre["disabled_full_029_datasets"] == 33
    )
    assert pre["source_input_sha256"] == binding_files()
    datasets, _ = inventory()
    if not (OUT / "protocol.json").exists():
        OUT.mkdir(exist_ok=False)
        snap = OUT / "source_snapshot"
        snap.mkdir()
        for n in ("app", "methods", "reference_data"):
            shutil.copytree(
                REPO / n,
                snap / n,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        verifier = Path(__file__).with_name("verify_qe_add_only_class_experiment.py")
        assert verifier.is_file()
        protocol = dict(
            name="041-conservative-add-only-class-completion",
            previous=str(OLD),
            production_enabled=False,
            curation_runtime_input=False,
            classifier_fits=0,
            deliberately_reused_029_parents=25,
            deliberately_reused_029_mode_classifiers=100,
            planned_verifier_sha256=sha(verifier),
            source_input_sha256=pre["source_input_sha256"],
            preflight_sha256=sha(PREFLIGHT / "receipt.json"),
            source_snapshot_sha256={
                p.relative_to(snap).as_posix(): sha(p)
                for p in snap.rglob("*")
                if p.is_file()
            },
            plan=(
                REPO / "docs/validation/next_add_only_class_completion.txt"
            ).read_text(),
            resource_receipt=configure_frozen_resources(snap),
            run_kind="Previously exposed saved-input grouped development replay; unchanged029 models/pools/scores, INNER-only new class thresholds and immutable029calls. Not fresh integrated deployment or analytical FDR.",
        )
        dump(OUT / "protocol.json", protocol)
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        dump(
            OUT / "run_manifest.json",
            dict(
                datasets=datasets,
                fresh_raw_input_runs=False,
                evaluation_note=protocol["run_kind"],
            ),
        )
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    assert pre["source_input_sha256"] == protocol["source_input_sha256"]
    for p, h in protocol["source_snapshot_sha256"].items():
        assert sha(OUT / "source_snapshot" / p) == h
    return protocol


def calibrate(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.add_only_class_completion import (
        class_completion_curves,
        choose_completion_thresholds,
    )

    destination = OUT / "calibration" / f"fold_{outer}"
    destination.mkdir(parents=True, exist_ok=False)
    p, m, counts, allowed, excluded, audits = calibration_data(outer)
    table = class_completion_curves(
        p, m, counts, training_groups=allowed, excluded_groups=excluded
    )
    thresholds, chosen = choose_completion_thresholds(table)
    table.to_csv(destination / "training_class_operating_points.csv", index=False)
    chosen.to_csv(destination / "selected_training_points.csv", index=False)
    pd.DataFrame(audits).to_csv(
        destination / "scope_reproduction_audit.csv", index=False
    )
    dump(
        destination / "operating_points.json",
        dict(
            thresholds=thresholds,
            training_groups=sorted(allowed),
            outer_excluded=sorted(excluded),
        ),
    )
    print(
        "041 INNER CALIBRATION",
        outer,
        "enabled",
        int(chosen.threshold.lt(1).sum()),
        "addedcorrect",
        int(chosen.added_correct.sum()),
        "extras",
        int(chosen.extras.sum()),
        flush=True,
    )


def run_fold(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.add_only_class_completion import select_completion

    datasets, folds = inventory()
    model = pd.read_pickle(OLD / f"fold_{outer}_nested_model.pkl")
    assert set(model["training_cohorts"]) == {
        d[:7] for d in datasets if folds[d] != outer
    }
    assert set(model["excluded_groups"]) == {
        d[:7] for d in datasets if folds[d] == outer
    }
    cuts = json.loads(
        (OUT / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )["thresholds"]
    for d in datasets:
        if folds[d] != outer:
            continue
        destination = OUT / "runs" / d
        destination.mkdir(parents=True, exist_ok=False)
        raw, scores = reproduce(
            d, {outer}, model, OLD / "runs" / d / "scoring_audit.pkl"
        )
        previous = common.read_previous(d)
        free = proposals_for(d, {outer}, raw, scores, previous)
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
        print("041 OUTER ADDITIONS", d, len(filled), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
        return
    initialize()
    with ProcessPoolExecutor(max_workers=3) as ex:
        list(ex.map(calibrate, range(5)))
    # Explicit barrier: ALLcalibrations finish before any outer annotation.
    with ProcessPoolExecutor(max_workers=3) as ex:
        list(ex.map(run_fold, range(5)))
    print(
        "041 ALL33DONE; full132chat report/wholedecision required before042", flush=True
    )


if __name__ == "__main__":
    main()
