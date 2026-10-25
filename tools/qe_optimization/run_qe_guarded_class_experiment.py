"""034: guarded class acceptance, EXACT frozen029 candidates/models/scores.

All five class calibrations finish before any final query is selected. No
classifier refits or rejected033 input/model reuse. All query scores are
reproduced, not merely trusted because an old score file is present.
"""

from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import shutil

import pandas as pd

from context import WORKSPACE, REPO
from frozen_resources import configure_frozen_resources
from run_qe_standard_mode_experiment import sha, dump, species, adduct

QE = WORKSPACE / "outputs/qe_optimization_20261022"
BASE = QE / "baseline"
OLD = QE / "experiment_029"
OUT = QE / "experiment_034"
PREPARED = QE / "standard_context_preparation_v1"
STD = QE / "experiment_005"
MASS = QE / "experiment_027"


def prepared_path(dataset, mask):
    return (
        PREPARED
        / "prepared"
        / ("exclude_" + "_".join(map(str, sorted(mask))))
        / (dataset + ".pkl")
    )


def inventory():
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    assert len(projects) == 33 and len({d[:7] for d in projects}) == 30
    assert all(
        len({folds[d] for d in projects if d[:7] == group}) == 1
        for group in {d[:7] for d in projects}
    )
    return projects, folds


def initialize():
    projects, folds = inventory()
    assert json.loads((OLD / "integrity_receipt.json").read_text())["all_passed"]
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
            OLD / "protocol.json",
            OLD / "integrity_receipt.json",
            OLD / "runner_source.py",
            PREPARED / "protocol.json",
            PREPARED / "summary.json",
            MASS / "protocol.json",
            MASS / "integrity_receipt.json",
            QE / "rt_atlas_development/fold_assignment.json",
            BASE / "run_manifest.json",
        ]
        inputs.extend(
            Path(__file__).with_name(n)
            for n in (
                "frozen_resources.py",
                "run_qe_atlas_experiment.py",
                "run_qe_standard_mode_experiment.py",
            )
        )
        for outer in range(5):
            folder = OLD / "calibration" / f"fold_{outer}"
            inputs.extend(
                [
                    OLD / f"fold_{outer}_nested_model.pkl",
                    folder / "operating_points.json",
                    folder / "selected_training_points.csv",
                    STD / "models" / f"fold_{outer}_standard_models.json",
                ]
            )
            inputs.extend(
                folder / f"inner_block_{j}_model.pkl" for j in range(5) if j != outer
            )
            for dataset in projects:
                mask = {outer} if folds[dataset] == outer else {outer, folds[dataset]}
                inputs.append(prepared_path(dataset, mask))
                inputs.append(
                    OLD / "runs" / dataset / "scoring_audit.pkl"
                    if folds[dataset] == outer
                    else folder / (dataset + "_blind_scores.pkl")
                )
        for dataset in projects:
            inputs.extend(
                [
                    BASE / "runs" / dataset / "full_annotation.pkl",
                    BASE / "evaluation_v2" / dataset / "manual_comparison.pkl",
                    MASS / "runs" / dataset / "private_mass_candidates.pkl",
                ]
            )
        old_protocol = json.loads((OLD / "protocol.json").read_text())
        protocol = dict(
            name="034-guarded-class-cutoffs-on-frozen029-inner-scores",
            previous=str(OLD),
            baseline=str(BASE),
            production_enabled=False,
            curation_runtime_input=False,
            numerical_workers=2,
            minimum_manual=100,
            minimum_projects=3,
            minimum_positive_winners=20,
            ambiguity_margin=0.05,
            extras_budget_per_endogenous_manual=0.095,
            target=0.9,
            solver_time_limit_per_stage_seconds=120,
            class_policy="Eligibility from EACH outer-training split only: >=100 endogenous manual rows, >=3 independent projects, >=20 correct fixed winners from >=3 projects. Sparse/unsupported classes and zero-manual classes keep029modecutoff. Eligible cuts: fixed positive-score coverage quantiles5%..100% plus88/89/90/91/92%, exactbaseline-correct boundary and1.0. Raw scores are NOT chemical probabilities; eligible class cutoffs may be below.05. SAME.05ambiguity, fixed label/identity winners; no losing-peak promotion.",
            threshold_policy="All five INNER-OOF calibrations BEFORE outer predictions. Every eligible class training correct>=029classcorrect; each mode wrong<=029; pooled9.5%endogenous manual extras budget, all fixed-class extra costs included. Lexicographic maximize class coverage capped90%,then totalcorrect,then minimize wrong/extras/cutoffchange. Nonoptimal solver returns WHOLE baseline, never partial winner or relaxed guard. Training guards are NOT held-out guarantees or FDR.",
            candidate_policy="EXACT029 physical pools,15independent-standard predictors and frozen25models/100experts. Reproduce ALL132inner and33outer scoreSeries bitexact. No new mass/RT/composition ranges, no033 inputs, no query curation inference.",
            run_kind="Cached original-feature replay; ONLY class acceptance cutoffs fitted from frozen029 genuinely nested INNER scores. NO classifier fits. Same005standardcleanup and fixed027massstage. NOT fresh production raw-input annotation.",
            validation=old_protocol["validation"],
            model_fit_note=old_protocol["model_fit_note"]
            + "034deliberately reuses ALL029 scoring models unchanged with hash bindings and prediction reproductions. Class parameters only use matching outer-training projects; baseline mode thresholds also came from these inner folds. Prior development exposure remains.",
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
    # Entire retained scientific implementation identical, except previously
    # verified deterministic composition cache and the TWO new unused files.
    for relative in json.loads((OLD / "protocol.json").read_text())[
        "source_snapshot_sha256"
    ]:
        if (
            relative.endswith("annotator/composition_ranges.py")
            or "/tests/" in relative
        ):
            continue
        if relative.endswith("matlab_annotation/evidence_training.py"):
            # Existing033 change only strengthens scope checks for a predictor
            # prefix NOT present in029 tables. The score function is unchanged.
            old_text = (OLD / "source_snapshot" / relative).read_text()
            expected = old_text.replace(
                'str(c).startswith("q_declared_standard_") for c in data["evidence"]',
                'str(c).startswith(("q_declared_standard_", "q_standard_aligned_"))\n'
                '        for c in data["evidence"]',
            )
            assert (OUT / "source_snapshot" / relative).read_text() == expected
            continue
        assert (OUT / "source_snapshot" / relative).read_bytes() == (
            OLD / "source_snapshot" / relative
        ).read_bytes(), relative
    return protocol


def old_runner():
    spec = importlib.util.spec_from_file_location(
        "frozen029_runner", OLD / "runner_source.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.OUT == OLD and module.PREPARED == PREPARED
    return module


def reproduce(dataset, mask, model, score_path):
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts

    raw = old_runner().load_predictors(dataset, mask)
    predicted = predict_mode_evidence_experts(raw, model)
    pd.testing.assert_series_equal(
        predicted, pd.read_pickle(score_path), check_exact=True
    )
    return raw, predicted


def calibration_data(outer):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.guarded_class_operating_point import clear_identity_winners
    from matlab_annotation.operating_point import select_operating_candidates

    protocol = json.loads((OUT / "protocol.json").read_text())
    projects, folds = inventory()
    allowed = {d[:7] for d in projects if folds[d] != outer}
    excluded = {d[:7] for d in projects if folds[d] == outer}
    folder = OLD / "calibration" / f"fold_{outer}"
    baseline = json.loads((folder / "operating_points.json").read_text())
    assert (
        set(baseline["training_groups"]) == allowed
        and set(baseline["outer_excluded"]) == excluded
    )
    cuts = baseline["thresholds"]
    winners, manuals, audits = [], [], []
    runner = old_runner()
    for inner in range(5):
        if inner == outer:
            continue
        model_path = folder / f"inner_block_{inner}_model.pkl"
        model = pd.read_pickle(model_path)
        assert set(model["training_cohorts"]) == {
            d[:7] for d in projects if folds[d] not in {outer, inner}
        }
        assert set(model["excluded_groups"]) == {
            d[:7] for d in projects if folds[d] in {outer, inner}
        }
        for dataset in projects:
            if folds[dataset] != inner:
                continue
            raw, scores = reproduce(
                dataset, {outer, inner}, model, folder / (dataset + "_blind_scores.pkl")
            )
            # Scopes checked BEFORE this training-only function opens labels.
            labeled, manual, identities = runner.with_training_labels(
                raw, permitted_groups=allowed, excluded_groups=excluded
            )
            manuals.append(manual.assign(project=dataset[:7]))
            pool = labeled["pool"].copy()
            pool["calibration_identity_in_manual"] = [
                (s, species(n), adduct(a)) in identities
                for s, n, a in pool[["stream", "name", "adduct"]].itertuples(
                    index=False, name=None
                )
            ]
            for stream, part in pool.groupby("stream", sort=False):
                selected = select_operating_candidates(
                    part,
                    scores.loc[part.index],
                    threshold=cuts[stream],
                    ambiguity_margin=protocol["ambiguity_margin"],
                )
                fixed = clear_identity_winners(
                    part,
                    scores.loc[part.index],
                    ambiguity_margin=protocol["ambiguity_margin"],
                )
                assert selected.index.equals(
                    fixed[fixed.class_selection_score.ge(cuts[stream])].index
                )
                winners.append(
                    fixed.assign(
                        lipid_class=fixed["class"].map(canonical_lipid_class),
                        project=dataset[:7],
                    )
                )
            audits.append(
                dict(
                    dataset=dataset,
                    outer=outer,
                    inner=inner,
                    bitexact_scores=True,
                    uniform_selector_parity=True,
                    model_sha256=sha(model_path),
                    prepared_sha256=sha(prepared_path(dataset, {outer, inner})),
                )
            )
    return (
        pd.concat(winners, ignore_index=True),
        pd.concat(manuals, ignore_index=True),
        cuts,
        allowed,
        excluded,
        audits,
    )


def calibrate(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.guarded_class_operating_point import (
        class_training_curves,
        choose_class_training_points,
    )

    protocol = json.loads((OUT / "protocol.json").read_text())
    destination = OUT / "calibration" / f"fold_{outer}"
    destination.mkdir(parents=True, exist_ok=True)
    assert not (destination / "operating_points.json").exists(), (
        "Do not overwrite a completed calibration"
    )
    winners, manual, cuts, allowed, excluded, audits = calibration_data(outer)
    table = class_training_curves(
        winners,
        manual,
        cuts,
        training_groups=allowed,
        excluded_groups=excluded,
        minimum_manual=protocol["minimum_manual"],
        minimum_projects=protocol["minimum_projects"],
        minimum_positives=protocol["minimum_positive_winners"],
    )
    reference = (
        table[table.baseline]
        .groupby("stream")[["manual", "correct", "extras", "different_peak"]]
        .sum()
        .sort_index()
    )
    old = (
        pd.read_csv(
            OLD / "calibration" / f"fold_{outer}" / "selected_training_points.csv"
        )
        .set_index("stream")[reference.columns]
        .sort_index()
    )
    pd.testing.assert_frame_equal(reference, old, check_exact=True, check_dtype=False)
    table.to_csv(destination / "training_class_operating_points.csv", index=False)
    learned = choose_class_training_points(
        table,
        training_groups=allowed,
        excluded_groups=excluded,
        extras_per_manual=protocol["extras_budget_per_endogenous_manual"],
        target=protocol["target"],
        time_limit=protocol["solver_time_limit_per_stage_seconds"],
    )
    chosen = table.loc[learned["selected_point_indices"]]
    chosen.to_csv(destination / "selected_class_training_points.csv", index=False)
    pd.DataFrame(audits).to_csv(
        destination / "scope_reproduction_audit.csv", index=False
    )
    learned.update(
        fallback_thresholds=cuts,
        training_table_sha256=sha(destination / "training_class_operating_points.csv"),
        selected_table_sha256=sha(destination / "selected_class_training_points.csv"),
        baseline_reference_parity_passed=True,
        inner_score_reproductions=len(audits),
    )
    dump(destination / "operating_points.json", learned)
    print(
        "034 TRAINING CALIBRATION",
        outer,
        "correct",
        int(reference.correct.sum()),
        "->",
        int(chosen.correct.sum()),
        "extras",
        int(chosen.extras.sum()),
        "fallback",
        learned["solver_fallback"],
        flush=True,
    )
    return outer


def annotate_dataset(dataset, outer, calibration, model, standards):
    from matlab_annotation.guarded_class_operating_point import (
        select_class_operating_candidates,
    )
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards
    from matlab_annotation.calibrated_mass_discovery import select_calibrated_additions

    dest = OUT / "runs" / dataset
    dest.mkdir(parents=True, exist_ok=False)
    data, scores = reproduce(
        dataset, {outer}, model, OLD / "runs" / dataset / "scoring_audit.pkl"
    )
    raw = data["pool"]
    baseline = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
    catalogue = declared_standard_targets(
        OUT / "source_snapshot/reference_data/matlab_pipeline",
        baseline["internal_standards_expected"],
    )
    mass_pool = pd.read_pickle(MASS / "runs" / dataset / "private_mass_candidates.pkl")
    assert not {"training_positive", "training_label_known", "same_manual_peak"} & set(
        mass_pool
    )
    automatic = pd.concat(
        [
            p["annotations"].copy().assign(stream=s)
            for s, p in baseline["streams"].items()
        ],
        ignore_index=True,
    )
    parts, candidates, audits, mass_audits = [], [raw, mass_pool], [], []
    for stream, payload in baseline["streams"].items():
        pool = raw[raw.stream.eq(stream)]
        scalar = calibration["fallback_thresholds"][stream]
        reference = select_operating_candidates(
            pool, scores.loc[pool.index], threshold=scalar, ambiguity_margin=0.05
        )
        uniform = select_class_operating_candidates(
            pool,
            scores.loc[pool.index],
            baseline_threshold=scalar,
            class_thresholds={},
            ambiguity_margin=0.05,
        )
        assert reference.index.equals(uniform.index)
        selected = select_class_operating_candidates(
            pool,
            scores.loc[pool.index],
            baseline_threshold=scalar,
            class_thresholds=calibration["thresholds"][stream],
            ambiguity_margin=0.05,
        )
        protected = (
            payload["annotations"][
                payload["annotations"].is_is.fillna(False).astype(bool)
            ]
            .copy()
            .assign(stream=stream)
        )
        combined = pd.concat([selected, protected], ignore_index=True)
        assert not combined.feat_feature_id.astype(str).duplicated().any()
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
        combined = pd.concat([corrected, added], ignore_index=True)
        assert not combined.feat_feature_id.astype(str).duplicated().any()
        parts.append(combined.assign(stream=stream))
        audits.append(audit.assign(dataset=dataset, stream=stream))
        mass_audits.append(
            dict(
                dataset=dataset,
                stream=stream,
                added_annotations=len(added),
                current_core_annotations=len(corrected),
                fixed_027_rule=True,
            )
        )
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
    keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
    candidate = (
        pd.concat(candidates, ignore_index=True)
        .drop_duplicates(keys, keep="last")
        .reset_index(drop=True)
    )
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
            bitexact029score_reproduction=True,
            uniform_cutoff_parity_passed=True,
            counts=annotations.groupby("stream").size().to_dict(),
            class_operating_points_sha256=sha(
                OUT / "calibration" / f"fold_{outer}" / "operating_points.json"
            ),
        ),
    )
    print("034 HELDOUT SELECTED", dataset, len(annotations), flush=True)


def run_fold(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    projects, folds = inventory()
    calibration = json.loads(
        (OUT / "calibration" / f"fold_{outer}" / "operating_points.json").read_text()
    )
    model = pd.read_pickle(OLD / f"fold_{outer}_nested_model.pkl")
    excluded = {d[:7] for d in projects if folds[d] == outer}
    assert set(model["excluded_groups"]) == excluded
    assert set(model["training_cohorts"]) == {
        d[:7] for d in projects if folds[d] != outer
    }
    standards = json.loads(
        (STD / "models" / f"fold_{outer}_standard_models.json").read_text()
    )
    assert not set(standards["training_groups"]) & excluded
    for dataset in projects:
        if folds[dataset] == outer:
            annotate_dataset(dataset, outer, calibration, model, standards)
    return outer


def main():
    protocol = initialize()
    with ProcessPoolExecutor(max_workers=protocol["numerical_workers"]) as executor:
        list(executor.map(calibrate, range(5)))
    assert all(
        (OUT / "calibration" / f"fold_{f}" / "operating_points.json").exists()
        for f in range(5)
    )
    dump(
        OUT / "all_calibrations_frozen.json",
        {
            str(f): sha(OUT / "calibration" / f"fold_{f}" / "operating_points.json")
            for f in range(5)
        },
    )
    with ProcessPoolExecutor(max_workers=protocol["numerical_workers"]) as executor:
        list(executor.map(run_fold, range(5)))
    print("034 COMPLETE; FULL132TABLE REQUIRED BEFORE NEXT EXPERIMENT", flush=True)


if __name__ == "__main__":
    main()
