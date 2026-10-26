"""048: new fixed CatBoost family + INNER class-aware cuts, old108 physics.

Requires the COMPLETE final no-fit preflight and an independent verifier BEFORE
any QE fit. All25fresh parents ->ALL5calibrations ->33blind outer outputs.
No app/default changes; no old041completion append or outer winner splice.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import catboost_experiment_data as data
import catboost_evidence_model as backend
import catboost_experiment_bindings as bindings
import catboost_operating_points as operating
from catboost_pipeline_assembly import assemble


def initialize():
    # No annotation imports/configuration until FINAL own snapshot is ready.
    preflight_path = data.PREFLIGHT / "receipt.json"
    gate = json.loads(preflight_path.read_text())
    assert gate["all_passed"] and gate["actual_training_fit_authorized"] is True
    hashes = bindings.source_inputs()
    runtime = backend.environment()
    assert gate["all_passed"] and gate["actual_training_fit_authorized"] is True
    assert (
        gate["classifier_fits"] == 0
        and gate["whole029_replays"] == gate["disabled_whole041_datasets"] == 33
    )
    assert (
        gate["legacy_label_row_parity_uses"] == 528
        and gate["source_input_sha256"] == hashes
    )
    assert gate["runtime"] == runtime
    verifier = Path(__file__).with_name("verify_qe_catboost_experiment.py")
    assert gate["planned_actual_verifier_sha256"] == data.sha(verifier)
    if (data.OUT / "protocol.json").exists():
        protocol = json.loads((data.OUT / "protocol.json").read_text())
        assert (
            protocol["source_input_sha256"] == hashes and protocol["runtime"] == runtime
        )
        assert protocol["preflight_sha256"] == data.sha(preflight_path)
    else:
        data.OUT.mkdir(exist_ok=False)
        snapshot = data.OUT / "source_snapshot"
        snapshot.mkdir()
        for folder in ("app", "reference_data", "methods"):
            shutil.copytree(
                REPO / folder,
                snapshot / folder,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        protocol = dict(
            name="048-CatBoost800-native-domain-INNER-class-calibrated-whole-core",
            production_enabled=False,
            curation_runtime_input=False,
            numerical_workers=2,
            classifier=dict(kind="CatBoostClassifier", **backend.PARAMETERS),
            mode_threshold_grid=list(operating.THRESHOLDS),
            ambiguity_margin=0.05,
            extras_budget_per_endogenous_manual=0.095,
            target=0.9,
            minimum_manual=100,
            minimum_projects=3,
            minimum_positive_winners=20,
            solver_time_limit_per_stage_seconds=120,
            previous=str(data.PREVIOUS),
            baseline=str(data.BASE),
            core_reference=str(data.CORE),
            runtime=runtime,
            source_input_sha256=hashes,
            preflight_sha256=data.sha(preflight_path),
            planned_actual_verifier_sha256=data.sha(verifier),
            source_snapshot_sha256={
                p.relative_to(snapshot).as_posix(): data.sha(p)
                for p in snapshot.rglob("*")
                if p.is_file()
            },
            resource_receipt=configure_frozen_resources(snapshot),
            candidate_policy="EXACT029original363physicalpools/108predictors/knownlabels/equalprojectweights. No newtargets/windows/ranges/physicalpilotfields. Allreference/query/outer/inner scopes unchanged.",
            scoring_policy="25freshparents/100fixedCPUCatBoostOrdered800treeexperts. No implicit evalset/earlystop/hyperparametersearch; nativecategoricalhandling is partofnewfamily. OldHGBnotclaimednewfits. All5INNERcalibrations BEFORE ANYouter outputs.",
            threshold_policy="Newnative0..1score-domain41modegrid/.05fixedcompetition,1disabled; pooled9.5%endogenousmanualbudget/weakestmodecapped90. Thenexisting034classcurves/MIPonNEWINNERscores,100manual/3projects/20positivewinnersfrom3projects. Sparsefixedcosts count;noeligibletrainclasscorrectloss/no modewrongincrease;lexicographiccapped90classmean/correct/wrong/extras/stability. UncertifiedsolverWHOLEnewmodebaselinefallback. Notchemicalprobability/FDR.",
            reconstruction_policy="WholeNEWclasscalibratedcore+fixed005standardmodels/adjudication+fixed027massphysicalproposals/rules. Changedcoreoccupancymaychangemassadmissions. No outer-onlymasscacheinINNERcalibration.04111correct/2extra addonEXPLICITLYOMITTED,notsplicedposthoc. CompareWHOLE048vsWHOLE041.",
            run_kind="Cachedoriginal-feature replay,NOTfreshproductionraw-inputappbenchmark. Newmodels/cuts properlygroup-held-out; inheritedbaseline/vendor/005/027developmentexposure remains. Notuntouchedexternalvalidation/analyticalFDR/deploymentreadiness.",
            acceptance_policy=">=100netcorrectgain;totalextras<=041;wrongpeaks<=041;prioritycapped90classmeanup/below90classcountnotup/noalready90priorityclassdropsbelow90/nopriorityloss>1pp;standardsnotregress. Noouterwinner/budget/guardrelaxation. Full132CHATtablebeforeanotheractualexperiment.",
        )
        data.dump(data.OUT / "protocol.json", protocol)
        shutil.copy2(Path(__file__), data.OUT / "runner_source.py")
        datasets, _ = data.inventory()
        data.dump(
            data.OUT / "run_manifest.json",
            dict(
                datasets=datasets,
                fresh_raw_input_runs=False,
                evaluation_note=protocol["run_kind"]
                + " "
                + protocol["reconstruction_policy"],
            ),
        )
    snapshot = data.OUT / "source_snapshot"
    for relative, h in protocol["source_snapshot_sha256"].items():
        assert data.sha(snapshot / relative) == h
    assert (data.OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    assert configure_frozen_resources(snapshot) == protocol["resource_receipt"]
    return protocol


def protocol():
    return json.loads((data.OUT / "protocol.json").read_text())


def pause_requested():
    """Human-requested pause at a completed model/calibration/dataset boundary."""
    return (data.OUT / "pause_requested").exists()


def checkpoint(stage):
    data.dump(
        data.OUT / "paused_checkpoint.json",
        dict(
            stage=stage,
            pause_requested=True,
            annotation_experiment_complete=False,
            protocol_sha256=data.sha(data.OUT / "protocol.json"),
        ),
    )
    print("PAUSED AT SAFE CHECKPOINT", stage, flush=True)


def read_model(outer, inner):
    path = data.parent_path(outer, inner)
    complete = json.loads(path.with_suffix(".complete.json").read_text())
    assert complete["model_sha256"] == data.sha(path)
    assert complete["protocol_sha256"] == data.sha(data.OUT / "protocol.json")
    return pd.read_pickle(path)


def check_expected_matrices(records, training, excluded, outer, inner):
    from matlab_annotation.mode_evidence_experts import _subset

    table = pd.read_csv(bindings.CONSTRUCTION / "all25_parent_mode_matrices.csv")
    part = table.loc[
        table.outer.eq(outer)
        & (table.inner.isna() if inner is None else table.inner.eq(inner))
    ]
    assert len(part) == 4
    for mode in data.MODES:
        x, y, groups, weights, _ = backend.fitting_matrix(
            {d: _subset(raw, mode) for d, raw in records.items()}, training, excluded
        )
        expected = part.loc[part.stream.eq(mode)].iloc[0]
        assert (
            backend.fingerprint(x, y, groups, weights) == expected.fitting_matrix_sha256
        )
    return dict(zip(part.stream, part.fitting_matrix_sha256))


def fit_parent(item):
    outer, inner = item
    if pause_requested():
        return dict(outer=outer, inner=inner, paused=True)
    configure_frozen_resources(data.OUT / "source_snapshot")
    config = protocol()
    path = data.parent_path(outer, inner)
    if path.exists():
        read_model(outer, inner)
        return dict(outer=outer, inner=inner, reused_own_completed_model=True)
    assert not path.with_suffix(".complete.json").exists()
    records, training, excluded = data.training_records(outer, inner)
    expected = check_expected_matrices(records, training, excluded, outer, inner)
    fitted = backend.fit_four_modes(
        records,
        training_groups=training,
        excluded_groups=excluded,
        parameters={k: v for k, v in config["classifier"].items() if k != "kind"},
    )
    assert (
        fitted["runtime"] == config["runtime"]
        and fitted["parameters"] == backend.PARAMETERS
    )
    assert all(
        fitted["experts"][s]["fitting_matrix_sha256"] == expected[s] for s in data.MODES
    )
    fitted["training_feature_masks"] = {
        d: raw["excluded_folds"] for d, raw in records.items()
    }
    fitted["standard_training_feature_scopes"] = {
        d: dict(
            training=raw["standard_reference_training_groups"],
            excluded=raw["standard_reference_excluded_groups"],
            reference_sha256=raw["standard_reference_sha256"],
        )
        for d, raw in records.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(fitted, path)
    data.dump(
        path.with_suffix(".complete.json"),
        dict(
            model_sha256=data.sha(path),
            protocol_sha256=data.sha(data.OUT / "protocol.json"),
            outer=outer,
            inner=inner,
            classifiers=4,
            iterations=800,
            source_matrix_sha256=expected,
        ),
    )
    print(
        "FRESH CATBOOST PARENT COMPLETE",
        outer,
        inner,
        fitted["training_pairs"],
        flush=True,
    )
    return dict(outer=outer, inner=inner, reused_own_completed_model=False)


def calibration_cases(outer, *, save=True):
    from matlab_annotation.guarded_class_operating_point import clear_identity_winners
    from annotator.composition_ranges import canonical_lipid_class

    datasets, folds = data.inventory()
    allowed = {d[:7] for d in datasets if folds[d] != outer}
    excluded = {d[:7] for d in datasets if folds[d] == outer}
    winners, manuals, audits = [], [], []
    for inner in range(5):
        if inner == outer:
            continue
        fitted = read_model(outer, inner)
        assert set(fitted["training_cohorts"]) == {
            d[:7] for d in datasets if folds[d] not in {outer, inner}
        }
        for d in datasets:
            if folds[d] != inner:
                continue
            raw = data.load_predictors(d, {outer, inner})
            scores = backend.predict_four_modes(raw, fitted)
            if save:
                path = (
                    data.OUT
                    / "calibration"
                    / f"fold_{outer}"
                    / (d + "_blind_scores.pkl")
                )
                assert not path.exists()
                scores.to_pickle(path)
            # Gold ONLY as authorized outer-TRAINING calibration, after scores.
            labelled, manual = data.authorize_labels(raw, allowed, excluded)
            manuals.append(manual.assign(project=d[:7]))
            for _stream, part in labelled["pool"].groupby("stream", sort=False):
                fixed = clear_identity_winners(
                    part, scores.loc[part.index], ambiguity_margin=0.05
                )
                winners.append(
                    fixed.assign(
                        lipid_class=fixed["class"].map(canonical_lipid_class),
                        project=d[:7],
                    )
                )
            audits.append(
                dict(
                    dataset=d,
                    outer=outer,
                    inner=inner,
                    query_reference_exclusion=sorted({outer, inner}),
                    blind_score_sha256=data.sha(path) if save else None,
                )
            )
    assert len(audits) == sum(folds[d] != outer for d in datasets)
    return (
        pd.concat(winners, ignore_index=True),
        pd.concat(manuals, ignore_index=True),
        allowed,
        excluded,
        audits,
    )


def mode_curves(winners, manual):
    rows = []
    counts = manual.stream.value_counts()
    for threshold in operating.THRESHOLDS:
        for mode in data.MODES:
            part = winners.loc[winners.stream.eq(mode)]
            rows.append(
                dict(
                    stream=mode,
                    threshold=threshold,
                    manual=int(counts[mode]),
                    **operating.mode_counts(part, threshold),
                )
            )
    table = pd.DataFrame(rows)
    operating.validate_curve_rows(table, winners)
    return table


def calibrate(outer):
    if pause_requested():
        return None
    configure_frozen_resources(data.OUT / "source_snapshot")
    from matlab_annotation.operating_point import choose_balanced_training_points
    from matlab_annotation.guarded_class_operating_point import (
        choose_class_training_points,
    )

    folder = data.OUT / "calibration" / f"fold_{outer}"
    path = folder / "operating_points.json"
    if path.exists():
        saved = json.loads(path.read_text())
        assert saved["protocol_sha256"] == data.sha(data.OUT / "protocol.json")
        for p, h in saved["calibration_artifacts_sha256"].items():
            assert data.sha(p) == h
        return outer
    folder.mkdir(parents=True, exist_ok=True)
    winners, manual, allowed, excluded, audits = calibration_cases(outer)
    table = mode_curves(winners, manual)
    cuts = choose_balanced_training_points(table, extras_per_manual=0.095, target=0.9)
    assert cuts is not None
    classes = operating.class_curves(
        winners,
        manual,
        cuts,
        training_groups=allowed,
        excluded_groups=excluded,
        minimum_manual=100,
        minimum_projects=3,
        minimum_positives=20,
    )
    learned = choose_class_training_points(
        classes,
        training_groups=allowed,
        excluded_groups=excluded,
        extras_per_manual=0.095,
        target=0.9,
        time_limit=120,
    )
    chosen = classes.loc[learned["selected_point_indices"]]
    table.to_csv(folder / "training_operating_points.csv", index=False)
    selected_mode = table.loc[
        [
            float(t) == cuts[s]
            for s, t in table[["stream", "threshold"]].itertuples(
                index=False, name=None
            )
        ]
    ]
    selected_mode.to_csv(folder / "selected_training_points.csv", index=False)
    classes.to_csv(folder / "training_class_operating_points.csv", index=False)
    chosen.to_csv(folder / "selected_class_training_points.csv", index=False)
    pd.DataFrame(audits).to_csv(folder / "scope_audit.csv", index=False)
    files = list(folder.glob("*.csv")) + list(folder.glob("*_blind_scores.pkl"))
    learned.update(
        fallback_thresholds=cuts,
        outer=outer,
        outer_excluded=sorted(excluded),
        protocol_sha256=data.sha(data.OUT / "protocol.json"),
        calibration_artifacts_sha256={str(p): data.sha(p) for p in sorted(files)},
    )
    data.dump(path, learned)
    print(
        "NEW INNER MODE+CLASS CALIBRATION COMPLETE",
        outer,
        "solverfallback",
        learned["solver_fallback"],
        flush=True,
    )
    return outer


def require_all_calibrations():
    actual = {}
    for outer in range(5):
        path = data.OUT / "calibration" / f"fold_{outer}" / "operating_points.json"
        saved = json.loads(path.read_text())
        assert saved["protocol_sha256"] == data.sha(data.OUT / "protocol.json")
        actual[str(outer)] = data.sha(path)
    frozen = json.loads((data.OUT / "all_calibrations_frozen.json").read_text())
    assert frozen == actual
    return True


def freeze_all_calibrations():
    hashes = {}
    for outer in range(5):
        path = data.OUT / "calibration" / f"fold_{outer}" / "operating_points.json"
        saved = json.loads(path.read_text())
        assert saved["protocol_sha256"] == data.sha(data.OUT / "protocol.json")
        hashes[str(outer)] = data.sha(path)
    target = data.OUT / "all_calibrations_frozen.json"
    if target.exists():
        assert json.loads(target.read_text()) == hashes
    else:
        data.dump(target, hashes)
    return require_all_calibrations()


def annotate(item):
    dataset, outer = item
    if pause_requested():
        return None
    configure_frozen_resources(data.OUT / "source_snapshot")
    require_all_calibrations()
    dest = data.OUT / "runs" / dataset
    if dest.exists():
        complete = json.loads((dest / "complete.json").read_text())
        assert complete["dataset"] == dataset and complete["fold"] == outer
        assert complete["model_sha256"] == data.sha(data.parent_path(outer, None))
        assert complete["calibration_sha256"] == data.sha(
            data.OUT / "calibration" / f"fold_{outer}" / "operating_points.json"
        )
        for p, h in complete["artifact_sha256"].items():
            assert data.sha(p) == h
        return dataset
    fitted = read_model(outer, None)
    raw = data.load_predictors(dataset, {outer})
    scores = backend.predict_four_modes(raw, fitted)
    calibration = json.loads(
        (
            data.OUT / "calibration" / f"fold_{outer}" / "operating_points.json"
        ).read_text()
    )
    classcuts = {s: calibration["thresholds"].get(s, {}) for s in data.MODES}
    result = assemble(
        dataset,
        outer,
        raw,
        scores,
        calibration["fallback_thresholds"],
        classcuts,
        data.OUT / "source_snapshot",
    )
    dest.mkdir(parents=True, exist_ok=False)
    result["annotations"].to_csv(dest / "automatic_annotations.csv", index=False)
    result["candidates"].to_pickle(dest / "candidates.pkl")
    scores.to_pickle(dest / "scoring_audit.pkl")
    result["standard_audit"].to_csv(dest / "standard_audit.csv", index=False)
    result["mass_audit"].to_csv(dest / "mass_discovery_audit.csv", index=False)
    files = list(dest.iterdir())
    data.dump(
        dest / "complete.json",
        dict(
            dataset=dataset,
            fold=outer,
            status="complete",
            production_enabled=False,
            model_sha256=data.sha(data.parent_path(outer, None)),
            calibration_sha256=data.sha(
                data.OUT / "calibration" / f"fold_{outer}" / "operating_points.json"
            ),
            artifact_sha256={str(p): data.sha(p) for p in files},
            counts=result["annotations"].groupby("stream").size().to_dict(),
        ),
    )
    print(
        "BLIND CATBOOST WHOLE-PIPELINE OUTPUT",
        dataset,
        len(result["annotations"]),
        flush=True,
    )
    return dataset


def main():
    config = initialize()
    with ProcessPoolExecutor(max_workers=config["numerical_workers"]) as executor:
        fits = list(executor.map(fit_parent, data.parents()))
    if pause_requested() or any(result.get("paused", False) for result in fits):
        checkpoint("model_parents")
        return
    assert len(fits) == 25
    with ProcessPoolExecutor(max_workers=config["numerical_workers"]) as executor:
        calibrations = list(executor.map(calibrate, range(5)))
    if pause_requested() or any(result is None for result in calibrations):
        checkpoint("inner_calibration")
        return
    assert sorted(calibrations) == list(range(5)) and freeze_all_calibrations()
    datasets, folds = data.inventory()
    with ProcessPoolExecutor(max_workers=config["numerical_workers"]) as executor:
        completed = list(executor.map(annotate, [(d, folds[d]) for d in datasets]))
    if pause_requested() or any(result is None for result in completed):
        checkpoint("outer_datasets")
        return
    assert sorted(completed) == sorted(datasets)
    data.dump(
        data.OUT / "producer_complete.json",
        dict(
            all_blind_outputs_complete=True,
            independent_verification_complete=False,
            performance_evaluation_complete=False,
            protocol_sha256=data.sha(data.OUT / "protocol.json"),
            parents=25,
            classifiers=100,
            calibrations=5,
            datasets=33,
            dataset_complete_sha256={
                d: data.sha(data.OUT / "runs" / d / "complete.json") for d in datasets
            },
        ),
    )
    print(
        "048 BLIND OUTPUTS COMPLETE; EVALUATION/INDEPENDENTVERIFICATION/FULL132CHAT REQUIRED",
        flush=True,
    )


if __name__ == "__main__":
    main()
