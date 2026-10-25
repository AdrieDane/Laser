"""043 full25model/528training/132INNER/fivecuts/33rawexport/gold replay."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
import json
from pathlib import Path

import pandas as pd

from frozen_resources import configure_frozen_resources
from verify_immutable_exports import check_export
import run_qe_raw_mass_profile_experiment as runner

OUT, QE = runner.OUT, runner.QE


def normalized_csv(frame):
    return pd.read_csv(
        StringIO(frame.to_csv(index=False)),
        float_precision="round_trip",
        low_memory=False,
    )


def check_enabled_curves(chosen):
    enabled = chosen.loc[chosen.threshold.lt(1)]
    assert enabled.feasible.all() and enabled.eligible.all()
    assert enabled.different_peak.eq(0).all() and enabled.added_correct.ge(10).all()
    assert (enabled.extras <= 0.2 * enabled.added_correct).all()
    assert (enabled.extras <= 0.02 * enabled.manual).all()
    assert enabled.minimum_leave_one_project_agreement.ge(0.8).all()


def verify_parent(job):
    configure_frozen_resources(OUT / "source_snapshot")
    outer, inner, datasets, folds, protocol = job
    data, training, excluded = runner.training_data(outer, inner, datasets, folds)
    path = runner.parent_path(outer, inner)
    model = pd.read_pickle(path)
    assert model["format"] == "laser-fixed-four-mode-evidence-experts-v1"
    assert set(model["experts"]) == set(runner.MODES)
    assert (
        set(model["training_cohorts"]) == training
        and set(model["excluded_groups"]) == excluded
    )
    assert model["training_feature_masks"] == {
        d: raw["excluded_folds"] for d, raw in data.items()
    }
    assert len(model["columns"]) == 131
    complete = json.loads(path.with_suffix(".complete.json").read_text())
    assert (
        complete["model_sha256"] == runner.sha(path)
        and complete["fresh_classifiers"] == 4
    )
    preflight = json.loads((runner.PREFLIGHT / "parent_support.json").read_text())
    expected = next(r for r in preflight if r["outer"] == outer and r["inner"] == inner)
    parameters = runner.inherited.common.estimator_parameters(protocol["classifier"])
    old_path = runner.CORE / (
        f"fold_{outer}_nested_model.pkl"
        if inner is None
        else f"calibration/fold_{outer}/inner_block_{inner}_model.pkl"
    )
    old = pd.read_pickle(old_path)
    total_rows = total_positive = 0
    for stream, expert in model["experts"].items():
        rows, y, projects = [], [], []
        for d, raw in data.items():
            selected = raw["pool"].stream.eq(stream) & raw["pool"].training_label_known
            rows.append(raw["evidence"].loc[selected])
            y.extend(raw["pool"].loc[selected, "training_positive"].astype(bool))
            projects.extend([d[:7]] * int(selected.sum()))
        x = pd.concat(rows, ignore_index=True)
        assert set(projects) == training and not set(projects) & excluded
        assert expert["training_cohorts"] == sorted(training) and expert[
            "excluded_groups"
        ] == sorted(excluded)
        assert expert["columns"] == model["columns"] == list(x)
        assert expert["training_pairs"] == len(y) and expert["positives"] == sum(y)
        assert old["experts"][stream]["training_pairs"] == len(y)
        assert old["experts"][stream]["positives"] == sum(y)
        categories = {
            c: {v: i for i, v in enumerate(sorted(x[c].astype(str).unique()))}
            for c in x
            if not pd.api.types.is_numeric_dtype(x[c])
        }
        assert (
            categories
            == expert["categories"]
            == expected["support"][stream]["categories"]
        )
        assert categories == old["experts"][stream]["categories"]
        assert all(c in list(x)[:108] for c in categories)
        assert expected["support"][stream] == dict(
            observations=len(y),
            positives=int(sum(y)),
            negatives=len(y) - int(sum(y)),
            actual_projects=sorted(set(projects)),
            project_counts=dict(Counter(projects)),
            columns=list(x),
            categories=categories,
        )
        params = expert["estimator"].get_params()
        assert all(params[k] == v for k, v in parameters.items())
        assert list(params["categorical_features"]) == [c in categories for c in x]
        assert expert["estimator"].n_features_in_ == 131
        assert expert["estimator"].n_iter_ == parameters["max_iter"]
        total_rows += len(y)
        total_positive += int(sum(y))
    assert (
        model["training_pairs"] == total_rows and model["positives"] == total_positive
    )
    print(
        "043 ALLMODEL/TRAINING/LABELS/SCHEMA/PARAMETERS PASS",
        outer,
        inner,
        len(data),
        flush=True,
    )
    return dict(training_queries=len(data), fresh_classifiers=4)


def verify_calibration(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts
    from matlab_annotation.add_only_class_completion import (
        class_completion_curves,
        choose_completion_thresholds,
    )

    datasets, folds = runner.inventory()
    allowed = {d[:7] for d in datasets if folds[d] != outer}
    excluded = {d[:7] for d in datasets if folds[d] == outer}
    folder = OUT / "calibration" / f"fold_{outer}"
    cases, manuals, counts, audits = [], [], Counter(), []
    for inner in range(5):
        if inner == outer:
            continue
        model = pd.read_pickle(runner.parent_path(outer, inner))
        oldmodel = pd.read_pickle(
            runner.CORE / "calibration" / f"fold_{outer}/inner_block_{inner}_model.pkl"
        )
        for d in datasets:
            if folds[d] != inner:
                continue
            oldraw, old_scores = runner.inherited.reproduce(
                d,
                {outer, inner},
                oldmodel,
                runner.CORE
                / "calibration"
                / f"fold_{outer}"
                / (d + "_blind_scores.pkl"),
            )
            raw = runner.load_predictors(d, {outer, inner})
            pd.testing.assert_frame_equal(raw["pool"], oldraw["pool"], check_exact=True)
            pd.testing.assert_frame_equal(
                raw["evidence"].iloc[:, :108], oldraw["evidence"], check_exact=True
            )
            scores = predict_mode_evidence_experts(raw, model)
            pd.testing.assert_series_equal(
                scores,
                pd.read_pickle(folder / (d + "_blind_scores.pkl")),
                check_exact=True,
            )
            labeled, manual, _ = runner.authorize_labels(raw, allowed, excluded)
            occupied, old_core = runner.inner_previous(
                d, outer, inner, labeled, old_scores
            )
            counts.update(
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
            cases.append(
                runner.proposals(d, {outer, inner}, labeled, scores, occupied).assign(
                    project=d[:7]
                )
            )
            manuals.append(manual.assign(project=d[:7]))
            audits.append(
                dict(
                    dataset=d,
                    outer=outer,
                    inner=inner,
                    new_model_sha256=runner.sha(runner.parent_path(outer, inner)),
                )
            )
    curves = class_completion_curves(
        pd.concat(cases, ignore_index=True),
        pd.concat(manuals, ignore_index=True),
        counts,
        training_groups=allowed,
        excluded_groups=excluded,
    )
    thresholds, chosen = choose_completion_thresholds(curves)
    for expected, name in (
        (curves, "training_class_operating_points.csv"),
        (chosen, "selected_training_points.csv"),
        (pd.DataFrame(audits), "scope_audit.csv"),
    ):
        pd.testing.assert_frame_equal(
            normalized_csv(expected),
            pd.read_csv(folder / name, float_precision="round_trip", low_memory=False),
            check_exact=True,
        )
    cuts = json.loads((folder / "operating_points.json").read_text())
    assert cuts == dict(
        thresholds=thresholds,
        training_groups=sorted(allowed),
        outer_excluded=sorted(excluded),
    )
    check_enabled_curves(chosen)
    print(
        "043 ALLINNER/SCORES/FULLCLASSGRID/CUTOFFS PASS", outer, len(audits), flush=True
    )
    return len(audits)


def verify_dataset(dataset):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts
    from matlab_annotation.add_only_class_completion import select_completion

    _, folds = runner.inventory()
    outer = folds[dataset]
    model = pd.read_pickle(runner.parent_path(outer, None))
    raw = runner.load_predictors(dataset, {outer})
    scores = predict_mode_evidence_experts(raw, model)
    folder = OUT / "runs" / dataset
    pd.testing.assert_series_equal(
        scores, pd.read_pickle(folder / "scoring_audit.pkl"), check_exact=True
    )
    previous = runner.previous_annotations(dataset)
    cuts = json.loads(
        (OUT / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )["thresholds"]
    free = runner.proposals(dataset, {outer}, raw, scores, previous)
    selected = select_completion(free, cuts)
    expected, candidates, filled = runner.assemble(dataset, previous, selected)
    pd.testing.assert_frame_equal(
        candidates, pd.read_pickle(folder / "candidates.pkl"), check_exact=True
    )
    pd.testing.assert_frame_equal(
        filled, pd.read_pickle(folder / "private_additions.pkl"), check_exact=True
    )
    actual = pd.read_csv(
        folder / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    normalized = pd.read_csv(
        StringIO(expected.to_csv(index=False)),
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    pd.testing.assert_frame_equal(actual, normalized, check_exact=True)
    report = check_export(
        folder / "automatic_annotations.csv",
        previous,
        candidates,
        pd.read_pickle(runner.PREVIOUS / "runs" / dataset / "candidates.pkl"),
        pd.read_pickle(runner.BASE / "runs" / dataset / "full_annotation.pkl"),
    )
    identities = [
        json.loads(
            (
                QE / v / "evaluation_v2" / dataset / "evaluation_identity.json"
            ).read_text()
        )
        for v in ("baseline", "experiment_029", "experiment_041", OUT.name)
    ]
    assert len({i["manual_sha256"] for i in identities}) == 1
    assert identities[-1]["selected_sha256"] == runner.sha(
        folder / "automatic_annotations.csv"
    )
    assert identities[-1]["candidates_sha256"] == runner.sha(folder / "candidates.pkl")
    complete = json.loads((folder / "complete.json").read_text())
    assert complete["previous_calls_immutable"] and complete[
        "added_annotations"
    ] == len(filled)
    assert complete["counts"] == actual.groupby("stream").size().to_dict()
    print(
        "043 ALL4MODES/OLDALLCELLS/NEWRAWQUANT/EXPORT/GOLD PASS",
        dataset,
        len(filled),
        flush=True,
    )
    return dict(dataset=dataset, added_annotations=len(filled), **report)


def main():
    protocol, datasets = runner.initialize()
    _, folds = runner.inventory()
    assert protocol["planned_verifier_sha256"] == runner.sha(Path(__file__))
    paths = {
        Path(__file__),
        Path(__file__).with_name("verify_immutable_exports.py"),
        OUT / "protocol.json",
        OUT / "runner_source.py",
        runner.PREFLIGHT / "receipt.json",
        runner.PREFLIGHT / "parent_support.json",
    }
    paths.update(OUT.glob("fold_*_nested_model*"))
    for name in ("calibration", "runs", "evaluation_v2", "reports_v2"):
        paths.update(p for p in (OUT / name).rglob("*") if p.is_file())
    hashes = {str(p): runner.sha(p) for p in sorted(paths)}
    jobs = [
        (o, i, datasets, folds, protocol)
        for o in range(5)
        for i in [None, *sorted(set(folds.values()) - {o})]
    ]
    with ProcessPoolExecutor(max_workers=3) as executor:
        parents = list(executor.map(verify_parent, jobs))
        inner = list(executor.map(verify_calibration, range(5)))
        outputs = list(executor.map(verify_dataset, datasets))
    assert len(parents) == 25 and sum(p["training_queries"] for p in parents) == 528
    assert sum(p["fresh_classifiers"] for p in parents) == 100
    assert (
        sum(inner) == 132
        and len(outputs) == 33
        and sum(r["raw_stream_checks"] for r in outputs) == 132
    )
    assert runner.source_inputs() == protocol["source_input_sha256"]
    assert all(runner.sha(Path(p)) == h for p, h in hashes.items())
    assert all(
        runner.sha(OUT / "source_snapshot" / p) == h
        for p, h in protocol["source_snapshot_sha256"].items()
    )
    assert (
        runner.sha(runner.PREFLIGHT / "parent_support.json")
        == protocol["preflight_support_sha256"]
    )
    runner.dump(
        OUT / "integrity_receipt.json",
        dict(
            all_passed=True,
            source_sha256=runner.sha(Path(__file__)),
            planned_verifier_sha256=protocol["planned_verifier_sha256"],
            source_input_hash_checks=len(protocol["source_input_sha256"]),
            source_snapshot_hash_checks=len(protocol["source_snapshot_sha256"]),
            bound_verification_artifacts=len(hashes),
            fresh_four_mode_parents=25,
            fresh_classifiers=100,
            training_query_checks=528,
            inner_score_replays=132,
            full_class_grid_replays=5,
            outer_datasets=33,
            original_coordinate_stream_checks=132,
            old_allcell_preservation_checks=33,
            candidate_export_parity_checks=33,
            identical_manual_reference_hash_checks=33,
            added_annotations=sum(r["added_annotations"] for r in outputs),
            available_measurement_cells_checked=sum(
                r["available_measurement_cells"] for r in outputs
            ),
            inherited_empty_old_cache_cells=sum(
                r["inherited_empty_old_measurement_cells"] for r in outputs
            ),
            inherited_accepted_candidate_flags=sum(r["legacy_flags"] for r in outputs),
            preexisting_roundoff_cells=sum(
                r["preexisting_roundoff_cells"] for r in outputs
            ),
            note="Complete engineering replay,not chemicalidentity/FDR/externalvalidation/deploymentapproval. ALL041callsimmutable;100freshbinarymodeclassifiers,131inputs,all33/132samegold. Primaryhistorical92100includes1456STDs;endogenous90644companionseparate.",
        ),
    )
    print("043 MASTER ALLPASS", flush=True)


if __name__ == "__main__":
    main()
