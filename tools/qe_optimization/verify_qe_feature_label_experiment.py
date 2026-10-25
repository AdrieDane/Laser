"""042fullmodel/preparation/INNER/outer/originalmeasurement engineering replay."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
import json
from pathlib import Path

import pandas as pd

from frozen_resources import configure_frozen_resources
from verify_immutable_exports import check_export
import run_qe_feature_label_experiment as runner

OUT, QE = runner.OUT, runner.QE


def normalized_csv(frame):
    return pd.read_csv(
        StringIO(frame.to_csv(index=False)),
        float_precision="round_trip",
        low_memory=False,
    )


def verify_parent(job):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.feature_label_model import (
        prepare_feature_label_training,
        _supported_classes,
        FORMAT,
    )

    outer, inner, datasets, folds, protocol = job
    data, training, excluded = runner.training_data(outer, inner, datasets, folds)
    records, audit, original_columns, feature_columns = prepare_feature_label_training(
        data, training_groups=training, excluded_groups=excluded
    )
    path = runner.parent_path(outer, inner)
    model = pd.read_pickle(path)
    assert model["format"] == FORMAT and set(model["experts"]) == set(runner.MODES)
    assert model["training_reference_permissions"] == sorted(training) and model[
        "excluded_groups"
    ] == sorted(excluded)
    assert model["training_query_audit"] == audit
    assert model["training_feature_masks"] == {
        d: raw["excluded_folds"] for d, raw in data.items()
    }
    assert (
        model["columns"] == original_columns
        and model["feature_columns"] == feature_columns
    )
    assert len(original_columns) == 108 and len(feature_columns) == 315
    metadata = json.loads(path.with_suffix(".complete.json").read_text())
    assert metadata["model_sha256"] == runner.sha(path)
    preflight = json.loads((runner.PREFLIGHT / "parent_support.json").read_text())
    expected = next(r for r in preflight if r["outer"] == outer and r["inner"] == inner)
    parameters = runner.inherited.common.estimator_parameters(protocol["classifier"])
    fresh = 0
    for stream, r in records.items():
        expert = model["experts"][stream]
        assert expert["singleton_supported_classes"] == _supported_classes(
            r["class_counts"], r["class_projects"]
        )
        assert expert["ambiguous_supported_classes"] == _supported_classes(
            r["conditional_counts"], r["conditional_projects"]
        )
        assert (
            expert["class_positive_observations"]
            == dict(r["class_counts"])
            == expected["support"][stream]["class_counts"]
        )
        assert (
            expert["class_positive_projects"]
            == {c: sorted(g) for c, g in r["class_projects"].items()}
            == expected["support"][stream]["class_projects"]
        )
        assert (
            expert["conditional_class_positive_observations"]
            == dict(r["conditional_counts"])
            == expected["support"][stream]["conditional_class_counts"]
        )
        assert (
            expert["conditional_class_positive_projects"]
            == {c: sorted(g) for c, g in r["conditional_projects"].items()}
            == expected["support"][stream]["conditional_class_projects"]
        )
        for stage, prefix in (("detector", "detector"), ("conditional", "label")):
            x = pd.concat(r[prefix + "_x"], ignore_index=True)
            y = pd.Series(r[prefix + "_y"], dtype=bool)
            groups = pd.Series(r[prefix + "_groups"], dtype=str)
            support = dict(
                observations=len(y),
                positives=int(y.sum()),
                negatives=int((~y).sum()),
                actual_projects=sorted(set(groups)),
                positive_projects=sorted(set(groups[y])),
                negative_projects=sorted(set(groups[~y])),
            )
            assert expert[stage + "_support"] == support
            assert all(
                support[k] == expected["support"][stream][prefix][k]
                for k in ("observations", "positives", "negatives", "actual_projects")
            )
            supported = (
                support["positives"] >= 20
                and support["negatives"] >= 20
                and len(support["positive_projects"]) >= 3
                and len(support["negative_projects"]) >= 3
            )
            fitted = expert[stage]
            assert (fitted is not None) == supported
            if fitted is None:
                continue
            fresh += 1
            assert all(fitted[k] == v for k, v in support.items())
            assert fitted["columns"] == list(x)
            categories = {
                c: {v: i for i, v in enumerate(sorted(x[c].astype(str).unique()))}
                for c in x
                if not pd.api.types.is_numeric_dtype(x[c])
            }
            assert fitted["categories"] == categories
            assert fitted["estimator"].n_features_in_ == len(x.columns)
            params = fitted["estimator"].get_params()
            assert all(params[k] == v for k, v in parameters.items())
            assert list(params["categorical_features"]) == [c in categories for c in x]
            assert not set(groups) & excluded and set(groups).issubset(training)
    assert fresh == model["fresh_classifier_fits"] == metadata["fresh_classifiers"]
    print(
        "042 ALLTRAINING/PREPARATION/STAGES/SUPPORT PASS",
        outer,
        inner,
        len(data),
        fresh,
        flush=True,
    )
    return dict(training_queries=len(data), fresh_classifiers=fresh)


def verify_calibration(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.feature_label_model import predict_feature_label_model
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
            raw, old_scores = runner.inherited.reproduce(
                d,
                {outer, inner},
                oldmodel,
                runner.CORE
                / "calibration"
                / f"fold_{outer}"
                / (d + "_blind_scores.pkl"),
            )
            ranks = predict_feature_label_model(raw, model)
            pd.testing.assert_frame_equal(
                ranks,
                pd.read_pickle(folder / (d + "_blind_scores.pkl")),
                check_exact=True,
            )
            labeled, manual, _ = runner.authorize_labels(raw, allowed, excluded)
            occupied, oldcore = runner.inner_previous(
                d, outer, inner, labeled, old_scores
            )
            counts.update(
                {
                    (s, c): int(n)
                    for (s, c), n in oldcore.assign(
                        _cls=oldcore["class"].map(canonical_lipid_class)
                    )
                    .groupby(["stream", "_cls"])
                    .training_positive.sum()
                    .items()
                }
            )
            cases.append(
                runner.proposals(d, {outer, inner}, labeled, ranks, occupied).assign(
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
    table = class_completion_curves(
        pd.concat(cases, ignore_index=True),
        pd.concat(manuals, ignore_index=True),
        counts,
        training_groups=allowed,
        excluded_groups=excluded,
    )
    cuts, chosen = choose_completion_thresholds(table)
    for name, frame in (
        ("training_class_operating_points.csv", table),
        ("selected_training_points.csv", chosen),
        ("scope_audit.csv", pd.DataFrame(audits)),
    ):
        pd.testing.assert_frame_equal(
            pd.read_csv(folder / name, float_precision="round_trip"),
            normalized_csv(frame),
            check_exact=True,
        )
    assert json.loads((folder / "operating_points.json").read_text()) == dict(
        thresholds=cuts,
        training_groups=sorted(allowed),
        outer_excluded=sorted(excluded),
    )
    enabled = chosen.loc[chosen.threshold.lt(1)]
    assert enabled.feasible.all() and enabled.different_peak.eq(0).all()
    assert (enabled.extras <= 0.2 * enabled.added_correct).all() and (
        enabled.extras <= 0.02 * enabled.manual
    ).all()
    assert enabled.minimum_leave_one_project_agreement.ge(0.8).all()
    print(
        "042 ALLINNER/RANKS/FULLCLASSGRID/CUTOFFS PASS", outer, len(audits), flush=True
    )
    return len(audits)


def verify_dataset(dataset):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.feature_label_model import predict_feature_label_model
    from matlab_annotation.add_only_class_completion import select_completion

    _, folds = runner.inventory()
    outer = folds[dataset]
    model = pd.read_pickle(runner.parent_path(outer, None))
    raw = runner.load_predictors(dataset, {outer})
    ranks = predict_feature_label_model(raw, model)
    folder = OUT / "runs" / dataset
    pd.testing.assert_frame_equal(
        ranks, pd.read_pickle(folder / "scoring_audit.pkl"), check_exact=True
    )
    previous = runner.previous_annotations(dataset)
    cuts = json.loads(
        (OUT / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )["thresholds"]
    free = runner.proposals(dataset, {outer}, raw, ranks, previous)
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
        "042 ALL4MODES/OLDALLCELLS/NEWRAWQUANT/EXPORT/GOLD PASS",
        dataset,
        len(filled),
        flush=True,
    )
    return dict(dataset=dataset, added_annotations=len(filled), **report)


def main():
    protocol, datasets = runner.initialize()
    _, folds = runner.inventory()
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
    assert (
        sum(inner) == 132
        and len(outputs) == 33
        and sum(r["raw_stream_checks"] for r in outputs) == 132
    )
    assert runner.source_inputs() == protocol["source_input_sha256"]
    assert all(runner.sha(Path(p)) == h for p, h in hashes.items())
    for p, h in protocol["source_snapshot_sha256"].items():
        assert runner.sha(OUT / "source_snapshot" / p) == h
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
            fresh_feature_label_parents=25,
            fresh_mode_stage_classifiers=sum(p["fresh_classifiers"] for p in parents),
            training_query_checks=528,
            inner_rank_table_replays=132,
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
            note="Complete engineering replay,not chemicalidentity/FDR/externalvalidation/deploymentapproval. Allold041calls immutable;newfeaturedetectorsandconditional-labelmodels freshly nested;all33datasets/132modes/samegoldreference.",
        ),
    )
    print("042 MASTER ALLPASS", flush=True)


if __name__ == "__main__":
    main()
