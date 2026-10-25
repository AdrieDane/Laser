"""039: ALL nested conditional fits, blind grids/cuts and full005/027 outputs."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_qe_isomer_label_experiment as runner
from frozen_resources import configure_frozen_resources
from preflight_qe_isomer_labels import verify_pairs, accepted_keys
from verify_qe_conditional_peak_experiment import independent_counts

OUT, OLD, QE, BASE = runner.OUT, runner.OLD, runner.QE, runner.BASE
sha = runner.sha


def inventory():
    return json.loads((OUT / "run_manifest.json").read_text())["datasets"], json.loads(
        (QE / "rt_atlas_development/fold_assignment.json").read_text()
    )


def verify_model(path):
    from matlab_annotation.evidence_training import validate_feature_scope

    model = pd.read_pickle(path)
    old_path = OLD / path.relative_to(OUT)
    old = pd.read_pickle(old_path)
    training, excluded = (
        set(model["training_reference_permissions"]),
        set(model["excluded_groups"]),
    )
    assert training == set(old["training_cohorts"]) and excluded == set(
        old["excluded_groups"]
    )
    assert set(model["training_cohorts"]) == training
    assert model["base_pointwise_model_sha256"] == sha(old_path)
    assert model["original_columns"] == old["columns"]
    for key in ("training_feature_masks", "standard_training_feature_scopes"):
        assert model[key] == old[key]
    records = {}
    for dataset, mask in model["training_feature_masks"].items():
        raw = runner.load_predictors(dataset, set(mask))
        validate_feature_scope(raw, training, excluded, query_is_training=True)
        records[dataset], _, _ = runner.with_training_labels(
            raw, permitted_groups=training, excluded_groups=excluded
        )
        pd.testing.assert_frame_equal(
            raw["evidence"], records[dataset]["evidence"], check_exact=True
        )
        scope = model["standard_training_feature_scopes"][dataset]
        assert (
            set(scope["training"]).issubset(training)
            and not set(scope["training"]) & excluded
        )
        assert dataset[:7] not in scope["training"] and dataset[:7] in scope["excluded"]
        reference = (
            QE
            / "nested_standard_development_v1"
            / ("exclude_" + "_".join(map(str, sorted(mask))) + ".json")
        )
        assert sha(reference) == scope["reference_sha256"]
    expected = verify_pairs(records, training, excluded)
    protocol = json.loads((OUT / "protocol.json").read_text())
    fitted = 0
    for mode, expert in model["experts"].items():
        values = expected[mode]
        assert expert["pair_rows"] == values["pair_rows"]
        assert (
            expert["supervised_contributing_projects"]
            == values["supervised_contributing_projects"]
        )
        assert (
            set(expert["training_reference_permissions"]) == training
            and set(expert["excluded_groups"]) == excluded
        )
        enough = (
            values["pair_rows"] >= protocol["minimum_pair_rows"]
            and len(values["supervised_contributing_projects"])
            >= protocol["minimum_contributing_projects"]
        )
        assert expert["supported"] == enough
        if values["pair_rows"]:
            for key in (
                "pair_rows_by_project",
                "features_by_project",
                "class_pair_projects",
            ):
                assert expert[key] == values[key]
        if enough:
            fitted += 1
            assert (
                expert["columns"] == values["columns"]
                and expert["categories"] == values["categories"]
            )
            assert expert["estimator"].n_iter_ == 180
            for key, value in protocol["classifier"].items():
                if key != "kind":
                    assert expert["estimator"].get_params()[key] == value
    print(
        "039 ALL TRAINING LABELPAIRS/WEIGHTS/SCOPES PASS",
        path.relative_to(OUT),
        flush=True,
    )
    return dict(
        model=path.relative_to(OUT).as_posix(),
        training_queries=len(records),
        fresh_experts=fitted,
    )


def verify_calibration(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.isomer_label_resolution import isomer_label_preferences
    from run_qe_conditional_peak_experiment import base_core

    protocol = json.loads((OUT / "protocol.json").read_text())
    projects, folds = inventory()
    folder = OUT / "calibration" / f"fold_{outer}"
    base_thresholds = json.loads(
        (OLD / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )["thresholds"]
    allowed = {d[:7] for d in projects if folds[d] != outer}
    excluded = {d[:7] for d in projects if folds[d] == outer}
    cases, totals, support = {}, Counter(), defaultdict(set)
    baseline_c, baseline_w, baseline_e = Counter(), Counter(), Counter()
    empty = pd.DataFrame(columns=["annotation_index", "new_feature_id"])
    for inner in range(5):
        if inner == outer:
            continue
        model = pd.read_pickle(folder / f"inner_block_{inner}_model.pkl")
        assert set(model["training_reference_permissions"]) == {
            d[:7] for d in projects if folds[d] not in {outer, inner}
        }
        assert set(model["excluded_groups"]) == {
            d[:7] for d in projects if folds[d] in {outer, inner}
        }
        for dataset in projects:
            if folds[dataset] != inner:
                continue
            raw = runner.load_predictors(dataset, {outer, inner})
            scores = runner.pointwise_scores(dataset, outer, raw, inner=inner)
            pd.testing.assert_series_equal(
                scores,
                pd.read_pickle(folder / (dataset + "_pointwise_scores.pkl")),
                check_exact=True,
            )
            protected, masses = runner.physical_context(dataset, {outer, inner})
            prefs = isomer_label_preferences(
                raw,
                model,
                protected_features_by_stream=protected,
                protected_masses_by_stream=masses,
            )
            pd.testing.assert_frame_equal(
                prefs,
                pd.read_pickle(folder / (dataset + "_blind_preferences.pkl")),
                check_exact=True,
            )
            disabled, same_scores, _ = runner.select_core(
                raw["pool"],
                scores,
                prefs,
                {s: 1.0 for s in runner.MODES},
                base_thresholds,
            )
            pd.testing.assert_series_equal(same_scores, scores, check_exact=True)
            assert accepted_keys(disabled) == accepted_keys(
                base_core(dataset, outer, raw)
            )
            _, manual, _ = runner.with_training_labels(
                raw, permitted_groups=allowed, excluded_groups=excluded
            )
            c, w, e = independent_counts(manual, disabled, empty)
            assert (c, w, e) == runner.case_counts(
                runner.truth_case(manual, disabled), empty
            )
            baseline_c.update(c)
            baseline_w.update(w)
            baseline_e.update(e)
            cases[dataset] = (raw["pool"], scores, prefs, manual)
            for r in manual.itertuples():
                cell = (r.stream, r.lipid_class)
                totals[cell] += 1
                support[cell].add(dataset[:7])
    assert len(cases) == sum(folds[d] != outer for d in projects)
    common = {c for c, n in totals.items() if n >= 100 and len(support[c]) >= 3}
    class_rows, mode_rows = [], []
    for threshold in protocol["confidence_grid"]:
        correct, wrong, extras, resolved = Counter(), Counter(), Counter(), Counter()
        for pool, scores, prefs, manual in cases.values():
            selected, _, audit = runner.select_core(
                pool,
                scores,
                prefs,
                {s: threshold for s in runner.MODES},
                base_thresholds,
            )
            c, w, e = independent_counts(manual, selected, empty)
            assert (c, w, e) == runner.case_counts(
                runner.truth_case(manual, selected), empty
            )
            correct.update(c)
            wrong.update(w)
            extras.update(e)
            if len(audit):
                resolved.update(
                    audit[audit.decision.eq("resolved_isomer_label_detection_proxy")]
                    .stream.value_counts()
                    .to_dict()
                )
        class_rows.extend(
            dict(
                stream=s,
                lipid_class=cls,
                threshold=threshold,
                manual=n,
                training_project_groups=len(support[(s, cls)]),
                eligible_priority=(s, cls) in common,
                correct_before=baseline_c[(s, cls)],
                correct_after=correct[(s, cls)],
                wrong_before=baseline_w[(s, cls)],
                wrong_after=wrong[(s, cls)],
            )
            for (s, cls), n in totals.items()
        )
        for mode in runner.MODES:
            cells = [c for c in sorted(common) if c[0] == mode]
            cb, wb = (
                sum(n for c, n in baseline_c.items() if c[0] == mode),
                sum(n for c, n in baseline_w.items() if c[0] == mode),
            )
            ca, wa = (
                sum(n for c, n in correct.items() if c[0] == mode),
                sum(n for c, n in wrong.items() if c[0] == mode),
            )
            objective = (
                sum(min(0.9, correct[c] / totals[c]) for c in cells) / len(cells)
                if cells
                else 0.0
            )
            mode_rows.append(
                dict(
                    stream=mode,
                    threshold=threshold,
                    manual=sum(n for c, n in totals.items() if c[0] == mode),
                    correct_before=cb,
                    correct_after=ca,
                    wrong_before=wb,
                    wrong_after=wa,
                    extras_before=baseline_e[mode],
                    extras_after=extras[mode],
                    resolved_clusters=resolved[mode],
                    priority_class_modes=len(cells),
                    capped90_class_mean=objective,
                    feasible=(
                        ca >= cb and wa <= wb and extras[mode] <= baseline_e[mode]
                    ),
                )
            )
    classes, modes = pd.DataFrame(class_rows), pd.DataFrame(mode_rows)
    for frame, name in (
        (classes, "class_isomer_calibration.csv"),
        (modes, "mode_isomer_calibration.csv"),
    ):
        pd.testing.assert_frame_equal(
            frame,
            pd.read_csv(folder / name, float_precision="round_trip"),
            check_exact=True,
            check_dtype=False,
        )
    chosen = {}
    for mode in runner.MODES:
        valid = modes[modes.stream.eq(mode) & modes.feasible]
        assert valid.threshold.eq(1.0).any()
        chosen[mode] = float(
            valid.sort_values(
                [
                    "capped90_class_mean",
                    "correct_after",
                    "wrong_after",
                    "extras_after",
                    "threshold",
                ],
                ascending=[False, False, True, True, False],
                kind="stable",
            )
            .iloc[0]
            .threshold
        )
    actual = json.loads((folder / "operating_points.json").read_text())
    assert (
        chosen == actual["thresholds"] and base_thresholds == actual["base_thresholds"]
    )
    assert actual["training_groups"] == sorted(allowed) and actual[
        "outer_excluded"
    ] == sorted(excluded)
    assert actual["class_table_sha256"] == sha(folder / "class_isomer_calibration.csv")
    assert actual["mode_table_sha256"] == sha(folder / "mode_isomer_calibration.csv")
    selected = modes[
        [
            t == chosen[s]
            for s, t in modes[["stream", "threshold"]].itertuples(
                index=False, name=None
            )
        ]
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        selected,
        pd.read_csv(
            folder / "selected_training_points.csv", float_precision="round_trip"
        ),
        check_exact=True,
        check_dtype=False,
    )
    print(
        "039 ALL INNER SCORES/PREFERENCES/GRIDS/CUTS PASS",
        outer,
        len(cases),
        flush=True,
    )
    return len(cases)


def independent_final_pipeline(dataset, outer, selected):
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards
    from matlab_annotation.calibrated_mass_discovery import select_calibrated_additions

    baseline = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
    catalogue = declared_standard_targets(
        OUT / "source_snapshot/reference_data/matlab_pipeline",
        baseline["internal_standards_expected"],
    )
    model = json.loads(
        (runner.STD / "models" / f"fold_{outer}_standard_models.json").read_text()
    )
    mass = pd.read_pickle(
        runner.MASS / "runs" / dataset / "private_mass_candidates.pkl"
    )
    anchors = pd.concat(
        [
            p["annotations"].copy().assign(stream=s)
            for s, p in baseline["streams"].items()
        ],
        ignore_index=True,
    )
    parts = []
    for mode, payload in baseline["streams"].items():
        core = selected[selected.stream.eq(mode)]
        std = (
            payload["annotations"][
                payload["annotations"].is_is.fillna(False).astype(bool)
            ]
            .copy()
            .assign(stream=mode)
        )
        accepted, _, _ = complete_declared_standards(
            pd.concat([core, std], ignore_index=True),
            payload["features"],
            catalogue[catalogue.stream.eq(mode)],
            model["library"],
            model["atlas"],
            model["transfer"],
            model["channels"],
            stream=mode,
            alignment_annotations=payload["annotations"],
        )
        additions = select_calibrated_additions(
            mass[mass.stream.eq(mode)] if len(mass) else mass,
            anchors,
            accepted,
            stream=mode,
        )
        parts.append(
            pd.concat([accepted, additions], ignore_index=True).assign(stream=mode)
        )
    return pd.concat(parts, ignore_index=True), baseline


def verify_dataset(dataset):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.isomer_label_resolution import isomer_label_preferences

    _, folds = inventory()
    outer = folds[dataset]
    folder = OUT / "runs" / dataset
    model = pd.read_pickle(OUT / f"fold_{outer}_nested_model.pkl")
    saved_cuts = json.loads(
        (OUT / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )
    raw = runner.load_predictors(dataset, {outer})
    scores = runner.pointwise_scores(dataset, outer, raw)
    pd.testing.assert_series_equal(
        scores, pd.read_pickle(folder / "scoring_audit.pkl"), check_exact=True
    )
    protected, masses = runner.physical_context(dataset, {outer})
    prefs = isomer_label_preferences(
        raw,
        model,
        protected_features_by_stream=protected,
        protected_masses_by_stream=masses,
    )
    pd.testing.assert_frame_equal(
        prefs, pd.read_pickle(folder / "blind_preferences.pkl"), check_exact=True
    )
    selected, modified, audit = runner.select_core(
        raw["pool"],
        scores,
        prefs,
        saved_cuts["thresholds"],
        saved_cuts["base_thresholds"],
    )
    pd.testing.assert_series_equal(
        modified,
        pd.read_pickle(folder / "adjusted_scoring_audit.pkl"),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        audit,
        pd.read_pickle(folder / "private_isomer_resolution_audit.pkl"),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        selected, pd.read_pickle(folder / "private_selected_core.pkl"), check_exact=True
    )
    expected, baseline = independent_final_pipeline(dataset, outer, selected)
    actual = pd.read_csv(
        folder / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    assert accepted_keys(expected) == accepted_keys(actual) and len(expected) == len(
        actual
    )
    assert set(actual) == set(expected)
    # Exact CSV semantics (including empty/nullable metadata), followed below
    # by independent ORIGINAL raw-coordinate checks, not just CSV roundtrips.
    normalized = pd.read_csv(
        StringIO(expected[list(actual)].to_csv(index=False)),
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    pd.testing.assert_frame_equal(actual, normalized, check_exact=True)
    rounding = 0
    for mode, payload in baseline["streams"].items():
        output = actual[actual.stream.eq(mode)]
        assert not output.feat_feature_id.astype(str).duplicated().any()
        original = payload["features"].set_index(
            payload["features"].feature_id.astype(str)
        )
        source = original.loc[output.feat_feature_id]
        for dest, origin in (
            ("feat_mz", "mz"),
            ("feat_rt", "rt"),
            ("feat_intensity", "intensity"),
        ):
            a, b = output[dest].to_numpy(float), source[origin].to_numpy(float)
            np.testing.assert_allclose(
                a, b, rtol=8 * np.finfo(float).eps, atol=0, equal_nan=True
            )
            rounding += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
        # Every quantitative/boundary raw field available in the accepted
        # core schema must retain the original feature's actual measurement.
        core_keys = accepted_keys(selected[selected.stream.eq(mode)])
        core_rows = output[
            [
                k in core_keys
                for k in output[
                    ["stream", "name", "adduct", "feat_feature_id", "is_is"]
                ]
                .astype(str)
                .itertuples(index=False, name=None)
            ]
        ]
        for field in original:
            col = "feat_" + str(field)
            if col not in core_rows or field == "feature_id":
                continue
            measured = original.loc[core_rows.feat_feature_id, field]
            observed = core_rows[col]
            if pd.api.types.is_numeric_dtype(
                measured
            ) and pd.api.types.is_numeric_dtype(observed):
                np.testing.assert_allclose(
                    observed.to_numpy(float),
                    measured.to_numpy(float),
                    rtol=8 * np.finfo(float).eps,
                    atol=0,
                    equal_nan=True,
                    err_msg=col,
                )
            else:
                assert (
                    observed.fillna("").astype(str).tolist()
                    == measured.fillna("").astype(str).tolist()
                ), col
    candidates = pd.read_pickle(folder / "candidates.pkl")
    accepted = candidates[candidates.automatic_accepted]
    assert accepted_keys(accepted) == accepted_keys(actual) and len(accepted) == len(
        actual
    )
    assert not candidates.loc[
        candidates.automatic_accepted, "automatic_candidate"
    ].any()
    hashes = [
        json.loads(
            (
                QE / v / "evaluation_v2" / dataset / "evaluation_identity.json"
            ).read_text()
        )["manual_sha256"]
        for v in ("baseline", "experiment_029", "experiment_039")
    ]
    assert len(set(hashes)) == 1
    resolved = (
        int(audit.decision.eq("resolved_isomer_label_detection_proxy").sum())
        if len(audit)
        else 0
    )
    complete = json.loads((folder / "complete.json").read_text())
    assert complete["resolved_isomer_clusters"] == resolved
    print(
        "039 FULL CORE/005/027/RAWQUANT/EXPORT/GOLD PASS", dataset, resolved, flush=True
    )
    return dict(
        dataset=dataset,
        raw_stream_checks=4,
        roundoff=rounding,
        resolved_isomer_clusters=resolved,
    )


def main():
    protocol, projects = runner.initialize()
    assert (
        len(projects) == 33
        and json.loads((OUT / "reports_v2/summary.json").read_text())["complete"]
    )
    configure_frozen_resources(OUT / "source_snapshot")
    paths = sorted(
        list(OUT.glob("fold_*_nested_model.pkl"))
        + list(OUT.glob("calibration/fold_*/inner_block_*_model.pkl"))
    )
    assert len(paths) == 25
    with ProcessPoolExecutor(max_workers=2) as executor:
        models = list(executor.map(verify_model, paths))
        inner = list(executor.map(verify_calibration, range(5)))
        outer = list(executor.map(verify_dataset, projects))
    assert sum(m["training_queries"] for m in models) == 528
    assert sum(inner) == 132 and sum(d["raw_stream_checks"] for d in outer) == 132
    runner.initialize()
    result = dict(
        all_passed=True,
        annotation_experiment="039",
        integrity_tool_sha256=sha(Path(__file__)),
        source_file_hash_checks=len(protocol["source_snapshot_sha256"]),
        input_file_hash_checks=len(protocol["input_sha256"]),
        fresh_conditional_parent_fits=25,
        fresh_conditional_mode_classifiers=sum(m["fresh_experts"] for m in models),
        deliberately_reused_pointwise_parents=25,
        all_training_query_scope_pair_weight_reconstructions=528,
        inner_pointwise_preference_reproductions=132,
        disabled_inner_core029parity_checks=132,
        full_training_grid_cutoff_reproductions=5,
        heldout_pointwise_preference_reproductions=33,
        full005standard027mass_pipeline_reproductions=132,
        raw_coordinate_stream_checks=132,
        accepted_candidate_export_parity_checks=33,
        identical_manual_reference_hash_checks=33,
        preexisting_roundoff_cells=sum(d["roundoff"] for d in outer),
        production_enabled=False,
        note="Saved-input groupeddevelopment,NOTexternalvalidation/FDR. Original029pointwisescores/physicalrows fixed;onlynewconditionalisomerlabelscores/cuts. Full independent005/027outerpipeline andrawquantitychecks;all28unresolved remainprimaryN,onlyX4userexcludedtarget.",
    )
    pd.DataFrame(models).to_csv(
        OUT / "conditional_label_training_audit.csv", index=False
    )
    pd.DataFrame(outer).to_csv(OUT / "isomer_resolution_output_audit.csv", index=False)
    runner.dump(OUT / "integrity_receipt.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
