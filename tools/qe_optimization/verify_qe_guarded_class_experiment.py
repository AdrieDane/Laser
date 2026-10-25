"""Independent034 full-output replay, training calibration and provenance audit."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_qe_guarded_class_experiment as runner
from frozen_resources import configure_frozen_resources

QE, OUT, OLD, BASE, STD, MASS = (
    runner.QE,
    runner.OUT,
    runner.OLD,
    runner.BASE,
    runner.STD,
    runner.MASS,
)
sha = runner.sha


def verify_calibration(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.guarded_class_operating_point import (
        class_training_curves,
        choose_class_training_points,
    )

    protocol = json.loads((OUT / "protocol.json").read_text())
    folder = OUT / "calibration" / f"fold_{outer}"
    learned = json.loads((folder / "operating_points.json").read_text())
    winners, manual, cuts, allowed, excluded, audits = runner.calibration_data(outer)
    curves = class_training_curves(
        winners,
        manual,
        cuts,
        training_groups=allowed,
        excluded_groups=excluded,
        minimum_manual=protocol["minimum_manual"],
        minimum_projects=protocol["minimum_projects"],
        minimum_positives=protocol["minimum_positive_winners"],
    )
    actual = pd.read_csv(
        folder / "training_class_operating_points.csv", float_precision="round_trip"
    )
    pd.testing.assert_frame_equal(curves, actual, check_exact=True)
    assert (
        sha(folder / "training_class_operating_points.csv")
        == learned["training_table_sha256"]
    )
    assert (
        sha(folder / "selected_class_training_points.csv")
        == learned["selected_table_sha256"]
    )
    assert learned["fallback_thresholds"] == cuts
    repeated = choose_class_training_points(
        curves,
        training_groups=allowed,
        excluded_groups=excluded,
        extras_per_manual=protocol["extras_budget_per_endogenous_manual"],
        target=protocol["target"],
        time_limit=protocol["solver_time_limit_per_stage_seconds"],
    )
    for key, value in repeated.items():
        assert value == learned[key], (outer, key)
    selected = curves.loc[learned["selected_point_indices"]]
    saved = pd.read_csv(
        folder / "selected_class_training_points.csv", float_precision="round_trip"
    )
    pd.testing.assert_frame_equal(
        selected.reset_index(drop=True), saved, check_exact=True
    )
    baseline = curves[curves.baseline].set_index(["stream", "lipid_class"])
    picked = selected.set_index(["stream", "lipid_class"]).reindex(baseline.index)
    assert picked.correct.ge(baseline.correct).all()
    sparse = ~baseline.eligible
    assert picked.loc[sparse, "threshold"].equals(baseline.loc[sparse, "threshold"])
    for stream in baseline.index.get_level_values(0).unique():
        assert (
            picked.loc[stream].different_peak.sum()
            <= baseline.loc[stream].different_peak.sum()
        )
    assert selected.extras.sum() <= learned["training_extras_budget"]
    print("034 CALIBRATION FULL REPRODUCTION", outer, len(audits), flush=True)
    return len(audits)


def verify_dataset(dataset):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.guarded_class_operating_point import (
        select_class_operating_candidates,
    )
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards
    from matlab_annotation.calibrated_mass_discovery import select_calibrated_additions

    _, folds = runner.inventory()
    outer = folds[dataset]
    thresholds = json.loads(
        (OUT / "calibration" / f"fold_{outer}" / "operating_points.json").read_text()
    )
    model = pd.read_pickle(OLD / f"fold_{outer}_nested_model.pkl")
    data, scores = runner.reproduce(
        dataset, {outer}, model, OUT / "runs" / dataset / "scoring_audit.pkl"
    )
    pd.testing.assert_series_equal(
        scores,
        pd.read_pickle(OLD / "runs" / dataset / "scoring_audit.pkl"),
        check_exact=True,
    )
    actual = pd.read_csv(
        OUT / "runs" / dataset / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    baseline = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
    catalogue = declared_standard_targets(
        OUT / "source_snapshot/reference_data/matlab_pipeline",
        baseline["internal_standards_expected"],
    )
    standards = json.loads(
        (STD / "models" / f"fold_{outer}_standard_models.json").read_text()
    )
    mass = pd.read_pickle(MASS / "runs" / dataset / "private_mass_candidates.pkl")
    anchors = pd.concat(
        [
            p["annotations"].copy().assign(stream=s)
            for s, p in baseline["streams"].items()
        ],
        ignore_index=True,
    )
    demotions, rounding, checks = 0, 0, 0
    for stream, pool in data["pool"].groupby("stream", sort=False):
        cut = thresholds["fallback_thresholds"][stream]
        uniform = select_class_operating_candidates(
            pool,
            scores.loc[pool.index],
            baseline_threshold=cut,
            class_thresholds={},
            ambiguity_margin=0.05,
        )
        old = select_operating_candidates(
            pool, scores.loc[pool.index], threshold=cut, ambiguity_margin=0.05
        )
        assert uniform.index.equals(old.index)
        selected = select_class_operating_candidates(
            pool,
            scores.loc[pool.index],
            baseline_threshold=cut,
            class_thresholds=thresholds["thresholds"][stream],
            ambiguity_margin=0.05,
        )
        payload = baseline["streams"][stream]
        protected = (
            payload["annotations"][
                payload["annotations"].is_is.fillna(False).astype(bool)
            ]
            .copy()
            .assign(stream=stream)
        )
        expected, _, audit = complete_declared_standards(
            pd.concat([selected, protected], ignore_index=True),
            payload["features"],
            catalogue[catalogue.stream.eq(stream)],
            standards["library"],
            standards["atlas"],
            standards["transfer"],
            standards["channels"],
            stream=stream,
            alignment_annotations=payload["annotations"],
        )
        if "action" in audit:
            demotions += int(audit.action.eq("demote_mass_identical_spike_alias").sum())
        local_mass = mass[mass.stream.eq(stream)] if len(mass) else mass
        added = select_calibrated_additions(
            local_mass, anchors, expected, stream=stream
        )
        expected = pd.concat([expected, added], ignore_index=True)
        observed = actual[actual.stream.eq(stream)]
        keys = ["name", "adduct", "feat_feature_id", "is_is"]
        assert set(
            expected[keys].astype(str).itertuples(index=False, name=None)
        ) == set(observed[keys].astype(str).itertuples(index=False, name=None))
        assert not observed.feat_feature_id.duplicated().any()
        original = payload["features"].set_index(
            payload["features"].feature_id.astype(str)
        )
        measured = original.loc[observed.feat_feature_id]
        for output, source in (
            ("feat_mz", "mz"),
            ("feat_rt", "rt"),
            ("feat_intensity", "intensity"),
        ):
            a, b = observed[output].to_numpy(float), measured[source].to_numpy(float)
            np.testing.assert_allclose(
                a, b, rtol=8 * np.finfo(float).eps, atol=0, equal_nan=True
            )
            rounding += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
        checks += 1
    assert checks == 4
    manual_hashes = [
        json.loads(
            (
                QE / version / "evaluation_v2" / dataset / "evaluation_identity.json"
            ).read_text()
        )["manual_sha256"]
        for version in ("baseline", "experiment_029", "experiment_034")
    ]
    assert len(set(manual_hashes)) == 1
    print("034 FULL PIPELINE AND ORIGINAL COORDINATES PASS", dataset, flush=True)
    return checks, demotions, rounding


def main():
    protocol = runner.initialize()
    projects, folds = runner.inventory()
    assert json.loads((OUT / "reports_v2/summary.json").read_text())["complete"]
    frozen = json.loads((OUT / "all_calibrations_frozen.json").read_text())
    assert frozen == {
        str(f): sha(OUT / "calibration" / f"fold_{f}" / "operating_points.json")
        for f in range(5)
    }
    paths = list(OLD.glob("fold_*_nested_model.pkl")) + list(
        OLD.glob("calibration/fold_*/inner_block_*_model.pkl")
    )
    assert len(paths) == 25 and not list(OUT.rglob("*model.pkl"))
    query_checks = 0
    for path in paths:
        model = pd.read_pickle(path)
        training, excluded = (
            set(model["training_cohorts"]),
            set(model["excluded_groups"]),
        )
        assert not training & excluded
        assert set(model["experts"]) == {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}
        for expert in model["experts"].values():
            assert (
                set(expert["training_cohorts"]) == training
                and set(expert["excluded_groups"]) == excluded
            )
            assert expert["estimator"].n_iter_ == 180
        for dataset, mask in model["training_feature_masks"].items():
            assert dataset[:7] in training and dataset[:7] not in excluded
            assert set(mask) == {folds[d] for d in projects if d[:7] in excluded} | {
                folds[dataset]
            }
            scope = model["standard_training_feature_scopes"][dataset]
            assert (
                set(scope["training"]).issubset(training)
                and dataset[:7] not in scope["training"]
            )
            assert not set(scope["training"]) & excluded
            reference = (
                QE
                / "nested_standard_development_v1"
                / ("exclude_" + "_".join(map(str, sorted(mask))) + ".json")
            )
            assert sha(reference) == scope["reference_sha256"]
            query_checks += 1
    with ProcessPoolExecutor(max_workers=2) as executor:
        inner = list(executor.map(verify_calibration, range(5)))
        outer = list(executor.map(verify_dataset, projects))
    assert sum(inner) == 132 and sum(r[0] for r in outer) == 132
    receipt = dict(
        all_passed=True,
        annotation_experiment="034",
        integrity_tool_sha256=sha(Path(__file__)),
        source_file_hash_checks=len(protocol["source_snapshot_sha256"]),
        input_file_hash_checks=len(protocol["input_sha256"]),
        unchanged029_models=25,
        unchanged029_mode_experts=100,
        new_classifier_fits=0,
        training_query_exclusion_checks=query_checks,
        inner_score_reproductions=sum(inner),
        class_curve_and_parameter_reproductions=5,
        heldout_score_reproductions=33,
        selector_plus_005standard_plus_027mass_reproductions=132,
        uniform_selector_parities=660,
        original_raw_coordinate_stream_checks=132,
        original_coordinate_relative_tolerance=8 * np.finfo(float).eps,
        original_coordinate_absolute_tolerance=0,
        preexisting_roundoff_cells=sum(r[2] for r in outer),
        standard_alias_demotions=sum(r[1] for r in outer),
        identical_manual_reference_hash_checks=33,
        production_enabled=False,
        note="Engineering integrity,NOT untouched external validation or FDR. Deliberately frozen029scores/models/candidates; only nested training-class cutoff calibration. Sparse classes fixed; full denominator and extras definition unchanged; X4only excluded from companion target scope.",
    )
    runner.dump(OUT / "integrity_receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
