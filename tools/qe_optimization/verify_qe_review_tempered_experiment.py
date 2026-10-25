"""037: verify unchanged labels/pools, review weights, ALL nested scores/cuts/peaks."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_qe_review_tempered_experiment as runner
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


def inventory():
    return json.loads((OUT / "run_manifest.json").read_text())["datasets"], json.loads(
        (QE / "rt_atlas_development/fold_assignment.json").read_text()
    )


def verify_calibration(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    import run_qe_review_tempered_experiment as runner
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts
    from matlab_annotation.operating_point import (
        select_operating_candidates,
        choose_balanced_training_points,
    )

    protocol = json.loads((OUT / "protocol.json").read_text())
    projects = json.loads((OUT / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    folder = OUT / "calibration" / f"fold_{outer}"
    excluded = {d[:7] for d in projects if folds[d] == outer}
    allowed = {d[:7] for d in projects if folds[d] != outer}
    cases, scores, manual_counts, fingerprints = {}, {}, Counter(), {}
    for inner in range(5):
        if inner == outer:
            continue
        model_path = folder / f"inner_block_{inner}_model.pkl"
        model = pd.read_pickle(model_path)
        fingerprints[model_path.relative_to(OUT).as_posix()] = sha(model_path)
        assert set(model["training_cohorts"]) == {
            d[:7] for d in projects if folds[d] not in {outer, inner}
        }
        assert set(model["excluded_groups"]) == {
            d[:7] for d in projects if folds[d] in {outer, inner}
        }
        for dataset in projects:
            if folds[dataset] != inner:
                continue
            data = runner.load_predictors(dataset, {outer, inner})
            predicted = predict_mode_evidence_experts(data, model)
            score_path = folder / (dataset + "_blind_scores.pkl")
            pd.testing.assert_series_equal(
                predicted, pd.read_pickle(score_path), check_exact=True
            )
            fingerprints[score_path.relative_to(OUT).as_posix()] = sha(score_path)
            labeled, manual, identities = runner.with_training_labels(
                data,
                permitted_groups=allowed,
                excluded_groups=excluded,
                apply_review=False,
            )
            pool = labeled["pool"]
            assert "training_review_uncertain" not in pool
            pool["calibration_identity_in_manual"] = [
                (s, runner.species(n), runner.adduct(a)) in identities
                for s, n, a in pool[["stream", "name", "adduct"]].itertuples(
                    index=False, name=None
                )
            ]
            cases[dataset], scores[dataset] = pool, predicted
            manual_counts.update(manual.stream.value_counts().to_dict())
    assert set(cases) == {d for d in projects if folds[d] != outer}
    records = []
    for threshold in protocol["threshold_grid"]:
        counts = {s: Counter() for s in manual_counts}
        for dataset, pool in cases.items():
            for stream, part in pool.groupby("stream", sort=False):
                selected = select_operating_candidates(
                    part,
                    scores[dataset].loc[part.index],
                    threshold=threshold,
                    ambiguity_margin=protocol["ambiguity_margin"],
                )
                counts[stream]["correct"] += int(selected.training_positive.sum())
                counts[stream]["extras"] += int(
                    (~selected.calibration_identity_in_manual).sum()
                )
                counts[stream]["different_peak"] += int(
                    (
                        selected.calibration_identity_in_manual
                        & selected.training_label_known
                        & ~selected.training_positive
                    ).sum()
                )
        records.extend(
            dict(stream=s, threshold=threshold, manual=n, **dict(counts[s]))
            for s, n in manual_counts.items()
        )
    table = pd.DataFrame(records).fillna(0)
    actual = pd.read_csv(
        folder / "training_operating_points.csv", float_precision="round_trip"
    )
    pd.testing.assert_frame_equal(table, actual, check_exact=True, check_dtype=False)
    fitted = choose_balanced_training_points(
        table,
        extras_per_manual=protocol["extras_budget_per_endogenous_manual"],
        target=0.9,
    )
    recorded = json.loads((folder / "operating_points.json").read_text())
    assert fitted == recorded["thresholds"]
    assert set(recorded["training_groups"]) == allowed
    assert set(recorded["outer_excluded"]) == excluded
    for name in (
        "operating_points.json",
        "training_operating_points.csv",
        "selected_training_points.csv",
    ):
        path = folder / name
        fingerprints[path.relative_to(OUT).as_posix()] = sha(path)
    print(
        "ALL INNER SCORES, COUNTS AND OPERATING-POINT CHOICE REPRODUCED",
        outer,
        len(cases),
        flush=True,
    )
    return dict(
        outer=outer,
        blind_dataset_score_checks=len(cases),
        operating_point_rows=len(table),
        artifact_sha256=fingerprints,
    )


def verify_dataset(dataset):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards
    from matlab_annotation.calibrated_mass_discovery import select_calibrated_additions

    _, folds = inventory()
    outer = folds[dataset]
    thresholds = json.loads(
        (OUT / "calibration" / f"fold_{outer}" / "operating_points.json").read_text()
    )
    model = pd.read_pickle(OUT / f"fold_{outer}_nested_model.pkl")
    data = runner.load_predictors(dataset, {outer})
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts

    scores = predict_mode_evidence_experts(data, model)
    pd.testing.assert_series_equal(
        scores,
        pd.read_pickle(OUT / "runs" / dataset / "scoring_audit.pkl"),
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
        selected = select_operating_candidates(
            pool,
            scores.loc[pool.index],
            threshold=thresholds["thresholds"][stream],
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
        for version in ("baseline", "experiment_029", "experiment_037")
    ]
    assert len(set(manual_hashes)) == 1
    print("037 FULL PIPELINE AND ORIGINAL COORDINATES PASS", dataset, flush=True)
    candidates = pd.read_pickle(OUT / "runs" / dataset / "candidates.pkl")
    k = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
    accepted = candidates[candidates.automatic_accepted]
    assert set(accepted[k].astype(str).itertuples(index=False, name=None)) == set(
        actual[k].astype(str).itertuples(index=False, name=None)
    )
    return checks, demotions, rounding


def verify_model(path):
    from matlab_annotation.evidence_training import validate_feature_scope
    from matlab_annotation.review_aware_supervision import review_tempered_weights
    from annotator.composition_ranges import canonical_lipid_class
    from diagnose_qe_negative_supervision import categorize

    model = pd.read_pickle(path)
    old_path = OLD / path.relative_to(OUT)
    old = pd.read_pickle(old_path)
    assert sha(old_path) == model["core_model_sha256"]
    training, excluded = set(model["training_cohorts"]), set(model["excluded_groups"])
    assert not training & excluded
    assert training == set(old["training_cohorts"]) and excluded == set(
        old["excluded_groups"]
    )
    assert model["columns"] == old["columns"]
    assert model["training_feature_masks"] == old["training_feature_masks"]
    assert (
        model["standard_training_feature_scopes"]
        == old["standard_training_feature_scopes"]
    )
    assert (
        model["training_pairs"] == old["training_pairs"]
        and model["positives"] == old["positives"]
    )
    assert set(model["experts"]) == {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}
    observations = {
        mode: dict(groups=[], positive=[], uncertain=[]) for mode in model["experts"]
    }
    for dataset, mask in model["training_feature_masks"].items():
        raw = runner.load_predictors(dataset, set(mask))
        validate_feature_scope(raw, training, excluded, query_is_training=True)
        assert (
            "training_review_uncertain" not in raw["pool"]
            and "training_review_uncertain" not in raw["evidence"]
        )
        labeled, _, _ = runner.with_training_labels(
            raw, permitted_groups=training, excluded_groups=excluded
        )
        original, _, _ = runner.with_training_labels(
            raw, permitted_groups=training, excluded_groups=excluded, apply_review=False
        )
        pool, truth = labeled["pool"], original["pool"]
        pd.testing.assert_frame_equal(
            pool.drop(columns="training_review_uncertain"), truth, check_exact=True
        )
        pd.testing.assert_frame_equal(
            labeled["evidence"], raw["evidence"], check_exact=True
        )
        assert not (
            pool.training_review_uncertain
            & (pool.training_positive | ~pool.training_label_known)
        ).any()
        assert model["review_training_audit"][dataset] == dict(
            known=int(pool.training_label_known.sum()),
            positive=int(pool.training_positive.sum()),
            uncertain_negative=int(pool.training_review_uncertain.sum()),
        )
        # Independent retrospective category implementation, not the fit helper.
        manual = pd.read_pickle(
            BASE / "evaluation_v2" / dataset / "manual_comparison.pkl"
        )
        categories = categorize(raw["pool"], manual, canonical_lipid_class)
        expected_uncertain = {
            (r["stream"], r["lipid_class"]): r["candidate_rows"]
            for r in categories
            if r["category"] == "negative_class_absent_in_mode_present_in_other_mode"
        }
        observed_uncertain = (
            pool[pool.training_review_uncertain]
            .assign(canonical_class=lambda p: p["class"].map(canonical_lipid_class))
            .groupby(["stream", "canonical_class"])
            .size()
            .to_dict()
        )
        assert expected_uncertain == observed_uncertain
        for mode, part in pool[pool.training_label_known].groupby("stream", sort=False):
            observations[mode]["groups"].extend([dataset[:7]] * len(part))
            observations[mode]["positive"].extend(part.training_positive.astype(bool))
            observations[mode]["uncertain"].extend(
                part.training_review_uncertain.astype(bool)
            )
        scope = model["standard_training_feature_scopes"][dataset]
        assert (
            set(scope["training"]).issubset(training)
            and not set(scope["training"]) & excluded
        )
        assert dataset[:7] not in scope["training"] and dataset[:7] in scope["excluded"]
        reference = (
            runner.QE
            / "nested_standard_development_v1"
            / ("exclude_" + "_".join(map(str, sorted(mask))) + ".json")
        )
        assert sha(reference) == scope["reference_sha256"]
    protocol = json.loads((OUT / "protocol.json").read_text())
    for mode, expert in model["experts"].items():
        obs = observations[mode]
        assert (
            expert["training_pairs"]
            == len(obs["groups"])
            == old["experts"][mode]["training_pairs"]
        )
        assert (
            expert["positives"]
            == sum(obs["positive"])
            == old["experts"][mode]["positives"]
        )
        assert (
            set(expert["training_cohorts"]) == training
            and set(expert["excluded_groups"]) == excluded
        )
        assert expert["columns"] == model["columns"] and set(
            expert["categories"]["stream"]
        ) == {mode}
        assert expert["estimator"].n_iter_ == 180
        for key, value in protocol["classifier"].items():
            if key != "kind":
                assert expert["estimator"].get_params()[key] == value
        actual = review_tempered_weights(
            obs["groups"], obs["positive"], obs["uncertain"]
        )
        factors = [0.25 if u else 1.0 for u in obs["uncertain"]]
        totals = Counter()
        for g, f in zip(obs["groups"], factors):
            totals[g] += f
        target = len(factors) / len(totals)
        expected = np.asarray(
            [f * target / totals[g] for g, f in zip(obs["groups"], factors)]
        )
        np.testing.assert_array_equal(actual, expected)
        policy = expert["sample_weight_policy"]
        assert (
            policy["kind"] == "review_aware_negative"
            and policy["uncertain_negative_factor"] == 0.25
        )
        assert policy["uncertain_negative_rows"] == sum(obs["uncertain"])
        assert policy["uncertain_negative_by_project"] == dict(
            Counter(g for g, u in zip(obs["groups"], obs["uncertain"]) if u)
        )
        actual_totals = Counter()
        for g, w in zip(obs["groups"], actual):
            actual_totals[g] += float(w)
        error = max(abs(v - target) for v in actual_totals.values())
        assert (
            policy["cohort_total_target"] == target
            and policy["cohort_total_max_abs_error"] == error
            and error < 1e-6
        )
    print(
        "037ALLLABELS/SCOPES/QUARTERWEIGHTS REPRODUCED",
        path.relative_to(OUT),
        flush=True,
    )
    return dict(
        model=path.relative_to(OUT).as_posix(),
        training_queries=len(model["training_feature_masks"]),
        known=model["training_pairs"],
        positive=model["positives"],
        uncertain_negative=sum(sum(v["uncertain"]) for v in observations.values()),
        fresh_experts=4,
    )


def main():
    protocol, _ = runner.initialize()
    projects, _ = inventory()
    assert (
        len(projects) == 33
        and json.loads((OUT / "reports_v2/summary.json").read_text())["complete"]
    )
    paths = list(OUT.glob("fold_*_nested_model.pkl")) + list(
        OUT.glob("calibration/fold_*/inner_block_*_model.pkl")
    )
    assert len(paths) == 25
    audited = [verify_model(path) for path in sorted(paths)]
    pd.DataFrame(audited).to_csv(OUT / "training_supervision_audit.csv", index=False)
    with ProcessPoolExecutor(max_workers=2) as executor:
        inner = list(executor.map(verify_calibration, range(5)))
        outer = list(executor.map(verify_dataset, projects))
    assert sum(r["blind_dataset_score_checks"] for r in inner) == 132
    assert sum(r[0] for r in outer) == 132
    result = dict(
        all_passed=True,
        annotation_experiment="037",
        integrity_tool_sha256=sha(Path(__file__)),
        source_file_hash_checks=len(protocol["source_snapshot_sha256"]),
        input_file_hash_checks=len(protocol["input_sha256"]),
        new_parent_fits=25,
        new_mode_classifiers=100,
        all_training_query_scope_label_weight_reconstructions=sum(
            r["training_queries"] for r in audited
        ),
        inner_score_reproductions=132,
        full_training_grid_and_cutoff_reproductions=5,
        heldout_score_reproductions=33,
        full005standard027mass_pipeline_reproductions=132,
        raw_coordinate_stream_checks=132,
        accepted_candidate_export_parity_checks=33,
        identical_manual_reference_hash_checks=33,
        raw_coordinate_rtol=8 * np.finfo(float).eps,
        raw_coordinate_atol=0,
        preexisting_roundoff_cells=sum(r[2] for r in outer),
        standard_alias_demotions=sum(r[1] for r in outer),
        production_enabled=False,
        note="Engineering integrity,NOTuntouched validation/FDR. EXACT029physicalrows/predictors/labels;ONLYfixedquarterweight uncertain cross-mode negativeTRAININGevidence,equalprojecttotals. All4modeexperts freshlyfitted withquery/outer/innerexcludedreferences. Calibrationtruthunchanged,same009/005/027. OnlyX4targetexcluded.",
    )
    runner.dump(OUT / "integrity_receipt.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
