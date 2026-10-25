"""036: verify ALLlabels/scopes, empirical refs, scores, cuts and original peaks."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import run_qe_empirical_experiment as runner
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
    import run_qe_empirical_experiment as runner
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
            )
            pool = labeled["pool"]
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
        for version in ("baseline", "experiment_029", "experiment_036")
    ]
    assert len(set(manual_hashes)) == 1
    print("036 FULL PIPELINE AND ORIGINAL COORDINATES PASS", dataset, flush=True)
    return checks, demotions, rounding


def verify_model(path):
    from matlab_annotation.evidence_training import validate_feature_scope

    model = pd.read_pickle(path)
    old_path = OLD / path.relative_to(OUT)
    old = pd.read_pickle(old_path)
    assert sha(old_path) == model["core_model_sha256"]
    training, excluded = set(model["training_cohorts"]), set(model["excluded_groups"])
    assert not training & excluded
    assert training == set(old["training_cohorts"]) and excluded == set(
        old["excluded_groups"]
    )
    assert (
        set(old["columns"]).issubset(model["columns"])
        and len(model["columns"]) == len(old["columns"]) + 8
    )
    assert {c for c in model["columns"] if c.startswith("q_empirical_")} == set(
        model["columns"]
    ) - set(old["columns"])
    assert model["training_feature_masks"] == old["training_feature_masks"]
    assert (
        model["standard_training_feature_scopes"]
        == old["standard_training_feature_scopes"]
    )
    assert set(model["experts"]) == {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}
    counts = {mode: Counter() for mode in model["experts"]}
    old_known = old_positive = 0
    for dataset, mask in model["training_feature_masks"].items():
        raw = runner.load_predictors(dataset, set(mask))
        validate_feature_scope(raw, training, excluded, query_is_training=True)
        labeled, _, _ = runner.with_training_labels(
            raw, permitted_groups=training, excluded_groups=excluded
        )
        pool = labeled["pool"]
        old_file = (
            QE
            / "standard_context_preparation_v1/prepared"
            / ("exclude_" + "_".join(map(str, sorted(mask))))
            / (dataset + ".pkl")
        )
        n = len(pd.read_pickle(old_file)["pool"])
        old_known += int(pool.iloc[:n].training_label_known.sum())
        old_positive += int(pool.iloc[:n].training_positive.sum())
        for mode, part in pool[pool.training_label_known].groupby("stream"):
            counts[mode]["known"] += len(part)
            counts[mode]["positive"] += int(part.training_positive.sum())
        scope = model["empirical_training_feature_scopes"][dataset]
        assert scope == dict(
            training=raw["empirical_reference_training_groups"],
            excluded=raw["empirical_reference_excluded_groups"],
            reference_sha256=raw["empirical_reference_sha256"],
        )
        assert (
            set(scope["training"]).issubset(training)
            and not set(scope["training"]) & excluded
        )
        assert dataset[:7] not in scope["training"] and dataset[:7] in scope["excluded"]
        reference = runner.EMPIRICAL / (
            "exclude_" + "_".join(map(str, sorted(mask))) + ".json"
        )
        assert sha(reference) == scope["reference_sha256"]
        ref = json.loads(reference.read_text())
        assert set(ref["training_groups"]) == set(scope["training"]) and set(
            ref["excluded_groups"]
        ) == set(scope["excluded"])
    assert old_known == old["training_pairs"] and old_positive == old["positives"]
    assert sum(v["known"] for v in counts.values()) == model["training_pairs"]
    assert sum(v["positive"] for v in counts.values()) == model["positives"]
    p = json.loads((OUT / "protocol.json").read_text())
    for mode, expert in model["experts"].items():
        assert (
            expert["training_pairs"] == counts[mode]["known"]
            and expert["positives"] == counts[mode]["positive"]
        )
        assert (
            set(expert["training_cohorts"]) == training
            and set(expert["excluded_groups"]) == excluded
        )
        assert expert["columns"] == model["columns"] and set(
            expert["categories"]["stream"]
        ) == {mode}
        assert (
            expert["estimator"].n_iter_ == 180 and "sample_weight_policy" not in expert
        )
        for name, value in p["classifier"].items():
            if name != "kind":
                assert expert["estimator"].get_params()[name] == value
    print(
        "036ALLTRAININGLABELS/OLDPAIRS/NEWSCOPES VERIFIED",
        path.relative_to(OUT),
        flush=True,
    )
    return dict(
        model=path.relative_to(OUT).as_posix(),
        training_queries=len(model["training_feature_masks"]),
        old_known=old_known,
        new_known=model["training_pairs"],
        old_positives=old_positive,
        new_positives=model["positives"],
        fresh_mode_experts=4,
    )


def main():
    verifier_hash = sha(Path(__file__))
    protocol, _ = runner.initialize()
    projects, _ = inventory()
    assert (
        len(projects) == 33
        and json.loads((OUT / "reports_v2/summary.json").read_text())["complete"]
    )
    prep = runner.PREPARED
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
    # Independent annotation checks can run while the much slower physical
    # reconstruction finishes. Never issue a master PASS without both checks.
    deadline = time.monotonic() + 4 * 60 * 60
    receipt_path = prep / "integrity_receipt.json"
    while not receipt_path.exists():
        assert time.monotonic() < deadline, "Preparation verification timed out"
        assert sha(Path(__file__)) == verifier_hash, "Verifier changed during run"
        print("ANNOTATION CHECKS COMPLETE; waiting for FULL preparation replay", flush=True)
        time.sleep(60)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["all_passed"] and receipt["full_masked_table_reproductions"] == 363
    assert receipt["independent_standard_mask_measurement_reproductions"] == 363
    assert receipt["numerical_reference_exact_reproductions"] == 25
    assert receipt["source_sha256"] == sha(
        Path(__file__).with_name("verify_qe_empirical_preparation_fast.py")
    )
    assert sha(Path(__file__)) == verifier_hash, "Verifier changed during run"
    result = dict(
        all_passed=True,
        annotation_experiment="036",
        integrity_tool_sha256=sha(Path(__file__)),
        source_file_hash_checks=len(protocol["source_snapshot_sha256"]),
        input_file_hash_checks=len(protocol["input_sha256"]),
        new_parent_fits=25,
        new_mode_classifiers=100,
        all_training_query_scope_label_old_pair_reconstructions=sum(
            r["training_queries"] for r in audited
        ),
        inner_score_reproductions=132,
        full_training_grid_and_cutoff_reproductions=5,
        heldout_score_reproductions=33,
        full005standard027mass_pipeline_reproductions=132,
        raw_coordinate_stream_checks=132,
        identical_manual_reference_hash_checks=33,
        raw_coordinate_rtol=8 * np.finfo(float).eps,
        raw_coordinate_atol=0,
        preexisting_roundoff_cells=sum(r[2] for r in outer),
        standard_alias_demotions=sum(r[1] for r in outer),
        full_preparation_mask_reproductions=363,
        independent_standard_mask_measurement_reproductions=363,
        numerical_reference_exact_reproductions=25,
        preparation_integrity_receipt_sha256=sha(prep / "integrity_receipt.json"),
        production_enabled=False,
        note="Engineering integrity,NOTuntouched validation/FDR. ALL029oldqueries/labels/coordinates preserved;newstrict3ppm/10s empiricalRTpairs+8numericfields,15exactSTDfields. All4modeexperts freshlyfitted;source/reference/supervision scopes excludedquery/outer/inner. Same009modecalibration/005STD/fixed027massstage. OnlyX4target-excluded.",
    )
    runner.dump(OUT / "integrity_receipt.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
