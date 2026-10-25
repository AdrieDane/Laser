"""Verify033 raw aligned-RT discovery, old identities, nested scopes and outputs."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from context import WORKSPACE
from frozen_resources import configure_frozen_resources

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "experiment_033"
CORE = QE / "experiment_029"
OLD = QE / "experiment_027"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_raw_preparation(dataset):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.chemical_targets import compile_additional_targets
    from matlab_annotation.standard_aligned_candidates import (
        prepare_standard_aligned_candidates,
        aligned_retention_evidence,
    )
    from matlab_annotation.declared_standard_context import (
        independent_standard_measurements,
        declared_standard_context_evidence,
    )
    from matlab_annotation.candidate_preparation import enrich_candidate_pools, _quality
    from matlab_annotation.evidence_training import validate_feature_scope

    prep = QE / "standard_aligned_candidate_preparation_v2"
    oldroot = QE / "standard_context_preparation_v1"
    checkpoint = json.loads((prep / "checkpoints" / (dataset + ".json")).read_text())
    baseline = pd.read_pickle(QE / "baseline/runs" / dataset / "full_annotation.pkl")
    staged = pd.read_pickle(
        QE / "nested_standard_development_v1/automatic_inputs" / (dataset + ".pkl")
    )
    blind = dict(
        baseline,
        streams={
            stream: dict(payload, annotations=staged["automatic"][stream])
            for stream, payload in baseline["streams"].items()
        },
    )
    physical, priors, contexts, measured_tables, saved_tables = {}, {}, {}, {}, {}
    coords = [
        "stream",
        "name",
        "class",
        "adduct",
        "carbon",
        "db",
        "ref_mz",
        "ref_rt",
        "feat_feature_id",
        "feat_mz",
        "feat_rt",
        "feat_intensity",
    ]
    keys = ["stream", "name", "adduct", "feat_feature_id"]
    for path, digest in checkpoint["prepared_sha256"].items():
        path = Path(path)
        assert sha(path) == digest
        name = path.parent.name
        data = pd.read_pickle(path)
        old = pd.read_pickle(oldroot / "prepared" / name / path.name)
        bundle_path = QE / "nested_reference_development_v2" / (name + ".json")
        standard_path = QE / "nested_standard_development_v1" / (name + ".json")
        bundle = json.loads(bundle_path.read_text())
        model = json.loads(standard_path.read_text())
        assert sha(bundle_path) == data["reference_sha256"]
        assert sha(standard_path) == data["standard_reference_sha256"]
        assert set(model["training_groups"]) == set(bundle["training_groups"])
        assert set(model["excluded_groups"]) == set(bundle["excluded_groups"])
        assert model["atlas"] == bundle["atlas"]
        validate_feature_scope(
            data,
            model["training_groups"],
            model["excluded_groups"],
            query_is_training=False,
        )
        measured, _ = independent_standard_measurements(
            blind, staged["declared"], model
        )
        saved_measured = pd.read_pickle(
            oldroot / "measurements" / name / (dataset + ".pkl")
        )
        pd.testing.assert_frame_equal(measured, saved_measured, check_exact=True)
        contexts[name], extra = {}, []
        for stream, payload in baseline["streams"].items():
            declared = staged["declared"][staged["declared"].stream.eq(stream)].copy()
            generated, ranges, _ = compile_additional_targets(
                payload["reference"],
                bundle["composition_envelope"],
                stream=stream,
                declared_standards=declared,
            )
            targets = pd.concat([payload["reference"], generated], ignore_index=True)
            targets["chemical_target_generated"] = (
                targets.get(
                    "chemical_target_generated", pd.Series(False, index=targets.index)
                )
                .fillna(False)
                .astype(bool)
            )
            proposals, targets, predictions = prepare_standard_aligned_candidates(
                targets,
                payload["features"],
                staged["automatic"][stream],
                declared,
                measured[measured.stream.eq(stream)].copy(),
                model,
                stream=stream,
                query_group=dataset[:7],
                ranges=ranges,
            )
            contexts[name][stream] = (targets, predictions)
            if len(proposals):
                assert proposals.ppm_error.abs().lt(3).all()
                assert (proposals.rt_error.abs() * 60).le(10 + 1e-9).all()
                proposals = _quality(proposals, payload["features"], spreads=True)
                extra.append(proposals)
        expanded = pd.concat([old["pool"], *extra], ignore_index=True)
        expanded = expanded.drop_duplicates(keys, keep="first").reset_index(drop=True)
        pd.testing.assert_frame_equal(
            data["pool"][coords],
            expanded[coords],
            check_dtype=False,
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            data["pool"].iloc[: len(old["pool"])][coords].reset_index(drop=True),
            old["pool"][coords].reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
        physical[name] = expanded.drop(
            columns=[c for c in expanded if c.startswith("q_cross_mode_")]
        )
        priors[name] = bundle["composition_priors"]
        measured_tables[name], saved_tables[name] = measured, data
    quality = {}
    for path in (QE / "experiment_004/raw_evidence" / dataset).glob("*.pkl"):
        stream, charge = path.stem.rsplit("_z", 1)
        raw = pd.read_pickle(path).set_index("feature_id")
        quality[(stream, int(charge))] = raw[[c for c in raw if c.startswith("q_")]]
    reconstructed, _ = enrich_candidate_pools(
        physical, baseline, priors, raw_quality_cache=quality
    )
    for name, fresh in reconstructed.items():
        saved = saved_tables[name]
        positions = pd.MultiIndex.from_frame(fresh["pool"][keys]).get_indexer(
            pd.MultiIndex.from_frame(physical[name][keys])
        )
        assert (positions >= 0).all()
        fresh = {k: v.iloc[positions].reset_index(drop=True) for k, v in fresh.items()}
        exact = declared_standard_context_evidence(fresh["pool"], measured_tables[name])
        extra = []
        for stream, part in fresh["pool"].groupby("stream", sort=False):
            targets, predictions = contexts[name][stream]
            extra.append(aligned_retention_evidence(part, targets, predictions))
        extra = pd.concat(extra).reindex(fresh["pool"].index)
        evidence = pd.concat([fresh["evidence"], exact, extra], axis=1)
        evidence = evidence.reindex(columns=saved["evidence"].columns)
        pd.testing.assert_frame_equal(evidence, saved["evidence"], check_exact=True)
        pd.testing.assert_frame_equal(fresh["pool"], saved["pool"], check_exact=True)
    assert len(reconstructed) == 11
    print(
        "RAW SPIKES, ALIGNED DISCOVERY AND ALL PREDICTORS REPRODUCED",
        dataset,
        flush=True,
    )
    return len(reconstructed)


def verify_heldout_dataset(dataset):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.mode_evidence_experts import (
        predict_mode_evidence_experts as predict_grouped_evidence,
    )
    from matlab_annotation.calibrated_mass_discovery import select_calibrated_additions
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards

    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    projects = json.loads((OUT / "run_manifest.json").read_text())["datasets"]
    assert (OUT / "runs" / dataset / "complete.json").exists()
    parity, demoted_aliases, roundoff_cells = 0, 0, 0
    outer = folds[dataset]
    mask = {d[:7] for d in projects if folds[d] == outer}
    path = (
        QE
        / "standard_aligned_candidate_preparation_v2/prepared"
        / f"exclude_{outer}"
        / (dataset + ".pkl")
    )
    data = pd.read_pickle(path)
    assert not {"training_positive", "training_label_known"} & set(data["pool"])
    model = pd.read_pickle(OUT / f"fold_{outer}_nested_model.pkl")
    assert set(model["excluded_groups"]) == mask
    recorded = pd.read_pickle(OUT / "runs" / dataset / "scoring_audit.pkl")
    pd.testing.assert_series_equal(
        predict_grouped_evidence(data, model), recorded, check_exact=True
    )
    automatic = pd.read_csv(
        OUT / "runs" / dataset / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        low_memory=False,
        float_precision="round_trip",
    )
    baseline = pd.read_pickle(QE / "baseline/runs" / dataset / "full_annotation.pkl")
    catalogue = declared_standard_targets(
        OUT / "source_snapshot/reference_data/matlab_pipeline",
        baseline["internal_standards_expected"],
    )
    standards = json.loads(
        (
            QE / "experiment_005/models" / f"fold_{outer}_standard_models.json"
        ).read_text()
    )
    mass_pool = pd.read_pickle(OLD / "runs" / dataset / "private_mass_candidates.pkl")
    automatic_anchors = pd.concat(
        [
            p["annotations"].copy().assign(stream=stream)
            for stream, p in baseline["streams"].items()
        ],
        ignore_index=True,
    )
    thresholds = json.loads(
        (OUT / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )["thresholds"]
    for stream, part in data["pool"].groupby("stream", sort=False):
        expected = select_operating_candidates(
            part,
            recorded.loc[part.index],
            threshold=thresholds[stream],
            ambiguity_margin=0.05,
        )
        # Existing005 cleanup can demote a newly selected endogenous
        # mass-identical spike alias. Compare the FULL unchanged pipeline,
        # not an assumption that all raw selector calls survive cleanup.
        payload = baseline["streams"][stream]
        protected = (
            payload["annotations"][
                payload["annotations"].is_is.fillna(False).astype(bool)
            ]
            .copy()
            .assign(stream=stream)
        )
        expected, _, audit = complete_declared_standards(
            pd.concat([expected, protected], ignore_index=True),
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
            demoted_aliases += int(
                audit.action.eq("demote_mass_identical_spike_alias").sum()
            )
        local_mass = (
            mass_pool[mass_pool.stream.eq(stream)] if len(mass_pool) else mass_pool
        )
        additions = select_calibrated_additions(
            local_mass, automatic_anchors, expected, stream=stream
        )
        expected = pd.concat([expected, additions], ignore_index=True)
        actual = automatic[automatic.stream.eq(stream)]
        original = payload["features"].set_index(
            payload["features"].feature_id.astype(str)
        )
        measured = original.loc[actual.feat_feature_id.astype(str)]
        for observed, source in (
            ("feat_mz", "mz"),
            ("feat_rt", "rt"),
            ("feat_intensity", "intensity"),
        ):
            a, b = actual[observed].to_numpy(float), measured[source].to_numpy(float)
            # Legacy observed RTs can carry one seconds/minutes conversion ULP.
            # Permit IEEE arithmetic roundoff only, not analytical movement.
            np.testing.assert_allclose(
                a, b, rtol=8 * np.finfo(float).eps, atol=0, equal_nan=True
            )
            roundoff_cells += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
        keys = ["name", "adduct", "feat_feature_id", "is_is"]
        assert set(
            expected[keys].astype(str).itertuples(index=False, name=None)
        ) == set(actual[keys].astype(str).itertuples(index=False, name=None))
        parity += 1
    assert parity == 4
    print("HELDOUT FULL PIPELINE REPRODUCED", dataset, flush=True)
    return parity, demoted_aliases, roundoff_cells


def main():
    import run_qe_standard_aligned_experiment as runner

    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (
        configure_frozen_resources(OUT / "source_snapshot")
        == protocol["resource_receipt"]
    )
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).with_name(
        "run_qe_standard_aligned_experiment.py"
    ).read_bytes()
    for path, digest in protocol["source_snapshot_sha256"].items():
        assert sha(OUT / "source_snapshot" / path) == digest
    for path, digest in protocol["input_sha256"].items():
        assert sha(Path(path)) == digest
    summary = json.loads((OUT / "reports_v2/summary.json").read_text())
    assert summary["complete"] and summary["datasets_evaluated"] == 33
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    projects = json.loads((OUT / "run_manifest.json").read_text())["datasets"]
    paths = list(OUT.glob("fold_*_nested_model.pkl")) + list(
        OUT.glob("calibration/fold_*/inner_block_*_model.pkl")
    )
    assert len(paths) == 25
    audits, training_queries = [], 0
    for path in sorted(paths):
        model = pd.read_pickle(path)
        old_path = CORE / path.relative_to(OUT)
        old = pd.read_pickle(old_path)
        assert sha(old_path) == model["core_model_sha256"]
        training, excluded = (
            set(model["training_cohorts"]),
            set(model["excluded_groups"]),
        )
        assert training == set(old["training_cohorts"]) and excluded == set(
            old["excluded_groups"]
        )
        assert training.isdisjoint(excluded)
        assert model["positives"] >= old["positives"]
        assert model["training_pairs"] >= old["training_pairs"]
        assert len(model["columns"]) == len(old["columns"]) + 8
        assert set(old["columns"]).issubset(model["columns"])
        assert (
            sum(str(c).startswith("q_standard_aligned_") for c in model["columns"]) == 8
        )
        assert (
            sum(str(c).startswith("q_declared_standard_") for c in model["columns"])
            == 15
        )
        assert set(model["experts"]) == {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}
        assert (
            sum(m["training_pairs"] for m in model["experts"].values())
            == model["training_pairs"]
        )
        assert (
            sum(m["positives"] for m in model["experts"].values()) == model["positives"]
        )
        for mode, expert in model["experts"].items():
            assert expert["estimator"].n_iter_ == 180
            assert (
                set(expert["training_cohorts"]) == training
                and set(expert["excluded_groups"]) == excluded
            )
            assert expert["columns"] == model["columns"]
            assert set(expert["categories"]["stream"]) == {mode}
            assert "sample_weight_policy" not in expert
        training_counts = {s: [0, 0] for s in model["experts"]}
        for dataset, mask in model["training_feature_masks"].items():
            query_group = dataset[:7]
            assert dataset[:7] in training and dataset[:7] not in excluded
            excluded_folds = {folds[d] for d in projects if d[:7] in excluded}
            assert set(mask) == excluded_folds | {folds[dataset]}
            scope = model["standard_training_feature_scopes"][dataset]
            assert set(scope["training"]).issubset(training)
            assert not set(scope["training"]) & excluded
            assert (
                query_group not in scope["training"]
                and query_group in scope["excluded"]
            )
            reference = (
                QE
                / "nested_standard_development_v1"
                / ("exclude_" + "_".join(map(str, sorted(mask))) + ".json")
            )
            assert sha(reference) == scope["reference_sha256"]
            bundle = json.loads(reference.read_text())
            assert set(bundle["training_groups"]) == set(scope["training"])
            assert set(bundle["excluded_groups"]) == set(scope["excluded"])
            data = runner.load_predictors(dataset, mask)
            labeled, _, _ = runner.with_training_labels(
                data, permitted_groups=training, excluded_groups=excluded
            )
            for stream, part in labeled["pool"].groupby("stream", sort=False):
                known = part.training_label_known.astype(bool)
                training_counts[stream][0] += int(known.sum())
                training_counts[stream][1] += int(
                    part.loc[known, "training_positive"].sum()
                )
            training_queries += 1
        for stream, expert in model["experts"].items():
            assert training_counts[stream] == [
                expert["training_pairs"],
                expert["positives"],
            ]
        audits.append(
            dict(
                model=path.relative_to(OUT).as_posix(),
                training_pairs_before=old["training_pairs"],
                training_pairs_after=model["training_pairs"],
                positives_before=old["positives"],
                positives_after=model["positives"],
                fitted_mode_experts=4,
                new_context_predictors=8,
            )
        )
    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(verify_heldout_dataset, projects))
    assert len(results) == 33
    parity = sum(p for p, _, _ in results)
    demoted_aliases = sum(n for _, n, _ in results)
    roundoff_cells = sum(n for _, _, n in results)
    assert parity == 132
    calibration_checks = 0
    for outer in range(5):
        folder = OUT / "calibration" / f"fold_{outer}"
        checkpoint = json.loads((folder / "operating_points.json").read_text())
        assert set(checkpoint["training_groups"]) == {
            d[:7] for d in projects if folds[d] != outer
        }
        assert set(checkpoint["outer_excluded"]) == {
            d[:7] for d in projects if folds[d] == outer
        }
        assert (
            sha(folder / "training_operating_points.csv")
            == checkpoint["training_table_sha256"]
        )
        table = pd.read_csv(folder / "selected_training_points.csv")
        assert table.extras.sum() <= 0.095 * table.manual.sum() + 1e-8
        calibration_checks += 1
    pd.DataFrame(audits).to_csv(OUT / "training_supervision_audit.csv", index=False)
    prep = QE / "standard_aligned_candidate_preparation_v2"
    preparation = json.loads((prep / "summary.json").read_text())
    assert preparation["all_complete"] and preparation["dataset_mask_tables"] == 363
    assert preparation["original_candidate_coordinate_parity_checks"] == 363
    assert preparation["predictor_schema_checks"] == 363
    assert preparation["query_exclusion_checks"] == 363
    p = json.loads((prep / "protocol.json").read_text())
    for path, digest in p["source_sha256"].items():
        assert sha(prep / "source_snapshot" / path) == digest
    for path, digest in p["input_sha256"].items():
        assert sha(Path(path)) == digest
    # Reproduce EVERY raw measurement and all363 old/new masked tables.
    with ProcessPoolExecutor(max_workers=3) as executor:
        counts = list(executor.map(verify_raw_preparation, projects))
    measured_datasets, prepared_tables = len(counts), sum(counts)
    assert measured_datasets == 33 and prepared_tables == 363
    receipt = dict(
        integrity_tool_sha256=sha(Path(__file__)),
        source_file_hash_checks=len(protocol["source_snapshot_sha256"]),
        input_file_hash_checks=len(protocol["input_sha256"]),
        fitted_model_scope_and_preserved_positive_checks=len(paths),
        training_query_exclusion_checks=training_queries,
        heldout_score_reproduction_checks=33,
        selector_plus_unchanged_standard_cleanup_parity_checks=parity,
        original_raw_coordinate_final_annotation_stream_checks=parity,
        raw_coordinate_comparison_relative_tolerance=8 * np.finfo(float).eps,
        raw_coordinate_comparison_absolute_tolerance=0,
        preexisting_floating_arithmetic_roundoff_cells=roundoff_cells,
        existing_standard_alias_demotions_reproduced=demoted_aliases,
        calibration_scope_hash_budget_checks=calibration_checks,
        completed_datasets=33,
        all_passed=True,
        annotation_experiment="033",
        new_standard_context_used=True,
        new_adduct_family_context_used=False,
        new_standard_aligned_discovery_used=True,
        independent_spike_measurement_dataset_checks=measured_datasets,
        all_mask_discovery_and_all_predictor_reproductions=prepared_tables,
        original_physical_identity_and_coordinate_preservation_checks=363,
        standard_context_preparation_inputs_verified=len(p["input_sha256"]),
        fitted_mode_experts=100,
        core_reference="029",
        fixed_mass_stage_reference="027",
        note="Engineering integrity,not external validation/FDR. Preserve029old physical queries/coordinates; newstrict3ppm/10s independent-standard alignedRTproposals and8numeric descriptors. Competition context recomputed;15exactSTDfields reproduced. Fresh4modeexperts,005cleanup/fixed027massstage. Samehistorical92102/target92100denominators,X4aloneexcluded. No productiondefault",
    )
    (OUT / "integrity_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
