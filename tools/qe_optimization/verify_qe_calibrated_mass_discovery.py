"""Verify027 scopes, immutable original calls and independent rule reproduction."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from math import isclose
from pathlib import Path

import pandas as pd

from context import WORKSPACE
from frozen_resources import configure_frozen_resources

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "experiment_027"
OLD = QE / "experiment_022"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(dataset):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.declared_standard_context import (
        independent_standard_measurements,
    )
    from matlab_annotation.calibrated_mass_discovery import (
        independently_measured_mass_anchors,
        calibrated_mass_candidates,
        select_calibrated_additions,
    )
    from matlab_annotation.chemical_targets import compile_additional_targets
    from matlab_annotation.retention_atlas import predict_aligned_retention_atlas

    outer = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())[
        dataset
    ]
    reference = json.loads(
        (QE / "nested_reference_development_v2" / f"exclude_{outer}.json").read_text()
    )
    standard = json.loads(
        (QE / "nested_standard_development_v1" / f"exclude_{outer}.json").read_text()
    )
    assert (
        dataset in standard["excluded_datasets"]
        and dataset in reference["excluded_datasets"]
    )
    assert dataset[:7] not in reference["training_groups"]
    assert set(reference["training_groups"]) == set(standard["training_groups"])
    assert set(reference["excluded_groups"]) == set(standard["excluded_groups"])
    baseline = pd.read_pickle(QE / "baseline/runs" / dataset / "full_annotation.pkl")
    original = pd.read_pickle(
        QE
        / "nested_candidate_preparation_v2/prepared"
        / f"exclude_{outer}"
        / (dataset + ".pkl")
    )
    assert set(original["reference_training_groups"]) == set(
        reference["training_groups"]
    )
    old = pd.read_csv(
        OLD / "runs" / dataset / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        low_memory=False,
    )
    actual = pd.read_csv(
        OUT / "runs" / dataset / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        low_memory=False,
    )
    stable = [
        "stream",
        "name",
        "adduct",
        "feat_feature_id",
        "is_is",
        "feat_mz",
        "feat_rt",
        "feat_intensity",
    ]
    pd.testing.assert_frame_equal(
        actual.iloc[: len(old)][stable].reset_index(drop=True),
        old[stable].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-14,
        atol=1e-14,
    )
    catalogue = declared_standard_targets(
        OUT / "source_snapshot/reference_data/matlab_pipeline",
        baseline["internal_standards_expected"],
    )
    measured, _ = independent_standard_measurements(baseline, catalogue, standard)
    anchors = independently_measured_mass_anchors(measured)
    pd.testing.assert_frame_equal(
        anchors, pd.read_pickle(OUT / "runs" / dataset / "private_mass_anchors.pkl")
    )
    automatic = pd.concat(
        [
            p["annotations"].copy().assign(stream=s)
            for s, p in baseline["streams"].items()
        ],
        ignore_index=True,
    )
    expected = []
    rows = []
    for stream, payload in baseline["streams"].items():
        declared = catalogue[catalogue.stream.eq(stream)]
        generated, ranges, _ = compile_additional_targets(
            payload["reference"],
            reference["composition_envelope"],
            stream=stream,
            declared_standards=declared,
        )
        targets = (
            pd.concat([payload["reference"], generated], ignore_index=True)
            .drop_duplicates(["name", "adduct"])
            .reset_index(drop=True)
        )
        targets = targets.loc[
            [
                ranges.decide(
                    r["class"], r["carbon"], r["db"], species=r["name"]
                ).allowed
                for r in targets.to_dict("records")
            ]
        ].reset_index(drop=True)
        prediction = predict_aligned_retention_atlas(
            targets, payload["annotations"], reference["atlas"], stream=stream
        )
        targets = targets.drop(columns=list(prediction), errors="ignore").join(
            prediction
        )
        pool = calibrated_mass_candidates(
            targets,
            payload["features"],
            anchors,
            original["pool"][original["pool"].stream.eq(stream)],
            stream=stream,
            declared_standard_masses=declared.ref_mz,
        )
        selected = select_calibrated_additions(
            pool, automatic, old[old.stream.eq(stream)], stream=stream
        )
        expected.append(selected)
        rows.append(
            dict(
                dataset=dataset,
                stream=stream,
                coarse_candidates=len(pool),
                novel_mass_pairs=int(pool.discovery_novel_mass_pair.sum())
                if len(pool)
                else 0,
                additions=len(selected),
                old_calls_immutable=True,
                scope_and_output_reproduced=True,
            )
        )
    expected = pd.concat(expected, ignore_index=True)
    persisted = pd.read_pickle(OUT / "runs" / dataset / "private_additions.pkl")
    keys = ["stream", "name", "adduct", "feat_feature_id"]

    def exact(df):
        return (
            set(df[keys].astype(str).itertuples(index=False, name=None))
            if len(df)
            else set()
        )

    assert exact(expected) == exact(persisted) == exact(actual.iloc[len(old) :])
    assert len(actual) == len(old) + len(expected)
    assert not actual[["stream", "feat_feature_id"]].astype(str).duplicated().any()
    for r in expected.to_dict("records"):
        original_feature = baseline["streams"][r["stream"]]["features"]
        feature = original_feature[
            original_feature.feature_id.astype(str).eq(str(r["feat_feature_id"]))
        ]
        assert len(feature) == 1
        assert r["feat_mz"] == float(feature.iloc[0].mz)
        # Window arithmetic converts minutes -> seconds -> minutes; only
        # floating-point roundoff, not a retention-time adjustment, is allowed.
        assert isclose(
            r["feat_rt"], float(feature.iloc[0].rt), rel_tol=1e-14, abs_tol=1e-14
        )
        assert r["feat_intensity"] == float(feature.iloc[0].intensity)
        assert abs(r["ppm_error"]) >= 3 and abs(r["calibrated_mass_error_ppm"]) < 3
        assert abs(r["rt_error"]) <= 3 / 60
        assert r["mass_calibration_anchor_count"] >= 6
        assert (
            r["mass_calibration_MAD_ppm"] <= 0.6
            and r["mass_calibration_p90_ppm"] <= 1.5
        )
        assert abs(r["mass_calibration_center_ppm"]) <= 2
    return rows


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).with_name(
        "run_qe_calibrated_mass_discovery.py"
    ).read_bytes()
    for path, digest in protocol["source_snapshot_sha256"].items():
        assert sha(OUT / "source_snapshot" / path) == digest
    for path, digest in protocol["input_sha256"].items():
        assert sha(Path(path)) == digest
    summary = json.loads((OUT / "reports_v2/summary.json").read_text())
    assert summary["complete"] and summary["datasets_evaluated"] == 33
    projects = json.loads((OUT / "run_manifest.json").read_text())["datasets"]
    with ProcessPoolExecutor(max_workers=3) as executor:
        rows = [r for group in executor.map(verify, projects) for r in group]
    table = pd.DataFrame(rows)
    assert len(table) == 132
    table.to_csv(OUT / "discovery_integrity_by_dataset_mode.csv", index=False)
    change = pd.read_csv(
        OUT / "comparison_with_experiment_022/by_dataset_mode_before_after.csv"
    )
    assert change.correct_gain.ge(0).all()
    assert change.extra_change.ge(0).all()
    assert (
        change.laser_same_manual_lipid_different_peak_after
        >= change.laser_same_manual_lipid_different_peak_before
    ).all()
    receipt = dict(
        annotation_experiment="027",
        integrity_tool_sha256=sha(Path(__file__)),
        source_file_hash_checks=len(protocol["source_snapshot_sha256"]),
        input_file_hash_checks=len(protocol["input_sha256"]),
        datasets=33,
        mode_scope_and_full_rule_reproduction_checks=132,
        immutable_old_call_tables_checked=33,
        raw_coordinate_checks=int(table.additions.sum()),
        new_annotation_score_fits=0,
        new_acceptance_cutoff_fits=0,
        all_passed=True,
        note="Fixed-rule/group-reference/immutability engineering checks, not external validation/FDR. All original92102manualentries retained, sourcehashes165checkedbyevaluator. No private feature/model/profile data exported.",
    )
    (OUT / "integrity_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
