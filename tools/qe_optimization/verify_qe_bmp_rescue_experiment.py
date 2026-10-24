"""Verify scoped030 specialist supervision, immutable core029 and all selections."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from context import WORKSPACE
from frozen_resources import configure_frozen_resources
from run_qe_bmp_rescue_experiment import (
    OUT,
    PREVIOUS,
    load_predictors,
    label_allowed_query,
    sha,
)

QE = WORKSPACE / "outputs/qe_optimization_20261022"


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (
        configure_frozen_resources(OUT / "source_snapshot")
        == protocol["resource_receipt"]
    )
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).with_name(
        "run_qe_bmp_rescue_experiment.py"
    ).read_bytes()
    for path, digest in protocol["source_snapshot_sha256"].items():
        assert sha(OUT / "source_snapshot" / path) == digest
    for path, digest in protocol["input_sha256"].items():
        assert sha(Path(path)) == digest
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    from matlab_annotation.bmp_rescue import bmp_queries, select_bmp_rescue, NP_MODES
    from matlab_annotation.evidence_training import predict_grouped_evidence
    from matlab_annotation.operating_point import choose_balanced_training_points

    summary = json.loads((OUT / "reports_v2/summary.json").read_text())
    assert summary["complete"] and summary["datasets_evaluated"] == 33
    assert summary["manual_rows"] == 92102 and summary["unresolved_manual_rows"] == 28
    projects = json.loads((OUT / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    models = list(OUT.glob("fold_*_nested_model.pkl")) + list(
        OUT.glob("calibration/fold_*/inner_block_*_model.pkl")
    )
    assert len(models) == 25
    audits, query_checks, models_with_zero_label_groups = [], 0, 0
    for path in models:
        model = pd.read_pickle(path)
        train, excluded = set(model["training_cohorts"]), set(model["excluded_groups"])
        assert train and excluded and train.isdisjoint(excluded)
        assert (
            model["supervision_scope_policy"] == "explicit_allowed_and_contributing_v1"
        )
        assert model["fit_class"] == "BMP" and model["estimator"].n_iter_ == 180
        assert set(model["categories"]["stream"]) == set(NP_MODES)
        known, positives, cohorts = 0, 0, Counter()
        positive_before, withheld = 0, 0
        for dataset, mask in model["training_feature_masks"].items():
            assert dataset[:7] in train and dataset[:7] not in excluded
            excluded_folds = {folds[d] for d in projects if d[:7] in excluded}
            assert set(mask) == excluded_folds | {folds[dataset]}
            raw = load_predictors(dataset, set(mask))
            full, _, audit = label_allowed_query(raw, train, excluded, partial=True)
            assert audit == model["partial_supervision_audit"][dataset]
            assert audit["positive_rows_before"] == audit["positive_rows_after"]
            assert not audit["evaluation_labels_changed"]
            subset = bmp_queries(full)
            original, _, _ = label_allowed_query(raw, train, excluded, partial=False)
            original = bmp_queries(original)
            pd.testing.assert_series_equal(
                subset["pool"].training_positive, original["pool"].training_positive
            )
            pd.testing.assert_frame_equal(
                subset["pool"].drop(columns="training_label_known"),
                original["pool"].drop(columns="training_label_known"),
            )
            pd.testing.assert_frame_equal(subset["evidence"], original["evidence"])
            n = int(subset["pool"].training_label_known.sum())
            assert model["training_query_known_rows"][dataset] == n
            if n:
                cohorts[dataset[:7]] += n
            known += n
            positives += int(subset["pool"].training_positive.sum())
            positive_before += int(original["pool"].training_positive.sum())
            withheld += int(original["pool"].training_label_known.sum()) - n
            scope = model["training_feature_scopes"][dataset]
            assert set(scope["training"]) == set(raw["reference_training_groups"])
            assert set(scope["excluded"]) == set(raw["reference_excluded_groups"])
            assert set(scope["standard_training"]) == set(
                raw["standard_reference_training_groups"]
            )
            assert set(scope["standard_excluded"]) == set(
                raw["standard_reference_excluded_groups"]
            )
            assert set(scope["training"]).issubset(train)
            assert dataset[:7] not in scope["training"]
            query_checks += 1
        assert {d[:7] for d in model["training_feature_masks"]} == train
        assert model["training_pairs"] == known
        assert model["positives"] == positives == positive_before
        assert set(model["supervised_training_cohorts"]) == set(cohorts)
        assert set(model["allowed_without_supervision"]) == train - set(cohorts)
        models_with_zero_label_groups += bool(model["allowed_without_supervision"])
        audits.append(
            dict(
                model=path.relative_to(OUT).as_posix(),
                allowed_projects=len(train),
                contributing_projects=len(cohorts),
                explicit_zero_label_projects=len(train - set(cohorts)),
                known_rows=known,
                positives_before=positive_before,
                positives_after=positives,
                known_negatives_withheld=withheld,
                all_scope_checks_passed=True,
            )
        )
    old_rows, new_rows, parity = 0, 0, 0
    keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
    for dataset in projects:
        outer = folds[dataset]
        model = pd.read_pickle(OUT / f"fold_{outer}_nested_model.pkl")
        assert set(model["excluded_groups"]) == {
            d[:7] for d in projects if folds[d] == outer
        }
        query = bmp_queries(load_predictors(dataset, {outer}))
        assert not {"training_positive", "training_label_known"} & set(query["pool"])
        recorded = pd.read_pickle(OUT / "runs" / dataset / "scoring_audit.pkl")
        pd.testing.assert_series_equal(predict_grouped_evidence(query, model), recorded)
        previous = pd.read_csv(
            PREVIOUS / "runs" / dataset / "automatic_annotations.csv",
            dtype={"feat_feature_id": str},
            low_memory=False,
        )
        after = pd.read_csv(
            OUT / "runs" / dataset / "automatic_annotations.csv",
            dtype={"feat_feature_id": str},
            low_memory=False,
        )
        old_keys = set(previous[keys].itertuples(index=False, name=None))
        all_keys = set(after[keys].itertuples(index=False, name=None))
        assert (
            old_keys <= all_keys
            and not after.duplicated(["stream", "feat_feature_id"]).any()
        )
        retained = after.iloc[: len(previous)]
        pd.testing.assert_frame_equal(
            previous,
            retained[previous.columns],
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        thresholds = json.loads(
            (OUT / "calibration" / f"fold_{outer}/operating_points.json").read_text()
        )["thresholds"]
        expected_keys = set(old_keys)
        for stream in ("RP_Pos", "RP_Neg", *NP_MODES):
            if stream in NP_MODES:
                part = query["pool"].loc[query["pool"].stream.eq(stream)]
                selected = select_bmp_rescue(
                    part,
                    recorded.loc[part.index],
                    previous.loc[previous.stream.eq(stream)],
                    stream=stream,
                    threshold=thresholds[stream],
                )
                expected_keys.update(selected[keys].itertuples(index=False, name=None))
            else:
                assert set(
                    after.loc[after.stream.eq(stream), keys].itertuples(
                        index=False, name=None
                    )
                ) == set(
                    previous.loc[previous.stream.eq(stream), keys].itertuples(
                        index=False, name=None
                    )
                )
            parity += 1
        assert all_keys == expected_keys
        additions = after.iloc[len(previous) :]
        assert (
            additions.stream.isin(NP_MODES).all() and additions["class"].eq("BMP").all()
        )
        assert not additions.is_is.fillna(False).astype(bool).any()
        baseline = pd.read_pickle(
            QE / "baseline/runs" / dataset / "full_annotation.pkl"
        )
        for stream, part in additions.groupby("stream"):
            raw = baseline["streams"][stream]["features"].copy()
            raw.index = raw.feature_id.astype(str)
            matched = raw.loc[part.feat_feature_id.astype(str)]
            for source, target in (
                ("mz", "feat_mz"),
                ("rt", "feat_rt"),
                ("intensity", "feat_intensity"),
            ):
                np.testing.assert_allclose(
                    matched[source].to_numpy(float),
                    part[target].to_numpy(float),
                    rtol=1e-12,
                    atol=1e-12,
                )
        old_rows += len(previous)
        new_rows += len(additions)
    for outer in range(5):
        folder = OUT / "calibration" / f"fold_{outer}"
        checkpoint = json.loads((folder / "operating_points.json").read_text())
        assert set(checkpoint["training_groups"]) == {
            d[:7] for d in projects if folds[d] != outer
        }
        assert set(checkpoint["outer_excluded"]) == {
            d[:7] for d in projects if folds[d] == outer
        }
        assert checkpoint["training_table_sha256"] == sha(
            folder / "training_operating_points.csv"
        )
        table = pd.read_csv(folder / "training_operating_points.csv")
        feasible = table.loc[table.different_peak.le(0.02 * table.manual)]
        thresholds = choose_balanced_training_points(
            feasible, extras_per_manual=0.10, target=0.9
        )
        assert thresholds == checkpoint["thresholds"]
        selected = pd.read_csv(folder / "selected_training_points.csv")
        assert selected.extras.sum() <= 0.10 * selected.manual.sum() + 1e-8
        assert selected.different_peak.le(0.02 * selected.manual + 1e-8).all()
    pd.DataFrame(audits).to_csv(OUT / "training_supervision_audit.csv", index=False)
    receipt = dict(
        all_passed=True,
        experiment="030_scoped_v2",
        datasets=33,
        dataset_modes=parity,
        source_hash_checks=len(protocol["source_snapshot_sha256"]),
        input_hash_checks=len(protocol["input_sha256"]),
        model_scope_and_supervision_checks=25,
        training_query_checks=query_checks,
        models_with_explicit_zero_label_groups=models_with_zero_label_groups,
        heldout_score_reproductions=33,
        immutable_core_rows_checked=old_rows,
        new_annotation_raw_coordinate_checks=new_rows,
        full_selection_parity_checks=parity,
        calibration_scope_hash_budget_checks=5,
        integrity_tool_sha256=sha(Path(__file__)),
        production_enabled=False,
        note="Initial strict generic scope failure preserved; ALL25fits freshly "
        "restarted with explicit allowed/reference vs contributing label scopes. "
        "No label invention or weakened query/test exclusions. All029calls immutable; "
        "extras/manual denominator unchanged,not analyticalFDR/externalvalidation.",
    )
    (OUT / "integrity_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
