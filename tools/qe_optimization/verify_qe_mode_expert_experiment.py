"""Verify028 four-mode expert scopes, unchanged labels and fixed mass stage."""

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from context import WORKSPACE
from frozen_resources import configure_frozen_resources

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "experiment_028"
CORE = QE / "experiment_022"
OLD = QE / "experiment_027"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (
        configure_frozen_resources(OUT / "source_snapshot")
        == protocol["resource_receipt"]
    )
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).with_name(
        "run_qe_mode_expert_experiment.py"
    ).read_bytes()
    for path, digest in protocol["source_snapshot_sha256"].items():
        assert sha(OUT / "source_snapshot" / path) == digest
    for path, digest in protocol["input_sha256"].items():
        assert sha(Path(path)) == digest
    summary = json.loads((OUT / "reports_v2/summary.json").read_text())
    assert summary["complete"] and summary["datasets_evaluated"] == 33
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    from matlab_annotation.mode_evidence_experts import (
        predict_mode_evidence_experts as predict_grouped_evidence,
    )
    from matlab_annotation.calibrated_mass_discovery import select_calibrated_additions
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards

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
        assert model["positives"] == old["positives"]
        assert model["training_pairs"] == old["training_pairs"]
        assert model["columns"] == old["columns"]
        assert not any(
            str(c).startswith("q_declared_standard_") for c in model["columns"]
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
        for dataset, mask in model["training_feature_masks"].items():
            assert dataset[:7] in training and dataset[:7] not in excluded
            excluded_folds = {folds[d] for d in projects if d[:7] in excluded}
            assert set(mask) == excluded_folds | {folds[dataset]}
            training_queries += 1
        audits.append(
            dict(
                model=path.relative_to(OUT).as_posix(),
                training_pairs_before=old["training_pairs"],
                training_pairs_after=model["training_pairs"],
                positives_before=old["positives"],
                positives_after=model["positives"],
                fitted_mode_experts=4,
            )
        )
    parity, demoted_aliases = 0, 0
    for dataset in projects:
        outer = folds[dataset]
        mask = {d[:7] for d in projects if folds[d] == outer}
        path = (
            QE
            / "nested_candidate_preparation_v2/prepared"
            / f"exclude_{outer}"
            / (dataset + ".pkl")
        )
        data = pd.read_pickle(path)
        assert not {"training_positive", "training_label_known"} & set(data["pool"])
        model = pd.read_pickle(OUT / f"fold_{outer}_nested_model.pkl")
        assert set(model["excluded_groups"]) == mask
        recorded = pd.read_pickle(OUT / "runs" / dataset / "scoring_audit.pkl")
        pd.testing.assert_series_equal(predict_grouped_evidence(data, model), recorded)
        automatic = pd.read_csv(
            OUT / "runs" / dataset / "automatic_annotations.csv",
            dtype={"feat_feature_id": str},
            low_memory=False,
        )
        baseline = pd.read_pickle(
            QE / "baseline/runs" / dataset / "full_annotation.pkl"
        )
        catalogue = declared_standard_targets(
            OUT / "source_snapshot/reference_data/matlab_pipeline",
            baseline["internal_standards_expected"],
        )
        standards = json.loads(
            (
                QE / "experiment_005/models" / f"fold_{outer}_standard_models.json"
            ).read_text()
        )
        mass_pool = pd.read_pickle(
            OLD / "runs" / dataset / "private_mass_candidates.pkl"
        )
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
            keys = ["name", "adduct", "feat_feature_id", "is_is"]
            assert set(
                expected[keys].astype(str).itertuples(index=False, name=None)
            ) == set(actual[keys].astype(str).itertuples(index=False, name=None))
            parity += 1
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
    receipt = dict(
        integrity_tool_sha256=sha(Path(__file__)),
        source_file_hash_checks=len(protocol["source_snapshot_sha256"]),
        input_file_hash_checks=len(protocol["input_sha256"]),
        fitted_model_scope_and_unchanged_positive_checks=len(paths),
        training_query_exclusion_checks=training_queries,
        heldout_score_reproduction_checks=33,
        selector_plus_unchanged_standard_cleanup_parity_checks=parity,
        existing_standard_alias_demotions_reproduced=demoted_aliases,
        calibration_scope_hash_budget_checks=calibration_checks,
        completed_datasets=33,
        all_passed=True,
        annotation_experiment="028",
        new_standard_context_used=False,
        fitted_mode_experts=100,
        core_reference="022",
        fixed_mass_stage_reference="027",
        note="Engineering integrity,not external validation/FDR. Same022corefeatures/labels;fourfixedmodeexperts forALLmodes,notposthocphasewinners. Same005STDandfixed027massselection. Calibrationbudgetcoverscoreonlyasin027;no outer-onlymassproposals enterinnercutofffitting. All92102manualentries/extras unchangeddefinition;privateprofiles/models stayoutsideGit.",
    )
    (OUT / "integrity_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
