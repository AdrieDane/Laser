"""Synthetic orchestration/export/calibration guards; no example model fits."""

import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources
from test_np_context_bmp_experiment import inputs
from verify_immutable_exports import check_export
from verify_qe_raw_mass_profile_experiment import check_enabled_curves
import run_qe_raw_mass_profile_experiment as runner

configure_frozen_resources(REPO)


def test_runner_refuses_missing_full_prefit_before_output(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PREFLIGHT", tmp_path / "missing")
    monkeypatch.setattr(runner, "OUT", tmp_path / "outputs")
    with pytest.raises(FileNotFoundError):
        runner.initialize()
    assert not runner.OUT.exists()


def test_scope_rejection_happens_before_any_query_truth_read(monkeypatch):
    reads = []
    monkeypatch.setattr(
        runner.inherited.common.pd, "read_pickle", lambda p: reads.append(p)
    )
    with pytest.raises(ValueError):
        runner.authorize_labels(dict(query_group="test"), {"train"}, {"test"})
    assert reads == []


def test_disabled_full_export_and_added_quantification_remain_exact(
    tmp_path, monkeypatch
):
    previous, candidates, added, features = inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "PREVIOUS", tmp_path / "old")
    annotations, actual, filled = runner.assemble("example", previous, added.iloc[:0])
    pd.testing.assert_frame_equal(annotations, previous, check_exact=True)
    pd.testing.assert_frame_equal(actual, candidates, check_exact=True)
    assert filled.empty
    # This fixture supplies raw quantification only for the new "other" peak;
    # use a separate empty-acceptance case for the all-raw-field corruption test.
    previous = previous.iloc[:0]
    candidates = candidates.assign(automatic_accepted=False, automatic_candidate=True)
    candidates.to_pickle(tmp_path / "old/runs/example/candidates.pkl")
    annotations, actual, filled = runner.assemble("example", previous, added)
    assert filled.auto_annotation_status.eq(runner.METHOD).all()
    assert filled.feat_sample_A.iloc[0] == 123.4
    path = tmp_path / "annotations.csv"
    annotations.to_csv(path, index=False)
    report = check_export(
        path,
        previous,
        actual,
        candidates,
        dict(streams={"NP_Pos": dict(features=features)}),
    )
    assert report["raw_stream_checks"] == 1
    broken = pd.read_csv(path)
    broken.loc[broken.feat_feature_id.eq("other"), "feat_sample_A"] = 0
    broken.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        check_export(
            path,
            previous,
            actual,
            candidates,
            dict(streams={"NP_Pos": dict(features=features)}),
        )


def test_verifier_uses_actual_curve_schema_and_refuses_bad_tradeoff():
    row = pd.DataFrame(
        [
            dict(
                threshold=0.5,
                feasible=True,
                eligible=True,
                different_peak=0,
                added_correct=12,
                extras=1,
                manual=100,
                minimum_leave_one_project_agreement=0.9,
            )
        ]
    )
    check_enabled_curves(row)
    check_enabled_curves(row.assign(threshold=1, feasible=True, eligible=False))
    for broken in (
        row.assign(different_peak=1),
        row.assign(extras=3),
        row.assign(eligible=False),
        row.assign(added_correct=9),
        row.assign(minimum_leave_one_project_agreement=0.7),
    ):
        with pytest.raises(AssertionError):
            check_enabled_curves(broken)
