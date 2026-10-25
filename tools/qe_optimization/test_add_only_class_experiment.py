"""Synthetic engineering checks; no example data or classification fits."""

import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources
import run_qe_add_only_class_experiment as runner
from test_np_context_bmp_experiment import inputs

configure_frozen_resources(REPO)


def test_disabled_export_is_entire_029_including_legacy_candidate_flags(
    tmp_path, monkeypatch
):
    previous, pool, added, _ = inputs(tmp_path, monkeypatch)
    pool.loc[0, "automatic_candidate"] = True
    pool.to_pickle(runner.common.PREVIOUS / "runs/example/candidates.pkl")
    annotations, candidates, filled = runner.assemble(
        "example", previous, added.iloc[:0]
    )
    pd.testing.assert_frame_equal(annotations, previous, check_exact=True)
    pd.testing.assert_frame_equal(candidates, pool, check_exact=True)
    assert filled.empty


def test_new_export_carries_all_original_quantification_and_correct_method_metadata(
    tmp_path, monkeypatch
):
    previous, pool, added, features = inputs(tmp_path, monkeypatch)
    annotations, candidates, filled = runner.assemble("example", previous, added)
    assert set(runner.common.accepted_keys(annotations)) == set(
        runner.common.accepted_keys(previous)
    ) | set(runner.common.accepted_keys(filled))
    pd.testing.assert_frame_equal(
        annotations.iloc[:1][list(previous)],
        previous,
        check_dtype=False,
        check_exact=True,
    )
    for frame in (
        annotations.iloc[1:],
        candidates.loc[candidates.feat_feature_id.eq("other")],
        filled,
    ):
        assert frame.feat_sample_A.iloc[0] == features.sample_A.iloc[0]
        assert (
            frame.auto_annotation_status.iloc[0]
            == "conservative_add_only_class_completion"
        )
        assert "context_completion_score" not in frame
        assert frame.class_completion_score.iloc[0] == 0.97
    assert not candidates.loc[
        candidates.feat_feature_id.eq("other"), "automatic_candidate"
    ].any()


def test_initialize_refuses_missing_prefit_receipt_before_any_outputs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "PREFLIGHT", tmp_path / "missing")
    monkeypatch.setattr(runner, "OUT", tmp_path / "outputs")
    with pytest.raises(FileNotFoundError):
        runner.initialize()
    assert not runner.OUT.exists()
