"""Synthetic engineering only; never read the example datasets."""

import sys

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
sys.path.insert(0, str(REPO / "app/tests"))

from test_isomer_label_pairs import record  # noqa: E402
from test_isomer_label_resolution import fixture  # noqa: E402
from matlab_annotation.isomer_label_resolution import isomer_label_preferences  # noqa: E402
from matlab_annotation.operating_point import select_operating_candidates  # noqa: E402
from preflight_qe_isomer_labels import verify_pairs, accepted_keys  # noqa: E402
from run_qe_isomer_label_experiment import select_core, MODES  # noqa: E402


def test_disabled_selector_preserves_exact_baseline_keys_scores_and_measurements(
    monkeypatch,
):
    data, model, protection, scores = fixture(monkeypatch)
    prefs = isomer_label_preferences(data, model, **protection)
    base = select_operating_candidates(data["pool"], scores, threshold=0.5)
    selected, modified, _ = select_core(
        data["pool"], scores, prefs, {s: 1.0 for s in MODES}, {s: 0.5 for s in MODES}
    )
    assert accepted_keys(selected) == accepted_keys(base)
    pd.testing.assert_series_equal(modified, scores, check_exact=True)
    assert selected.isomer_label_preference.isna().all()
    for row in selected.itertuples():
        physical = data["pool"][
            data["pool"].name.eq(row.name)
            & data["pool"].feat_feature_id.eq(row.feat_feature_id)
        ].iloc[0]
        assert (row.feat_mz, row.feat_rt, row.feat_intensity) == (
            physical.feat_mz,
            physical.feat_rt,
            physical.feat_intensity,
        )


def test_resolved_annotation_keeps_detection_proxy_and_original_label_score_separate(
    monkeypatch,
):
    data, model, protection, scores = fixture(monkeypatch)
    prefs = isomer_label_preferences(data, model, **protection)
    selected, _, _ = select_core(
        data["pool"], scores, prefs, {s: 0.7 for s in MODES}, {s: 0.5 for s in MODES}
    )
    row = selected[selected.name.eq("PC-P(36:2)")].iloc[0]
    assert row.class_selection_score == row.isomer_feature_detection_proxy == 0.9
    assert row.original_pointwise_class_selection_score == 0.6
    assert row.isomer_label_preference == 0.8
    assert row.auto_annotation_status == "isomer_resolved_feature_detection_proxy"
    assert len(selected) == 2


def test_independent_training_support_and_balanced_weights_match_all_modes():
    a, b = record("a"), record("b")
    b["pool"].loc[4, "ref_mz"] = 700.0
    summary = verify_pairs({"a": a, "b": b}, {"a", "b", "c"}, {"z"})
    assert summary["RP_Pos"]["pair_rows"] == 6
    assert summary["RP_Pos"]["pair_rows_by_project"] == {"a": 2, "b": 4}
    assert summary["RP_Pos"]["supervised_contributing_projects"] == ["a", "b"]
    assert summary["NP_Pos"]["pair_rows"] == 0
    assert summary["NP_Neg"]["supervised_contributing_projects"] == []


def test_string_candidate_indices_are_not_confused_with_original_feature_ids(
    monkeypatch,
):
    data, model, protection, scores = fixture(monkeypatch)
    labels = ["rowA", "rowB", "rowC", "rowD", "rowE"]
    for frame in (data["pool"], data["evidence"], scores):
        frame.index = labels
    prefs = isomer_label_preferences(data, model, **protection)
    selected, _, _ = select_core(
        data["pool"], scores, prefs, {s: 0.7 for s in MODES}, {s: 0.5 for s in MODES}
    )
    assert selected[selected.name.eq("PC-P(36:2)")].iloc[0].feat_feature_id == "1"
