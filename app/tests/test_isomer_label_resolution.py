import numpy as np
import pandas as pd
import pytest

import matlab_annotation.isomer_label_resolution as module
from matlab_annotation.mode_evidence_experts import MODES


def fixture(monkeypatch, preferences=(0.8, 0.9, 0.7)):
    pool = pd.DataFrame(
        dict(
            stream=["RP_Pos"] * 5,
            name=[
                "PC-P(36:2)",
                "PC-O(36:3)",
                "isomerC",
                "other_adduct",
                "other_feature",
            ],
            adduct=["H", "H", "H", "Na", "H"],
            **{"class": ["PC-P", "PC-O", "PC", "PC", "PC"]},
            feat_feature_id=["1", "1", "1", "1", "2"],
            ref_mz=[700.0] * 5,
            is_is=[False] * 5,
            feat_mz=[700.0001] * 5,
            feat_rt=[5.0] * 5,
            feat_intensity=[1000.0] * 5,
        )
    )
    evidence = pd.DataFrame(
        dict(
            stream=pool.stream,
            **{"class": pool["class"]},
            absolute_RT_error_seconds=[1.0, 2.0, 3.0, 1.0, 2.0],
        )
    )
    data = dict(
        query_group="z",
        reference_training_groups=["a", "b", "c"],
        reference_excluded_groups=["z"],
        pool=pool,
        evidence=evidence,
    )
    model = dict(training_reference_permissions=["a", "b", "c"], excluded_groups=["z"])
    monkeypatch.setattr(
        module,
        "predict_isomer_label_preferences",
        lambda data, model, left, right: np.asarray(preferences[: len(left)]),
    )
    protection = dict(
        protected_features_by_stream={s: set() for s in MODES},
        protected_masses_by_stream={s: [] for s in MODES},
    )
    scores = pd.Series([0.6, 0.9, 0.4, 0.3, 0.5], name="base_score")
    return data, model, protection, scores


def test_complete_unique_winner_transfers_detection_proxy_without_touching_inputs(
    monkeypatch,
):
    data, model, protection, scores = fixture(monkeypatch)
    before_pool, before_scores = data["pool"].copy(), scores.copy()
    prefs = module.isomer_label_preferences(data, model, **protection)
    assert len(prefs) == 3
    adjusted, audit = module.resolve_isomer_scores(
        data["pool"], scores, prefs, {s: 0.7 for s in MODES}
    )
    assert adjusted.iloc[0] == 0.9 and adjusted.iloc[[1, 2]].isna().all()
    pd.testing.assert_series_equal(adjusted.iloc[3:], scores.iloc[3:])
    assert audit.iloc[0].winner_index == 0
    assert audit.iloc[0].conditional_min_preference == 0.8
    assert audit.iloc[0].decision == "resolved_isomer_label_detection_proxy"
    pd.testing.assert_frame_equal(data["pool"], before_pool, check_exact=True)
    pd.testing.assert_series_equal(scores, before_scores, check_exact=True)


@pytest.mark.parametrize(
    "preferences", [(np.nan, 0.9, 0.7), (0.8, 0.2, 0.8), (0.6, 0.6, 0.6)]
)
def test_unknown_cyclic_or_insufficient_complete_preferences_do_not_prune_cluster(
    monkeypatch, preferences
):
    data, model, protection, scores = fixture(monkeypatch, preferences)
    prefs = module.isomer_label_preferences(data, model, **protection)
    adjusted, _ = module.resolve_isomer_scores(
        data["pool"], scores, prefs, {s: 0.7 for s in MODES}
    )
    pd.testing.assert_series_equal(adjusted, scores, check_exact=True)


def test_disabled_method_exactly_preserves_baseline_and_partial_pairs_fail(monkeypatch):
    data, model, protection, scores = fixture(monkeypatch)
    prefs = module.isomer_label_preferences(data, model, **protection)
    adjusted, _ = module.resolve_isomer_scores(
        data["pool"], scores, prefs, {s: 1.0 for s in MODES}
    )
    pd.testing.assert_series_equal(adjusted, scores, check_exact=True)
    with pytest.raises(ValueError, match="COMPLETE"):
        module.resolve_isomer_scores(
            data["pool"], scores, prefs.iloc[:2], {s: 0.7 for s in MODES}
        )


def test_spike_features_masses_named_standards_and_one_bad_rt_withhold_whole_cluster(
    monkeypatch,
):
    for case in ("feature", "mass", "named", "role", "rt"):
        data, model, protection, _ = fixture(monkeypatch)
        if case == "feature":
            protection["protected_features_by_stream"]["RP_Pos"] = {"1"}
        elif case == "mass":
            protection["protected_masses_by_stream"]["RP_Pos"] = [700.0]
        elif case == "named":
            data["pool"].loc[1, "name"] = "IS: PC"
        elif case == "role":
            data["pool"].loc[1, "is_is"] = True
        else:
            data["evidence"].loc[1, "absolute_RT_error_seconds"] = 11.0
        assert module.isomer_label_preferences(data, model, **protection).empty


def test_all_physical_versions_included_in_old_proxy_and_suppression(monkeypatch):
    data, model, protection, scores = fixture(monkeypatch)
    data["pool"] = pd.concat([data["pool"], data["pool"].iloc[[1]]], ignore_index=True)
    data["evidence"] = pd.concat(
        [data["evidence"], data["evidence"].iloc[[1]]], ignore_index=True
    )
    scores = pd.concat([scores, pd.Series([0.95], index=[5], name=scores.name)])
    prefs = module.isomer_label_preferences(data, model, **protection)
    adjusted, _ = module.resolve_isomer_scores(
        data["pool"], scores, prefs, {s: 0.7 for s in MODES}
    )
    assert adjusted.iloc[0] == 0.95 and adjusted.iloc[[1, 2, 5]].isna().all()
    data["pool"].loc[5, "ref_mz"] = 701.0
    assert module.isomer_label_preferences(data, model, **protection).empty


def test_non_clique_mass_chain_is_not_silently_split_into_supported_partial_cluster(
    monkeypatch,
):
    data, model, protection, _ = fixture(monkeypatch)
    data["pool"].loc[[0, 1, 2], "ref_mz"] = [700.0, 700.0 + 0.9e-8, 700.0 + 1.8e-8]
    assert module.isomer_label_preferences(data, model, **protection).empty


def test_high_preference_cannot_force_label_without_distinct_rt_support(monkeypatch):
    data, model, protection, scores = fixture(monkeypatch)
    data["evidence"].loc[[0, 1], "absolute_RT_error_seconds"] = 1.0
    prefs = module.isomer_label_preferences(data, model, **protection)
    modified, audit = module.resolve_isomer_scores(
        data["pool"], scores, prefs, {s: 0.7 for s in MODES}
    )
    pd.testing.assert_series_equal(modified, scores, check_exact=True)
    assert audit.iloc[0].decision == "isomer_rt_evidence_not_distinct"
