import numpy as np
import pandas as pd
import pytest

from matlab_annotation.paired_discovery import paired_discovery_candidates, select_paired_additions


def fixture():
    target = pd.DataFrame([dict(name="PE(36:2)", **{"class": "PE"}, carbon=36, db=2,
        adduct="[M+H]+", ref_mz=700., atlas_rt_seconds=300., atlas_training_cohorts=8,
        atlas_training_mad_seconds=2.)])
    features = pd.DataFrame([dict(feature_id="new", mz=700., rt=320/60, intensity=100.),
                             dict(feature_id="old", mz=700., rt=300/60, intensity=90.)])
    automatic = pd.DataFrame([dict(stream="NP_Neg", name="PE(36:2)", **{"class": "PE"},
        carbon=36, db=2, feat_feature_id="other", feat_rt=6., feat_intensity=100.,
        ppm_error=.1, post_regression_confidence=.99, is_is=False)])
    old = pd.DataFrame([dict(name="PE(36:2)", adduct="[M+H]+", feat_feature_id="old")])
    return target, features, automatic, old


def test_bounded_new_candidate_keeps_old_physical_alternative_and_raw_coordinates():
    t, f, a, old = fixture()
    original = f.copy(deep=True)
    result = paired_discovery_candidates(t, f, a, old, stream="RP_Pos")
    assert result.feat_feature_id.tolist() == ["new", "old"]
    assert result.discovery_novel_to007.tolist() == [True, False]
    assert result.feat_rt.iloc[0] == pytest.approx(320/60)
    pd.testing.assert_frame_equal(f, original)
    f.loc[0, "rt"] = 331/60
    assert paired_discovery_candidates(t, f, a, old, stream="RP_Pos").empty


def test_no_self_stream_isomer_or_standard_anchor_substitution():
    t, f, a, old = fixture()
    for change in [dict(stream="RP_Pos"), {"class": "PE[P]"}, dict(is_is=True)]:
        assert paired_discovery_candidates(t, f, a.assign(**change), old, stream="RP_Pos").empty


def test_spike_mass_withholds_entire_target_and_bad_rt_unavailable():
    t, f, a, old = fixture()
    assert paired_discovery_candidates(t, f, a, old, stream="RP_Pos", declared_standard_masses=[700.]).empty
    assert paired_discovery_candidates(t.assign(atlas_rt_seconds=np.nan), f, a, old, stream="RP_Pos").empty
    assert paired_discovery_candidates(t.assign(atlas_training_cohorts=2), f, a, old, stream="RP_Pos").empty
    with pytest.raises(ValueError, match="Curated labels"):
        paired_discovery_candidates(t.assign(training_positive=True), f, a, old, stream="RP_Pos")


def selection_fixture():
    t, f, a, old = fixture()
    p = paired_discovery_candidates(t, f, a, old, stream="RP_Pos")
    p["q_paired_best_log_correlation"] = .99
    p["q_paired_profile_specimens"] = 12
    occupied = pd.DataFrame([dict(name="PC(34:1)", adduct="[M+H]+", feat_feature_id="unrelated")])
    return p, occupied


def test_unique_novel_winner_only_adds_and_does_not_mutate_old_calls():
    p, old = selection_fixture()
    original = old.copy(deep=True)
    added = select_paired_additions(p, pd.Series([.95, .6]), old, threshold=.8)
    assert added.feat_feature_id.tolist() == ["new"]
    assert added.automatic_accepted.all()
    pd.testing.assert_frame_equal(old, original)


def test_occupied_or_old_pool_winner_cannot_force_weaker_free_feature():
    p, old = selection_fixture()
    assert select_paired_additions(p, pd.Series([.7, .99]), old, threshold=.5).empty
    assert select_paired_additions(p, pd.Series([.95, .7]), old.assign(feat_feature_id="new"), threshold=.5).empty
    assert select_paired_additions(p, pd.Series([.95, .9]), old, threshold=.5).empty


def test_same_feature_competing_identity_and_missing_profiles_withhold():
    p, old = selection_fixture()
    other = p.iloc[[0]].assign(name="Another(36:2)")
    p = pd.concat([p, other], ignore_index=True)
    assert select_paired_additions(p, pd.Series([.95, .6, .9]), old, threshold=.5).empty
    p = p.iloc[:2].copy()
    p["q_paired_best_log_correlation"] = np.nan
    assert select_paired_additions(p, pd.Series([.95, .6]), old, threshold=.5).empty
