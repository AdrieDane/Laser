import numpy as np
import pandas as pd

from matlab_annotation.rp_plane_peak_choice import repeak_with_rp_plane


def fixture():
    a = pd.DataFrame(
        [
            dict(
                name="PE(36:2)",
                **{"class": "PE"},
                carbon=36,
                db=2,
                stream="RP_Pos",
                adduct="[M+H]+",
                is_is=False,
                feat_feature_id="old",
                feat_mz=700.0,
                ref_mz=700.0,
                feat_rt=5.25,
                feat_intensity=1000.0,
                class_selection_score=0.7,
                q_raw_log_median=12.0,
                confidence=0.9,
            )
        ]
    )
    f = pd.DataFrame([dict(feature_id="free", mz=700.0, rt=5.0, intensity=2000.0)])
    p = pd.DataFrame(
        [
            dict(
                rp_plane_rt_seconds=300.0,
                rp_plane_status="independent_bounded_RP_plane",
                rp_plane_other_anchors=20,
                rp_plane_loo_median_seconds=0.3,
                rp_plane_loo_p90_seconds=0.8,
            )
        ]
    )
    return a, f, p


def test_replaces_only_original_coordinates_preserves_identity_and_clears_old_evidence():
    a, f, p = fixture()
    after, candidates, audit = repeak_with_rp_plane(a, f, p, stream="RP_Pos")
    assert len(after) == len(a) and after.feat_feature_id.iloc[0] == "free"
    assert (
        after.name.iloc[0] == a.name.iloc[0]
        and after.adduct.iloc[0] == a.adduct.iloc[0]
    )
    assert (
        after.previous_feature_id.iloc[0] == "old"
        and after.previous_class_selection_score.iloc[0] == 0.7
    )
    assert np.isnan(after.q_raw_log_median.iloc[0]) and np.isnan(
        after.class_selection_score.iloc[0]
    )
    assert np.isnan(after.confidence.iloc[0]) and candidates.automatic_accepted.all()
    assert audit.action.eq("replace_original_peak_from_independent_RP_plane").all()


def test_near_old_peak_standards_manual_and_NP_stay_untouched():
    a, f, p = fixture()
    for old in (
        a.assign(feat_rt=5.02),
        a.assign(is_is=True),
        a.assign(manual_confirmed=True),
    ):
        after, _, _ = repeak_with_rp_plane(old, f, p, stream="RP_Pos")
        pd.testing.assert_frame_equal(old, after[old.columns], check_dtype=False)
    after, _, _ = repeak_with_rp_plane(a, f, p, stream="NP_Pos")
    pd.testing.assert_frame_equal(a, after)


def test_raw_ambiguity_count_before_occupancy_and_declared_spikes_protected():
    a, f, p = fixture()
    f2 = pd.concat([f, f.assign(feature_id="old")], ignore_index=True)
    assert (
        repeak_with_rp_plane(a, f2, p, stream="RP_Pos")[0].feat_feature_id.iloc[0]
        == "old"
    )
    assert (
        repeak_with_rp_plane(a, f, p, stream="RP_Pos", declared_standard_masses=[700])[
            0
        ].feat_feature_id.iloc[0]
        == "old"
    )


def test_conflicting_new_claims_are_all_withheld_and_weak_certificates_fail():
    a, f, p = fixture()
    a = pd.concat(
        [a, a.assign(name="PG(36:2)", feat_feature_id="second")], ignore_index=True
    )
    p = pd.concat([p, p], ignore_index=True)
    after, raw, audit = repeak_with_rp_plane(a, f, p, stream="RP_Pos")
    assert (
        raw.empty and audit.action.eq("withhold_conflicting_RP_plane_proposals").all()
    )
    pd.testing.assert_frame_equal(a, after[a.columns], check_dtype=False)
    a, f, p = fixture()
    assert repeak_with_rp_plane(
        a, f, p.assign(rp_plane_other_anchors=11), stream="RP_Pos"
    )[1].empty
