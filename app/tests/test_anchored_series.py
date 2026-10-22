import numpy as np
import pandas as pd

from matlab_annotation.anchored_series import (
    fit_class_channels,
    predict_strong_anchor_rt,
    complete_strong_anchor_series,
)
from matlab_annotation.retention_atlas import fit_retention_atlas


def atlas_fixture():
    training = pd.DataFrame(
        [
            dict(
                cohort=str(g),
                stream="RP_Pos",
                **{"class": "PE"},
                carbon=c,
                db=d,
                rt_seconds=200 + 8 * (c - 30) - 10 * d,
            )
            for g in range(4)
            for c in range(30, 47, 2)
            for d in (1, 2)
        ]
    )
    atlas = fit_retention_atlas(training)
    automatic = training[training.cohort.eq("0") & training.db.eq(1)].head(7).copy()
    automatic["feat_rt"] = (automatic.rt_seconds + 20) / 60
    automatic["ppm_error"] = 0.3
    automatic["class_selection_score"] = 0.99
    automatic["post_regression_confidence"] = 0.01  # use NEW score, not old decision
    automatic["is_is"] = False
    automatic["atlas_rt_seconds"] = -100  # stale accepted-call metadata MUST not leak
    target = pd.DataFrame([dict(stream="RP_Pos", **{"class": "PE"}, carbon=44, db=1)])
    return atlas, automatic, target


def test_refined_rt_uses_six_distinct_other_compositions_and_cleans_old_metadata():
    atlas, a, t = atlas_fixture()
    old = a.copy(deep=True)
    p = predict_strong_anchor_rt(t, a, atlas, stream="RP_Pos")
    assert p.refined_anchor_count.iloc[0] == 7
    assert abs(p.anchor_refined_rt_seconds.iloc[0] - (200 + 8 * 14 - 10 + 20)) < 0.1
    pd.testing.assert_frame_equal(a, old)
    # Query cannot vote for itself: six anchors including query -> five OTHER.
    q = t.assign(carbon=30)
    assert (
        predict_strong_anchor_rt(q, a.head(6), atlas, stream="RP_Pos")
        .anchor_refined_rt_seconds.isna()
        .all()
    )


def test_weak_scores_repeated_adducts_standards_and_discordant_anchors_do_not_count():
    atlas, a, t = atlas_fixture()
    assert (
        predict_strong_anchor_rt(
            t, a.assign(class_selection_score=0.94), atlas, stream="RP_Pos"
        )
        .anchor_refined_rt_seconds.isna()
        .all()
    )
    assert (
        predict_strong_anchor_rt(t, a.assign(is_is=True), atlas, stream="RP_Pos")
        .anchor_refined_rt_seconds.isna()
        .all()
    )
    repeated = pd.concat([a.head(5), a.head(5)], ignore_index=True)
    assert (
        predict_strong_anchor_rt(t, repeated, atlas, stream="RP_Pos")
        .anchor_refined_rt_seconds.isna()
        .all()
    )
    bad = a.copy()
    bad.loc[bad.index[:2], "feat_rt"] += 2
    assert (
        predict_strong_anchor_rt(t, bad, atlas, stream="RP_Pos")
        .anchor_refined_rt_seconds.isna()
        .all()
    )


def completion_fixture():
    before = pd.DataFrame(
        [dict(name="PC(34:1)", adduct="[M+H]+", feat_feature_id="old", is_is=False)]
    )
    features = pd.DataFrame(
        [dict(feature_id="free", mz=700.0, rt=5.0, intensity=1000.0)]
    )
    targets = pd.DataFrame(
        [
            dict(
                name="PE(36:2)",
                **{"class": "PE"},
                adduct="[M+H]+",
                carbon=36,
                db=2,
                ref_mz=700.0,
                ref_rt=4.5,
                anchor_refined_rt_seconds=300.0,
                refined_anchor_count=8,
            )
        ]
    )
    priors = {"RP_Pos|PE": dict(cohorts=20, joint={"36:2": 10})}
    channels = fit_class_channels(
        pd.DataFrame(
            [
                dict(cohort=str(g), stream="RP_Pos", **{"class": "PE"}, adduct="[M+H]+")
                for g in range(5)
            ]
        )
    )
    return before, features, targets, priors, channels


def test_completion_adds_unique_supported_original_peak_and_preserves_every_old_call():
    a, f, t, p, c = completion_fixture()
    result, raw, audit = complete_strong_anchor_series(a, f, t, p, c, stream="RP_Pos")
    assert result.feat_feature_id.tolist() == ["old", "free"]
    assert np.isnan(result.class_selection_score.iloc[-1])
    assert raw.automatic_accepted.all()
    assert audit.action.eq("add_unique_strong_anchor_hit").all()


def test_completion_counts_all_raw_hits_before_occupancy_and_spike_filtering():
    a, f, t, p, c = completion_fixture()
    f = pd.concat([f, f.assign(feature_id="old")], ignore_index=True)
    assert len(complete_strong_anchor_series(a, f, t, p, c, stream="RP_Pos")[0]) == 1
    a, f, t, p, c = completion_fixture()
    assert (
        len(
            complete_strong_anchor_series(
                a, f, t, p, c, stream="RP_Pos", declared_standard_masses=[700]
            )[0]
        )
        == 1
    )


def test_unseen_channels_compositions_and_competing_new_identities_are_withheld():
    a, f, t, p, c = completion_fixture()
    assert len(complete_strong_anchor_series(a, f, t, {}, c, stream="RP_Pos")[0]) == 1
    assert (
        len(
            complete_strong_anchor_series(
                a, f, t.assign(adduct="[M+Na]+"), p, c, stream="RP_Pos"
            )[0]
        )
        == 1
    )
    t = pd.concat([t, t.assign(name="PG(36:2)", **{"class": "PG"})], ignore_index=True)
    p["RP_Pos|PG"] = p["RP_Pos|PE"]
    c["models"]["RP_Pos|PG"] = c["models"]["RP_Pos|PE"]
    assert len(complete_strong_anchor_series(a, f, t, p, c, stream="RP_Pos")[0]) == 1


def test_channel_prior_counts_groups_not_duplicate_adduct_rows():
    data = pd.DataFrame(
        [
            dict(cohort=str(g), stream="NP_Neg", **{"class": "PE"}, adduct="[M-H]-")
            for g in range(4)
        ]
    )
    data = pd.concat([data] * 10, ignore_index=True)
    model = fit_class_channels(data)["models"]["NP_Neg|PE"]
    assert model["cohorts"] == 4 and model["adducts"]["[M-H]-"] == 4
