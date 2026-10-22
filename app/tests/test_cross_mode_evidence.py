import numpy as np
import pandas as pd
import pytest

from matlab_annotation.cross_mode_evidence import cross_mode_evidence


def fixture():
    autos, features = [], {}
    for stream, offset in [("RP_Pos", 2), ("RP_Neg", 0)]:
        x = pd.DataFrame(
            [
                dict(
                    feature_id=str(i),
                    mz=500 + i,
                    rt=(100 + i + offset) / 60,
                    a=10 * (i + 1),
                    b=20 * (i + 1),
                    c=30 * (i + 1),
                    d=40 * (i + 1),
                )
                for i in range(10)
            ]
        )
        x.attrs = dict(
            sample_columns=list("abcd"),
            sample_group_mapping_status="validated",
            sample_group_map={c: "Group_" + c for c in "abcd"},
        )
        features[stream] = x
        autos.extend(
            [
                dict(
                    stream=stream,
                    name=f"PE({30 + i}:1)",
                    **{"class": "PE"},
                    carbon=30 + i,
                    db=1,
                    feat_feature_id=str(i),
                    feat_rt=(100 + i + offset) / 60,
                    feat_intensity=100 + i,
                    ppm_error=0.1,
                    post_regression_confidence=0.99,
                    is_is=False,
                )
                for i in range(10)
            ]
        )
    pool = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                **{"class": "PE"},
                carbon=30,
                db=1,
                feat_feature_id="0",
                feat_rt=102 / 60,
            )
        ]
    )
    return pool, features, pd.DataFrame(autos)


def test_independent_stream_profiles_and_leave_own_composition_out():
    pool, features, auto = fixture()
    original = features["RP_Pos"].copy(deep=True)
    result = cross_mode_evidence(pool, features, auto).iloc[0]
    assert result.q_cross_mode_supporting_streams == 1
    assert result.q_cross_mode_RT_anchor_count == 9
    assert result.q_cross_mode_RT_residual_seconds == pytest.approx(0)
    assert result.q_cross_mode_best_group_profile_correlation == pytest.approx(1)
    assert result.q_cross_mode_profile_groups == 4
    pd.testing.assert_frame_equal(features["RP_Pos"], original)


def test_no_guessed_sample_pairing_unknown_metadata_profiles_stay_unknown():
    pool, features, auto = fixture()
    features["RP_Neg"].attrs["sample_group_mapping_status"] = "unknown"
    result = cross_mode_evidence(pool, features, auto).iloc[0]
    assert np.isnan(result.q_cross_mode_best_group_profile_correlation)
    assert result.q_cross_mode_supporting_streams == 1
    assert np.isfinite(result.q_cross_mode_RT_residual_seconds)


def test_never_merge_ether_or_positional_isomers_and_ignore_standards():
    pool, features, auto = fixture()
    auto.loc[auto.stream.eq("RP_Neg"), "class"] = "PE[P]"
    assert (
        cross_mode_evidence(pool, features, auto)
        .iloc[0]
        .q_cross_mode_supporting_streams
        == 0
    )
    auto["class"] = "PE"
    auto["is_is"] = True
    assert (
        cross_mode_evidence(pool, features, auto)
        .iloc[0]
        .q_cross_mode_supporting_streams
        == 0
    )


def test_blank_qc_and_flat_profiles_cannot_supply_correlation():
    pool, features, auto = fixture()
    for x in features.values():
        x.attrs["sample_roles"] = dict(
            a="biological", b="biological", c="qc", d="blank"
        )
    assert np.isnan(
        cross_mode_evidence(pool, features, auto)
        .iloc[0]
        .q_cross_mode_best_group_profile_correlation
    )
    for x in features.values():
        x.attrs.pop("sample_roles")
        x[list("abcd")] = 100
    assert np.isnan(
        cross_mode_evidence(pool, features, auto)
        .iloc[0]
        .q_cross_mode_best_group_profile_correlation
    )


def test_rt_offsets_are_not_transferred_between_np_and_rp():
    pool, features, auto = fixture()
    features["NP_Neg"] = features.pop("RP_Neg")
    auto.loc[auto.stream.eq("RP_Neg"), "stream"] = "NP_Neg"
    result = cross_mode_evidence(pool, features, auto).iloc[0]
    assert result.q_cross_mode_supporting_streams == 1
    assert result.q_cross_mode_same_phase_support == 0
    assert np.isnan(result.q_cross_mode_RT_residual_seconds)
