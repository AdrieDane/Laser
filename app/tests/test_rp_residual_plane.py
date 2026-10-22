import numpy as np
import pandas as pd

from matlab_annotation.rp_residual_plane import predict_rp_residual_plane, _inside_hull
from matlab_annotation.retention_atlas import fit_retention_atlas


def fixture():
    data = pd.DataFrame(
        [
            dict(
                cohort=str(g),
                stream="RP_Pos",
                **{"class": "PE"},
                carbon=c,
                db=d,
                rt_seconds=300 + 8 * (c - 30) - 12 * d,
            )
            for g in range(4)
            for c in range(30, 43, 2)
            for d in range(4)
        ]
    )
    atlas = fit_retention_atlas(data)
    automatic = data[data.cohort.eq("0")].copy()
    automatic["name"] = [
        f"PE({c}:{d})"
        for c, d in automatic[["carbon", "db"]].itertuples(index=False, name=None)
    ]
    automatic["feat_rt"] = (
        automatic.rt_seconds
        + 20
        + 1.5 * (automatic.carbon - 36)
        + 2 * (automatic.db - 1)
    ) / 60
    automatic["class_selection_score"] = 0.99
    automatic["ppm_error"] = 0.2
    automatic["is_is"] = False
    automatic["atlas_rt_seconds"] = -12345
    target = pd.DataFrame([dict(stream="RP_Pos", **{"class": "PE"}, carbon=36, db=1)])
    return atlas, automatic, target


def test_shape_adjustment_from_other_compositions_and_stale_metadata_ignored():
    atlas, a, t = fixture()
    old = a.copy(deep=True)
    p = predict_rp_residual_plane(t, a, atlas, stream="RP_Pos")
    assert p.rp_plane_status.iloc[0] == "independent_bounded_RP_plane"
    assert abs(p.rp_plane_rt_seconds.iloc[0] - (300 + 8 * 6 - 12 + 20)) < 0.1
    assert p.rp_plane_other_anchors.iloc[0] == len(a) - 1
    assert p.rp_plane_loo_p90_seconds.iloc[0] < 0.1
    pd.testing.assert_frame_equal(a, old)


def test_query_cannot_change_own_prediction_by_self_anchoring():
    atlas, a, t = fixture()
    p = predict_rp_residual_plane(t, a, atlas, stream="RP_Pos")
    same = a.carbon.eq(36) & a.db.eq(1)
    a.loc[same, "feat_rt"] += 10
    q = predict_rp_residual_plane(t, a, atlas, stream="RP_Pos")
    pd.testing.assert_frame_equal(p, q)


def test_hull_no_extrapolation_and_repeated_channels_do_not_inflate_support():
    assert _inside_hull([[0, 0], [1, 0], [1, 1], [0, 1]], [0.5, 0.5])
    assert not _inside_hull([[0, 0], [1, 0], [1, 1], [0, 1]], [2, 0.5])
    atlas, a, t = fixture()
    assert (
        predict_rp_residual_plane(t.assign(carbon=44), a, atlas, stream="RP_Pos")
        .rp_plane_rt_seconds.isna()
        .all()
    )
    repeated = pd.concat([a.head(8)] * 10, ignore_index=True)
    assert (
        predict_rp_residual_plane(t, repeated, atlas, stream="RP_Pos")
        .rp_plane_rt_seconds.isna()
        .all()
    )


def test_weak_manual_standard_and_negative_phase_do_not_calibrate():
    atlas, a, t = fixture()
    for modified in (
        a.assign(class_selection_score=0.94),
        a.assign(is_is=True),
        a.assign(manual_confirmed=True),
    ):
        assert (
            predict_rp_residual_plane(t, modified, atlas, stream="RP_Pos")
            .rp_plane_rt_seconds.isna()
            .all()
        )
    assert (
        predict_rp_residual_plane(t, a, atlas, stream="NP_Pos")
        .rp_plane_rt_seconds.isna()
        .all()
    )


def test_incoherent_anchor_series_and_excessive_residual_slope_are_withheld():
    atlas, a, t = fixture()
    bad = a.copy()
    bad["feat_rt"] += np.tile([-0.2, 0.2], len(bad) // 2)
    assert (
        predict_rp_residual_plane(t, bad, atlas, stream="RP_Pos")
        .rp_plane_rt_seconds.isna()
        .all()
    )
    steep = a.copy()
    steep["feat_rt"] += (steep.carbon - 36) * 5 / 60
    assert (
        predict_rp_residual_plane(t, steep, atlas, stream="RP_Pos")
        .rp_plane_rt_seconds.isna()
        .all()
    )
