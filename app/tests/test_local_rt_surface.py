import numpy as np
import pandas as pd
import pytest

from matlab_annotation.local_rt_surface import predict_local_anchor_surfaces


def data():
    anchors = pd.DataFrame(
        [
            dict(
                name=f"PE({c}:{d})",
                **{"class": "PE"},
                carbon=c,
                db=d,
                feat_rt=(200 + 15 * (c - 34) - 22 * d) / 60,
                feat_intensity=1e6,
                ppm_error=0.3,
                post_regression_confidence=0.99,
                is_is=False,
            )
            for c in (32, 34, 36, 38)
            for d in (0, 1, 2, 3)
        ]
    )
    targets = pd.DataFrame(
        {"class": ["PE", "PE", "unknown"], "carbon": [36, 35, 36], "db": [2, 1, 2]}
    )
    return anchors, targets


def test_local_surface_recovers_independent_run_offset_without_a_manual_fit():
    anchors, targets = data()
    old = anchors.copy(deep=True)
    predictions = predict_local_anchor_surfaces(targets, anchors)
    assert predictions.local_surface_rt_seconds.iloc[0] == pytest.approx(186.0)
    assert predictions.local_surface_rt_seconds.iloc[1] == pytest.approx(193.0)
    assert np.isnan(predictions.local_surface_rt_seconds.iloc[2])
    assert predictions.local_surface_support.iloc[0] >= 8
    pd.testing.assert_frame_equal(old, anchors)


def test_target_never_calibrates_itself_even_through_repeated_adducts():
    anchors, targets = data()
    without = anchors[~(anchors.carbon.eq(36) & anchors.db.eq(2))]
    expected = predict_local_anchor_surfaces(targets.iloc[:1], without)
    fake = anchors[(anchors.carbon.eq(36) & anchors.db.eq(2))].assign(
        feat_rt=9999.0, feat_intensity=1e12
    )
    poisoned = pd.concat(
        [without, pd.concat([fake] * 20, ignore_index=True)], ignore_index=True
    )
    actual = predict_local_anchor_surfaces(targets.iloc[:1], poisoned)
    pd.testing.assert_frame_equal(expected, actual)


def test_standards_weak_mass_or_confidence_and_sparse_support_are_not_anchors():
    anchors, targets = data()
    for column, value in [
        ("is_is", True),
        ("ppm_error", 1.5),
        ("post_regression_confidence", 0.89),
    ]:
        result = predict_local_anchor_surfaces(
            targets, anchors.assign(**{column: value})
        )
        assert result.local_surface_rt_seconds.isna().all()
    repeated = pd.concat([anchors.iloc[:7]] * 10, ignore_index=True)
    assert (
        predict_local_anchor_surfaces(targets, repeated)
        .local_surface_rt_seconds.isna()
        .all()
    )
    with pytest.raises(ValueError):
        predict_local_anchor_surfaces(targets, anchors, minimum_anchors=4)
