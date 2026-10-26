"""Standard-only software checks, no QE annotations/classifier fits."""

import numpy as np
import pandas as pd
import pytest

import standard_mass_trend_pilot as m


def anchors():
    x = np.arange(300, 1000, 100, dtype=float)
    return pd.DataFrame(
        dict(
            stream="RP_Pos",
            neutral_key=[str(i) for i in range(len(x))],
            feature_id=[str(i) for i in range(len(x))],
            reference_mz=x,
            ppm=(x - 600) / 500,
        )
    )


def test_exact_slope_and_complete_neutral_holdout():
    a = anchors()
    table = m.leave_neutral_out(m.grouped_anchors(a))
    assert len(table) == 7 and table.training_neutrals.eq(6).all()
    assert table.trend_LOO_absolute_error_ppm.max() < 1e-14
    assert table.inside_training_mass_span.sum() == 5
    assert table.constant_LOO_absolute_error_ppm.sum() > 1
    assert table.trend_slope_ppm_per_1000Da.eq(2).all()


def test_multiple_adducts_do_not_multiply_neutral_votes():
    a = anchors()
    extra = a.iloc[[0]].assign(feature_id="other-ion", reference_mz=322)
    grouped = m.grouped_anchors(pd.concat([a, extra], ignore_index=True))
    assert len(grouped) == 7 and grouped.physical_ions.max() == 2
    assert m.leave_neutral_out(grouped).training_neutrals.eq(6).all()


def test_sparse_and_no_mass_span_are_not_zero_fits():
    assert m.leave_neutral_out(m.grouped_anchors(anchors().iloc[:5])).empty
    assert m.robust_line(np.arange(10), np.arange(10)) is None
    assert m.independent_line(np.arange(10), np.arange(10)) is None


def test_scalar_vector_parity_with_outlier_and_tied_masses():
    rng = np.random.default_rng(27)
    for size in (5, 6, 10, 25):
        x = rng.integers(300, 1200, size).astype(float)
        y = rng.normal(0, 0.2, size)
        y[0] = 2.5
        np.testing.assert_allclose(
            m.robust_line(x, y), m.independent_line(x, y), rtol=0, atol=1e-12
        )


def test_reject_labels_shared_features_and_uncensored_input():
    for a in (
        anchors().assign(training_positive=False),
        anchors().assign(feature_id="one-feature"),
        anchors().assign(ppm=4),
    ):
        with pytest.raises(ValueError):
            m.grouped_anchors(a)
