"""Original grid/selector semantics on synthetic INNER score cases."""

import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
import matrix_operating_points as op


def cases():
    rows, manual, scores = [], [], []
    for stream in op.MODES:
        for i, value in enumerate((1.0, 0.6, 0.4)):
            rows.append(
                dict(
                    stream=stream,
                    name=f"PC({34 + 2 * i}:1)",
                    adduct="[M+H]+",
                    feat_feature_id=stream + str(i),
                    feat_intensity=100.0,
                    training_positive=i < 2,
                    training_label_known=True,
                    calibration_identity_in_manual=i < 2,
                )
            )
            scores.append(value)
        manual.extend([dict(stream=stream, project="A", internal_standard=False)] * 2)
    pool = pd.DataFrame(rows)
    return {
        "case": dict(project="A", pool=pool, scores=pd.Series(scores))
    }, pd.DataFrame(manual)


def test_original17grid_including_native_one_score_not_disabled():
    c, m = cases()
    table = op.mode_curves(c, m, training_groups={"A"}, excluded_groups={"Q"})
    assert len(table) == 68 and list(table.threshold.unique()) == list(op.GRID)
    assert table.loc[table.threshold.eq(1), "correct"].eq(1).all()
    assert table.loc[table.threshold.eq(0.6), "correct"].eq(2).all()
    assert table.loc[table.threshold.eq(0.4), "extras"].eq(1).all()
    assert op.choose_modes(table) == {s: 0.6 for s in op.MODES}


def test_excluded_scope_and_invalid_score_are_refused():
    c, m = cases()
    with pytest.raises(ValueError):
        op.mode_curves(c, m, training_groups={"B"}, excluded_groups={"A"})
    c["case"]["scores"].iloc[0] = 1.1
    with pytest.raises(ValueError):
        op.mode_curves(c, m, training_groups={"A"}, excluded_groups={"Q"})
