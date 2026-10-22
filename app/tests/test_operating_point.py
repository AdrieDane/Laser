import pandas as pd
import pytest

from matlab_annotation.class_candidates import select_class_candidates
from matlab_annotation.operating_point import (
    choose_training_operating_point,
    select_operating_candidates,
)


def test_default_selector_parity_and_lower_cutoff_never_removes_ambiguity_guard():
    pool = pd.DataFrame(
        dict(
            name=["a", "b", "c", "d", "d"],
            adduct=["H"] * 5,
            feat_feature_id=["1", "1", "2", "3", "4"],
            feat_intensity=[100, 80, 30, 20, 10],
        )
    )
    scores = pd.Series([0.8, 0.78, 0.4, 0.9, 0.7])
    pd.testing.assert_frame_equal(
        select_operating_candidates(pool, scores, threshold=0.5),
        select_class_candidates(pool, scores, threshold=0.5),
    )
    selected = select_operating_candidates(pool, scores, threshold=0.3)
    assert selected.name.tolist() == ["d", "c"]
    assert not selected.feat_feature_id.duplicated().any()
    assert not selected[["name", "adduct"]].duplicated().any()


def test_training_budget_maximizes_correct_not_number_of_calls():
    table = pd.DataFrame(
        dict(
            threshold=[0.2, 0.3, 0.4, 0.5],
            manual=[1000] * 4,
            correct=[950, 930, 920, 920],
            extras=[150, 85, 70, 60],
        )
    )
    assert choose_training_operating_point(table) == pytest.approx(0.3)
    assert choose_training_operating_point(
        table, extras_per_manual=0.07
    ) == pytest.approx(0.5)
    assert choose_training_operating_point(table, extras_per_manual=0.01) is None


def test_invalid_denominators_or_scores_fail_closed():
    with pytest.raises(ValueError):
        choose_training_operating_point(
            pd.DataFrame(
                dict(
                    threshold=[0.2, 0.3],
                    manual=[100, 200],
                    correct=[90, 90],
                    extras=[1, 1],
                )
            )
        )
    with pytest.raises(ValueError):
        choose_training_operating_point(
            pd.DataFrame(dict(threshold=[0.3], manual=[10], correct=[11], extras=[0]))
        )
    with pytest.raises(ValueError):
        select_operating_candidates(
            pd.DataFrame(), pd.Series(dtype=float), threshold=0.01
        )
