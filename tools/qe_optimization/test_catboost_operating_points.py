import numpy as np
import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
from catboost_operating_points import select_candidates, mode_counts, class_curves
from matlab_annotation.operating_point import select_operating_candidates


def fixture():
    p = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                **{"class": cls},
                name=name,
                adduct="[M+H]+",
                feat_feature_id=fid,
                feat_intensity=intensity,
            )
            for cls, name, fid, intensity in (
                ("PC", "PC(34:1)", "one", 20),
                ("PE", "PE(37:1)", "one", 20),
                ("PC", "PC(34:1)", "two", 10),
                ("PE", "PE(36:1)", "three", 30),
            )
        ]
    )
    return p, pd.Series([0.9, 0.7, 0.8, 0.01])


@pytest.mark.parametrize("cut", [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95])
def test_uniform_cut_native_selector_parity_on_original_domain(cut):
    p, scores = fixture()
    old = select_operating_candidates(p, scores, threshold=cut)
    new = select_candidates(p, scores, baseline_threshold=cut)
    assert old.index.tolist() == new.index.tolist()
    pd.testing.assert_series_equal(old.class_selection_score, new.class_selection_score)


def test_native_low_domain_no_rescaling_and_explicit_disable_even_score1():
    p, scores = fixture()
    assert select_candidates(p, scores, baseline_threshold=0).index.tolist() == [0, 3]
    scores.iloc[0] = 1
    assert select_candidates(p, scores, baseline_threshold=1).empty
    changed = select_candidates(
        p, scores, baseline_threshold=0, class_thresholds={"PC": 1}
    )
    assert changed.index.tolist() == [3]
    assert changed.class_selection_score.iloc[0] == 0.01


def test_class_cutoffs_cannot_promote_a_losing_label_or_second_peak():
    p, scores = fixture()
    chosen = select_candidates(
        p, scores, baseline_threshold=0, class_thresholds={"PC": 1}
    )
    assert "PE(37:1)" not in set(chosen.name)
    assert "two" not in set(chosen.feat_feature_id)


def test_curve_disabled_counts_match_actual_selection_with_saturated_fixture():
    p, scores = fixture()
    p["project"] = "A"
    p["lipid_class"] = p["class"]
    p["class_selection_score"] = scores
    p["training_positive"] = True
    p["training_label_known"] = True
    p["calibration_identity_in_manual"] = True
    p.loc[0, "class_selection_score"] = 1
    manual = pd.DataFrame(
        dict(
            stream=["RP_Pos"],
            lipid_class=["PC"],
            project=["A"],
            internal_standard=[False],
        )
    )
    table = class_curves(
        p.iloc[[0]], manual, {"RP_Pos": 1}, training_groups={"A"}, excluded_groups={"T"}
    )
    assert table.threshold.eq(1).all() and table.correct.eq(0).all()
    assert mode_counts(p, 1) == dict(correct=0, extras=0, different_peak=0)
    with pytest.raises(ValueError):
        select_candidates(p, scores, baseline_threshold=np.nan)
