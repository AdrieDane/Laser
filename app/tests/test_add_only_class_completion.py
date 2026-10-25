import numpy as np
import pandas as pd
import pytest

from test_np_context_bmp_completion import candidates
from matlab_annotation.add_only_class_completion import (
    free_completion_winners,
    select_completion,
    class_completion_curves,
    choose_completion_thresholds,
)

SCOPE = dict(training_groups={"a", "b", "c"}, excluded_groups={"heldout"})


def test_free_winners_preserve_old_calls_and_original_physical_competition():
    p, scores, old = candidates()
    before = p.copy(deep=True)
    kwargs = dict(stream="NP_Pos", protected_features=set(), declared_masses=[])
    free = free_completion_winners(p, scores, old, **kwargs)
    assert len(free) == 2
    occupied = p.iloc[[0]]
    free = free_completion_winners(p, scores, occupied, **kwargs)
    assert free.feat_feature_id.tolist() == ["other"]
    kwargs["protected_features"] = {"other"}
    assert free_completion_winners(p, scores, occupied, **kwargs).empty
    pd.testing.assert_frame_equal(p, before)


def test_no_weaker_fallback_after_strongest_peak_is_ambiguous_or_occupied():
    p, scores, old = candidates()
    extra = p.iloc[[0]].assign(feat_feature_id="weak", feat_intensity=1)
    rival = p.iloc[[0]].assign(name="PG(36:2)", **{"class": "PG"})
    p = pd.concat([p, extra, rival], ignore_index=True)
    scores = pd.Series([0.95, 0.7, 0.9, 0.8, 0.94])
    result = free_completion_winners(
        p, scores, old, stream="NP_Pos", protected_features=set(), declared_masses=[]
    )
    assert result.feat_feature_id.tolist() == ["other"]


@pytest.mark.parametrize(
    "change",
    [
        {"feat_mz": 900.0},
        {"feat_rt": 15.0},
        {"isotopologue": "M+1"},
        {"adduct": "[M+unsupported]+"},
        {"is_is": True},
    ],
)
def test_physical_and_standard_guards_cannot_be_overridden_by_score(change):
    p, scores, old = candidates()
    for c, value in change.items():
        p[c] = value
    assert free_completion_winners(
        p, scores, old, stream="NP_Pos", protected_features=set(), declared_masses=[]
    ).empty


def calibration_input():
    manual = pd.DataFrame(
        dict(
            stream="RP_Pos",
            lipid_class="PE",
            project=["a", "b", "c"] * 40,
            internal_standard=False,
        )
    )
    proposals = pd.DataFrame(
        dict(
            stream="RP_Pos",
            lipid_class="PE",
            project=["a", "b", "c"] * 8,
            training_positive=True,
            training_label_known=True,
            calibration_identity_in_manual=True,
            class_selection_score=np.linspace(0.04, 0.02, 24),
        )
    )
    return proposals, manual, {("RP_Pos", "PE"): 80}


def test_training_support_and_strict_marginal_guards_choose_low_unambiguous_scores():
    p, m, baseline = calibration_input()
    table = class_completion_curves(p, m, baseline, **SCOPE)
    cuts, chosen = choose_completion_thresholds(table)
    assert chosen.added_correct.sum() == 24 and cuts["RP_Pos"]["PE"] == 0.02
    assert len(select_completion(p, cuts)) == 24
    assert select_completion(p.assign(class_selection_score=1.0), {}).empty
    # Existing high-coverage classes and sparse references cannot be expanded.
    for ref, counts in (
        (m, {("RP_Pos", "PE"): 110}),
        (m.iloc[:99], {("RP_Pos", "PE"): 70}),
    ):
        t = class_completion_curves(p, ref, counts, **SCOPE)
        c, selected = choose_completion_thresholds(t)
        assert selected.threshold.eq(1).all() and select_completion(p, c).empty


def test_new_wrong_peak_or_dominant_single_project_prevents_completion():
    p, m, baseline = calibration_input()
    # One wrong peak at the highest score makes every non-disabled level fail.
    wrong = p.iloc[[0]].assign(training_positive=False, class_selection_score=0.99)
    t = class_completion_curves(pd.concat([p, wrong]), m, baseline, **SCOPE)
    assert not t.loc[t.threshold.lt(1), "feasible"].any()
    p["project"] = "a"
    t = class_completion_curves(p, m, baseline, **SCOPE)
    assert not t.loc[t.threshold.lt(1), "feasible"].any()


def test_authorization_and_unknown_labels_are_not_silently_ignored():
    p, m, baseline = calibration_input()
    with pytest.raises(ValueError, match="Excluded"):
        class_completion_curves(p.assign(project="heldout"), m, baseline, **SCOPE)
    with pytest.raises(ValueError, match="known"):
        class_completion_curves(
            p.assign(training_label_known=False), m, baseline, **SCOPE
        )
    unknown = p.iloc[[0]].assign(
        training_positive=False, training_label_known=False, class_selection_score=0.99
    )
    t = class_completion_curves(pd.concat([p, unknown]), m, baseline, **SCOPE)
    assert (
        t.loc[t.threshold.lt(1), "added_calls"]
        .gt(t.loc[t.threshold.lt(1), "added_correct"])
        .all()
    )
