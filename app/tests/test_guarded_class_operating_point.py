import numpy as np
import pandas as pd
import pytest

from matlab_annotation.operating_point import select_operating_candidates
from matlab_annotation.guarded_class_operating_point import (
    clear_identity_winners,
    select_class_operating_candidates,
    class_training_curves,
    choose_class_training_points,
)

SCOPE = dict(training_groups={"g0", "g1", "g2"}, excluded_groups={"test"})


def pool():
    return pd.DataFrame(
        dict(
            stream=["RP_Pos"] * 6,
            **{"class": ["PE", "PC", "PE", "PE", "PE", "PC"]},
            name=[
                "PE(34:1)",
                "PC(32:1)",
                "PE(36:2)",
                "PE(36:2)",
                "PE(38:3)",
                "PC(34:2)",
            ],
            adduct=["[M+H]+"] * 6,
            feat_feature_id=["f1", "f1", "f2", "f3", "f4", "f5"],
            feat_intensity=[100, 100, 80, 90, 20, 60],
        )
    )


def test_uniform_fallback_exactly_preserves_existing_geometry_for_every_grid_point():
    data = pool()
    scores = pd.Series([0.9, 0.7, 0.8, 0.85, 0.03, 0.6])
    for threshold in np.arange(0.05, 1.001, 0.05):
        a = select_operating_candidates(data, scores, threshold=float(threshold))
        b = select_class_operating_candidates(
            data, scores, baseline_threshold=float(threshold), class_thresholds={}
        )
        assert a.index.tolist() == b.index.tolist()
    assert clear_identity_winners(data, scores).index.tolist() == [0, 3, 5, 4]


def test_class_cutoff_can_calibrate_unique_low_score_but_not_promote_losing_labels_or_peaks():
    data = pool()
    scores = pd.Series([0.9, 0.7, 0.8, 0.85, 0.03, 0.6])
    result = select_class_operating_candidates(
        data, scores, baseline_threshold=0.4, class_thresholds={"PE": 0.02, "PC": 0.8}
    )
    assert result.index.tolist() == [0, 3, 4]
    assert not result.feat_feature_id.duplicated().any()
    assert not result.duplicated(["name", "adduct"]).any()
    result = select_class_operating_candidates(
        data, scores, baseline_threshold=0.4, class_thresholds={"PE": 0.95, "PC": 0.1}
    )
    assert result.index.tolist() == [5]


def test_ambiguity_is_not_disabled_by_a_low_class_cutoff():
    data = pool()
    scores = pd.Series([0.9, 0.88, 0.8, 0.85, 0.03, 0.6])
    result = select_class_operating_candidates(
        data, scores, baseline_threshold=0.4, class_thresholds={"PE": 0.0, "PC": 0.0}
    )
    assert "f1" not in set(result.feat_feature_id)


def test_mixed_streams_inconsistent_identity_classes_or_invalid_scores_are_rejected():
    data = pool()
    scores = pd.Series([0.9, 0.7, 0.8, 0.85, 0.03, 0.6])
    bad = data.copy()
    bad.loc[0, "stream"] = "NP_Pos"
    with pytest.raises(ValueError, match="one stream"):
        clear_identity_winners(bad, scores)
    bad = data.copy()
    bad.loc[3, "class"] = "PC"
    with pytest.raises(ValueError, match="consistent class"):
        clear_identity_winners(bad, scores)
    for value in [np.nan, -0.1, 1.1]:
        bad = scores.copy()
        bad.iloc[0] = value
        with pytest.raises(ValueError, match="Finite agreement"):
            clear_identity_winners(data, bad)
    with pytest.raises(ValueError, match="Conflicting"):
        select_class_operating_candidates(
            data,
            scores,
            baseline_threshold=0.4,
            class_thresholds={"PE-O": 0.2, "PE[O]": 0.3},
        )


def curves_input():
    manual = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                lipid_class="PE",
                project=f"g{i % 3}",
                internal_standard=False,
            )
            for i in range(120)
        ]
    )
    winners = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                lipid_class="PE",
                project=f"g{i % 3}",
                class_selection_score=0.9 if i < 95 else 0.03,
                training_positive=True,
                training_label_known=True,
                calibration_identity_in_manual=True,
            )
            for i in range(110)
        ]
    )
    return winners, manual


def test_curves_use_full_manual_denominator_positive_quantiles_and_training_support():
    w, m = curves_input()
    t = class_training_curves(w, m, {"RP_Pos": 0.4}, **SCOPE)
    assert t.manual.eq(120).all() and t.eligible.all()
    assert t[t.baseline].correct.iloc[0] == 95
    assert 0.03 in t.threshold.to_list()
    assert t[t.threshold.eq(0.03)].correct.iloc[0] == 110
    m = m.assign(project="g0")
    w = w.assign(project="g0")
    t = class_training_curves(w, m, {"RP_Pos": 0.4}, **SCOPE)
    assert len(t) == 1 and not t.eligible.any() and t.threshold.iloc[0] == 0.4


def test_sparse_unsupported_and_standard_rows_cannot_get_new_cutoffs():
    w, m = curves_input()
    t = class_training_curves(w.iloc[:5], m, {"RP_Pos": 0.4}, **SCOPE)
    assert len(t) == 1 and not t.eligible.any()
    t = class_training_curves(w.iloc[:99], m.iloc[:99], {"RP_Pos": 0.4}, **SCOPE)
    assert len(t) == 1 and not t.eligible.any()
    with pytest.raises(ValueError, match="Standards"):
        class_training_curves(
            w, m.assign(internal_standard=True), {"RP_Pos": 0.4}, **SCOPE
        )
    with pytest.raises(ValueError, match="Excluded"):
        class_training_curves(w.assign(project="test"), m, {"RP_Pos": 0.4}, **SCOPE)


def table():
    records = []
    for stream, cls in [("RP_Pos", "PE"), ("RP_Pos", "PC"), ("NP_Neg", "BMP")]:
        records += [
            dict(
                stream=stream,
                lipid_class=cls,
                manual=100,
                threshold=0.4,
                correct=80,
                extras=3,
                different_peak=1,
                eligible=True,
                baseline=True,
            ),
            dict(
                stream=stream,
                lipid_class=cls,
                manual=100,
                threshold=0.2,
                correct=90,
                extras=7,
                different_peak=1,
                eligible=True,
                baseline=False,
            ),
            dict(
                stream=stream,
                lipid_class=cls,
                manual=100,
                threshold=0.1,
                correct=95,
                extras=12,
                different_peak=2,
                eligible=True,
                baseline=False,
            ),
        ]
    records += [
        dict(
            stream="NP_Neg",
            lipid_class="rare",
            manual=5,
            threshold=0.4,
            correct=2,
            extras=1,
            different_peak=0,
            eligible=False,
            baseline=True,
        )
    ]
    return pd.DataFrame(records)


def test_solver_balances_classes_without_losing_training_corrects_or_adding_wrong_peaks():
    t = table()
    result = choose_class_training_points(t, **SCOPE)
    selected = t.loc[result["selected_point_indices"]]
    assert not result["solver_fallback"]
    assert selected.correct.sum() == 272
    assert selected.extras.sum() == 22 <= result["training_extras_budget"]
    assert selected.different_peak.sum() == 3
    assert "rare" not in result["thresholds"]["NP_Neg"]
    assert all(
        v == 0.2 for classes in result["thresholds"].values() for v in classes.values()
    )


def test_solver_cannot_sacrifice_a_common_class_to_improve_another():
    t = table()
    bad = dict(
        stream="RP_Pos",
        lipid_class="PC",
        manual=100,
        threshold=0.9,
        correct=40,
        extras=0,
        different_peak=0,
        eligible=True,
        baseline=False,
    )
    t = pd.concat([t, pd.DataFrame([bad])], ignore_index=True)
    result = choose_class_training_points(t, **SCOPE)
    assert 0.9 not in result["thresholds"]["RP_Pos"].values()


def test_solver_retains_whole_baseline_on_unproven_solver_result(monkeypatch):
    from types import SimpleNamespace
    import scipy.optimize

    monkeypatch.setattr(
        scipy.optimize,
        "milp",
        lambda *a, **k: SimpleNamespace(status=1, success=False, x=None),
    )
    t = table()
    r = choose_class_training_points(t, **SCOPE)
    assert r["solver_fallback"]
    assert t.loc[r["selected_point_indices"], "baseline"].all()


def test_invalid_or_infeasible_training_counts_are_not_silently_relaxed():
    t = table()
    with pytest.raises(ValueError, match="not feasible"):
        choose_class_training_points(t, extras_per_manual=0.001, **SCOPE)
    bad = t.copy()
    bad.loc[0, "correct"] = 101
    with pytest.raises(ValueError, match="cannot exceed"):
        choose_class_training_points(bad, **SCOPE)
    bad = t.copy()
    bad.loc[1, "manual"] = 99
    with pytest.raises(ValueError, match="denominators"):
        choose_class_training_points(bad, **SCOPE)
    with pytest.raises(ValueError, match="Distinct explicit"):
        choose_class_training_points(
            t, training_groups={"same"}, excluded_groups={"same"}
        )
