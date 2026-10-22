import pandas as pd
import pytest

from matlab_annotation.class_operating_point import (
    choose_class_training_points,
    select_class_operating_candidates,
)


def fixture():
    rows = []
    for cls, points in {
        "PC": [(0.5, 98, 10, True), (0.8, 90, 2, False)],
        "PE": [(0.5, 80, 10, True), (0.2, 90, 18, False)],
    }.items():
        for threshold, correct, extras, reference in points:
            rows.append(
                dict(
                    stream="RP_Pos",
                    lipid_class=cls,
                    threshold=threshold,
                    manual=100,
                    correct=correct,
                    extras=extras,
                    wrong_peak=1,
                    training_cohorts=8,
                    reference=reference,
                )
            )
    return pd.DataFrame(rows)


def choose(table, **kwargs):
    return choose_class_training_points(
        table, training_groups={"train"}, excluded_groups={"test"}, **kwargs
    )


def test_class_target_allocation_protects_total_and_stays_inside_budget():
    result = choose(fixture(), extras_per_manual=0.1)
    assert result["thresholds"] == {"RP_Pos|PC": 0.8, "RP_Pos|PE": 0.2}
    assert result["correct"] == 180 >= result["reference_correct"]
    assert result["extras"] == 20
    assert result["capped_class_coverage_sum"] == pytest.approx(1.8)


def test_under_supported_classes_keep_reference_no_overfit_to_one_cohort():
    table = fixture()
    table.loc[table.lipid_class.eq("PE"), "training_cohorts"] = 1
    result = choose(table, extras_per_manual=0.1)
    assert result["thresholds"] == {"RP_Pos|PC": 0.5, "RP_Pos|PE": 0.5}


def test_zero_manual_classes_and_infeasible_budget_are_explicit():
    table = fixture()
    zero = table.iloc[[0]].assign(
        lipid_class="Unknown",
        manual=0,
        correct=0,
        extras=1,
        wrong_peak=0,
        training_cohorts=0,
    )
    result = choose(pd.concat([table, zero], ignore_index=True), extras_per_manual=0.11)
    assert result["thresholds"]["RP_Pos|Unknown"] == 0.5
    assert result["manual"] == 200
    assert choose(table, extras_per_manual=0) is None


def test_order_invariance_and_no_mutation():
    table = fixture()
    original = table.copy(deep=True)
    a = choose(table, extras_per_manual=0.1)
    b = choose(table.iloc[::-1], extras_per_manual=0.1)
    assert a["thresholds"] == b["thresholds"]
    pd.testing.assert_frame_equal(table, original)


def test_group_overlap_bad_counts_and_missing_reference_rejected():
    with pytest.raises(ValueError, match="Distinct explicit"):
        choose_class_training_points(
            fixture(), training_groups={"same"}, excluded_groups={"same"}
        )
    with pytest.raises(ValueError, match="fixed denominators"):
        choose(fixture().assign(reference=False))
    with pytest.raises(ValueError, match="integer"):
        choose(fixture().assign(correct=100.5))


def test_mode_coverage_cannot_fall_below_reference_or_target_to_pay_another_mode():
    table = fixture()
    table.loc[table.lipid_class.eq("PE"), "correct"] = 80
    table.loc[table.lipid_class.eq("PE"), "extras"] = 10
    extra = fixture()[fixture().lipid_class.eq("PE")].assign(
        stream="NP_Pos", lipid_class="FA"
    )
    result = choose(pd.concat([table, extra], ignore_index=True), extras_per_manual=0.1)
    assert result["thresholds"]["RP_Pos|PC"] == 0.5
    assert result["correct"] == 258


def test_class_cutoff_cannot_promote_a_losing_label_and_uniform_cutoff_parity():
    from matlab_annotation.operating_point import select_operating_candidates

    pool = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                name="PC(34:1)",
                **{"class": "PC"},
                adduct="[M+H]+",
                feat_feature_id="same",
                feat_intensity=100.0,
            ),
            dict(
                stream="RP_Pos",
                name="PE(36:1)",
                **{"class": "PE"},
                adduct="[M+H]+",
                feat_feature_id="same",
                feat_intensity=90.0,
            ),
        ]
    )
    scores = pd.Series([0.8, 0.6])
    assert select_class_operating_candidates(
        pool, scores, {"RP_Pos|PC": 0.9, "RP_Pos|PE": 0.5}, fallback_threshold=0.5
    ).empty
    actual = select_class_operating_candidates(pool, scores, {}, fallback_threshold=0.5)
    expected = select_operating_candidates(pool, scores, threshold=0.5)
    assert actual.feat_feature_id.tolist() == expected.feat_feature_id.tolist()
    assert actual.name.tolist() == expected.name.tolist()
