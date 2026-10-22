from io import StringIO

import pandas as pd
import pytest

from matlab_annotation.conditional_peak_choice import (
    conditional_training_weights,
    rerank_accepted_peaks,
)


def fixture():
    before = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                name="PE(36:2)",
                adduct="H",
                is_is=False,
                feat_feature_id="old",
                **{"class": "PE"},
                feat_mz=700.0,
                feat_rt=5.0,
                class_selection_score=0.8,
                automatic_accepted=True,
            )
        ]
    )
    pool = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                name="PE(36:2)",
                adduct="H",
                is_is=False,
                feat_feature_id=fid,
                **{"class": "PE"},
                feat_mz=700.001,
                feat_rt=5.01,
                training_positive=truth,
                training_label_known=True,
            )
            for fid, truth in [("old", False), ("new", True)]
        ]
    )
    return before, pool, pd.Series([0.3, 0.95])


def test_replaces_only_peak_with_no_new_identity_or_training_truth_in_output():
    before, pool, scores = fixture()
    result, audit = rerank_accepted_peaks(before, pool, scores)
    assert result.iloc[0].feat_feature_id == "new"
    assert result.iloc[0]["name"] == before.iloc[0]["name"]
    assert len(result) == len(before)
    assert not any(c.startswith("training_") for c in result)
    assert result.class_selection_score.isna().all()
    assert len(audit) == 1
    assert before.iloc[0].feat_feature_id == "old"


def test_csv_empty_text_column_can_receive_original_candidate_text_without_type_error():
    before, pool, scores = fixture()
    before["isotopologue"] = ""
    before = pd.read_csv(
        StringIO(before.to_csv(index=False)), dtype={"feat_feature_id": str}
    )
    pool["isotopologue"] = ""
    result, _ = rerank_accepted_peaks(before, pool, scores)
    assert result.iloc[0].feat_feature_id == "new"
    assert result.iloc[0].isotopologue == ""


def test_equal_total_true_and_alternative_weight_per_identity_then_per_cohort():
    _, pool, _ = fixture()
    larger = pd.concat(
        [pool, pool.iloc[[0]].assign(feat_feature_id="third")], ignore_index=True
    )
    a = larger.assign(name="A")
    b = pool.assign(name="B")
    x = pd.concat([a, b], ignore_index=True)
    dataset = pd.Series(["a"] * 3 + ["b"] * 2)
    cohort = pd.Series(["project1"] * 3 + ["project2"] * 2)
    weights = conditional_training_weights(x, dataset, cohort)
    assert weights[:3].sum() == pytest.approx(weights[3:].sum())
    assert weights[x.training_positive].sum() == pytest.approx(
        weights[~x.training_positive].sum()
    )


def test_no_resolved_true_peak_is_not_conditional_training_material():
    _, pool, _ = fixture()
    pool["training_positive"] = False
    assert (
        conditional_training_weights(pool, pd.Series(["a"] * 2), pd.Series(["p"] * 2))
        .eq(0)
        .all()
    )


def test_occupied_or_forbidden_feature_and_protected_manual_row_are_unchanged():
    before, pool, scores = fixture()
    occupied = pd.concat(
        [before, before.assign(name="PC(33:2)", feat_feature_id="new")],
        ignore_index=True,
    )
    result, _ = rerank_accepted_peaks(occupied, pool, scores)
    assert result.feat_feature_id.tolist() == ["old", "new"]
    result, _ = rerank_accepted_peaks(
        before, pool, scores, forbidden_features={"RP_Pos": {"new"}}
    )
    assert result.feat_feature_id.tolist() == ["old"]
    result, _ = rerank_accepted_peaks(
        before.assign(manual_confirmed=True), pool, scores
    )
    assert result.feat_feature_id.tolist() == ["old"]


def test_competing_peak_scores_need_a_clear_margin():
    before, pool, scores = fixture()
    pool = pd.concat(
        [pool, pool.iloc[[1]].assign(feat_feature_id="third")], ignore_index=True
    )
    result, _ = rerank_accepted_peaks(before, pool, pd.Series([0.3, 0.95, 0.94]))
    assert result.feat_feature_id.tolist() == ["old"]


def test_conflicting_free_feature_claims_are_all_withheld():
    before, pool, scores = fixture()
    second = before.assign(name="PC(33:2)", feat_feature_id="old2")
    two = pd.concat([before, second], ignore_index=True)
    candidates = pd.concat(
        [pool, pool.assign(name="PC(33:2)", feat_feature_id=["old2", "new"])],
        ignore_index=True,
    )
    result, audit = rerank_accepted_peaks(
        two, candidates, pd.Series([0.3, 0.95, 0.3, 0.96])
    )
    assert result.feat_feature_id.tolist() == ["old", "old2"]
    assert audit.action.eq("withhold_conflicting_peak_proposals").all()


def test_inherited_duplicate_standard_labels_stay_untouched_with_their_original_peaks():
    before, pool, scores = fixture()
    standards = pd.concat(
        [
            before.assign(name="IS", is_is=True, feat_feature_id="s1"),
            before.assign(name="IS", is_is=True, feat_feature_id="s2"),
        ],
        ignore_index=True,
    )
    combined = pd.concat([before, standards], ignore_index=True)
    result, _ = rerank_accepted_peaks(combined, pool, scores)
    assert result.feat_feature_id.tolist() == ["new", "s1", "s2"]
    pd.testing.assert_frame_equal(
        result.iloc[1:][standards.columns].reset_index(drop=True),
        standards,
        check_dtype=False,
    )


def test_duplicate_unprotected_identity_is_not_silently_reranked():
    before, pool, scores = fixture()
    repeated = pd.concat(
        [before, before.assign(feat_feature_id="second")], ignore_index=True
    )
    with pytest.raises(ValueError):
        rerank_accepted_peaks(repeated, pool, scores)
