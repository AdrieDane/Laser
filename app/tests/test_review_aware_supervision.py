import copy

import numpy as np
import pandas as pd
import pytest

from matlab_annotation.review_aware_supervision import (
    uncertain_review_negatives,
    review_tempered_weights,
)
from matlab_annotation.evidence_training import (
    fit_grouped_evidence_model,
    predict_grouped_evidence,
)
from test_evidence_training import fixture, data


def flags(pool, manual, **overrides):
    kwargs = dict(
        query_group="a",
        training_groups={"a", "b"},
        excluded_groups={"z"},
        normalize_species=str,
        normalize_adduct=str,
    )
    kwargs.update(overrides)
    return uncertain_review_negatives(pool, manual, **kwargs)


def examples():
    manual = pd.DataFrame(
        [
            ("NP_Pos", "BMP", "BMP(36:2)", "H", "10", True, False),
            ("NP_Neg", "PC", "PC(36:2)", "H", "20", True, False),
            ("RP_Pos", "PE-P", "PE-P(36:2)", "H", "30", True, False),
            ("NP_Neg", "BMP", "labelled_BMP", "H", "40", True, True),
        ],
        columns=[
            "stream",
            "lipid_class",
            "species",
            "adduct",
            "feature_id",
            "mapped",
            "internal_standard",
        ],
    )
    pool = pd.DataFrame(
        [
            ("NP_Neg", "BMP", "BMP(36:2)", "H", "11", False, True),
            ("NP_Neg", "BMP", "BMP(36:2)", "H", "20", False, True),
            ("NP_Neg", "PC", "PC(36:2)", "H", "21", False, True),
            ("NP_Neg", "PC", "PC(38:2)", "H", "22", False, True),
            ("NP_Neg", "BMP", "BMP(38:2)", "H", "12", True, True),
            ("NP_Neg", "BMP", "BMP(40:2)", "H", "13", False, False),
            ("NP_Neg", "PE-O", "PE-O(36:2)", "H", "14", False, True),
            ("NP_Neg", "PE-P", "PE-P(36:2)", "H", "15", False, True),
        ],
        columns=[
            "stream",
            "class",
            "name",
            "adduct",
            "feat_feature_id",
            "training_positive",
            "training_label_known",
        ],
    )
    pool.index = np.arange(8) * 2
    return pool, manual


def test_only_other_mode_absence_is_tempered_not_labels_coordinates_or_ether_aliases():
    pool, manual = examples()
    old_pool, old_manual = pool.copy(deep=True), manual.copy(deep=True)
    result = flags(pool, manual)
    assert result.tolist() == [True, False, False, False, False, False, False, True]
    assert result.index.equals(pool.index)
    pd.testing.assert_frame_equal(pool, old_pool, check_exact=True)
    pd.testing.assert_frame_equal(manual, old_manual, check_exact=True)


def test_standard_only_class_does_not_establish_endogenous_review():
    pool, manual = examples()
    manual.loc[0, "internal_standard"] = True
    assert not flags(pool, manual)[pool["class"].eq("BMP")].any()


@pytest.mark.parametrize(
    "overrides",
    [
        {"query_group": "z"},
        {"training_groups": {"a", "z"}},
        {"excluded_groups": set()},
    ],
)
def test_query_or_evaluation_curation_cannot_enter_fit(overrides):
    pool, manual = examples()
    with pytest.raises(ValueError, match="authorized"):
        flags(pool, manual, **overrides)


def test_weight_strength_project_totals_and_permutation():
    g = np.array(["a"] * 4 + ["b"] * 2)
    y = np.array([1, 0, 0, 0, 1, 0])
    uncertain = np.array([0, 0, 1, 1, 0, 1])
    w = review_tempered_weights(g, y, uncertain)
    assert w[:4].sum() == pytest.approx(3) and w[4:].sum() == pytest.approx(3)
    assert w[2] / w[0] == pytest.approx(0.25) and w[0] == w[1]
    order = [4, 3, 2, 5, 0, 1]
    np.testing.assert_allclose(
        w[order], review_tempered_weights(g[order], y[order], uncertain[order])
    )
    with pytest.raises(ValueError, match="Positive"):
        review_tempered_weights(g, y, np.ones(6, bool))
    with pytest.raises(ValueError, match="Aligned"):
        review_tempered_weights(g, y, [False])


def test_fit_is_opt_in_preserves_all_labels_and_never_predicts_review_flags():
    records, groups, parameters = fixture()
    old = fit_grouped_evidence_model(
        records, training_groups=groups, excluded_groups={"z"}, parameters=parameters
    )
    for r in records.values():
        r["pool"]["training_review_uncertain"] = [False, False, True, False]
    unchanged = fit_grouped_evidence_model(
        records, training_groups=groups, excluded_groups={"z"}, parameters=parameters
    )
    model = fit_grouped_evidence_model(
        records,
        training_groups=groups,
        excluded_groups={"z"},
        parameters=parameters,
        weight_policy="review_aware_negative",
    )
    assert model["training_pairs"] == old["training_pairs"] == 12
    assert model["positives"] == old["positives"] == 6
    assert model["sample_weight_policy"]["uncertain_negative_rows"] == 3
    assert model["sample_weight_policy"]["cohort_total_max_abs_error"] < 1e-12
    assert model["columns"] == old["columns"]
    query = data("z", sorted(groups), ["z"])
    np.testing.assert_array_equal(
        predict_grouped_evidence(query, old), predict_grouped_evidence(query, unchanged)
    )
    query["evidence"]["training_review_uncertain"] = False
    with pytest.raises(ValueError, match="Supervised"):
        predict_grouped_evidence(query, model)
    bad = copy.deepcopy(records)
    del bad["a"]["pool"]["training_review_uncertain"]
    with pytest.raises(ValueError, match="review-uncertain"):
        fit_grouped_evidence_model(
            bad,
            training_groups=groups,
            excluded_groups={"z"},
            parameters=parameters,
            weight_policy="review_aware_negative",
        )
