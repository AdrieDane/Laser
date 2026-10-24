import numpy as np
import pytest

from matlab_annotation.class_balanced_weights import tempered_class_mode_weights
from matlab_annotation.evidence_training import (
    fit_grouped_evidence_model,
    predict_grouped_evidence,
)
from test_evidence_training import fixture, data


def test_rare_cells_gain_weight_but_cohort_totals_remain_equal():
    groups = ["a"] * 102 + ["b"] * 8
    streams = ["NP_Pos"] * 110
    classes = ["PC"] * 100 + ["BMP"] * 2 + ["PC"] * 4 + ["BMP"] * 4
    w = tempered_class_mode_weights(groups, streams, classes)
    assert w[:102].sum() == pytest.approx(55)
    assert w[102:].sum() == pytest.approx(55)
    assert w[100] > w[0]
    assert w[0] == w[99] and w[100] == w[101]
    assert w[102] == w[-1]


def test_relative_factors_capped_and_permutation_invariant():
    groups = np.array(["a"] * 1002)
    streams = np.array(["RP_Pos"] * len(groups))
    classes = np.array(["PC"] * 1000 + ["BMP", "CL"])
    w = tempered_class_mode_weights(groups, streams, classes)
    assert w.max() / w.min() <= 16 + 1e-10
    order = np.arange(len(groups))[::-1]
    np.testing.assert_allclose(
        w[order],
        tempered_class_mode_weights(groups[order], streams[order], classes[order]),
    )


@pytest.mark.parametrize(
    "vectors",
    [([], [], []), (["a"], ["NP_Pos", "NP_Neg"], ["BMP"]), (["a"], [""], ["BMP"])],
)
def test_bad_metadata_rejected(vectors):
    with pytest.raises(ValueError):
        tempered_class_mode_weights(*vectors)


def test_weight_policy_is_opt_in_and_does_not_change_training_labels():
    records, groups, parameters = fixture()
    for record in records.values():
        record["evidence"]["stream"] = "NP_Pos"
    old = fit_grouped_evidence_model(
        records, training_groups=groups, excluded_groups={"z"}, parameters=parameters
    )
    explicit = fit_grouped_evidence_model(
        records,
        training_groups=groups,
        excluded_groups={"z"},
        parameters=parameters,
        weight_policy=None,
    )
    balanced = fit_grouped_evidence_model(
        records,
        training_groups=groups,
        excluded_groups={"z"},
        parameters=parameters,
        weight_policy="tempered_class_mode",
    )
    assert old["training_pairs"] == balanced["training_pairs"] == 12
    assert old["positives"] == balanced["positives"] == 6
    assert "sample_weight_policy" not in old
    assert balanced["sample_weight_policy"]["kind"] == "tempered_class_mode"
    query = data("z", sorted(groups), ["z"])
    query["evidence"]["stream"] = "NP_Pos"
    np.testing.assert_array_equal(
        predict_grouped_evidence(query, old), predict_grouped_evidence(query, explicit)
    )
    with pytest.raises(ValueError, match="Unknown"):
        fit_grouped_evidence_model(
            records,
            training_groups=groups,
            excluded_groups={"z"},
            parameters=parameters,
            weight_policy="invalid",
        )
