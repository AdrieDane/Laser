import copy

import numpy as np
import pandas as pd
import pytest

from matlab_annotation.feature_label_model import (
    fit_feature_label_model,
    predict_feature_label_model,
)


def data(query, references, excluded, *, supervised=True):
    rows = []
    for i in range(40):
        for j, cls in enumerate(("PC", "PE")):
            positive = i < 30 and j == i % 2
            rows.append(
                dict(
                    stream="RP_Pos",
                    feat_feature_id=f"f{i}",
                    name=f"{cls}({30 + i}:1)",
                    **{"class": cls},
                    adduct="[M+H]+",
                    is_is=False,
                    training_positive=positive,
                    training_label_known=True,
                    error=1.0 if positive else 8.0,
                    quality=1.0 if i < 30 else 0.0,
                )
            )
    frame = pd.DataFrame(rows)
    pool = frame.drop(columns=["error", "quality"])
    evidence = frame[["stream", "class", "error", "quality"]].copy()
    if not supervised:
        pool = pool.drop(columns=["training_positive", "training_label_known"])
    return dict(
        query_group=query,
        reference_training_groups=references,
        reference_excluded_groups=excluded,
        pool=pool,
        evidence=evidence,
    )


def fitted():
    groups = {"a", "b", "c"}
    records = {g: data(g, sorted(groups - {g}), [g, "z"]) for g in sorted(groups)}
    model = fit_feature_label_model(
        records,
        training_groups=groups,
        excluded_groups={"z"},
        parameters=dict(
            max_iter=8,
            max_leaf_nodes=3,
            min_samples_leaf=2,
            early_stopping=False,
            random_state=1,
        ),
    )
    return model, data("z", sorted(groups), ["z"], supervised=False)


def test_distinct_feature_and_conditional_targets_fit_two_models_not_one_candidate_score():
    model, query = fitted()
    assert model["fresh_classifier_fits"] == 2
    expert = model["experts"]["RP_Pos"]
    assert expert["detector"]["observations"] == 120
    assert expert["detector"]["positives"] == 90
    assert expert["conditional"]["observations"] == 180
    assert expert["conditional"]["positives"] == 90
    assert expert["singleton_supported_classes"] == ["PC", "PE"]
    assert expert["ambiguous_supported_classes"] == ["PC", "PE"]
    before = query["pool"].copy(deep=True)
    scores = predict_feature_label_model(query, model)
    assert scores.supported_cluster.all() and scores.index.equals(query["pool"].index)
    assert (
        scores.feature_detection_rank.iloc[0] > scores.feature_detection_rank.iloc[-1]
    )
    assert scores.conditional_label_rank.iloc[0] > scores.conditional_label_rank.iloc[1]
    for indices in query["pool"].groupby("feat_feature_id").groups.values():
        group = scores.loc[indices]
        assert np.isclose(group.conditional_label_rank.sum(), 1)
        assert np.isclose(
            group.joint_agreement_rank.sum(), group.feature_detection_rank.iloc[0]
        )
    pd.testing.assert_frame_equal(query["pool"], before, check_exact=True)


def test_unsupported_competing_class_withholds_whole_cluster_not_losing_label_promotion():
    model, query = fitted()
    query["pool"].loc[0, "class"] = "unsupported"
    result = predict_feature_label_model(query, model)
    assert result.loc[[0, 1], "joint_agreement_rank"].eq(0).all()
    assert not result.loc[[0, 1], "supported_cluster"].any()


def test_singleton_supported_labels_have_conditional_one_without_old_score_cap():
    model, query = fitted()
    query["pool"] = query["pool"].iloc[::2].copy()
    query["evidence"] = query["evidence"].iloc[::2].copy()
    result = predict_feature_label_model(query, model)
    assert result.supported_cluster.all() and result.conditional_label_rank.eq(1).all()
    pd.testing.assert_series_equal(
        result.joint_agreement_rank, result.feature_detection_rank, check_names=False
    )


def test_no_query_supervision_or_stage_training_scope_substitution():
    model, query = fitted()
    bad = copy.deepcopy(query)
    bad["pool"]["calibration_identity_in_manual"] = False
    with pytest.raises(ValueError, match="refuses supervision"):
        predict_feature_label_model(bad, model)
    bad = copy.deepcopy(model)
    bad["experts"]["RP_Pos"]["detector"]["actual_projects"] = ["z"]
    with pytest.raises(ValueError, match="contributors"):
        predict_feature_label_model(query, bad)
    query["reference_training_groups"].append("z")
    with pytest.raises(ValueError, match="Reference"):
        predict_feature_label_model(query, model)
