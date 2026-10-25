import numpy as np
import pandas as pd
import pytest

from matlab_annotation.feature_label_examples import (
    feature_evidence,
    feature_training_targets,
)


def example():
    pool = pd.DataFrame(
        dict(
            stream="RP_Pos",
            feat_feature_id=["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
            name=["PC(34:1)", "PE(36:4)"] * 4 + ["PC(32:1)"] * 2,
            adduct="[M+H]+",
            **{"class": ["PC", "PE"] * 4 + ["PC"] * 2},
            is_is=[False] * 6 + [True, False, False, False],
            training_positive=[
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
            ],
            training_label_known=[True] * 4 + [False, True] + [True] * 4,
        )
    )
    evidence = pd.DataFrame(
        dict(
            stream="RP_Pos",
            **{"class": pool["class"]},
            absolute_RT_error_seconds=np.arange(10, dtype=float),
            raw_quality=[1.0, 1.0, np.nan, np.nan, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
        )
    )
    return pool, evidence


def test_detection_targets_count_original_features_once_and_unknown_is_not_negative():
    pool, _ = example()
    targets, conditional = feature_training_targets(pool)
    assert len(targets) == 5
    assert targets.positive.tolist() == [True, False, False, False, True]
    assert targets.known.tolist() == [True, True, False, False, True]
    assert conditional.tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert targets.loc[("RP_Pos", "e"), "distinct_labels"] == 1


def test_aggregation_is_supervision_blind_preserves_missing_and_never_returns_ids_as_inputs():
    pool, evidence = example()
    result = feature_evidence(pool, evidence)
    assert result.loc[("RP_Pos", "a"), "feature_min__absolute_RT_error_seconds"] == 0
    assert result.loc[("RP_Pos", "a"), "feature_max__absolute_RT_error_seconds"] == 1
    assert np.isnan(result.loc[("RP_Pos", "b"), "feature_median__raw_quality"])
    assert result.loc[("RP_Pos", "e"), "feature_distinct_label_hypotheses"] == 1
    assert not {"name", "class", "feat_feature_id", "feature_id", "stream"} & set(
        result
    )
    modified = pool.assign(training_positive=False, training_label_known=False)
    pd.testing.assert_frame_equal(
        result, feature_evidence(modified, evidence), check_exact=True
    )


@pytest.mark.parametrize(
    "column",
    [
        "training_positive",
        "calibration_identity_in_manual",
        "manual_confirmed",
        "dataset",
        "name",
    ],
)
def test_supervised_or_identity_predictors_cannot_enter_feature_detector(column):
    pool, evidence = example()
    evidence[column] = 1
    with pytest.raises(ValueError, match="label-free"):
        feature_evidence(pool, evidence)


def test_contradictory_versions_and_multiple_positive_labels_are_hard_errors():
    pool, _ = example()
    bad = pool.copy()
    bad.loc[9, "training_positive"] = False
    with pytest.raises(ValueError, match="versions"):
        feature_training_targets(bad)
    bad = pool.copy()
    bad.loc[1, "training_positive"] = True
    with pytest.raises(ValueError, match="Multiple"):
        feature_training_targets(bad)
    bad = pool.copy()
    bad.loc[0, "training_label_known"] = False
    with pytest.raises(ValueError, match="known"):
        feature_training_targets(bad)


def test_unknown_alternative_to_a_positive_feature_is_not_used_as_conditional_negative():
    pool, _ = example()
    pool.loc[1, "training_label_known"] = False
    targets, conditional = feature_training_targets(pool)
    assert targets.loc[("RP_Pos", "a"), "known"]
    assert conditional.iloc[0] and not conditional.iloc[1]
