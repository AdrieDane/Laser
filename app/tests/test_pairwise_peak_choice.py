import numpy as np
import pandas as pd
import pytest

from matlab_annotation.pairwise_peak_choice import (
    pair_evidence,
    pair_training_data,
    pairwise_peak_scores,
)


def fixture_data():
    pool = pd.DataFrame(
        dict(
            stream=["RP_Pos"] * 3,
            name=["PE(36:2)"] * 3,
            adduct=["[M+H]+"] * 3,
            feat_feature_id=["a", "b", "c"],
            training_positive=[False, True, False],
            training_label_known=[True] * 3,
        )
    )
    evidence = pd.DataFrame(
        dict(
            stream=["RP_Pos"] * 3,
            **{"class": ["PE"] * 3},
            absolute_mass_error_ppm=[2.0, 0.1, 1.5],
            q_previously_selected=[1.0, 0.0, 0.0],
            q_fitted_post_regression_confidence=[0.99, 0.0, 0.0],
            carbon=[36] * 3,
        )
    )
    return pool, evidence


def test_pair_descriptors_remove_occurrence_and_old_choice_shortcuts():
    _, e = fixture_data()
    x = pair_evidence(e.iloc[[0]], e.iloc[[1]])
    assert x.absolute_mass_error_ppm__difference.iloc[0] == pytest.approx(1.9)
    assert x.absolute_mass_error_ppm__mean.iloc[0] == pytest.approx(1.05)
    assert not any("previous" in c or "confidence" in c or "carbon" in c for c in x)
    assert "feat_feature_id" not in x


def test_pair_training_is_symmetric_identity_balanced_and_label_free_inputs():
    p, e = fixture_data()
    x, y, w, metadata, positives = pair_training_data(
        p, e, pd.Series(["one"] * 3), pd.Series(["group"] * 3)
    )
    assert y.tolist() == [1, 1, 0, 0]
    assert np.allclose(w, w[0])
    assert set(metadata.cohort) == {"group"} and len(positives) == 1
    assert not any(c.startswith("training_") for c in x)
    assert np.allclose(
        x.absolute_mass_error_ppm__difference[:2],
        -x.absolute_mass_error_ppm__difference[2:],
    )


def test_pair_winner_beats_every_alternative_and_input_order_is_irrelevant():
    p, e = fixture_data()

    def predict(x):
        return 1 / (1 + np.exp(3 * x.absolute_mass_error_ppm__difference.to_numpy()))

    scores = pairwise_peak_scores(p, e, predict)
    assert scores.loc[1] > 0.98
    assert scores.loc[0] < 0.01 and scores.loc[2] < 0.02
    pd.testing.assert_series_equal(
        scores, pairwise_peak_scores(p.iloc[::-1], e.iloc[::-1], predict).sort_index()
    )


def test_missing_and_categorical_identity_guards():
    _, e = fixture_data()
    bad = e.iloc[[1]].copy()
    bad["class"] = "PC"
    with pytest.raises(ValueError):
        pair_evidence(e.iloc[[0]], bad)
    p, e = fixture_data()
    p.loc[2, "feat_feature_id"] = "a"
    with pytest.raises(ValueError):
        pairwise_peak_scores(p, e, lambda x: np.full(len(x), 0.5))


def test_pairwise_cycles_and_constant_classifier_cannot_force_a_winner():
    p, e = fixture_data()
    scores = pairwise_peak_scores(p, e, lambda x: np.full(len(x), 0.9))
    assert np.allclose(scores, 0.5)  # forward/reverse bias cancels


def test_unknown_labels_and_multi_true_identities_do_not_train():
    p, e = fixture_data()
    p.loc[0, "training_positive"] = True
    with pytest.raises(ValueError, match="No conditional"):
        pair_training_data(p, e, pd.Series(["one"] * 3), pd.Series(["group"] * 3))
