"""Synthetic alternate-learner software checks, NOT QE performance experiments."""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
import catboost_evidence_model as m


def data(group, count, *, query=False):
    frames = []
    for mode in m.MODES:
        x = np.linspace(-2, 2, count)
        evidence = pd.DataFrame(
            dict(
                stream=mode,
                **{"class": np.where(x > 0, "PC", "PE")},
                adduct="[M+H]+",
                stream_class=np.where(x > 0, mode + "|PC", mode + "|PE"),
                mass_error=x,
                missing_value=np.where(x > 1, np.nan, x),
            )
        )
        frames.append(evidence)
    evidence = pd.concat(frames, ignore_index=True)
    pool = evidence[["stream"]].copy()
    if not query:
        pool["training_positive"] = evidence.mass_error.gt(0)
        pool["training_label_known"] = True
        pool.loc[0, "training_label_known"] = False
    refs = ["A", "B"] if query else ["B" if group == "A" else "A"]
    return dict(
        pool=pool,
        evidence=evidence,
        query_group=group,
        reference_training_groups=refs,
        reference_excluded_groups=[group, "T"] if not query else ["T"],
    )


@pytest.fixture(scope="module")
def trained():
    records = {"a": data("A", 8), "b": data("B", 6)}
    parameters = {**m.PARAMETERS, "iterations": 16}
    model = m.fit_four_modes(
        records,
        training_groups={"A", "B"},
        excluded_groups={"T"},
        parameters=parameters,
    )
    return records, parameters, model


def test_original_known_row_and_equal_project_weight_contract():
    records = {"a": data("A", 8), "b": data("B", 6)}
    x, y, groups, weights, categorical = m.fitting_matrix(records, {"A", "B"}, {"T"})
    expected_x, expected_y, expected_groups = [], [], []
    for record in records.values():
        for idx, row in record["pool"].iterrows():
            if row.training_label_known:
                expected_x.append(record["evidence"].loc[idx].to_dict())
                expected_y.append(int(row.training_positive))
                expected_groups.append(record["query_group"])
    pd.testing.assert_frame_equal(x, m.matrix(pd.DataFrame(expected_x), categorical))
    assert y.tolist() == expected_y and groups.tolist() == expected_groups
    assert weights[groups == "A"].sum() == pytest.approx(len(y) / 2)
    assert weights[groups == "B"].sum() == pytest.approx(len(y) / 2)
    assert categorical == ["stream", "class", "adduct", "stream_class"]


def test_synthetic_all_modes_repeat_fit_and_pickle_predictions_are_exact(trained):
    records, parameters, model = trained
    again = m.fit_four_modes(
        records,
        training_groups={"A", "B"},
        excluded_groups={"T"},
        parameters=parameters,
    )
    query = data("T", 4, query=True)
    first = m.predict_four_modes(query, model)
    pd.testing.assert_series_equal(
        first, m.predict_four_modes(query, again), check_exact=True
    )
    pd.testing.assert_series_equal(
        first,
        m.predict_four_modes(query, pickle.loads(pickle.dumps(model))),
        check_exact=True,
    )
    assert set(model["experts"]) == set(m.MODES) and first.between(0, 1).all()
    assert all(e["estimator"].tree_count_ == 16 for e in model["experts"].values())


def test_unseen_categories_and_numeric_missing_values_remain_predictable(trained):
    _, _, model = trained
    query = data("T", 4, query=True)
    query["evidence"]["class"] = "unseen_class"
    query["evidence"]["missing_value"] = np.nan
    scores = m.predict_four_modes(query, model)
    assert scores.index.equals(query["pool"].index) and np.isfinite(scores).all()


def test_scope_schema_and_identity_fields_are_rejected(trained):
    records, _, model = trained
    invalid = copy.deepcopy(records)
    invalid["a"]["reference_training_groups"] = ["A", "T"]
    with pytest.raises(ValueError):
        m.fitting_matrix(invalid, {"A", "B"}, {"T"})
    invalid = copy.deepcopy(records)
    invalid["a"]["evidence"]["feature_id"] = "not_a_predictor"
    with pytest.raises(ValueError):
        m.fitting_matrix(invalid, {"A", "B"}, {"T"})
    query = data("T", 4, query=True)
    query["evidence"] = query["evidence"].drop(columns="missing_value")
    with pytest.raises(ValueError):
        m.predict_four_modes(query, model)


def test_no_implicit_early_stopping_or_variable_cpu_parameters():
    for changed in (
        {"thread_count": 2},
        {"allow_writing_files": True},
        {"use_best_model": True},
        {"early_stopping_rounds": 10},
    ):
        with pytest.raises(ValueError):
            m.fit_four_modes(
                {},
                training_groups={"A"},
                excluded_groups={"T"},
                parameters={**m.PARAMETERS, **changed},
            )
