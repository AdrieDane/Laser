"""Toy-only native HGB consumer/scopes; NEVER fits real QE examples."""

from collections import Counter
import copy

import numpy as np
import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
import matrix_experiment_data as data
import matrix_mode_model as backend


def record(query, refs, blocked, *, labels=True):
    streams = np.repeat(data.MODES, 8)
    columns = {"stream": streams, "class": ["PC", "PE"] * 16}
    columns.update({f"old{i}": np.arange(32, dtype=float) + i for i in range(106)})
    columns.update({data.peer.PREFIX + field: np.nan for field in data.peer.FIELDS})
    frame = pd.DataFrame(columns)
    pool = pd.DataFrame(dict(stream=streams))
    if labels:
        pool["training_positive"] = [False, True] * 16
        pool["training_label_known"] = True
    return dict(
        query_group=query,
        query_dataset="synthetic_" + query,
        reference_training_groups=refs,
        reference_excluded_groups=blocked,
        matrix_reference_training_groups=refs,
        matrix_reference_excluded_groups=blocked,
        matrix_reference_sha256="0" * 64,
        matrix_layer_sha256="1" * 64,
        evidence=frame,
        pool=pool,
    )


def test_real_experiment_path_is049_not_inherited048():
    assert data.model_path(0, None).parent == data.OUT
    assert "experiment_049" in str(data.model_path(0, 1))


def test_scope_and_route_mismatch_are_refused():
    r = record("A", ["B", "C"], ["A", "Q"])
    data.validate_record(r, {"A", "B", "C"}, {"Q"}, query_is_training=True)
    broken = copy.deepcopy(r)
    broken["matrix_reference_training_groups"].append("Q")
    with pytest.raises(ValueError):
        data.validate_record(broken, {"A", "B", "C"}, {"Q"}, query_is_training=True)
    broken = copy.deepcopy(r)
    broken["pool"].loc[0, "stream"] = "NP_Neg"
    with pytest.raises(ValueError):
        data.validate_record(broken, {"A", "B", "C"}, {"Q"}, query_is_training=True)


def test_toy_native_fit_exact_consumer_matrix_weights_and_prediction():
    from matlab_annotation.evidence_training import evidence_matrix
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts

    records = {g: record(g, [r for r in "ABC" if r != g], [g, "Q"]) for g in "ABC"}
    before = {g: r["evidence"].copy(deep=True) for g, r in records.items()}
    model = backend.fit(
        records,
        training_groups=set("ABC"),
        excluded_groups={"Q"},
        parameters=data.parameters(),
    )
    assert set(model["experts"]) == set(data.MODES)
    for mode in data.MODES:
        frames, y, groups = [], [], []
        for raw in records.values():
            use = raw["pool"].stream.eq(mode) & raw["pool"].training_label_known
            frames.append(raw["evidence"].loc[use])
            y.extend(raw["pool"].loc[use, "training_positive"].astype(int))
            groups.extend([raw["query_group"]] * int(use.sum()))
        frame = pd.concat(frames, ignore_index=True)
        native = evidence_matrix(frame, model["experts"][mode])
        counts = Counter(groups)
        weights = np.asarray([len(groups) / len(counts) / counts[g] for g in groups])
        ledger = model["independent_fitting_ledger"][mode]
        assert backend.digest(native) == ledger["matrix_sha256"]
        assert backend.digest(np.asarray(y, int)) == ledger["labels_sha256"]
        assert backend.digest(weights) == ledger["weights_sha256"]
    query = record("Q", list("ABC"), ["Q"], labels=False)
    actual = backend.predict(query, model)
    expected = predict_mode_evidence_experts(query, model)
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    assert actual.between(0, 1).all()
    for g in records:
        pd.testing.assert_frame_equal(
            before[g], records[g]["evidence"], check_exact=True
        )
    with pytest.raises(ValueError):
        backend.predict(
            dict(query, pool=query["pool"].assign(training_positive=True)), model
        )
