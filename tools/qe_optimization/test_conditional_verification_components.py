"""No real datasets; completed integrity receipt and nullable-ID schema checks."""

from io import StringIO
import json

import pandas as pd

import verify_qe_conditional_peak_experiment as verifier


def test_nullable_previous_original_feature_ids_preserve_text_and_leading_zeroes():
    csv = "feat_feature_id,peak_refinement_old_feature_id\n10,00123\n11,\n12,456\n"
    parsed = pd.read_csv(
        StringIO(csv),
        dtype={"feat_feature_id": str, "peak_refinement_old_feature_id": str},
    )
    assert parsed.peak_refinement_old_feature_id.fillna("").tolist() == [
        "00123",
        "",
        "456",
    ]


def test_only_matching_complete_hash_bound_model_receipts_resume(tmp_path):
    path = tmp_path / "case.json"
    assert verifier.completed_component(path, "model", "model.pkl", "hash") is None
    path.write_text('{"all_passed":')
    assert verifier.completed_component(path, "model", "model.pkl", "hash") is None
    record = dict(
        all_passed=True,
        kind="model",
        name="model.pkl",
        verification_identity_sha256="hash",
        result=dict(model="model.pkl", training_queries=19, fresh_experts=4),
    )
    path.write_text(json.dumps(record))
    assert (
        verifier.completed_component(path, "model", "model.pkl", "hash")
        == record["result"]
    )
    for kind, name, binding in (
        ("outer", "model.pkl", "hash"),
        ("model", "wrong", "hash"),
        ("model", "model.pkl", "changed"),
    ):
        assert verifier.completed_component(path, kind, name, binding) is None
    for field, value in (
        ("all_passed", False),
        ("result", dict(model="model.pkl", training_queries=0, fresh_experts=4)),
    ):
        path.write_text(json.dumps(dict(record, **{field: value})))
        assert verifier.completed_component(path, "model", "model.pkl", "hash") is None


def test_partial_inner_or_outer_checks_cannot_complete_integrity(tmp_path, monkeypatch):
    path = tmp_path / "case.json"
    monkeypatch.setattr(
        verifier, "inventory", lambda: (["a", "b", "c"], {"a": 0, "b": 1, "c": 1})
    )
    inner = dict(
        all_passed=True,
        kind="inner",
        name="0",
        verification_identity_sha256="hash",
        result=2,
    )
    path.write_text(json.dumps(inner))
    assert verifier.completed_component(path, "inner", "0", "hash") == 2
    path.write_text(json.dumps(dict(inner, result=1)))
    assert verifier.completed_component(path, "inner", "0", "hash") is None
    outer = dict(
        all_passed=True,
        kind="outer",
        name="a",
        verification_identity_sha256="hash",
        result=dict(dataset="a", raw_stream_checks=4, moved_peaks=0),
    )
    path.write_text(json.dumps(outer))
    assert verifier.completed_component(path, "outer", "a", "hash") == outer["result"]
    path.write_text(
        json.dumps(dict(outer, result=dict(dataset="a", raw_stream_checks=3)))
    )
    assert verifier.completed_component(path, "outer", "a", "hash") is None
