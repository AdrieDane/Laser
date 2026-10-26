"""Nested matrix scope/index controls, no example manual comparisons opened."""

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
import matrix_nested_data as data


def raw(query="Q", refs=("A", "B"), blocked=("Q",)):
    return dict(
        query_group=query,
        reference_training_groups=list(refs),
        reference_excluded_groups=list(blocked),
        pool=pd.DataFrame(index=[1, 5]),
        evidence=pd.DataFrame({"score": [0.3, 0.5]}, index=[1, 5]),
    )


def test_all_masks_cover_own_outer_inner_but_never_query_reference():
    masks = data.models()
    assert len(masks) == 25
    assert [sum(len(m) == n for m in masks) for n in (1, 2, 3)] == [5, 10, 10]
    for own in range(5):
        choices = data.original.masks_for(own)
        assert len(choices) == 11 and all(own in mask for mask in choices)


def test_exact_reference_mask_required_for_final_queries():
    d = raw()
    model = dict(training_groups=["A", "B"], excluded_groups=["Q"])
    data.validate_scope(d, model, {"A", "B"}, {"Q"}, query_is_training=False)
    for refs, excluded in (
        (["A", "Q"], ["B"]),
        (["A"], ["B", "Q"]),
        (["A", "B"], ["OTHER"]),
    ):
        with pytest.raises(ValueError):
            data.validate_scope(
                d,
                dict(training_groups=refs, excluded_groups=excluded),
                {"A", "B"},
                {"Q"},
                query_is_training=False,
            )


def test_training_queries_use_finer_reference_exclusion_than_scoring_fit():
    d = raw(blocked=("Q", "OUTER", "INNER"))
    model = dict(training_groups=["A", "B"], excluded_groups=["Q", "OUTER", "INNER"])
    data.validate_scope(
        d, model, {"A", "B", "Q"}, {"OUTER", "INNER"}, query_is_training=True
    )
    with pytest.raises(ValueError):
        data.validate_scope(
            d,
            dict(training_groups=["A", "B", "Q"], excluded_groups=["OUTER", "INNER"]),
            {"A", "B", "Q"},
            {"OUTER", "INNER"},
            query_is_training=True,
        )


def test_scientific_workers_configure_final_snapshot_first():
    import prepare_qe_matrix_nested as producer
    import verify_qe_matrix_nested as verifier

    for module, names in (
        (producer, ("fit", "prepare")),
        (verifier, ("reference", "query")),
    ):
        definitions = {
            n.name: n
            for n in ast.parse(Path(module.__file__).read_text()).body
            if isinstance(n, ast.FunctionDef)
        }
        for name in names:
            first = definitions[name].body[0]
            assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
            assert first.value.func.id == "configure_frozen_resources"


@pytest.mark.parametrize("tamper", [None, "index", "source_hash", "reference_hash"])
def test_delta_loader_preserves_all_old_columns_and_refuses_tampering(
    tmp_path, monkeypatch, tamper
):
    query, a, b = "24P0003", "24P0001", "24P0002"
    dataset = query + "_example"
    original = raw(query=query, refs=(a, b), blocked=(query,))
    original["evidence"] = pd.DataFrame(
        {f"old{n}": [float(n), float(n + 1)] for n in range(108)}, index=[1, 5]
    )
    original["query_dataset"] = dataset
    original["excluded_folds"] = [0]
    model = dict(training_groups=[a, b], excluded_groups=[query])
    old_path = tmp_path / "source.pkl"
    pd.to_pickle(original, old_path)
    monkeypatch.setattr(data, "OUT", tmp_path)
    monkeypatch.setattr(data.original, "load_predictors", lambda *_: original)
    monkeypatch.setattr(data.original, "prepared_path", lambda *_: old_path)
    monkeypatch.setattr(
        data.pilot,
        "inventory",
        lambda: ([dataset, a + "_x", b + "_x"], {dataset: 0, a + "_x": 1, b + "_x": 2}),
    )
    folder = tmp_path / "models/exclude_0"
    folder.mkdir(parents=True)
    reference = folder / "model.json"
    reference.write_text(json.dumps(model))
    added = pd.DataFrame(
        0.5, index=[1, 5], columns=[data.peer.PREFIX + f for f in data.peer.FIELDS]
    )
    saved = dict(
        dataset=dataset,
        excluded_folds=[0],
        source_pool_sha256=data.pilot.sha(old_path),
        reference_sha256=data.pilot.sha(reference),
        evidence=added,
    )
    if tamper == "index":
        saved["evidence"] = added.set_axis([1, 6])
    if tamper == "source_hash":
        saved["source_pool_sha256"] = "not-the-source-hash"
    if tamper == "reference_hash":
        saved["reference_sha256"] = "not-the-reference-hash"
    destination = data.layer_path(dataset, {0})
    destination.parent.mkdir(parents=True)
    pd.to_pickle(saved, destination)
    if tamper:
        with pytest.raises(AssertionError):
            data.load_predictors(dataset, {0})
    else:
        loaded = data.load_predictors(dataset, {0})
        pd.testing.assert_frame_equal(
            loaded["pool"], original["pool"], check_exact=True
        )
        pd.testing.assert_frame_equal(
            loaded["evidence"][original["evidence"].columns],
            original["evidence"],
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            loaded["evidence"][added.columns], added, check_exact=True
        )
