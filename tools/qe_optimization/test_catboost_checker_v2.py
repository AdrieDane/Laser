"""V2 memory-only fix, no original scientific helpers or tolerances changed."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import verify_qe_catboost_experiment as original
import verify_qe_catboost_experiment_v2 as fixed
from test_catboost_evidence_model import data
import catboost_evidence_model as backend


def functions(module):
    tree = ast.parse(Path(module.__file__).read_text())
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_every_original_scientific_helper_is_identical_except_owned_copy():
    a, b = functions(original), functions(fixed)
    for name, node in a.items():
        if name == "main":
            continue
        other = b[name]
        if name == "native_scores":
            edits = 0
            for call in ast.walk(other):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "to_numpy"
                ):
                    keep = []
                    for keyword in call.keywords:
                        if keyword.arg == "copy":
                            assert (
                                isinstance(keyword.value, ast.Constant)
                                and keyword.value.value is True
                            )
                            edits += 1
                        else:
                            keep.append(keyword)
                    call.keywords = keep
            assert edits == 1
        assert ast.dump(node, include_attributes=False) == ast.dump(
            other, include_attributes=False
        ), name


class Estimator:
    def predict_proba(self, x, thread_count):
        assert thread_count == 1
        p = 1 / (1 + np.exp(-x.mass_error.to_numpy(float)))
        return np.column_stack([1 - p, p])


def test_reproduce_cow_failure_and_fixed_prediction_exact_parity():
    query = data("T", 8, query=True)
    x = query["evidence"]
    assert not x.mass_error.to_numpy(float).flags.writeable
    columns = list(x)
    categorical = list(x.select_dtypes(exclude=["number", "bool"]))
    model = dict(
        format=backend.FORMAT,
        training_cohorts=["A", "B"],
        excluded_groups=["T"],
        columns=columns,
        experts={
            s: dict(
                estimator=Estimator(),
                columns=columns,
                categorical_columns=categorical,
                training_cohorts=["A", "B"],
                excluded_groups=["T"],
            )
            for s in backend.MODES
        },
    )
    before = x.copy(deep=True)
    with pytest.raises(ValueError, match="read-only"):
        original.native_scores(query, model)
    pd.testing.assert_series_equal(
        fixed.native_scores(query, model),
        backend.predict_four_modes(query, model),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(x, before, check_exact=True)
