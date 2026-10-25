"""Independent grid and unchanged whole guard/exports; no actual045cuts."""

import ast
from pathlib import Path

import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
import run_qe_bmp_headroom_experiment as runner
from verify_qe_bmp_headroom_experiment import independent_grid
from test_bmp_project_headroom import fixture


def test_scalar_independent_169grid_and_each_project_witness_match():
    projects = fixture()
    cuts, _, grid, witnesses = runner.headroom.choose_project_headroom(
        projects, training_groups=set("ABCD"), excluded_groups={"Q"}
    )
    assert independent_grid(projects, grid, witnesses, set("ABCD")) == cuts
    broken = witnesses.copy()
    broken.loc[0, "remaining_new_correct"] += 1
    with pytest.raises(AssertionError):
        independent_grid(projects, grid, broken, set("ABCD"))
    corrupted = grid.copy()
    corrupted.loc[0, "training_headroom_pass"] = not bool(
        corrupted.loc[0, "training_headroom_pass"]
    )
    with pytest.raises(AssertionError):
        independent_grid(projects, corrupted, witnesses, set("ABCD"))


def test_whole_guard_is_AST_identical_not_relaxed_after_044():
    root = Path(__file__).parent
    functions = []
    for name in (
        "report_qe_bmp_context_tradeoff.py",
        "report_qe_bmp_headroom_tradeoff.py",
    ):
        tree = ast.parse((root / name).read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "assess"
        )
        functions.append(ast.dump(fn, include_attributes=False))
    assert functions[0] == functions[1]


def test_model_paths_are_reused_not_new_fits_or_044_annotation_copies():
    assert runner.PREVIOUS == runner.QE / "experiment_041"
    assert runner.parent_path(0, 1) == runner.source.parent_path(0, 1)
    assert runner.OUT != runner.source.OUT
    tree = ast.parse(Path(runner.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"fit", "fit_context", "fit_parent"}
    caller = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "cases_and_curves"
    )
    reused = [
        n
        for n in ast.walk(caller)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "calibration_cases"
    ]
    assert len(reused) == 1 and any(
        k.arg == "fit" and isinstance(k.value, ast.Constant) and k.value.value is False
        for k in reused[0].keywords
    )


def test_explicit_disable_grid_replays_independently_and_inputs_unchanged():
    projects = fixture()
    projects[["added_correct", "extras"]] = 0
    before = projects.copy(deep=True)
    cuts, _, grid, witnesses = runner.headroom.choose_project_headroom(
        projects, training_groups=set("ABCD"), excluded_groups={"Q"}
    )
    assert (
        cuts
        == independent_grid(projects, grid, witnesses, set("ABCD"))
        == dict.fromkeys(runner.NP_MODES, 1.0)
    )
    pd.testing.assert_frame_equal(projects, before, check_exact=True)
