"""047 synthetic lower-domain/native-AST/unchanged-risk/484-grid safeguards."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
import bmp_extended_scores as extended
from matlab_annotation.np_context_bmp_completion import select_context_bmp_additions
from test_bmp_continuous_experiment import candidates
from test_bmp_project_headroom import fixture
from verify_qe_bmp_extended_experiment import independent_grid
import run_qe_bmp_extended_experiment as runner


def test_native_body_identical_except_one_lower_bound_and_low_scores_unmodified():
    extended.assert_native_body_parity()
    p, e = candidates()
    scores = pd.Series([0.3, 0.2])
    before = scores.copy(deep=True)
    args = dict(stream="NP_Pos", protected_features=set(), declared_masses=np.array([]))
    selected = extended.select_additions(
        p, scores, e, p.iloc[:0], threshold=0.2, **args
    )
    assert selected.feat_feature_id.tolist() == ["a"]
    assert selected.class_selection_score.tolist() == [0.3]
    pd.testing.assert_series_equal(scores, before, check_exact=True)
    with pytest.raises(ValueError):
        select_context_bmp_additions(p, scores, p.iloc[:0], threshold=0.2, **args)
    with pytest.raises(ValueError):
        extended.select_native_additions(p, scores, p.iloc[:0], threshold=0.049, **args)


@pytest.mark.parametrize("threshold", [0.5, 0.55, 0.9, 0.975, 0.99, 1.0])
def test_existing_native_domain_parity(threshold):
    p, _ = candidates()
    args = dict(stream="NP_Pos", protected_features=set(), declared_masses=np.array([]))
    scores = pd.Series([0.95, 0.9])
    a = extended.select_native_additions(
        p, scores, p.iloc[:0], threshold=threshold, **args
    )
    b = select_context_bmp_additions(p, scores, p.iloc[:0], threshold=threshold, **args)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_lower_domain_keeps_global_ambiguity_native_physics_and_no_fallback():
    p, e = candidates()
    args = dict(
        stream="NP_Pos",
        threshold=0.05,
        protected_features=set(),
        declared_masses=np.array([]),
    )
    # Shared identity on different features: strongest feature wins first.
    e.loc[0, extended.PREFIX + "source_physical"] = 0
    assert extended.select_additions(
        p, pd.Series([0.3, 0.2]), e, p.iloc[:0], **args
    ).empty
    e.loc[0, extended.PREFIX + "source_physical"] = 1
    p.loc[0, "feat_mz"] = 800.01
    assert extended.select_additions(
        p, pd.Series([0.3, 0.2]), e, p.iloc[:0], **args
    ).empty
    p, _ = candidates()
    # Different label on the SAME original feature needs .05 separation.
    p.loc[1, "feat_feature_id"] = "a"
    p.loc[1, "name"] = "PG(36:2)"
    p.loc[1, "class"] = "PG"
    assert extended.select_additions(
        p, pd.Series([0.3, 0.28]), e, p.iloc[:0], **args
    ).empty


def projects22():
    old = fixture()
    extra = []
    for t in extended.LEVELS:
        if t in old.threshold.unique():
            continue
        rows = old.loc[old.threshold.eq(0.5)].copy()
        rows["threshold"] = t
        rows[["added_correct", "extras"]] = 0
        if t == 0.2:
            good = rows.stream.eq("NP_Pos") & rows.project.isin(list("ABC"))
            rows.loc[good, ["added_correct", "extras"]] = [40, 10]
        extra.append(rows)
    return pd.concat([old, *extra], ignore_index=True)


def test_all484_grid_points_scalar_checked_and_same_risk_rules():
    projects = projects22()
    pooled = extended.pooled_curves(
        projects, training_groups=set("ABCD"), excluded_groups={"Q"}
    )
    cuts, grid = extended.choose_joint_grid(pooled)
    assert len(extended.LEVELS) == 22 and len(grid) == 484
    assert (
        cuts
        == independent_grid(projects, grid, set("ABCD"))
        == {"NP_Pos": 0.2, "NP_Neg": 1.0}
    )
    bad = grid.copy()
    bad.loc[0, "extras"] += 1
    with pytest.raises(AssertionError):
        independent_grid(projects, bad, set("ABCD"))
    projects[["added_correct", "extras"]] = 0
    pooled = extended.pooled_curves(
        projects, training_groups=set("ABCD"), excluded_groups={"Q"}
    )
    cuts, grid = extended.choose_joint_grid(pooled)
    assert (
        cuts
        == independent_grid(projects, grid, set("ABCD"))
        == dict.fromkeys(extended.MODES, 1.0)
    )


def test_whole_guard_is_unchanged_and_model_paths_reused():
    folder = Path(__file__).parent
    parsed = []
    for name in (
        "report_qe_bmp_context_tradeoff.py",
        "report_qe_bmp_extended_tradeoff.py",
    ):
        tree = ast.parse((folder / name).read_text())
        parsed.append(
            ast.dump(
                next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "assess"
                )
            )
        )
    assert parsed[0] == parsed[1]
    assert runner.PREVIOUS == runner.QE / "experiment_041"
    assert runner.parent_path(0, 1) == runner.source.parent_path(0, 1)
    tree = ast.parse(Path(runner.__file__).read_text())
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"fit", "fit_parent", "fit_context"}
        for n in ast.walk(tree)
    )
