"""Driver permission, pause, import-root and bounded-dispatch controls only."""

import ast
from concurrent.futures import Future
from pathlib import Path
import subprocess
import sys

import pytest

from context import REPO
from frozen_resources import configure_frozen_resources
import run_qe_matrix_context_experiment as runner

configure_frozen_resources(REPO)


def test_initialization_refuses_missing_permission_before_output_creation(
    tmp_path, monkeypatch
):
    def refuse():
        raise RuntimeError("no-final-permission")

    destination = tmp_path / "not-created"
    monkeypatch.setattr(runner.data, "OUT", destination)
    monkeypatch.setattr(runner.bindings, "require_fit_permission", refuse)
    with pytest.raises(RuntimeError, match="no-final-permission"):
        runner.initialize()
    assert not destination.exists()


def test_fresh_driver_import_cannot_cache_wrong_scientific_root():
    for module in (
        "run_qe_matrix_context_experiment",
        "verify_qe_matrix_context_experiment",
    ):
        code = f"import sys;sys.path.insert(0,'tools/qe_optimization');import {module};assert 'resource_paths' not in sys.modules"
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


def test_every_scientific_worker_configures_final_snapshot_first():
    source = ast.parse(Path(runner.__file__).read_text())
    definitions = {n.name: n for n in source.body if isinstance(n, ast.FunctionDef)}
    for name in ("fit_parent", "calibrate", "annotate"):
        first = definitions[name].body[0]
        assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
        assert first.value.func.id == "configure_frozen_resources"


def test_pause_marker_stops_dispatch_and_preserves_incomplete_status(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner.data, "OUT", tmp_path)
    (tmp_path / "pause_requested").write_text("explicit synthetic pause")

    def forbidden(_):
        raise AssertionError("paused task was dispatched")

    assert runner.execute([0, 1, 2], forbidden, "synthetic") is None
    assert (tmp_path / "pause_status.json").exists()


def test_dispatch_never_has_more_than_two_unconsumed_tasks(tmp_path, monkeypatch):
    class Pool:
        active = 0
        maximum = 0

        def __init__(self, *, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, item):
            Pool.active += 1
            Pool.maximum = max(Pool.maximum, Pool.active)

            class Result(Future):
                def result(self, *args, **kwargs):
                    Pool.active -= 1
                    return super().result(*args, **kwargs)

            future = Result()
            future.set_result(function(item))
            return future

    monkeypatch.setattr(runner.data, "OUT", tmp_path)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", Pool)
    result = runner.execute(range(7), lambda i: i * i, "synthetic")
    assert sorted(result) == [i * i for i in range(7)]
    assert Pool.maximum == 2 and Pool.active == 0


def test_independent_selector_matches_native_including_ambiguity_and_one_score():
    from context import REPO
    from frozen_resources import configure_frozen_resources

    configure_frozen_resources(REPO)
    import numpy as np
    import pandas as pd
    from matlab_annotation.operating_point import select_operating_candidates
    from verify_qe_matrix_context_experiment import select_scalar

    rng = np.random.default_rng(14)
    pool = pd.DataFrame(
        dict(
            feat_feature_id=rng.integers(0, 45, 300).astype(str),
            name=rng.integers(0, 80, 300).astype(str),
            adduct=rng.choice(["[M+H]+", "[M+Na]+"], 300),
            feat_intensity=rng.integers(0, 30, 300).astype(float),
        )
    )
    pool.loc[[0, 2], "feat_intensity"] = np.nan
    scores = pd.Series(rng.choice([0.2, 0.4, 0.6, 1.0], len(pool)), index=pool.index)
    for cut in (0.2, 0.6, 1.0):
        native = select_operating_candidates(pool, scores, threshold=cut)
        separate = select_scalar(pool, scores, cut)
        assert native.index.equals(separate.index)
        pd.testing.assert_series_equal(
            native.class_selection_score,
            separate.class_selection_score,
            check_exact=True,
        )


def test_independent_mode_choice_matches_original_objective():
    import pandas as pd
    from matlab_annotation.operating_point import choose_balanced_training_points
    from verify_qe_matrix_context_experiment import choose_scalar

    rows = [
        dict(
            stream=s,
            threshold=cut,
            manual=100,
            correct=correct,
            extras=extra,
            different_peak=wrong,
        )
        for s in runner.data.MODES
        for cut, correct, extra, wrong in (
            (1.0, 30, 0, 0),
            (0.6, 82, 8, 2),
            (0.2, 96, 20, 3),
        )
    ]
    table = pd.DataFrame(rows)
    assert choose_scalar(table) == choose_balanced_training_points(
        table, extras_per_manual=0.095, target=0.9
    )


def test_independent_completion_levels_counts_guards_and_choice():
    import pandas as pd
    from matlab_annotation.add_only_class_completion import (
        class_completion_curves,
        choose_completion_thresholds,
    )
    from verify_qe_matrix_context_experiment import check_completion_curves

    proposals, manual = [], []
    for project in ("A", "B", "C"):
        for i in range(4):
            proposals.append(
                dict(
                    stream="RP_Pos",
                    lipid_class="PC",
                    project=project,
                    class_selection_score=0.9 - i * 0.01,
                    training_positive=True,
                    training_label_known=True,
                    calibration_identity_in_manual=True,
                )
            )
        manual.extend(
            [
                dict(
                    stream="RP_Pos",
                    lipid_class="PC",
                    project=project,
                    internal_standard=False,
                )
            ]
            * 40
        )
    proposals.append(
        dict(
            stream="RP_Pos",
            lipid_class="PC",
            project="A",
            class_selection_score=0.95,
            training_positive=False,
            training_label_known=True,
            calibration_identity_in_manual=False,
        )
    )
    p, m = pd.DataFrame(proposals), pd.DataFrame(manual)
    base = {("RP_Pos", "PC"): 100}
    table = class_completion_curves(
        p, m, base, training_groups={"A", "B", "C"}, excluded_groups={"Q"}
    )
    expected, _ = choose_completion_thresholds(table)
    assert check_completion_curves(p, m, base, table) == expected
    broken = table.copy()
    broken.loc[0, "extras"] += 1
    with pytest.raises(AssertionError):
        check_completion_curves(p, m, base, broken)
