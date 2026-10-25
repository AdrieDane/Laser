"""046 synthetic physical-only admission/independent grid/unchanged whole guard."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
from bmp_continuous_admission import select_additions
from bmp_context_measurements import PREFIX
import run_qe_bmp_continuous_experiment as runner
from verify_qe_bmp_continuous_experiment import independent_grid
from test_bmp_project_headroom import fixture


def candidates():
    pool = pd.DataFrame(
        [
            dict(
                stream="NP_Pos",
                name="BMP(36:2)",
                **{"class": "BMP"},
                adduct="[M+NH4]+",
                is_is=False,
                feat_feature_id=fid,
                feat_intensity=100.0,
                ref_mz=800.0,
                feat_mz=800.0001,
                feat_rt=5.0,
                ref_rt=5.0,
            )
            for fid in ("a", "b")
        ]
    )
    evidence = pd.DataFrame(
        {
            PREFIX + "source_physical": [1.0, 1.0],
            PREFIX + "counterpart_raw": [np.nan, 0.9],
        }
    )
    return pool, evidence


def choose(pool, evidence, **extra):
    return select_additions(
        pool,
        pd.Series([0.95, 0.9]),
        evidence,
        pool.iloc[:0],
        stream="NP_Pos",
        threshold=extra.pop("threshold", 0.5),
        protected_features=extra.pop("protected_features", set()),
        declared_masses=np.array([]),
        **extra,
    )


def test_unknown_profile_not_a_veto_but_no_physical_fallback():
    pool, evidence = candidates()
    before = pool.copy(deep=True)
    assert choose(pool, evidence).feat_feature_id.tolist() == ["a"]
    evidence.loc[0, PREFIX + "source_physical"] = 0
    assert choose(pool, evidence).empty  # never promote weaker b
    evidence.loc[0, PREFIX + "source_physical"] = np.nan
    assert choose(pool, evidence).empty
    pd.testing.assert_frame_equal(pool, before, check_exact=True)


@pytest.mark.parametrize(
    "kind", ["ppm", "rt", "standard", "isotope", "protection", "disable"]
)
def test_unchanged_native_guard_still_blocks_best_candidate(kind):
    pool, evidence = candidates()
    kwargs = {}
    if kind == "ppm":
        pool.loc[0, "feat_mz"] = 800.01
    elif kind == "rt":
        pool.loc[0, "feat_rt"] = 5.5
    elif kind == "standard":
        pool.loc[0, "is_is"] = True
    elif kind == "isotope":
        pool["isotopologue"] = ["M+1", ""]
    elif kind == "protection":
        kwargs["protected_features"] = {"a"}
    elif kind == "disable":
        kwargs["threshold"] = 1.0
    assert choose(pool, evidence, **kwargs).empty


def test_original044_169grid_independent_and_corruption_refused():
    projects = fixture()
    pooled = runner.headroom.pooled_curves(
        projects, training_groups=set("ABCD"), excluded_groups={"Q"}
    )
    cuts, grid = runner.completion.choose_joint_grid(pooled)
    assert cuts == independent_grid(projects, grid, set("ABCD"))
    bad = grid.copy()
    bad.loc[0, "eligible"] = not bool(bad.loc[0, "eligible"])
    with pytest.raises(AssertionError):
        independent_grid(projects, bad, set("ABCD"))


def test_whole_guard_identical_and_no_new_model_fits():
    folder = Path(__file__).parent
    dumps = []
    for name in (
        "report_qe_bmp_context_tradeoff.py",
        "report_qe_bmp_continuous_tradeoff.py",
    ):
        tree = ast.parse((folder / name).read_text())
        dumps.append(
            ast.dump(
                next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "assess"
                )
            )
        )
    assert dumps[0] == dumps[1]
    assert runner.PREVIOUS == runner.QE / "experiment_041"
    assert runner.parent_path(0, 1) == runner.source.parent_path(0, 1)
    tree = ast.parse(Path(runner.__file__).read_text())
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"fit", "fit_parent", "fit_context"}
        for n in ast.walk(tree)
    )
