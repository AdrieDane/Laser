"""Initialization-only restart: no scientific changes or snapshot guard bypass."""

import ast
from pathlib import Path
import shutil
import subprocess
import sys

from context import REPO


def functions(name):
    tree = ast.parse((Path(__file__).parent / name).read_text())
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_new_startup_imports_native_only_after_final_snapshot_configuration():
    init = functions("run_qe_bmp_extended_experiment_v2.py")["initialize"]
    config, parity = [], []
    for n in ast.walk(init):
        if isinstance(n, ast.Call):
            if (
                isinstance(n.func, ast.Name)
                and n.func.id == "configure_frozen_resources"
            ):
                config.append(n)
                assert not any(
                    isinstance(x, ast.Name) and x.id == "source"
                    for x in ast.walk(n.args[0])
                )
            if (
                isinstance(n.func, ast.Attribute)
                and n.func.attr == "assert_native_body_parity"
            ):
                parity.append(n)
    assert len(config) == 2 and len(parity) == 1
    assert parity[0].lineno > max(c.lineno for c in config)


def test_initialization_restart_changes_no_calibration_or_selection_algorithm():
    for stem, names in (
        (
            "run_qe_bmp_extended_experiment",
            ["project_curves", "cases_and_curves", "assemble", "calibrate", "run_fold"],
        ),
        (
            "verify_qe_bmp_extended_experiment",
            [
                "check_parent",
                "independent_proposals",
                "independent_grid",
                "check_calibration",
                "check_outer",
            ],
        ),
        ("report_qe_bmp_extended_tradeoff", ["assess"]),
    ):
        a, b = functions(stem + ".py"), functions(stem + "_v2.py")
        for name in names:
            assert ast.dump(a[name]) == ast.dump(b[name])
    # Actual scientific helper remains the one tested/frozen before first attempt.
    assert (REPO / "tools/qe_optimization/bmp_extended_scores.py").is_file()


def test_fresh_process_final_root_works_and_stale_root_guard_still_refuses(tmp_path):
    roots = []
    for name in ("first", "final"):
        root = tmp_path / name
        for folder in ("app", "reference_data", "methods"):
            (root / folder).mkdir(parents=True)
        shutil.copy2(REPO / "app/resource_paths.py", root / "app/resource_paths.py")
        shutil.copy2(
            REPO / "reference_data/standard_lipid_composition_ranges.tsv",
            root / "reference_data/standard_lipid_composition_ranges.tsv",
        )
        roots.append(root)
    setup = "import sys; sys.path.insert(0,sys.argv[1]); from frozen_resources import configure_frozen_resources\n"
    guard = (
        setup
        + """configure_frozen_resources(sys.argv[2])
try:
    configure_frozen_resources(sys.argv[3])
except RuntimeError as error:
    assert 'escaped' in str(error)
    print('stale-root correctly refused')
else:
    raise AssertionError('Snapshot guard was bypassed')
"""
    )
    fresh = (
        setup
        + """receipt=configure_frozen_resources(sys.argv[3])
assert receipt['reference_data'] == str(__import__('pathlib').Path(sys.argv[3])/'reference_data')
print('final-root correctly configured')
"""
    )
    for code, expected in (
        (guard, "stale-root correctly refused"),
        (fresh, "final-root correctly configured"),
    ):
        completed = subprocess.run(
            [sys.executable, "-c", code, str(Path(__file__).parent), *map(str, roots)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert expected in completed.stdout
