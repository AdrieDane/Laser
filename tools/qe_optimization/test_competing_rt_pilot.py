"""Structural single-change checks; no example-data access."""

import ast
import copy
from pathlib import Path


def functions(name):
    tree = ast.parse(Path(__file__).with_name(name).read_text())
    return {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}


def test_only_scientific_runner_change_is_explicit_anchor_guard():
    old = functions("diagnose_qe_coherent_rt.py")
    new = functions("diagnose_qe_competing_class_rt.py")
    assert old.keys() == new.keys()
    for name in old.keys() - {"initialize", "predict"}:
        assert ast.dump(old[name]) == ast.dump(new[name]), name
    target = copy.deepcopy(new["predict"])
    removed = 0
    for node in ast.walk(target):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "coherent_retention_predictions"
        ):
            for keyword in list(node.keywords):
                if keyword.arg == "competing_class_anchor_guard":
                    assert (
                        isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                    )
                    node.keywords.remove(keyword)
                    removed += 1
    assert removed == 1
    assert ast.dump(target) == ast.dump(old["predict"])


def test_full_integrity_replay_functions_are_unchanged():
    old = functions("verify_qe_coherent_rt_pilot.py")
    new = functions("verify_qe_competing_class_rt_pilot.py")
    assert old.keys() == new.keys()
    for name in old:
        assert ast.dump(old[name]) == ast.dump(new[name]), name
