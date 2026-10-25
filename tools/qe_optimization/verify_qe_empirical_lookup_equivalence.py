"""Exact scalar/vector numerical AND ALL completed full-table equivalence gate."""

import ast
import json
from pathlib import Path

import pandas as pd

from context import WORKSPACE
from run_qe_standard_mode_experiment import sha, dump
from verify_qe_empirical_preparation import same

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OLD = QE / "empirical_RT_candidate_preparation_v1"
NEW = QE / "empirical_RT_candidate_preparation_v2"


def function(tree, name):
    return next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def main():
    old = json.loads((OLD / "protocol.json").read_text())
    new = json.loads((NEW / "protocol.json").read_text())
    inputs = {
        str(root / "protocol.json"): sha(root / "protocol.json") for root in (OLD, NEW)
    }
    scalar = OLD / "source_snapshot/app/matlab_annotation/empirical_retention.py"
    vector = NEW / "source_snapshot/app/matlab_annotation/empirical_retention.py"
    a, b = ast.parse(scalar.read_text()), ast.parse(vector.read_text())
    for name in (
        "_verify_atlas_scope",
        "fit_empirical_retention",
        "predict_aligned_empirical_retention",
    ):
        assert ast.dump(function(a, name), include_attributes=False) == ast.dump(
            function(b, name), include_attributes=False
        ), name
    original = function(a, "predict_empirical_reference")
    fallback = function(b, "_predict_empirical_reference_scalar")
    fallback.name = original.name
    assert ast.dump(original, include_attributes=False) == ast.dump(
        fallback, include_attributes=False
    )
    # Every other scientific source must be unchanged. Test-only differences
    # cannot change runtime but remain covered by their own frozen hashes.
    for relative, digest in old["source_sha256"].items():
        assert sha(OLD / "source_snapshot" / relative) == digest
        if "/tests/" not in relative and not relative.endswith(
            "/empirical_retention.py"
        ):
            assert digest == new["source_sha256"][relative], relative
    for relative, digest in new["source_sha256"].items():
        assert sha(NEW / "source_snapshot" / relative) == digest
    for source, digest in old["input_sha256"].items():
        assert sha(Path(source)) == digest
        assert new["input_sha256"][source] == digest
    checkpoints = sorted((OLD / "checkpoints").glob("*.json"))
    assert len(checkpoints) >= 9, "ALLcompleted original datasets required,not a sample"
    tables = 0
    for checkpoint in checkpoints:
        current = NEW / "checkpoints" / checkpoint.name
        previous, actual = (
            json.loads(checkpoint.read_text()),
            json.loads(current.read_text()),
        )
        assert previous["dataset"] == actual["dataset"]
        same(previous["tables"], actual["tables"])
        inputs[str(checkpoint)] = sha(checkpoint)
        inputs[str(current)] = sha(current)
        assert len(previous["prepared_sha256"]) == 11
        for old_path, digest in previous["prepared_sha256"].items():
            old_path = Path(old_path)
            new_path = NEW / "prepared" / old_path.parent.name / old_path.name
            assert sha(old_path) == digest
            assert sha(new_path) == actual["prepared_sha256"][str(new_path)]
            same(pd.read_pickle(old_path), pd.read_pickle(new_path))
            inputs[str(old_path)] = digest
            inputs[str(new_path)] = sha(new_path)
            tables += 1
        print("SCALAR/VECTOR ALL11FULLTABLES EXACT", previous["dataset"], flush=True)
    assert tables == len(checkpoints) * 11
    result = dict(
        all_passed=True,
        annotation_changes=False,
        classifier_fits=0,
        completed_scalar_datasets=len(checkpoints),
        exact_full_masked_table_comparisons=tables,
        unchanged_fitter_alignment_scope_function_AST_checks=3,
        original_scalar_fallback_AST_preserved=True,
        verifier_sha256=sha(Path(__file__)),
        inputs_sha256=inputs,
        note="Engineering ONLY. Same full physical pools,ALLpredictors,observed coordinates and reference metadata. Equivalent vectorized lookup,no learned prediction cache or scientific gate change. ALLcompleted scalarv1datasets checked,not sampling. Oldpartialfiles preserved;no old annotation fits.",
    )
    dump(NEW / "lookup_equivalence_receipt.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "inputs_sha256"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
