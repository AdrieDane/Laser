"""Exact ALLcompleted scalar/vector/fast table gate; not annotation performance."""

import ast
import copy
import json
import marshal
from pathlib import Path
import types

import pandas as pd

from context import WORKSPACE
from run_qe_standard_mode_experiment import sha, dump
from verify_qe_empirical_preparation import same

QE = WORKSPACE / "outputs/qe_optimization_20261022"
ORIGINAL = QE / "empirical_RT_candidate_preparation_v1"
VECTOR = QE / "empirical_RT_candidate_preparation_v2"
NEW = QE / "empirical_RT_candidate_preparation_v3"


def function(tree, name):
    return next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def tree(root, name):
    return ast.parse(
        (root / "source_snapshot/app/matlab_annotation" / name).read_text()
    )


def code_signature(value):
    """Own frozen pyc semantics,excluding ONLY embedded snapshot filename."""
    if isinstance(value, types.CodeType):
        fields = (
            "co_argcount",
            "co_posonlyargcount",
            "co_kwonlyargcount",
            "co_nlocals",
            "co_stacksize",
            "co_flags",
            "co_code",
            "co_consts",
            "co_names",
            "co_varnames",
            "co_freevars",
            "co_cellvars",
            "co_name",
            "co_qualname",
            "co_firstlineno",
            "co_linetable",
            "co_exceptiontable",
        )
        return tuple((name, code_signature(getattr(value, name))) for name in fields)
    if isinstance(value, tuple):
        return tuple(code_signature(v) for v in value)
    return value


def main():
    current = json.loads((NEW / "protocol.json").read_text())
    inputs, compared, total, dataset_counts = {}, set(), 0, {}
    # The RT fitter, scope checking and complete anchor/alignment algorithm are
    # unchanged. The original scalar lookup remains verbatim as a fallback.
    a, b = tree(ORIGINAL, "empirical_retention.py"), tree(NEW, "empirical_retention.py")
    for name in (
        "_verify_atlas_scope",
        "fit_empirical_retention",
        "predict_aligned_empirical_retention",
    ):
        assert ast.dump(function(a, name), include_attributes=False) == ast.dump(
            function(b, name), include_attributes=False
        )
    fallback = copy.deepcopy(function(b, "_predict_empirical_reference_scalar"))
    fallback.name = "predict_empirical_reference"
    assert ast.dump(
        function(a, "predict_empirical_reference"), include_attributes=False
    ) == ast.dump(fallback, include_attributes=False)
    # Remove ONLY the four empty-mass-interval prefilter assignments and restore
    # the old iteration domain. Then the ENTIRE physical proposal function,
    # including every acceptance inequality, must be AST-identical.
    original = function(
        tree(ORIGINAL, "empirical_candidates.py"), "prepare_empirical_candidates"
    )
    modified = copy.deepcopy(
        function(tree(NEW, "empirical_candidates.py"), "prepare_empirical_candidates")
    )
    names = {"_target_masses", "_target_lo", "_target_hi", "_mass_possible"}
    removed = []
    kept = []
    for node in modified.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in names
        ):
            removed.append(node.targets[0].id)
        else:
            kept.append(node)
    assert set(removed) == names and len(removed) == 4
    modified.body = kept
    loops = [
        n
        for n in modified.body
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Tuple)
        and [v.id for v in n.target.elts] == ["i", "target"]
    ]
    assert len(loops) == 1
    assert ast.unparse(loops[0].iter) == "targets.loc[_mass_possible].iterrows()"
    loops[0].iter = ast.parse("targets.iterrows()", mode="eval").body
    assert ast.dump(original, include_attributes=False) == ast.dump(
        modified, include_attributes=False
    )
    for root in (ORIGINAL, VECTOR, NEW):
        protocol = json.loads((root / "protocol.json").read_text())
        inputs[str(root / "protocol.json")] = sha(root / "protocol.json")
        for relative, digest in protocol["source_sha256"].items():
            assert sha(root / "source_snapshot" / relative) == digest
            if relative.endswith(".pyc"):
                old_code = marshal.loads(
                    (root / "source_snapshot" / relative).read_bytes()[16:]
                )
                new_code = marshal.loads(
                    (NEW / "source_snapshot" / relative).read_bytes()[16:]
                )
                assert code_signature(old_code) == code_signature(new_code), relative
                continue
            if "/tests/" not in relative and not relative.endswith(
                ("/empirical_retention.py", "/empirical_candidates.py")
            ):
                assert current["source_sha256"][relative] == digest, relative
        for path, digest in protocol["input_sha256"].items():
            assert sha(Path(path)) == digest
            assert current["input_sha256"][path] == digest
    for root, minimum in ((ORIGINAL, 12), (VECTOR, 3)):
        checkpoints = sorted((root / "checkpoints").glob("*.json"))
        assert len(checkpoints) >= minimum
        dataset_counts[root.name] = len(checkpoints)
        for checkpoint in checkpoints:
            current_checkpoint = NEW / "checkpoints" / checkpoint.name
            previous = json.loads(checkpoint.read_text())
            actual = json.loads(current_checkpoint.read_text())
            same(previous["tables"], actual["tables"])
            inputs[str(checkpoint)] = sha(checkpoint)
            inputs[str(current_checkpoint)] = sha(current_checkpoint)
            assert len(previous["prepared_sha256"]) == 11
            for path, digest in previous["prepared_sha256"].items():
                path = Path(path)
                new_path = NEW / "prepared" / path.parent.name / path.name
                assert sha(path) == digest
                assert sha(new_path) == actual["prepared_sha256"][str(new_path)]
                same(pd.read_pickle(path), pd.read_pickle(new_path))
                inputs[str(path)] = digest
                inputs[str(new_path)] = sha(new_path)
                compared.add((previous["dataset"], path.parent.name))
                total += 1
            print(
                "ALL11SCALAR/VECTOR/FAST FULLTABLES EXACT",
                root.name,
                previous["dataset"],
                flush=True,
            )
    result = dict(
        all_passed=True,
        annotation_changes=False,
        classifier_fits=0,
        completed_datasets_by_original_implementation=dataset_counts,
        exact_full_masked_table_comparisons=total,
        unique_full_masked_tables=len(compared),
        original_RTfitter_alignment_scope_AST_checks=3,
        original_scalar_lookup_AST_preserved=True,
        original_full_physical_gate_body_AST_preserved=True,
        verifier_sha256=sha(Path(__file__)),
        inputs_sha256=inputs,
        note="Engineering ONLY. Every completed older case checked,ALLphysical rows/observed coordinates/ALLpredictors/reference metadata EXACT. Totalcomparisons may include same query/mask frombotholdversions; unique count separate. Onlyparameter-free lookup/empty-mass-interval efficiency changes,no learnedprediction cache/scientific threshold/model change. Oldpartialoutputs preserved,ZEROoldscorefits.",
    )
    dump(NEW / "lookup_equivalence_receipt.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "inputs_sha256"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
