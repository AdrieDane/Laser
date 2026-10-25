"""Engineering-only deterministic cache equivalence; not annotation accuracy."""

import ast
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from context import WORKSPACE
from frozen_resources import configure_frozen_resources

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OLD = QE / "standard_aligned_candidate_preparation_v1"
NEW = QE / "standard_aligned_candidate_preparation_v2"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    configure_frozen_resources(NEW / "source_snapshot")
    from annotator.composition_ranges import (
        canonical_lipid_class,
        ClassCompositionRange,
    )
    from matlab_annotation.chemical_targets import envelope_range_settings

    before = json.loads((OLD / "protocol.json").read_text())
    after = json.loads((NEW / "protocol.json").read_text())
    changed = []
    for rel, digest in before["source_sha256"].items():
        assert sha(OLD / "source_snapshot" / rel) == digest
        if rel.startswith("app/tests/") or rel.endswith(".pyc"):
            continue
        if digest != after["source_sha256"].get(rel):
            changed.append(rel)
    assert changed == ["app/annotator/composition_ranges.py"], changed
    assert set(after["source_sha256"]) - set(before["source_sha256"]) == {
        "app/tests/test_composition_range_cache.py"
    }
    old_source = OLD / "source_snapshot/app/annotator/composition_ranges.py"
    tree = ast.parse(old_source.read_text())
    original_function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "canonical_lipid_class"
    )
    original_class = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "CompositionRangeSettings"
    )
    original_lookup = next(
        n
        for n in original_class.body
        if isinstance(n, ast.FunctionDef) and n.name == "class_lookup"
    )
    namespace = dict(re=re, ClassCompositionRange=ClassCompositionRange)
    exec(
        compile(
            ast.Module(body=[original_function, original_lookup], type_ignores=[]),
            str(old_source),
            "exec",
        ),
        namespace,
    )
    inputs = {
        str(OLD / "protocol.json"): sha(OLD / "protocol.json"),
        str(NEW / "protocol.json"): sha(NEW / "protocol.json"),
    }
    chemical_strings, datasets, tables = set(), 0, 0
    for receipt in sorted((OLD / "checkpoints").glob("*.json")):
        old = json.loads(receipt.read_text())
        current_receipt = NEW / "checkpoints" / receipt.name
        current = json.loads(current_receipt.read_text())
        inputs[str(receipt)], inputs[str(current_receipt)] = (
            sha(receipt),
            sha(current_receipt),
        )
        assert len(old["prepared_sha256"]) == len(current["prepared_sha256"]) == 11
        for path, digest in old["prepared_sha256"].items():
            path = Path(path)
            new_path = NEW / "prepared" / path.parent.name / path.name
            assert sha(path) == digest
            assert sha(new_path) == current["prepared_sha256"][str(new_path)]
            a, b = pd.read_pickle(path), pd.read_pickle(new_path)
            pd.testing.assert_frame_equal(a["pool"], b["pool"], check_exact=True)
            pd.testing.assert_frame_equal(
                a["evidence"], b["evidence"], check_exact=True
            )
            assert a.keys() == b.keys()
            for key in a.keys() - {"pool", "evidence"}:
                assert a[key] == b[key], key
            chemical_strings.update(
                a["pool"][["class", "name"]].itertuples(index=False, name=None)
            )
            inputs[str(path)], inputs[str(new_path)] = digest, sha(new_path)
            tables += 1
        datasets += 1
        print("EXACT UNCACHED/CACHED ALL11TABLE PARITY", receipt.stem, flush=True)
    assert datasets >= 6 and tables == 11 * datasets
    for label, species in chemical_strings:
        assert canonical_lipid_class(label, species=species) == namespace[
            "canonical_lipid_class"
        ](label, species=species)
    lookups = 0
    for path in sorted((QE / "nested_reference_development_v2").glob("exclude_*.json")):
        bundle = json.loads(path.read_text())
        inputs[str(path)] = sha(path)
        for stream in ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"):
            settings = envelope_range_settings(bundle["composition_envelope"], stream)
            expected = namespace["class_lookup"](settings)
            assert settings.class_lookup() == expected
            settings.class_lookup().clear()
            assert settings.class_lookup() == expected
            lookups += 1
    assert lookups == 100
    result = dict(
        all_passed=True,
        annotation_accuracy_claim=False,
        changed_non_test_sources=changed,
        exact_full_dataset_parity_checks=datasets,
        exact_pool_evidence_and_metadata_table_checks=tables,
        original_AST_canonical_class_species_checks=len(chemical_strings),
        original_AST_nested_range_dictionary_checks=lookups,
        inputs_sha256=inputs,
        verifier_sha256=sha(Path(__file__)),
        note="Deterministic text/range cache only. Lookup key contains ALL supplied immutable class rules, including numerical bounds and flags; return dictionaries are fresh. No atlas/standard model, RTprediction, raw measurement, score, label or annotation decision cached. Original v1preparation preserved. Full v2raw replay still required for all363tables. Not annotation improvement.",
    )
    (NEW / "cache_parity_receipt.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in result.items() if k != "inputs_sha256"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
