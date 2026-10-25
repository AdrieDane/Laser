"""Checkpoint the UNCHANGED036full-table verifier, not annotation/model caching.

Only this wrapper's own hash-bound, fully completed case receipts may resume.
Old stdout/partial scratch copies are NEVER used as evidence. No frozen checker,
preparer, scientific source, inputs or experiment protocol is modified.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import verify_qe_empirical_preparation_fast as checker

QE, PREPARED, EMPIRICAL, OLD = (
    checker.QE,
    checker.PREPARED,
    checker.EMPIRICAL,
    checker.OLD,
)
OUT = QE / "empirical_RT_preparation_verification_v1"
sha, dump = checker.sha, checker.dump


def digest_manifest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def completed_case(path, dataset, identity):
    """Truncated, stale, wrong-case or non-PASS receipts never count."""
    try:
        receipt = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (
        receipt.get("all_passed") is True
        and receipt.get("dataset") == dataset
        and receipt.get("verification_identity_sha256") == identity
        and receipt.get("full_masked_tables") == 11
        and receipt.get("fresh_standard_mask_checks") == 11
    )


def verify_case(dataset, identity):
    path = OUT / "completed_cases" / (dataset + ".json")
    if completed_case(path, dataset, identity):
        return dataset, "hash-bound completed check reused"
    count = checker.verify_dataset(dataset)
    assert count == 11
    dump(
        path,
        dict(
            all_passed=True,
            dataset=dataset,
            verification_identity_sha256=identity,
            full_masked_tables=count,
            fresh_standard_mask_checks=count,
            original_checker_sha256=sha(Path(checker.__file__)),
            wrapper_sha256=sha(Path(__file__)),
        ),
    )
    return dataset, "fresh complete reconstruction"


def input_manifest(projects):
    assert (PREPARED / "runner_source.py").read_bytes() == Path(__file__).with_name(
        "prepare_qe_empirical_candidates_fast.py"
    ).read_bytes()
    assert (EMPIRICAL / "runner_source.py").read_bytes() == Path(__file__).with_name(
        "build_qe_empirical_rt_references.py"
    ).read_bytes()
    ready = json.loads((PREPARED / "summary.json").read_text())
    assert ready["all_complete"] and ready["dataset_mask_tables"] == 363
    files = {
        Path(__file__),
        Path(checker.__file__),
        Path(__file__).with_name("frozen_resources.py"),
        Path(__file__).with_name("context.py"),
        Path(__file__).with_name("run_qe_standard_mode_experiment.py"),
        QE / "baseline/run_manifest.json",
    }
    source_count = input_count = 0
    for root in (EMPIRICAL, PREPARED):
        files.update(
            (root / "protocol.json", root / "summary.json", root / "runner_source.py")
        )
        protocol = json.loads((root / "protocol.json").read_text())
        for relative, digest in protocol["source_sha256"].items():
            path = root / "source_snapshot" / relative
            assert sha(path) == digest
            files.add(path)
            source_count += 1
        for name, digest in protocol["input_sha256"].items():
            path = Path(name)
            assert sha(path) == digest
            files.add(path)
            input_count += 1
    for dataset in projects:
        files.update(
            (
                QE / "baseline/runs" / dataset / "full_annotation.pkl",
                QE
                / "nested_standard_development_v1/automatic_inputs"
                / (dataset + ".pkl"),
            )
        )
        path = PREPARED / "checkpoints" / (dataset + ".json")
        files.add(path)
        receipt = json.loads(path.read_text())
        assert len(receipt["prepared_sha256"]) == 11
        for name, digest in receipt["prepared_sha256"].items():
            path = Path(name)
            assert sha(path) == digest
            files.update(
                (
                    path,
                    OLD / "measurements" / path.parent.name / (dataset + ".pkl"),
                    QE
                    / "nested_standard_development_v1"
                    / (path.parent.name + ".json"),
                )
            )
    for path in EMPIRICAL.glob("exclude_*.json"):
        files.update((path, QE / "nested_reference_development_v2" / path.name))
    files.add(QE / "rt_atlas_development/observations.pkl")
    return {str(p): sha(p) for p in sorted(files)}, source_count, input_count


def main():
    projects = json.loads((QE / "baseline/run_manifest.json").read_text())["datasets"]
    assert len(projects) == len(set(projects)) == 33
    manifest, source_count, input_count = input_manifest(projects)
    identity = digest_manifest(manifest)
    OUT.mkdir(exist_ok=True)
    (OUT / "completed_cases").mkdir(exist_ok=True)
    binding_path = OUT / "binding.json"
    binding = dict(verification_identity_sha256=identity, input_sha256=manifest)
    if binding_path.exists():
        assert json.loads(binding_path.read_text()) == binding, (
            "Inputs/checker changed; use a NEW verification directory"
        )
    else:
        dump(binding_path, binding)
    # Refit all25 numerical references even on a resumed invocation.
    references = checker.verify_references()
    with ProcessPoolExecutor(max_workers=2) as executor:
        pending = [
            executor.submit(verify_case, dataset, identity) for dataset in projects
        ]
        for future in as_completed(pending):
            print("036 RESUMABLE CHECK", *future.result(), flush=True)
    assert all(
        completed_case(OUT / "completed_cases" / (d + ".json"), d, identity)
        for d in projects
    )
    # Guard against input/source changes DURING the long reconstruction too.
    assert all(sha(Path(path)) == digest for path, digest in manifest.items())
    result = dict(
        all_passed=True,
        annotation_changes=False,
        classifier_fits=0,
        source_sha256=sha(Path(checker.__file__)),
        resumable_wrapper_sha256=sha(Path(__file__)),
        verification_binding_sha256=sha(binding_path),
        verification_identity_sha256=identity,
        numerical_reference_exact_reproductions=references,
        full_masked_table_reproductions=363,
        independent_standard_mask_measurement_reproductions=363,
        datasets=33,
        source_hash_checks=source_count,
        input_hash_checks=input_count,
        additional_bound_verification_files=len(manifest),
        note="UNCHANGED originalverify_dataset executed forALL33queries/363fulltables andfreshSTDmasks;ALL25numericalrefsrefitted. Case receipts written ONLYAFTERcompletechecks, bound toALLsource/input/prepared/checkpoint/measurement hashes. No oldstdout/partialcopies reused. EngineeringONLY,notannotationaccuracy/FDR;no scientificchanges.",
    )
    dump(PREPARED / "integrity_receipt.json", result)
    dump(OUT / "integrity_receipt.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
