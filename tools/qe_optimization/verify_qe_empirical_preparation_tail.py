"""Scheduling ONLY: verify tail cases with unchanged hash-bound case function.

Main coordinator remains responsible for ALL33/363master verification. These
checks only populate the same validated case receipts, so its later queued
cases can safely reuse them. No annotation/scientific/checker source changes.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import verify_qe_empirical_preparation_resumable as original


def main():
    binding_path = original.OUT / "binding.json"
    binding = json.loads(binding_path.read_text())
    manifest = binding["input_sha256"]
    identity = original.digest_manifest(manifest)
    assert identity == binding["verification_identity_sha256"]
    assert all(original.sha(Path(p)) == h for p, h in manifest.items())
    own_hash = original.sha(Path(__file__))
    projects = json.loads((original.QE / "baseline/run_manifest.json").read_text())[
        "datasets"
    ]
    # Order-based scheduling, never choosing cases by annotation performance.
    tail = list(reversed(projects[-8:]))
    with ProcessPoolExecutor(max_workers=2) as executor:
        jobs = [
            executor.submit(original.verify_case, dataset, identity) for dataset in tail
        ]
        for future in as_completed(jobs):
            print(
                "SUPPLEMENTAL SCHEDULING: SAME FULL CASE CHECK",
                *future.result(),
                flush=True,
            )
    assert all(
        original.completed_case(
            original.OUT / "completed_cases" / (d + ".json"), d, identity
        )
        for d in tail
    )
    assert all(original.sha(Path(p)) == h for p, h in manifest.items())
    assert original.sha(Path(__file__)) == own_hash
    result = dict(
        all_selected_case_checks_passed=True,
        complete_experiment_integrity=False,
        annotation_changes=False,
        classifier_fits=0,
        datasets=len(tail),
        full_masked_table_checks=11 * len(tail),
        source_sha256=own_hash,
        original_case_wrapper_sha256=original.sha(Path(original.__file__)),
        original_verification_binding_sha256=original.sha(binding_path),
        verification_identity_sha256=identity,
        case_receipt_sha256={
            d: original.sha(original.OUT / "completed_cases" / (d + ".json"))
            for d in tail
        },
        note="SchedulingONLY,NOTmasterPASS. CallsUNCHANGEDoriginal.verify_case/verify_dataset,ALLsource/input hashes checked before/after. Oldstdout neverused. Maincoordinator stillrequiresALL33/363cases,25refreplays,allsource/inputfinalhashchecks. No extraindependentdatasets or annotation accuracy claims.",
    )
    original.dump(original.OUT / "supplemental_scheduling_receipt.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
