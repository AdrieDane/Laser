"""Final049 permission gate; incomplete actual checker/source set FAILS closed."""

import json
from concurrent.futures import ProcessPoolExecutor
import subprocess
import sys
from pathlib import Path

from context import REPO
from frozen_resources import configure_frozen_resources
import matrix_experiment_data as data
import matrix_experiment_bindings as bindings


def main():
    configure_frozen_resources(REPO)
    assert not data.OUT.exists(), "No actual049 outputs before final permission"
    assert all(Path(__file__).with_name(n).is_file() for n in bindings.SOURCES)
    assert set(bindings.environment()["thread_environment"].values()) == {"1"}
    inputs = bindings.source_inputs()
    stage = json.loads((bindings.STAGE / "receipt.json").read_text())
    assert (
        stage["all_passed"]
        and stage["original_training_query_uses"] == 528
        and stage["original_score_checks"] == 165
        and stage["whole041_control_datasets"] == 33
    )
    assert stage["actual049_fit_permission"] is False
    tests = [
        "test_matrix_mode_model.py",
        "test_matrix_pipeline_assembly.py",
        "test_matrix_operating_points.py",
        "test_matrix_experiment_controls.py",
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *[str(Path(__file__).with_name(n)) for n in tests],
            "-q",
            "--tb=short",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    print(result.stdout, flush=True)
    assert result.returncode == 0, result.stderr
    from verify_qe_matrix_context_experiment import legacy_control

    datasets, _ = data.inventory()
    with ProcessPoolExecutor(max_workers=2) as executor:
        controls = list(executor.map(legacy_control, datasets))
    assert len(controls) == 33 and set(controls) == set(datasets)
    assert inputs == bindings.source_inputs()
    data.PREFLIGHT.mkdir(exist_ok=False)
    (data.PREFLIGHT / "software_tests.txt").write_text(result.stdout + result.stderr)
    data.nested.pilot.dump(
        data.PREFLIGHT / "receipt.json",
        dict(
            all_passed=True,
            actual049_fit_permission=True,
            actual_QE_classifier_fits=0,
            synthetic_native_model_tests_executed=True,
            final_independent_whole041_control_datasets=33,
            source_input_sha256=inputs,
            numeric_runtime=bindings.environment(),
            planned_actual_verifier_sha256=data.nested.pilot.sha(bindings.CHECKER),
            stage_a_receipt_sha256=data.nested.pilot.sha(
                bindings.STAGE / "receipt.json"
            ),
            note="Full363+528knownlabels/100native matrices/165oldscores/33whole041controls andsoftwaretests complete; actualdriver/master sources bound. Plannedactualverifier andallsourcefreeze nowrequired unchanged. No actualnewQE scores/cuts/annotations yet.",
        ),
    )
    print("049 FINAL PREFLIGHT ALLPASS; ACTUAL FITS NOW PERMITTED", flush=True)


if __name__ == "__main__":
    main()
