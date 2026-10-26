"""Manually invoke ONLY after ALL33blind049outputs; no auto-next experiment."""

import json
from pathlib import Path
import subprocess
import sys

import matrix_experiment_data as data


def run(tool, *args):
    print("049 POSTPROCESS", tool, flush=True)
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name(tool)), *map(str, args)],
        check=True,
    )


def main():
    receipt = json.loads((data.OUT / "producer_complete.json").read_text())
    assert receipt["all_blind_outputs_complete"] and receipt["datasets"] == 33
    assert receipt["protocol_sha256"] == data.nested.pilot.sha(
        data.OUT / "protocol.json"
    )
    datasets, _ = data.inventory()
    for dataset in datasets:
        assert receipt["dataset_complete_sha256"][dataset] == data.nested.pilot.sha(
            data.OUT / "runs" / dataset / "complete.json"
        )
    run(
        "evaluate_qe_baseline.py",
        data.OUT,
        "--manual-reference",
        data.QE / "manual_reference_v2",
    )
    run("report_qe_experiment.py", "--experiment", data.OUT)
    run("report_qe_experiment.py", "--experiment", data.OUT, "--before", data.PREVIOUS)
    comparison = data.OUT / "comparison_with_experiment_041"
    run(
        "report_qe_target_scope.py", "--comparison", comparison, "--exclude-class", "X4"
    )
    run("report_qe_priority_classes.py", "--comparison", comparison)
    run(
        "report_qe_endogenous_companion.py",
        "--before",
        data.PREVIOUS,
        "--experiment",
        data.OUT,
        "--output",
        comparison / "endogenous_and_spiked_companion",
    )
    run("verify_qe_matrix_context_experiment.py")
    run("report_qe_matrix_context_decision.py")
    print(
        "049 EVALUATION/MASTER/WHOLEDECISION COMPLETE; FULL132CHAT BEFORE NEXT ANNOTATION EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
