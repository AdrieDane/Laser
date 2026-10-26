"""048 all33 evaluation, independent verification and prospective decision.

Manual invocation only AFTER all25fits/all5calibrations/all33blind exports.
No next experiment launched; complete132CHATtable still required afterward.
"""

from pathlib import Path
import json
import subprocess
import sys

import catboost_experiment_data as data


def run(tool, *args):
    print("048 POSTPROCESS", tool, flush=True)
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name(tool)), *map(str, args)],
        check=True,
    )


def main():
    receipt = json.loads((data.OUT / "producer_complete.json").read_text())
    assert receipt["all_blind_outputs_complete"] and receipt["datasets"] == 33
    assert receipt["protocol_sha256"] == data.sha(data.OUT / "protocol.json")
    datasets, _ = data.inventory()
    for d in datasets:
        assert receipt["dataset_complete_sha256"][d] == data.sha(
            data.OUT / "runs" / d / "complete.json"
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
    run("verify_qe_catboost_experiment.py")
    run("report_qe_catboost_decision.py")
    print(
        "048 EVALUATION/INTEGRITY/DECISION COMPLETE; FULL132CHAT BEFORE NEXT ACTUAL EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
