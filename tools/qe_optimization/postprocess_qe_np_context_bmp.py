"""Wait for THIS040producer,then full evaluation/reports/integrity;no new fits.

Does not publish a decision or start another annotation experiment. The complete
132-row user-facing comparison and whole-method decision remain mandatory.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from context import WORKSPACE

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "experiment_040_v2"
TOOLS = Path(__file__).parent


def run(name, *arguments):
    print("040 POSTPROCESS", name, flush=True)
    subprocess.run(
        [sys.executable, str(TOOLS / name), *map(str, arguments)], check=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer-pid", type=int, required=True)
    args = parser.parse_args()
    assert args.producer_pid > 1
    datasets = json.loads((OUT / "run_manifest.json").read_text())["datasets"]
    assert len(datasets) == 33
    while not all((OUT / "runs" / d / "complete.json").is_file() for d in datasets):
        try:
            os.kill(args.producer_pid, 0)
        except ProcessLookupError:
            raise RuntimeError(
                "040producer stopped before ALL33outputs;do not infer complete results"
            ) from None
        time.sleep(5)
    run(
        "evaluate_qe_baseline.py", OUT, "--manual-reference", QE / "manual_reference_v2"
    )
    run("report_qe_experiment.py", "--experiment", OUT)
    run(
        "report_qe_experiment.py",
        "--experiment",
        OUT,
        "--before",
        QE / "experiment_029",
    )
    comparison = OUT / "comparison_with_experiment_029"
    run(
        "report_qe_target_scope.py", "--comparison", comparison, "--exclude-class", "X4"
    )
    run("report_qe_priority_classes.py", "--comparison", comparison)
    run("verify_qe_np_context_bmp_experiment.py")
    print(
        "040 ALL33BENCHMARK/INTEGRITY COMPLETE;report FULL132chat+wrongtotals/decision BEFORE041",
        flush=True,
    )


if __name__ == "__main__":
    main()
