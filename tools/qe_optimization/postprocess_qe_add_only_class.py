"""Evaluate/report/audit complete041; never publishes or starts another cycle."""

import json
from pathlib import Path
import subprocess
import sys

from context import WORKSPACE

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "experiment_041"
TOOLS = Path(__file__).parent


def run(tool, *args):
    print("041 POSTPROCESS", tool, flush=True)
    subprocess.run([sys.executable, str(TOOLS / tool), *map(str, args)], check=True)


def main():
    datasets = json.loads((OUT / "run_manifest.json").read_text())["datasets"]
    assert len(datasets) == 33
    assert all((OUT / "runs" / d / "complete.json").exists() for d in datasets)
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
    run("verify_qe_add_only_class_experiment.py")
    print(
        "041 ALL33EVAL/AUDITDONE; full132chat table/wholedecision before042", flush=True
    )


if __name__ == "__main__":
    main()
