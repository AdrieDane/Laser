"""042 complete-output evaluation/report/audit; no next experiment/publication."""

import json
from pathlib import Path
import subprocess
import sys

from context import WORKSPACE

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "experiment_042"
TOOLS = Path(__file__).parent


def run(tool, *args):
    print("042 POSTPROCESS", tool, flush=True)
    subprocess.run([sys.executable, str(TOOLS / tool), *map(str, args)], check=True)


def main():
    datasets = json.loads((OUT / "run_manifest.json").read_text())["datasets"]
    assert len(datasets) == 33 and all(
        (OUT / "runs" / d / "complete.json").exists() for d in datasets
    )
    run(
        "evaluate_qe_baseline.py", OUT, "--manual-reference", QE / "manual_reference_v2"
    )
    run("report_qe_experiment.py", "--experiment", OUT)
    run(
        "report_qe_experiment.py",
        "--experiment",
        OUT,
        "--before",
        QE / "experiment_041",
    )
    comparison = OUT / "comparison_with_experiment_041"
    run(
        "report_qe_target_scope.py", "--comparison", comparison, "--exclude-class", "X4"
    )
    run("report_qe_priority_classes.py", "--comparison", comparison)
    run("verify_qe_feature_label_experiment.py")
    print(
        "042 ALL33BENCHMARK/INTEGRITY COMPLETE;FULL132chat/wholedecision BEFORE043",
        flush=True,
    )


if __name__ == "__main__":
    main()
