"""046 ALL33 evaluation/primary+endogenous reports/fullmaster, no nextcycle."""

import json
from pathlib import Path
import subprocess
import sys

from context import WORKSPACE

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "experiment_046"
TOOLS = Path(__file__).parent


def run(tool, *args):
    print("046 POSTPROCESS", tool, flush=True)
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
    run(
        "report_qe_endogenous_companion.py",
        "--before",
        QE / "experiment_041",
        "--experiment",
        OUT,
        "--output",
        comparison / "endogenous_and_spiked_companion",
    )
    run("verify_qe_bmp_continuous_experiment.py")
    print(
        "046 ALL33BENCHMARK/INTEGRITY COMPLETE;FULL132chat/endogenouscompanion/wholedecision BEFORE046",
        flush=True,
    )


if __name__ == "__main__":
    main()
