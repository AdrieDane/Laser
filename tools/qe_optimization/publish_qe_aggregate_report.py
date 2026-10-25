"""Copy ONLY audited aggregate experiment reports, never models/raw feature IDs.

Explicit opt-in destination/decision; no Git operations, no annotation changes.
Private input paths and per-feature comparisons are intentionally not exported.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


AGGREGATES = (
    "by_dataset_mode_before_after.csv",
    "by_dataset_mode_class_before_after.csv",
    "by_mode_before_after.csv",
    "by_mode_class_before_after.csv",
)
PUBLIC_COLUMNS = {
    "dataset",
    "stream",
    "lipid_class",
    "correct_gain",
    "extra_change",
    "manual_entries",
    "dataset_folders",
    "independent_project_groups",
    "optimization_priority",
    *(
        f + "_" + side
        for side in ("before", "after")
        for f in (
            "manual_rows",
            "manual_same_peak",
            "unresolved_manual_rows",
            "laser_not_in_manual_list",
            "laser_same_manual_lipid_different_peak",
            "unique_laser_annotations",
            "correct_manual_percent",
        )
    ),
}


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def public_json(value):
    if isinstance(value, dict):
        return {
            k: public_json(v)
            for k, v in value.items()
            if k
            not in {
                "input_sha256",
                "source_input_sha256",
                "resource_receipt",
                "artifact_sha256",
                "fold_results",
                "manual_reference_identity_inputs_sha256",
            }
        }
    if isinstance(value, list):
        return [public_json(v) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        return Path(value).name
    return value


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def copy_aggregate(source, destination):
    table = pd.read_csv(source)
    if not set(table).issubset(PUBLIC_COLUMNS):
        raise ValueError(
            f"Unapproved non-aggregate columns in {source.name}: {set(table) - PUBLIC_COLUMNS}"
        )
    if "dataset" in table and (
        table.dataset.nunique() != 33
        or table.groupby(["dataset", "stream"]).ngroups != 132
    ):
        raise ValueError("Full33dataset/132mode report required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def publish(experiment, before, destination, decision):
    ready = json.loads((experiment / "reports_v2/summary.json").read_text())
    receipt = json.loads((experiment / "integrity_receipt.json").read_text())
    if (
        not ready["complete"]
        or ready["datasets_evaluated"] != 33
        or not receipt["all_passed"]
    ):
        raise ValueError("Complete benchmark AND passing integrity receipt required")
    if not (destination / "decision.txt").is_file():
        raise ValueError("Write the human-readable retain/reject decision first")
    scopes = {
        "": experiment / "full_comparison",
        "versus_" + before.name.removeprefix("experiment_"): experiment
        / ("comparison_with_" + before.name),
    }
    for relative, source in list(scopes.items()):
        if relative:
            scopes[relative + "/target_scope_without_X4"] = (
                source / "target_scope_without_X4"
            )
            scopes[relative + "/optimization_priority"] = (
                source / "optimization_priority"
            )
    for relative, source in scopes.items():
        folder = destination / relative
        folder.mkdir(parents=True, exist_ok=True)
        copied = 0
        for name in AGGREGATES:
            if (source / name).is_file():
                copy_aggregate(source / name, folder / name)
                copied += 1
        if copied == 0:
            raise ValueError(f"No aggregate reports in {source}")
        summary = json.loads((source / "summary.json").read_text())
        write_json(folder / "summary.json", public_json(summary))
        if (source / "performance_before_after.txt").is_file():
            shutil.copy2(
                source / "performance_before_after.txt",
                folder / "performance_before_after.txt",
            )
    write_json(destination / "integrity_receipt.json", public_json(receipt))
    protocol = public_json(json.loads((experiment / "protocol.json").read_text()))
    protocol.update(
        decision=decision,
        retained_research_reference=before.name
        if decision == "rejected"
        else experiment.name,
    )
    write_json(destination / "protocol_summary.json", protocol)
    for relative, source in scopes.items():
        if (source / "summary.json").exists():
            assert sha(source / "by_mode_class_before_after.csv") == sha(
                destination / relative / "by_mode_class_before_after.csv"
            )
    print(f"Published audited aggregate reports to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--decision", choices=("retained_research_only", "rejected"), required=True
    )
    args = parser.parse_args()
    publish(
        args.experiment.resolve(),
        args.before.resolve(),
        args.destination.resolve(),
        args.decision,
    )
