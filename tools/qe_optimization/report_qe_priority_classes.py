"""Companion prioritization view; does not change any benchmark denominator."""

import argparse
import json
from pathlib import Path

import pandas as pd


def priority_table(rows, minimum_manual=100, minimum_projects=3):
    required = {
        "dataset",
        "stream",
        "lipid_class",
        "manual_rows_before",
        "manual_rows_after",
    }
    if not required.issubset(rows):
        raise ValueError("Complete dataset/mode/class comparison required")
    if not rows.manual_rows_before.equals(rows.manual_rows_after):
        raise ValueError("Manual denominators must be identical")
    if rows.duplicated(["dataset", "stream", "lipid_class"]).any():
        raise ValueError("Unique dataset/mode/class rows required")
    labels = rows[rows.manual_rows_after.gt(0)].copy()
    labels["project_group"] = labels.dataset.str[:7]
    counts = labels.groupby(["stream", "lipid_class"], as_index=False).agg(
        manual_entries=("manual_rows_after", "sum"),
        independent_project_groups=("project_group", "nunique"),
        dataset_folders=("dataset", "nunique"),
    )
    counts["optimization_priority"] = "defer_sparse"
    priority = counts.manual_entries.ge(
        minimum_manual
    ) & counts.independent_project_groups.ge(minimum_projects)
    counts.loc[priority, "optimization_priority"] = "prioritize"
    counts.loc[counts.lipid_class.eq("X4"), "optimization_priority"] = (
        "user_excluded_X4"
    )
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args()
    rows = pd.read_csv(args.comparison / "by_dataset_mode_class_before_after.csv")
    assert rows.dataset.nunique() == 33
    assert rows.groupby(["dataset", "stream"]).ngroups == 132
    counts = priority_table(rows)
    aggregate = pd.read_csv(args.comparison / "by_mode_class_before_after.csv")
    result = counts.merge(
        aggregate, on=["stream", "lipid_class"], validate="one_to_one"
    )
    assert result.manual_entries.equals(result.manual_rows_after)
    out = args.comparison / "optimization_priority"
    out.mkdir(exist_ok=True)
    result.to_csv(out / "by_mode_class_before_after.csv", index=False)
    summary = dict(
        minimum_manual_entries=100,
        minimum_independent_project_groups=3,
        purpose="Optimization priority only,not revised success criteria or benchmark exclusions. "
        "Full all-class132row comparison remains authoritative; sparse classes stay visible. "
        "Only X4 is user-excluded from target scope. Extras are unconfirmed,not false positives/FDR.",
        groups={},
    )
    for name, part in result.groupby("optimization_priority"):
        record = dict(
            class_modes=len(part), manual_entries=int(part.manual_entries.sum())
        )
        for side in ("before", "after"):
            n = int(part["manual_rows_" + side].sum())
            correct = int(part["manual_same_peak_" + side].sum())
            record["correct_" + side] = correct
            record["percent_" + side] = 100 * correct / n if n else None
            record["unconfirmed_extras_" + side] = int(
                part["laser_not_in_manual_list_" + side].sum()
            )
            record["different_feature_" + side] = int(
                part["laser_same_manual_lipid_different_peak_" + side].sum()
            )
            record["class_modes_below_90_" + side] = int(
                part["correct_manual_percent_" + side].lt(90).sum()
            )
        summary["groups"][name] = record
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
