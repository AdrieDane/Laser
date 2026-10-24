"""Companion target-scope report; never rewrite historical all-manual reports."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = [
    "manual_rows",
    "manual_same_peak",
    "unresolved_manual_rows",
    "laser_not_in_manual_list",
    "laser_same_manual_lipid_different_peak",
    "unique_laser_annotations",
]


def target_tables(all_classes, excluded):
    excluded = set(excluded)
    if not excluded or not excluded.issubset(set(all_classes.lipid_class)):
        raise ValueError("Explicit existing classes must be excluded")
    if all_classes.duplicated(["dataset", "stream", "lipid_class"]).any():
        raise ValueError("Dataset/mode/class rows must be unique")
    kept = all_classes.loc[~all_classes.lipid_class.isin(excluded)].copy()
    tables = {}
    for name, keys in {
        "by_dataset_mode": ["dataset", "stream"],
        "by_mode": ["stream"],
        "by_mode_class": ["stream", "lipid_class"],
    }.items():
        columns = [f + "_" + s for s in ("before", "after") for f in FIELDS]
        table = kept.groupby(keys, as_index=False)[columns].sum()
        assert table.manual_rows_before.equals(table.manual_rows_after)
        for side in ("before", "after"):
            table["correct_manual_percent_" + side] = (
                100
                * table["manual_same_peak_" + side]
                / table["manual_rows_" + side].replace(0, np.nan)
            )
        table["correct_gain"] = (
            table.manual_same_peak_after - table.manual_same_peak_before
        )
        table["extra_change"] = (
            table.laser_not_in_manual_list_after - table.laser_not_in_manual_list_before
        )
        tables[name] = table
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--exclude-class", action="append", required=True)
    args = parser.parse_args()
    source = args.comparison / "by_dataset_mode_class_before_after.csv"
    original = pd.read_csv(source)
    tables = target_tables(original, args.exclude_class)
    assert len(tables["by_dataset_mode"]) == 132
    assert tables["by_dataset_mode"].dataset.nunique() == 33
    out = args.comparison / (
        "target_scope_without_" + "_".join(sorted(args.exclude_class))
    )
    out.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / (name + "_before_after.csv"), index=False)
    pooled = tables["by_mode"]
    cls = tables["by_mode_class"]
    result = dict(
        excluded_classes=sorted(args.exclude_class),
        datasets=33,
        dataset_modes=132,
        all_manual_rows=int(original.manual_rows_after.sum()),
        excluded_manual_rows=int(
            original.loc[
                original.lipid_class.isin(args.exclude_class), "manual_rows_after"
            ].sum()
        ),
        target_manual_rows=int(pooled.manual_rows_after.sum()),
        note="Companion target-scope view,NOT an annotation improvement or reference "
        "relabeling. Historical all-manual reports/inputs unchanged. X4 was explicitly "
        "excluded by the user on24October2026. Unconfirmed extras,not false positives/FDR.",
    )
    for side in ("before", "after"):
        n = int(pooled["manual_rows_" + side].sum())
        c = int(pooled["manual_same_peak_" + side].sum())
        result["correct_" + side] = c
        result["percent_" + side] = 100 * c / n
        result["extras_" + side] = int(pooled["laser_not_in_manual_list_" + side].sum())
        result["different_feature_" + side] = int(
            pooled["laser_same_manual_lipid_different_peak_" + side].sum()
        )
        result["unresolved_manual_" + side] = int(
            pooled["unresolved_manual_rows_" + side].sum()
        )
        result["class_modes_with_manual_" + side] = int(
            cls["manual_rows_" + side].gt(0).sum()
        )
        result["class_modes_below_90_" + side] = int(
            (
                cls["manual_rows_" + side].gt(0)
                & cls["correct_manual_percent_" + side].lt(90)
            ).sum()
        )
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
