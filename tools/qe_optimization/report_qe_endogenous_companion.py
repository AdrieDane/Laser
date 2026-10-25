"""Separate endogenous/spiked-reference views; NEVER replace primary history."""

import argparse
import json
from pathlib import Path

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
from report_qe_experiment import pair


def endpoint(root, datasets, *, standards):
    from standard_lipidomics.validation import (
        AnnotationComparison,
        summarize_comparison,
    )

    manuals, calls = [], []
    for d in datasets:
        directory = root / "evaluation_v2" / d
        m, a = (
            pd.read_pickle(directory / name)
            for name in ("manual_comparison.pkl", "laser_comparison.pkl")
        )
        assert (
            m.internal_standard.isin([True, False]).all()
            and a.internal_standard.isin([True, False]).all()
        )
        manuals.append(
            m.loc[m.internal_standard.eq(standards) & ~m.lipid_class.eq("X4")]
        )
        calls.append(a.loc[a.internal_standard.eq(standards) & ~a.lipid_class.eq("X4")])
    comparison = AnnotationComparison(
        pd.concat(manuals, ignore_index=True), pd.concat(calls, ignore_index=True)
    )
    tables = {}
    for name, keys in {
        "by_dataset_mode": ["dataset", "stream"],
        "by_dataset_mode_class": ["dataset", "stream", "lipid_class"],
        "by_mode": ["stream"],
        "by_mode_class": ["stream", "lipid_class"],
    }.items():
        t = summarize_comparison(comparison, group_by=tuple(keys))
        t["correct_manual_percent"] = (
            100 * t.manual_same_peak / t.manual_rows.replace(0, float("nan"))
        )
        tables[name] = (t, keys)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_frozen_resources(REPO)
    datasets = json.loads((args.before / "run_manifest.json").read_text())["datasets"]
    assert len(datasets) == 33 and set(datasets) == set(
        json.loads((args.experiment / "run_manifest.json").read_text())["datasets"]
    )
    for d in datasets:
        identity = [
            json.loads(
                (root / "evaluation_v2" / d / "evaluation_identity.json").read_text()
            )
            for root in (args.before, args.experiment)
        ]
        assert identity[0]["manual_sha256"] == identity[1]["manual_sha256"]
    args.output.mkdir(exist_ok=False, parents=True)
    summaries = {}
    for standard, name in ((False, "endogenous"), (True, "spiked_references")):
        before, after = (
            endpoint(args.before, datasets, standards=standard),
            endpoint(args.experiment, datasets, standards=standard),
        )
        directory = args.output / name
        directory.mkdir()
        results = {}
        for table, (left, keys) in before.items():
            combined = pair(left, after[table][0], keys)
            combined.to_csv(directory / (table + "_before_after.csv"), index=False)
            results[table] = combined
        modes = results["by_mode"]
        assert len(results["by_dataset_mode"]) == 132
        result = {"manual_entries": int(modes.manual_rows_before.sum())}
        for side in ("before", "after"):
            result["correct_" + side] = int(modes["manual_same_peak_" + side].sum())
            result["correct_percent_" + side] = (
                100 * result["correct_" + side] / result["manual_entries"]
            )
            result["extras_" + side] = int(
                modes["laser_not_in_manual_list_" + side].sum()
            )
            result["different_feature_" + side] = int(
                modes["laser_same_manual_lipid_different_peak_" + side].sum()
            )
            result["unresolved_manual_" + side] = int(
                modes["unresolved_manual_rows_" + side].sum()
            )
        (directory / "summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        summaries[name] = result
    assert summaries["endogenous"]["manual_entries"] == 90644
    assert summaries["spiked_references"]["manual_entries"] == 1456
    assert (
        summaries["endogenous"]["manual_entries"]
        + summaries["spiked_references"]["manual_entries"]
        == 92100
    )
    receipt = dict(
        datasets=33,
        dataset_modes=132,
        scopes=summaries,
        primary_historical_manual_entries=92100,
        excluded_classes=["X4"],
        note="Companion ONLY: endogenous90644 vs spiked1456. Historicalprimary92100 unchanged. RemovingstandardsNOTannotationgain. Samev2manualreferences/unresolvedrows retained. Extrasareunconfirmed,notFP/FDR. Both endpoints use unique-pair/duplicate semantics of originalvalidator. Previouslyexposedgroupeddevelopment,notdeployment/externalvalidation.",
    )
    (args.output / "summary.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
