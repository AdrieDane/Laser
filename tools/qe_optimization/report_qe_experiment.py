"""Full all-class before/after report; refuses incomplete collection results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from context import WORKSPACE as ROOT

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
EXPERIMENT = ROOT / "outputs/qe_optimization_20261022/experiment_001"
STREAMS = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")


def pair(before, after, keys):
    fields = [
        "manual_rows",
        "manual_same_peak",
        "unresolved_manual_rows",
        "laser_not_in_manual_list",
        "laser_same_manual_lipid_different_peak",
        "unique_laser_annotations",
        "correct_manual_percent",
    ]
    result = before[[*keys, *fields]].merge(
        after[[*keys, *fields]],
        on=keys,
        how="outer",
        suffixes=("_before", "_after"),
        validate="one_to_one",
    )
    # Absent class rows mean no observations for that class, not a failed run.
    for c in fields:
        if c != "correct_manual_percent":
            result[c + "_before"] = result[c + "_before"].fillna(0).astype(int)
            result[c + "_after"] = result[c + "_after"].fillna(0).astype(int)
    assert result.manual_rows_before.equals(result.manual_rows_after), (
        "Manual denominators changed"
    )
    result["correct_gain"] = (
        result.manual_same_peak_after - result.manual_same_peak_before
    )
    result["extra_change"] = (
        result.laser_not_in_manual_list_after - result.laser_not_in_manual_list_before
    )
    return result


def fraction(row, side):
    n = int(row["manual_rows_" + side])
    correct = int(row["manual_same_peak_" + side])
    return f"{100 * correct / n:.1f}% ({correct}/{n})" if n else "n.a. (0 manual)"


def main():
    global EXPERIMENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    args = parser.parse_args()
    EXPERIMENT = args.experiment.resolve()
    protocol = json.loads((EXPERIMENT / "protocol.json").read_text())
    for root in (BASE, EXPERIMENT):
        summary = json.loads((root / "reports_v2/summary.json").read_text())
        assert summary["complete"] and summary["datasets_evaluated"] == 33, root
    report = EXPERIMENT / "full_comparison"
    report.mkdir(exist_ok=True)
    tables = {}
    groups = {
        "by_dataset_mode": ["dataset", "stream"],
        "by_dataset_mode_class": ["dataset", "stream", "lipid_class"],
        "by_mode_class": ["stream", "lipid_class"],
        "by_mode": ["stream"],
    }
    for name, keys in groups.items():
        before = pd.read_csv(BASE / "reports_v2" / (name + ".csv"))
        after = pd.read_csv(EXPERIMENT / "reports_v2" / (name + ".csv"))
        table = pair(before, after, keys)
        table.to_csv(report / (name + "_before_after.csv"), index=False)
        tables[name] = table
    assert len(tables["by_dataset_mode"]) == 132
    pooled = tables["by_mode"]
    correct_before = int(pooled.manual_same_peak_before.sum())
    correct_after = int(pooled.manual_same_peak_after.sum())
    manual = int(pooled.manual_rows_before.sum())
    extras_before = int(pooled.laser_not_in_manual_list_before.sum())
    extras_after = int(pooled.laser_not_in_manual_list_after.sum())
    guard = extras_after <= extras_before
    byclass = tables["by_mode_class"]
    class_below = byclass[
        (byclass.manual_rows_after > 0) & (byclass.correct_manual_percent_after < 90)
    ]
    class_below.to_csv(report / "class_modes_still_below_90.csv", index=False)
    result = dict(
        datasets=33,
        dataset_modes=132,
        manual=manual,
        correct_before=correct_before,
        correct_after=correct_after,
        percent_before=100 * correct_before / manual,
        percent_after=100 * correct_after / manual,
        extras_before=extras_before,
        extras_after=extras_after,
        wrong_peak_before=int(
            pooled.laser_same_manual_lipid_different_peak_before.sum()
        ),
        wrong_peak_after=int(pooled.laser_same_manual_lipid_different_peak_after.sum()),
        extras_no_increase_guard_pass=guard,
        class_modes_below_90=len(class_below),
        production_enabled=False,
        experiment=protocol["name"],
        manual_reference="v2, identical standard normalization mapping correction at both endpoints; not an annotation gain",
        target_achieved=False,
        note="A passing pooled score does not establish >=90% in every curated class/mode. Goal also requires validated production integration, not just this offline experiment.",
    )
    (report / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        protocol["name"],
        "",
        "Scope: all 33 active datasets, four modes each. Correct means the same lipid/adduct/mode AND original feature as manual curation.",
        "Correct percentage uses ALL manual entries, including unresolved feature mappings. Extras mean absent from that dataset/mode manual list, not proven false identities.",
        "Before: fresh default-engine raw-input baseline from frozen 4da94ef sources, optional PE selector OFF.",
        "After: "
        + protocol["run_kind"]
        + " Grouped five-fold held-out RT/prior/classifier fits. Not a fresh production run.",
        "Algorithm development inspected example-data RT diagnostics; these results are not untouched external validation.",
        "Recorded validation scope: "
        + protocol.get("validation", "See frozen protocol."),
        "Recorded fitting limitations: "
        + protocol.get("model_fit_note", "See frozen protocol."),
        "Both endpoints use reference v2: 960 independently verified internal-standard raw-profile mapping corrections; all prior matches and manual denominators preserved. Not counted as algorithm gains.",
        f"Pooled correct: {100 * correct_before / manual:.1f}% ({correct_before}/{manual}) -> {100 * correct_after / manual:.1f}% ({correct_after}/{manual}).",
        f"Total unconfirmed extras: {extras_before} -> {extras_after}. No-increase guard: {guard}.",
        f"Class/mode combinations still below 90%: {len(class_below)}. No production default changed.",
        "",
        "COMPLETE DATASET/MODE BEFORE AND AFTER TABLES",
    ]
    table = tables["by_dataset_mode"]
    for stream in STREAMS:
        lines.extend(
            [
                "",
                stream,
                "Dataset | Correct/manual before | Correct/manual after | Extras before | Extras after | Wrong-peak calls before | Wrong-peak calls after | Unresolved manual entries",
            ]
        )
        for row in (
            table[table.stream.eq(stream)].sort_values("dataset").to_dict("records")
        ):
            name = row["dataset"].split("__")[0]
            lines.append(
                f"{name} | {fraction(row, 'before')} | {fraction(row, 'after')} | {row['laser_not_in_manual_list_before']} | {row['laser_not_in_manual_list_after']} | {row['laser_same_manual_lipid_different_peak_before']} | {row['laser_same_manual_lipid_different_peak_after']} | {row['unresolved_manual_rows_after']}"
            )
    lines.extend(
        [
            "",
            "COMPLETE POOLED CLASS/MODE TABLE (counts pooled before division)",
            "Mode | Lipid class | Correct/manual before | Correct/manual after | Extras before | Extras after | Wrong-peak calls before | Wrong-peak calls after",
        ]
    )
    for row in byclass.sort_values(["stream", "lipid_class"]).to_dict("records"):
        lines.append(
            f"{row['stream']} | {row['lipid_class']} | {fraction(row, 'before')} | {fraction(row, 'after')} | {row['laser_not_in_manual_list_before']} | {row['laser_not_in_manual_list_after']} | {row['laser_same_manual_lipid_different_peak_before']} | {row['laser_same_manual_lipid_different_peak_after']}"
        )
    lines.extend(
        [
            "",
            "Every dataset/mode/class, including unchanged rows, is in by_dataset_mode_class_before_after.csv.",
            "Do not start another annotation-changing experiment until the complete tables have been delivered to the user.",
        ]
    )
    (report / "performance_before_after.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
