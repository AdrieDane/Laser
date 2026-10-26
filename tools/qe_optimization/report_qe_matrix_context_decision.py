"""049 prospective WHOLE decision and full132 user-facing report, after master."""

import json

import pandas as pd

import matrix_experiment_data as data
from report_qe_catboost_decision import decide as original_guard_logic


def decide(summary, priority, standards):
    result = original_guard_logic(summary, priority, standards)
    result["note"] = (
        "WHOLE049 versus WHOLE041;8new nested sample-context descriptors/freshnativeHGB180experts. ENTIREoriginal029/005/027/041policy,includingrecomputedcompletion. No posthocclass/modewinner splice,guardrelaxation orappdefaults. Researchcachedreplay,notdeployment/externalvalidation/chemicalFDR."
    )
    return result


def main():
    integrity_path = data.OUT / "integrity_receipt.json"
    assert json.loads(integrity_path.read_text())["all_passed"]
    comparison = data.OUT / "comparison_with_experiment_041"
    target = comparison / "target_scope_without_X4"
    summary = json.loads((target / "summary.json").read_text())
    assert summary["target_manual_rows"] == 92100
    assert (
        summary["correct_before"],
        summary["extras_before"],
        summary["different_feature_before"],
    ) == (82740, 9270, 1604)
    priority = pd.read_csv(
        comparison / "optimization_priority/by_mode_class_before_after.csv"
    )
    standards = json.loads(
        (
            comparison
            / "endogenous_and_spiked_companion/spiked_references/summary.json"
        ).read_text()
    )
    result = decide(summary, priority, standards)
    result.update(
        primary=summary, integrity_receipt_sha256=data.nested.pilot.sha(integrity_path)
    )
    data.nested.pilot.dump(data.OUT / "decision.json", result)
    table = pd.read_csv(target / "by_dataset_mode_before_after.csv")
    assert len(table) == 132 and table.dataset.nunique() == 33
    lines = [
        "049 whole-method decision: "
        + (
            "RETAIN research method"
            if result["retain_whole_method"]
            else "REJECT whole method"
        ),
        "Correct means SAME lipid,adduct,mode AND original peak as manual curation. Percentages use ALL manual entries,including unresolved; X4onlyexcluded,standardsincluded. Extras are TOTAL calls absent from manual list,not proven false identities. Wrongpeaks separate.",
        "All33datasets/132modes/30projects. Cached grouped development replay,not fresh integrated-app or external validation. New8context descriptors/freshHGB;old029modepolicy/005STD/027mass/041completionpolicy retained. Output rawsamplecells copied fromoriginal,notrescaled/imputed;notannotationgains.",
        "App defaults unchanged; target not yet established for deployment.",
    ]
    for stream in data.MODES:
        lines += [
            "",
            stream,
            "Dataset | Correct/manual before | Correct/manual after | TOTAL extras before | TOTAL extras after | Wrong peak before | Wrong peak after",
        ]
        for row in (
            table.loc[table.stream.eq(stream)].sort_values("dataset").to_dict("records")
        ):
            values = [row["dataset"]]
            for side in ("before", "after"):
                n, c = (
                    int(row["manual_rows_" + side]),
                    int(row["manual_same_peak_" + side]),
                )
                values.append(
                    f"{100 * c / n:.1f}% ({c}/{n})" if n else "n.a. (0 manual)"
                )
            values += [
                str(row[field + "_" + side])
                for field in (
                    "laser_not_in_manual_list",
                    "laser_same_manual_lipid_different_peak",
                )
                for side in ("before", "after")
            ]
            lines.append(" | ".join(values))
    lines += [
        "",
        "Prospective guards:",
        *[f"{k}: {v}" for k, v in result["guards"].items()],
        "",
        result["note"],
    ]
    (data.OUT / "decision_and_full132_table.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
