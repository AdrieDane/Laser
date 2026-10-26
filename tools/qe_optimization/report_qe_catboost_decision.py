"""048 prospective whole-method decision; no outer-class output splicing."""

import json

import numpy as np
import pandas as pd

import catboost_experiment_data as data


def decide(summary, priority, standards):
    p = priority.loc[priority.optimization_priority.eq("prioritize")].copy()
    n = p.manual_rows_before.to_numpy(float)
    assert (n > 0).all() and np.array_equal(n, p.manual_rows_after.to_numpy(float))
    before, after = (
        100 * p["manual_same_peak_" + side].to_numpy(float) / n
        for side in ("before", "after")
    )
    guards = dict(
        at_least_100_net_correct=summary["correct_after"] - summary["correct_before"]
        >= 100,
        no_total_extra_increase=summary["extras_after"] <= summary["extras_before"],
        no_wrong_peak_increase=summary["different_feature_after"]
        <= summary["different_feature_before"],
        priority_capped90_mean_improves=bool(
            np.minimum(after, 90).mean() > np.minimum(before, 90).mean()
        ),
        priority_below90_count_not_increased=int((after < 90).sum())
        <= int((before < 90).sum()),
        no_already90_priority_class_falls_below=not bool(
            ((before >= 90) & (after < 90)).any()
        ),
        no_priority_class_loses_over_one_percentage_point=bool(
            (after >= before - 1).all()
        ),
        standards_no_correct_loss=standards["correct_after"]
        >= standards["correct_before"],
        standards_no_extra_increase=standards["extras_after"]
        <= standards["extras_before"],
        standards_no_wrong_peak_increase=standards["different_feature_after"]
        <= standards["different_feature_before"],
    )
    return dict(
        retain_whole_method=all(guards.values()),
        guards=guards,
        priority_class_modes=len(p),
        priority_capped90_mean_before=float(np.minimum(before, 90).mean()),
        priority_capped90_mean_after=float(np.minimum(after, 90).mean()),
        priority_below90_before=int((before < 90).sum()),
        priority_below90_after=int((after < 90).sum()),
        production_enabled=False,
        target_achieved=False,
        note="WHOLE048 versus WHOLE041; no winning-class/mode splice or guard relaxation. New learner+INNERclass calibration;041 completion add-on omitted. Retention,ifguards pass,is research only. Not chemical FDR, untouched external validation or production readiness.",
    )


def main():
    integrity = json.loads((data.OUT / "integrity_receipt.json").read_text())
    assert integrity["all_passed"]
    folder = data.OUT / "comparison_with_experiment_041"
    target = folder / "target_scope_without_X4"
    summary = json.loads((target / "summary.json").read_text())
    assert summary["target_manual_rows"] == 92100
    assert (
        summary["correct_before"] == 82740
        and summary["extras_before"] == 9270
        and summary["different_feature_before"] == 1604
    )
    priority = pd.read_csv(
        folder / "optimization_priority/by_mode_class_before_after.csv"
    )
    standards = json.loads(
        (
            folder / "endogenous_and_spiked_companion/spiked_references/summary.json"
        ).read_text()
    )
    result = decide(summary, priority, standards)
    result.update(
        primary=summary,
        integrity_receipt_sha256=data.sha(data.OUT / "integrity_receipt.json"),
    )
    data.dump(data.OUT / "decision.json", result)
    table = pd.read_csv(target / "by_dataset_mode_before_after.csv")
    assert len(table) == 132 and table.dataset.nunique() == 33
    lines = [
        "048 whole-method research decision: "
        + ("RETAIN" if result["retain_whole_method"] else "REJECT"),
        "Correct means same lipid,adduct,mode AND original peak; percentage denominator is every manual entry,including28unresolved. X4only excluded; standards included. Extras absent from manual list are unconfirmed,not proven false identities.",
        "New CatBoost learner+INNERclass calibration; old041completion omitted. Cached original-feature replay,not fresh deployment. All33datasets/30projectgroups; prior development exposure remains.",
        "No production defaults changed. Full dataset/mode table follows; separate class/mode,priority,endo/STD and wrong-peak tables saved alongside.",
    ]
    for mode in data.MODES:
        lines += [
            "",
            mode,
            "Dataset | Correct/manual before | Correct/manual after | TOTAL extras before | TOTAL extras after | Wrong peak before | Wrong peak after",
        ]
        for row in (
            table.loc[table.stream.eq(mode)].sort_values("dataset").to_dict("records")
        ):
            fractions = []
            for side in ("before", "after"):
                n, c = (
                    int(row["manual_rows_" + side]),
                    int(row["manual_same_peak_" + side]),
                )
                fractions.append(
                    f"{100 * c / n:.1f}% ({c}/{n})" if n else "n.a. (0 manual)"
                )
            lines.append(
                " | ".join(
                    [
                        row["dataset"],
                        *fractions,
                        *[
                            str(row[f + "_" + s])
                            for f in (
                                "laser_not_in_manual_list",
                                "laser_same_manual_lipid_different_peak",
                            )
                            for s in ("before", "after")
                        ],
                    ]
                )
            )
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
