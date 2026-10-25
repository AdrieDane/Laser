"""Read-only WHOLE046 decision under the already frozen prospective plan.

Run manually AFTER complete postprocessing/master integrity. Never a model,
runtime filter, cutoff re-selection or exemption from primary extra counts.
"""

from collections import Counter
import json
from pathlib import Path

import pandas as pd

from frozen_resources import configure_frozen_resources
import run_qe_bmp_continuous_experiment as runner


def assess(
    *,
    gains,
    extras,
    unsupported_extras,
    wrong_by_mode,
    endogenous_by_mode,
    old_correct_lost,
    changed_outside_BMP,
    BMP_modes_improved,
):
    checks = dict(
        at_least_100_new_manual_matches=gains >= 100,
        new_total_extras_no_more_than_new_correct=extras <= gains,
        new_total_extras_no_more_than_571=extras <= 571,
        unsupported_composition_extras_budget=unsupported_extras
        <= 0.1 * sum(endogenous_by_mode.values()),
        new_wrong_no_more_than_five=sum(wrong_by_mode.values()) <= 5,
        new_wrong_per_mode_budget=all(
            wrong_by_mode[s] <= 0.005 * endogenous_by_mode[s] for s in runner.NP_MODES
        ),
        no_old_manual_match_loss=old_correct_lost == 0,
        no_change_outside_NP_BMP=changed_outside_BMP == 0,
        at_least_one_BMP_mode_improves=BMP_modes_improved >= 1,
    )
    return dict(
        retain_whole_research_method=bool(all(checks.values())),
        checks={k: bool(v) for k, v in checks.items()},
    )


def main():
    configure_frozen_resources(runner.OUT / "source_snapshot")
    master_path = runner.OUT / "integrity_receipt.json"
    master = json.loads(master_path.read_text())
    assert (
        master["all_passed"]
        and master["datasets"] == 33
        and master["dataset_mode_cases"] == 132
    )
    datasets, _ = runner.inventory()
    hashes = {
        str(Path(__file__)): runner.sha(Path(__file__)),
        str(runner.PLAN): runner.sha(runner.PLAN),
        str(master_path): runner.sha(master_path),
    }
    totals = Counter()
    wrong = Counter(dict.fromkeys(runner.NP_MODES, 0))
    endogenous = Counter(dict.fromkeys(runner.NP_MODES, 0))
    improved = Counter(dict.fromkeys(runner.NP_MODES, 0))
    rows = []
    for d in datasets:
        before_m = pd.read_pickle(
            runner.PREVIOUS / "evaluation_v2" / d / "manual_comparison.pkl"
        )
        after_m = pd.read_pickle(
            runner.OUT / "evaluation_v2" / d / "manual_comparison.pkl"
        )
        before_l = pd.read_pickle(
            runner.PREVIOUS / "evaluation_v2" / d / "laser_comparison.pkl"
        )
        after_l = pd.read_pickle(
            runner.OUT / "evaluation_v2" / d / "laser_comparison.pkl"
        )
        for endpoint in (runner.PREVIOUS, runner.OUT):
            for filename in ("manual_comparison.pkl", "laser_comparison.pkl"):
                p = endpoint / "evaluation_v2" / d / filename
                hashes[str(p)] = runner.sha(p)
        fixed = [
            "stream",
            "species",
            "adduct",
            "feature_id",
            "mapped",
            "internal_standard",
            "lipid_class",
        ]
        pd.testing.assert_frame_equal(before_m[fixed], after_m[fixed], check_exact=True)
        target = ~before_m.lipid_class.eq("X4")
        totals["gains"] += int(
            (target & after_m.same_manual_peak & ~before_m.same_manual_peak).sum()
        )
        totals["old_correct_lost"] += int(
            (target & before_m.same_manual_peak & ~after_m.same_manual_peak).sum()
        )
        totals["manual"] += int(target.sum())
        other = {
            (r.stream, runner.original.inherited.common.species(r.species))
            for r in after_m.loc[
                ~after_m.internal_standard & after_m.lipid_class.eq("BMP")
            ].itertuples()
        }
        oldkeys = {
            (r.stream, r.species, r.adduct, r.feature_id) for r in before_l.itertuples()
        }
        for r in after_l.itertuples():
            if (r.stream, r.species, r.adduct, r.feature_id) in oldkeys:
                continue
            totals["new_calls"] += 1
            totals["changed_outside_BMP"] += int(
                r.stream not in runner.NP_MODES
                or r.lipid_class != "BMP"
                or r.internal_standard
            )
            if r.comparison == "not_in_manual_list":
                totals["extras"] += 1
                opposite = "NP_Neg" if r.stream == "NP_Pos" else "NP_Pos"
                totals["unsupported_extras"] += int(
                    (opposite, runner.original.inherited.common.species(r.species))
                    not in other
                )
            if r.comparison == "same_manual_lipid_different_peak":
                wrong[r.stream] += 1
        for s in runner.NP_MODES:
            bm = before_m.loc[before_m.stream.eq(s) & before_m.lipid_class.eq("BMP")]
            am = after_m.loc[after_m.stream.eq(s) & after_m.lipid_class.eq("BMP")]
            bl = before_l.loc[before_l.stream.eq(s) & before_l.lipid_class.eq("BMP")]
            al = after_l.loc[after_l.stream.eq(s) & after_l.lipid_class.eq("BMP")]
            endogenous[s] += int((~am.internal_standard).sum())
            improved[s] += int(am.same_manual_peak.sum() - bm.same_manual_peak.sum())
            rows.append(
                dict(
                    dataset=d,
                    stream=s,
                    manual_rows=len(am),
                    correct_before=int(bm.same_manual_peak.sum()),
                    correct_after=int(am.same_manual_peak.sum()),
                    correct_manual_percent_before=100
                    * int(bm.same_manual_peak.sum())
                    / len(bm)
                    if len(bm)
                    else float("nan"),
                    correct_manual_percent_after=100
                    * int(am.same_manual_peak.sum())
                    / len(am)
                    if len(am)
                    else float("nan"),
                    unresolved_manual_rows=int((~am.mapped).sum()),
                    total_annotations_before=len(bl),
                    total_annotations_after=len(al),
                    extras_before=int(bl.comparison.eq("not_in_manual_list").sum()),
                    extras_after=int(al.comparison.eq("not_in_manual_list").sum()),
                    different_peak_before=int(
                        bl.comparison.eq("same_manual_lipid_different_peak").sum()
                    ),
                    different_peak_after=int(
                        al.comparison.eq("same_manual_lipid_different_peak").sum()
                    ),
                )
            )
    assert totals["manual"] == 92100
    assert not set(wrong) - set(runner.NP_MODES)
    result = assess(
        gains=totals["gains"],
        extras=totals["extras"],
        unsupported_extras=totals["unsupported_extras"],
        wrong_by_mode=wrong,
        endogenous_by_mode=endogenous,
        old_correct_lost=totals["old_correct_lost"],
        changed_outside_BMP=totals["changed_outside_BMP"],
        BMP_modes_improved=sum(n > 0 for n in improved.values()),
    )
    result.update(
        counts=dict(totals),
        new_wrong_by_mode=dict(wrong),
        endogenous_BMP_manual_by_mode=dict(endogenous),
        new_correct_BMP_by_mode=dict(improved),
        all_extras_still_primary=True,
        production_enabled=False,
        note="WHOLE method only. No class/polarity/dataset splicing after rejection; decision is a research retention check, not deployment approval or chemical FDR.",
    )
    destination = runner.OUT / "prospective_tradeoff_decision"
    destination.mkdir(exist_ok=False)
    pd.DataFrame(rows).to_csv(destination / "BMP_all66_before_after.csv", index=False)
    assert all(runner.sha(Path(p)) == h for p, h in hashes.items())
    runner.dump(destination / "decision.json", result)
    runner.dump(
        destination / "receipt.json",
        dict(
            all_passed=True,
            source_input_sha256=hashes,
            annotation_changes=False,
            thresholds_reselected=False,
            decision_sha256=runner.sha(destination / "decision.json"),
        ),
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
