"""Read-only: target deficits and mutually exclusive selector failure counts.

No new annotations, thresholds, classifiers or successful-class output mixing.
Reuses explicit completed029 diagnostics, not a fresh annotation benchmark.
"""

from pathlib import Path

import pandas as pd

from context import REPO, WORKSPACE
from run_qe_standard_mode_experiment import sha, dump

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "below_target_bottlenecks_029_v1"
PRIORITY = (
    REPO
    / "docs/validation/qe_20261025_039/versus_029/optimization_priority/by_mode_class_before_after.csv"
)
SELECTION = REPO / "docs/validation/qe_20261025_selection_029/by_mode_class_reason.csv"
DISCOVERY = QE / "competing_class_RT_opportunities_v1/by_mode_class_reason.csv"
REASONS = {
    "winning_correct_original_peak_below_mode_score_cutoff": "correct_peak_wins_but_is_below_cutoff",
    "correct_original_peak_absent_from_core_candidates": "correct_peak_absent_from_core",
    "different_original_peak_won_same_lipid_identity": "different_peak_wins_same_identity",
    "other_lipid_label_ranked_higher_on_original_peak": "other_label_wins_correct_peak",
    "correct_original_peak_withheld_for_close_label_scores": "close_label_scores_withhold_correct_peak",
    "standard_specialist_not_core_selector": "standard_specialist_gap",
    "manual_original_peak_unresolved": "unresolved_original_peak",
}


def main():
    inputs = {str(p): sha(p) for p in (PRIORITY, SELECTION, DISCOVERY)}
    source_hash = sha(Path(__file__))
    priority = pd.read_csv(PRIORITY)
    priority = priority[priority.optimization_priority.eq("prioritize")].copy()
    assert len(priority) == 86
    priority = priority[priority.correct_manual_percent_before.lt(90)].copy()
    assert len(priority) == 43
    table = priority[
        [
            "stream",
            "lipid_class",
            "manual_rows_before",
            "manual_same_peak_before",
            "correct_manual_percent_before",
            "laser_not_in_manual_list_before",
            "laser_same_manual_lipid_different_peak_before",
        ]
    ].rename(
        columns={
            "manual_rows_before": "manual_entries",
            "manual_same_peak_before": "correct_original_peak_matches",
            "correct_manual_percent_before": "correct_manual_percent",
            "laser_not_in_manual_list_before": "unconfirmed_extras",
            "laser_same_manual_lipid_different_peak_before": "different_original_peak_choices",
        }
    )
    table["additional_correct_needed_for_90pct"] = (
        9 * table.manual_entries + 9
    ) // 10 - table.correct_original_peak_matches
    reasons = pd.read_csv(SELECTION)
    assert set(reasons.reason) == set(REASONS)
    reasons["reason"] = reasons.reason.map(REASONS)
    pivot = (
        reasons.pivot(
            index=["stream", "lipid_class"],
            columns="reason",
            values="missing_manual_annotations",
        )
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    table = table.merge(
        pivot, on=["stream", "lipid_class"], how="left", validate="one_to_one"
    )
    for c in REASONS.values():
        table[c] = table[c].fillna(0).astype(int)
    assert (
        table[list(REASONS.values())]
        .sum(axis=1)
        .eq(table.manual_entries - table.correct_original_peak_matches)
        .all()
    )
    fresh = pd.read_csv(DISCOVERY)
    fresh = fresh[fresh.reason.eq("mechanically_free_if_all_old_calls_preserved")][
        ["stream", "lipid_class", "manual_entries"]
    ].rename(
        columns={"manual_entries": "new_coherent_candidates_mechanically_free_ONLY"}
    )
    table = table.merge(
        fresh, on=["stream", "lipid_class"], how="left", validate="one_to_one"
    )
    table["new_coherent_candidates_mechanically_free_ONLY"] = (
        table.new_coherent_candidates_mechanically_free_ONLY.fillna(0).astype(int)
    )
    table = table.sort_values("additional_correct_needed_for_90pct", ascending=False)
    OUT.mkdir(exist_ok=False)
    table.to_csv(OUT / "below90_priority_class_modes.csv", index=False)
    assert sha(Path(__file__)) == source_hash
    for p, h in inputs.items():
        assert sha(Path(p)) == h
    summary = dict(
        annotation_changes=False,
        new_classifier_fits=0,
        reference="retained029",
        below90_priority_class_modes=43,
        priority_class_modes=86,
        manual_entries=int(table.manual_entries.sum()),
        correct_original_peak_matches=int(table.correct_original_peak_matches.sum()),
        total_additional_correct_needed_to_reach90_in_each=int(
            table.additional_correct_needed_for_90pct.sum()
        ),
        mutually_exclusive_missing_reason_totals={
            c: int(table[c].sum()) for c in REASONS.values()
        },
        mechanically_free_new_coherent_candidate_opportunities=int(
            table.new_coherent_candidates_mechanically_free_ONLY.sum()
        ),
        source_sha256=source_hash,
        input_sha256=inputs,
        note="Reuses completed all33development diagnostics/counts,not a new annotation benchmark. Deficits are per-class/mode ceilings,not pooled90%. New coherent candidate opportunities overlap other categories/old033036discoveries and are upper bounds,not accepted matches or guaranteed gains. All43prioritybelow90classes shown; sparse classes deferred,not excluded from fullbenchmark.",
    )
    dump(OUT / "summary.json", summary)
    print(table.to_string(index=False), flush=True)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
