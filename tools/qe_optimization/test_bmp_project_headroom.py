"""Synthetic project robustness/headroom, no actual045 operatingpoint choices."""

import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
from bmp_project_headroom import (
    choose_project_headroom,
    pooled_curves,
    combine_project_folders,
)
from bmp_context_completion import MODES, LEVELS


def fixture():
    rows = []
    for p in "ABCD":
        for s in MODES:
            for t in LEVELS:
                correct = extra = 0
                if s == "NP_Pos" and p in "ABC":
                    if t == 0.5:
                        correct, extra = (60 if p == "A" else 10), 20
                    elif t == 0.9:
                        correct, extra = (
                            (30 if p == "A" else 25),
                            (25 if p == "A" else 20),
                        )
                    elif t == 0.95:
                        correct, extra = 20, (6 if p == "C" else 7)
                rows.append(
                    dict(
                        project=p,
                        stream=s,
                        threshold=t,
                        manual=250,
                        endogenous_manual=240,
                        previous_correct=50,
                        added_correct=correct,
                        extras=extra,
                        extras_without_opposite_composition=0,
                        different_peak=0,
                        unresolved_calls=0,
                    )
                )
    return pd.DataFrame(rows)


def test_headroom_and_leave_one_project_are_distinct_additional_guards():
    cuts, _, grid, witnesses = choose_project_headroom(
        fixture(), training_groups=set("ABCD"), excluded_groups={"Q"}
    )
    assert cuts == {"NP_Pos": 0.95, "NP_Neg": 1.0}
    borderline = grid.loc[grid.NP_Pos.eq(0.5) & grid.NP_Neg.eq(1.0)].iloc[0]
    assert borderline.original_safety_eligible and borderline.training_headroom_pass
    assert not borderline.leave_one_project_pass
    assert borderline.minimum_leave_one_project_correct_surplus == -20
    too_many = grid.loc[grid.NP_Pos.eq(0.9) & grid.NP_Neg.eq(1.0)].iloc[0]
    assert too_many.original_safety_eligible and too_many.leave_one_project_pass
    assert not too_many.training_headroom_pass
    assert len(grid) == len(LEVELS) ** 2 and len(witnesses) == len(grid) * 4


def test_all_unsupported_modes_explicitly_disable_without_weak_fallback():
    table = fixture()
    table[["added_correct", "extras"]] = 0
    cuts, _, grid, _ = choose_project_headroom(
        table, training_groups=set("ABCD"), excluded_groups={"Q"}
    )
    assert cuts == dict.fromkeys(MODES, 1.0) and not grid.eligible.any()


def test_related_folders_contribute_once_to_positive_project_support():
    table = fixture()
    duplicate_folder = table.loc[table.project.eq("A")].copy()
    duplicate_folder["added_correct"] = 0
    duplicate_folder["extras"] = 0
    combined = combine_project_folders(
        pd.concat([table, duplicate_folder], ignore_index=True)
    )
    pooled = pooled_curves(combined, training_groups=set("ABCD"), excluded_groups={"Q"})
    assert pooled.loc[
        pooled.stream.eq("NP_Pos") & pooled.threshold.eq(0.95), "positive_projects"
    ].tolist() == [3]
    with pytest.raises(ValueError, match="combined per PROJECT"):
        pooled_curves(
            pd.concat([table, duplicate_folder]),
            training_groups=set("ABCD"),
            excluded_groups={"Q"},
        )


def test_heldout_project_or_missing_level_cannot_enter_curves():
    table = fixture()
    with pytest.raises(ValueError, match="Held-out/missing"):
        pooled_curves(table, training_groups=set("ABC"), excluded_groups={"D"})
    with pytest.raises(ValueError, match="every declared"):
        pooled_curves(
            table.iloc[1:], training_groups=set("ABCD"), excluded_groups={"Q"}
        )
    table["extras"] = table.extras.astype(float)
    table.loc[0, "extras"] = 0.5
    with pytest.raises(ValueError, match="integers"):
        pooled_curves(table, training_groups=set("ABCD"), excluded_groups={"Q"})
