"""Research runner bookkeeping tests; never read real example annotations."""

import pandas as pd
import pytest

import run_qe_conditional_peak_experiment as runner
from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
from verify_qe_conditional_peak_experiment import independent_counts  # noqa: E402


def calls():
    result = pd.DataFrame(
        [
            ("RP_Pos", "PE(36:2)", "[M+H]+", "10", False),
            ("RP_Pos", "PC(34:2)", "[M+H]+", "20", False),
            ("RP_Pos", "TG(54:3)", "[M+H]+", "30", False),
        ],
        columns=["stream", "name", "adduct", "feat_feature_id", "is_is"],
    )
    result["class"] = ["PE", "PC", "TG"]
    return result


def test_replacement_updates_acceptance_and_review_status_without_new_identities():
    before = calls()
    after = before.copy()
    after.loc[0, "feat_feature_id"] = "11"
    candidates = pd.concat([before, after.iloc[[0]]], ignore_index=True)
    candidates["automatic_accepted"] = [True, True, True, False]
    candidates["automatic_candidate"] = ~candidates.automatic_accepted
    candidates["auto_annotation_status"] = "original"
    result = runner.updated_candidates(candidates, before, after)
    assert result.automatic_accepted.tolist() == [False, True, True, True]
    assert result.automatic_candidate.tolist() == [True, False, False, False]
    assert (
        result.loc[0, "auto_annotation_status"]
        == "original_peak_released_by_conditional_refinement"
    )
    assert (
        result.loc[3, "auto_annotation_status"] == "conditional_original_peak_refined"
    )
    pd.testing.assert_frame_equal(result.iloc[1:3], candidates.iloc[1:3])
    assert candidates.automatic_accepted.tolist() == [True, True, True, False]


def test_duplicate_or_unknown_standard_rows_cannot_silently_enter_export():
    before = calls()
    with pytest.raises(AssertionError):
        runner.updated_candidates(pd.concat([before, before.iloc[[0]]]), before, before)
    bad = before.astype({"is_is": object})
    bad.loc[0, "is_is"] = None
    with pytest.raises(AssertionError):
        runner.feature_keys(bad)


def test_inner_counts_preserve_unresolved_and_extras_and_require_same_original_peak():
    manual = pd.DataFrame(
        [
            ("RP_Pos", "PE(36:2)", "[M+H]+", False, "11", True, "PE"),
            ("RP_Pos", "PC(34:2)", "[M+H]+", False, "20", True, "PC"),
            ("RP_Pos", "PI(38:4)", "[M+H]+", False, "", False, "PI"),
        ],
        columns=[
            "stream",
            "species",
            "adduct",
            "internal_standard",
            "feature_id",
            "mapped",
            "lipid_class",
        ],
    )
    before = calls()
    empty = pd.DataFrame(columns=["annotation_index", "new_feature_id"])
    moves = pd.DataFrame([dict(annotation_index=0, new_feature_id="11.0")])
    case = runner.truth_case(manual, before)
    assert runner.case_counts(case, empty) == independent_counts(manual, before, empty)
    # Runner canonicalizes move FIDs; real candidate FIDs are already canonical.
    canonical_moves = moves.assign(new_feature_id="11")
    after = runner.case_counts(case, moves)
    assert after == independent_counts(manual, before, canonical_moves)
    assert after[0] == {("RP_Pos", "PE"): 1, ("RP_Pos", "PC"): 1}
    assert not after[1]
    assert after[2] == {"RP_Pos": 1}
    assert len(case["rows"]) == 3
