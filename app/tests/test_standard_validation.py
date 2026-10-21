from __future__ import annotations

from dataclasses import replace
import json

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from standard_lipidomics.validation import (
    AnnotationComparison,
    canonical_feature_id,
    compare_manual_annotations,
    summarize_comparison,
)


def manual(name="PE(36:2)", feature="17", **changes):
    return dict(
        stream="RP_Pos",
        matlab_species=name,
        matlab_class="PE",
        matlab_adduct="[M+H]+",
        xcms_feature_id=feature,
        crosswalk_status="profile_match",
        **changes,
    )


def laser(name="PE(36:2)", feature="17", **changes):
    return dict(
        stream="RP_Pos",
        name=name,
        **{"class": "PE"},
        adduct="[M+H]+",
        feat_feature_id=feature,
        **changes,
    )


def compare(m=None, a=None, **kwargs):
    return compare_manual_annotations(
        pd.DataFrame([manual()] if m is None else m),
        pd.DataFrame([laser()] if a is None else a),
        **kwargs,
    )


def test_exact_match_is_same_original_feature_not_name_only():
    result = compare(a=[laser(feature="18")])
    assert result.manual.outcome.tolist() == ["same_lipid_different_peak"]
    assert result.laser.comparison.tolist() == ["same_manual_lipid_different_peak"]
    s = summarize_comparison(result).iloc[0]
    assert s.manual_same_peak == 0 and s.manual_same_lipid_different_peak == 1


def test_other_identity_on_manual_peak_and_combined_conflict_are_separate():
    result = compare(a=[laser("PC(33:2)"), laser(feature="18")])
    r = result.manual.iloc[0]
    assert r.outcome == "different_peak_and_other_lipid_on_manual_peak"
    assert r.same_lipid_different_peak and r.other_lipid_on_manual_peak
    s = summarize_comparison(result).iloc[0]
    assert s.manual_mapped_not_matched == 1
    assert s.manual_same_lipid_different_peak == s.manual_peak_other_lipid == 1


def test_other_lipid_only_and_missing_identity():
    assert (
        compare(a=[laser("PC(33:2)")]).manual.outcome.iloc[0]
        == "other_lipid_on_manual_peak"
    )
    assert (
        compare(a=[laser("PC(33:2)", "99")]).manual.outcome.iloc[0]
        == "manual_lipid_not_selected"
    )


def test_unresolved_truth_never_establishes_wrong_peak():
    m = manual()
    m.update(crosswalk_status="profile_below_threshold", xcms_feature_id="")
    result = compare(m=[m])
    r = result.manual.iloc[0]
    assert r.outcome == "manual_peak_unresolved" and not r.same_manual_peak
    assert not r.same_lipid_different_peak and not r.other_lipid_on_manual_peak
    assert result.laser.comparison.iloc[0] == "manual_identity_peak_unresolved"
    s = summarize_comparison(result).iloc[0]
    assert pd.isna(s.percent_mapped_manual_same_peak)
    assert s.percent_all_manual_rows_same_peak == 0 and s.unresolved_manual_rows == 1


def test_absent_from_saved_candidates_is_not_never_considered():
    unknown = compare(a=[laser(feature="99")])
    assert unknown.manual.candidate_evidence.iloc[0] == "diagnostics_not_supplied"
    observed = compare(
        a=[laser(feature="99")], candidates=pd.DataFrame([laser(feature="88")])
    )
    assert (
        observed.manual.candidate_evidence.iloc[0]
        == "absent_from_available_candidate_tables"
    )
    assert not any("never" in c or "rejected" in c for c in observed.manual.columns)


@pytest.mark.parametrize(
    "final,pe,expected",
    [
        (True, False, "present_in_final_candidate_table"),
        (False, True, "present_in_PE_discovery_table"),
        (True, True, "present_in_both_candidate_tables"),
    ],
)
def test_candidate_table_locations(final, pe, expected):
    result = compare(
        candidates=pd.DataFrame([laser()]) if final else None,
        pe_decisions=pd.DataFrame([laser()]) if pe else None,
    )
    assert result.manual.candidate_evidence.iloc[0] == expected


def test_explicit_checks_and_support_are_observed_not_inferred_rejection():
    pool = pd.DataFrame(
        [
            laser(
                grid_coherence_assessed="True",
                grid_coherence_pass="False",
                post_regression_eligible="False",
            )
        ]
    )
    pe = pd.DataFrame(
        [
            laser(
                pe_decision="below_pe_score",
                pe_completion_score=0.99,
                pe_support_count=3,
            )
        ]
    )
    result = compare(candidates=pool, pe_decisions=pe)
    r = result.manual.iloc[0]
    assert r.same_manual_peak
    assert json.loads(r.recorded_final_check_failures) == {
        "grid_coherence_not_passed": 1,
        "post_regression_not_eligible": 1,
    }
    assert json.loads(r.recorded_PE_decisions) == ["below_pe_score"]
    assert r.best_recorded_PE_score == 0.99 and r.all_recorded_PE_support_below_five


def test_unknown_checks_do_not_become_passes_or_failures():
    result = compare(
        candidates=pd.DataFrame(
            [laser(grid_coherence_assessed=True, grid_coherence_pass=None)]
        ),
        pe_decisions=pd.DataFrame([laser(pe_support_count=None)]),
    )
    r = result.manual.iloc[0]
    assert r.recorded_final_check_failures == "{}"
    assert r.all_recorded_PE_support_below_five is None


def test_duplicate_rows_do_not_inflate_unique_agreement():
    result = compare(m=[manual(), manual()], a=[laser(), laser()])
    s = summarize_comparison(result).iloc[0]
    assert s.manual_rows == s.mapped_manual_rows == s.laser_rows == 2
    assert (
        s.manual_same_peak
        == s.unique_mapped_manual_annotations
        == s.unique_laser_annotations
        == 1
    )
    assert s.duplicate_mapped_manual_rows == s.duplicate_laser_rows == 1
    assert (
        s.percent_mapped_manual_same_peak == s.percent_all_manual_rows_same_peak == 100
    )


def test_conflicting_manual_labels_on_one_feature_remain_visible():
    result = compare(m=[manual(), manual("PC(33:2)")])
    assert result.manual.conflicting_manual_feature.all()
    assert summarize_comparison(result).conflicting_manual_rows.iloc[0] == 2


def test_multiple_accepted_adducts_on_same_feature_are_visible():
    second = laser()
    second["adduct"] = "[M+Na]+"
    result = compare(a=[laser(), second])
    assert result.laser.multiple_labels_on_feature.all()


def test_internal_standards_do_not_match_endogenous_identity():
    m = manual(matlab_is_internal_standard=True)
    result = compare(m=[m])
    assert not result.manual.same_manual_peak.iloc[0]
    assert result.laser.comparison.iloc[0] == "not_in_manual_list"
    result = compare(m=[m], a=[laser("PE(36:2) (IS)")])
    assert result.manual.same_manual_peak.iloc[0]


def test_explicit_standard_flag_contradiction_is_recorded():
    result = compare(a=[laser("PE(36:2) (IS)", is_is=False)])
    assert result.laser.standard_flag_conflict.iloc[0]


def test_positional_spacing_alias_but_not_position_or_ether_alias():
    m = manual("1-acyl LPC(18:1)")
    m["matlab_class"] = "1-acyl LPC"
    a = laser("1-acylLPC(18:1)")
    a["class"] = "1-acylLPC"
    assert compare(m=[m], a=[a]).manual.same_manual_peak.iloc[0]
    a["name"] = "2-acylLPC(18:1)"
    assert not compare(m=[m], a=[a]).manual.same_manual_peak.iloc[0]
    assert not compare(
        m=[manual("PE-O(36:2)")], a=[laser("PE-P(36:2)")]
    ).manual.same_manual_peak.iloc[0]


@pytest.mark.parametrize(
    "value,expected",
    [
        (17.0, "17"),
        ("17.00", "17"),
        ("0017", "0017"),
        ("0017.0", "0017.0"),
        ("NA", "NA"),
        ("peak-17", "peak-17"),
    ],
)
def test_feature_ids_preserve_text_identifiers(value, expected):
    assert canonical_feature_id(value) == expected


def test_feature_identity_never_crosses_measurement_stream():
    a = laser()
    a.update(stream="RP_Neg", adduct="[M-H]-")
    result = compare(a=[a])
    assert not result.manual.other_lipid_on_manual_peak.iloc[0]
    assert not result.manual.same_manual_peak.iloc[0]


def test_inputs_not_modified():
    m, a = pd.DataFrame([manual()]), pd.DataFrame([laser(grid_coherence_pass=False)])
    old_m, old_a = m.copy(deep=True), a.copy(deep=True)
    compare_manual_annotations(m, a, candidates=a, pe_decisions=a)
    assert_frame_equal(m, old_m)
    assert_frame_equal(a, old_a)


def test_empty_candidate_tables_and_empty_inputs():
    m, a = pd.DataFrame([manual()]).iloc[:0], pd.DataFrame([laser()]).iloc[:0]
    empty = compare_manual_annotations(m, a, candidates=a, pe_decisions=a)
    assert empty.manual.empty and empty.laser.empty
    assert summarize_comparison(empty).empty
    result = compare_manual_annotations(pd.DataFrame([manual()]), a, candidates=a)
    assert result.manual.outcome.iloc[0] == "manual_lipid_not_selected"
    assert (
        result.manual.candidate_evidence.iloc[0]
        == "absent_from_available_candidate_tables"
    )


def test_laser_only_class_is_kept_with_unavailable_manual_denominator():
    m = pd.DataFrame([manual()]).iloc[:0]
    result = compare_manual_annotations(m, pd.DataFrame([laser()]))
    s = summarize_comparison(result).iloc[0]
    assert s.laser_not_in_manual_list == 1 and s.manual_rows == 0
    assert pd.isna(s.percent_mapped_manual_same_peak)


def test_duplicate_dataset_excluded_only_from_pool_not_detail():
    first = compare(dataset="first")
    second = compare(dataset="duplicate", include_in_pooled=False)
    combined = AnnotationComparison(
        pd.concat([first.manual, second.manual]), pd.concat([first.laser, second.laser])
    )
    assert len(summarize_comparison(combined)) == 2
    s = summarize_comparison(combined, group_by=("stream",), pooled_only=True).iloc[0]
    assert s.manual_same_peak == s.unique_laser_annotations == 1


def test_invalid_required_fields_fail_loudly():
    with pytest.raises(ValueError, match="missing columns"):
        compare_manual_annotations(pd.DataFrame(), pd.DataFrame())
    with pytest.raises(ValueError, match="no original feature ID"):
        compare(m=[manual(feature="")])
    bad = laser()
    bad["stream"] = "RP_unknown"
    with pytest.raises(ValueError, match="Unknown measurement streams"):
        compare(a=[bad])


def test_unresolved_identity_also_has_resolved_manual_entry():
    unresolved = manual()
    unresolved.update(crosswalk_status="no_candidates", xcms_feature_id="")
    result = compare(m=[manual(), unresolved], a=[laser(feature="99")])
    assert result.laser.has_unresolved_manual_entry.iloc[0]
    assert result.laser.comparison.iloc[0] == "same_manual_lipid_different_peak"


def test_existing_comparison_can_be_reused_without_mutation():
    result = compare()
    old = replace(result, manual=result.manual.copy(), laser=result.laser.copy())
    summarize_comparison(result, group_by=("stream", "lipid_class"))
    assert_frame_equal(result.manual, old.manual)
    assert_frame_equal(result.laser, old.laser)
