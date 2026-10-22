import pandas as pd
import pytest

from matlab_annotation.selection_diagnostics import audit_operating_candidates


def fixture():
    frame = pd.DataFrame(
        dict(
            name=["A", "B", "C", "D", "E", "F", "F", "G"],
            adduct=["[M+H]+"] * 8,
            feat_feature_id=["01", "01", "02", "03", "03", "04", "05", "06"],
            feat_intensity=[100] * 8,
        )
    )
    scores = pd.Series([0.9, 0.6, 0.3, 0.8, 0.78, 0.9, 0.8, 0.5])
    return frame, scores


def test_every_primary_gate_and_exact_existing_selector_parity():
    pool, scores = fixture()
    before = pool.copy(deep=True)
    audit = audit_operating_candidates(pool, scores, threshold=0.5)
    assert audit.primary_gate.tolist() == [
        "selected",
        "another_label_ranks_above_on_feature",
        "below_acceptance_cutoff",
        "ambiguous_feature_label",
        "another_label_ranks_above_on_feature",
        "selected",
        "same_lipid_adduct_preferred_on_another_feature",
        "selected",
    ]
    assert audit.selector_accepts.tolist() == [
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
    ]
    pd.testing.assert_frame_equal(pool, before)


def test_stable_ties_duplicate_pairs_and_exact_cutoff_are_explained():
    pool = pd.DataFrame(
        dict(
            name=["A", "A", "B"],
            adduct=["ion"] * 3,
            feat_feature_id=["01"] * 3,
            feat_intensity=[10, 10, 10],
        )
    )
    scores = pd.Series([0.5, 0.5, 0.5])
    audit = audit_operating_candidates(pool, scores, threshold=0.5, ambiguity_margin=0)
    assert audit.primary_gate.tolist() == [
        "selected",
        "duplicate_candidate_pair",
        "another_label_ranks_above_on_feature",
    ]


def test_stream_and_row_alignment_guards():
    pool, scores = fixture()
    pool["stream"] = ["NP_Pos", "RP_Pos"] * 4
    with pytest.raises(ValueError, match="separately"):
        audit_operating_candidates(pool, scores, threshold=0.5)
    pool = pool.drop(columns="stream")
    with pytest.raises(ValueError, match="indices"):
        audit_operating_candidates(
            pool, scores.rename(index=lambda i: i + 1), threshold=0.5
        )
    pool.index = [0] * len(pool)
    with pytest.raises(ValueError, match="Unique"):
        audit_operating_candidates(pool, scores, threshold=0.5)
