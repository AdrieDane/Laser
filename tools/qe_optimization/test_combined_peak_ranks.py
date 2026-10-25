"""Synthetic read-only diagnostic safeguards, not annotation experiments."""

import numpy as np
import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources
from diagnose_qe_combined_peak_ranks import blind_moves, add_truth, physical_masks

configure_frozen_resources(REPO)


def fixture():
    pool = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                name="PE(36:2)",
                **{"class": "PE"},
                adduct="[M+H]+",
                feat_feature_id=fid,
                feat_intensity=10 - i,
                is_is=False,
                ref_mz=700.0,
                feat_mz=700.0001,
                feat_rt=5.0,
                ref_rt=5.0,
            )
            for i, fid in enumerate(["a", "b", "c"])
        ]
    )
    scores = pd.Series([0.4, 0.9, 0.8], index=pool.index)
    evidence = pd.DataFrame(
        dict(q_raw_counterpart_profile_measurement_available=[1.0, 1.0, 1.0])
    )
    current = pool.iloc[[0]].copy()
    return pool, scores, evidence, current


def test_fixed_best_cannot_fallback_after_ambiguity_physics_or_occupancy_filters():
    pool, scores, evidence, current = fixture()
    original = pool.copy(deep=True)
    physical = pd.Series([True, False, True])
    protected = pd.Series(False, index=pool.index)
    moves, _ = blind_moves(
        pool, scores, evidence, current, current, physical, protected, {2}
    )
    assert moves.new_feature_id.tolist() == ["b"]
    assert moves.identity_only.all()
    assert not moves.clear_physical_free_target.any()
    # Even with physical/clear evidence, an occupied best cannot be replaced
    # by the weaker but free "c" feature.
    occupied = pd.concat([current, pool.iloc[[1]].assign(name="PC(34:0)")])
    moves, _ = blind_moves(
        pool,
        scores,
        evidence,
        current,
        occupied,
        pd.Series(True, index=pool.index),
        protected,
        {1, 2},
    )
    assert moves.new_feature_id.tolist() == ["b"] and not moves.target_is_free.any()
    pd.testing.assert_frame_equal(pool, original, check_exact=True)


def test_clear_free_profile_supported_move_remains_only_diagnostic():
    pool, scores, evidence, current = fixture()
    physical = pd.Series(True, index=pool.index)
    protected = pd.Series(False, index=pool.index)
    moves, _ = blind_moves(
        pool, scores, evidence, current, current, physical, protected, {1}
    )
    assert moves.clear_physical_free_profile.all() and moves.rank_gap.iloc[
        0
    ] == pytest.approx(0.5)
    assert not {"automatic_accepted", "training_positive", "diagnostic_change"} & set(
        moves
    )
    protected.loc[0] = True
    moves, _ = blind_moves(
        pool, scores, evidence, current, current, physical, protected, {1}
    )
    assert not moves.clear_physical_free_target.any()


def test_no_unknown_old_score_fallback_or_ambiguous_old_identity():
    pool, scores, evidence, current = fixture()
    old = current.assign(feat_feature_id="absent")
    moves, skipped = blind_moves(
        pool,
        scores,
        evidence,
        old,
        old,
        pd.Series(True, index=pool.index),
        pd.Series(False, index=pool.index),
        {1},
    )
    assert moves.empty and skipped["old_feature_absent_from_core"] == 1
    duplicate = pd.concat([current, current]).reset_index(drop=True)
    moves, skipped = blind_moves(
        pool,
        scores,
        evidence,
        duplicate,
        duplicate,
        pd.Series(True, index=pool.index),
        pd.Series(False, index=pool.index),
        {1},
    )
    assert moves.empty and skipped["standard_or_duplicate_old_identity"] == 2


def test_supervision_and_invalid_rank_are_rejected_before_blind_selection():
    pool, scores, evidence, current = fixture()
    args = (
        scores,
        evidence,
        current,
        current,
        pd.Series(True, index=pool.index),
        pd.Series(False, index=pool.index),
        {1},
    )
    with pytest.raises(ValueError, match="supervised"):
        blind_moves(pool.assign(training_positive=False), *args)
    with pytest.raises(ValueError, match="Finite"):
        blind_moves(pool, scores * float("nan"), *args[1:])


def test_gold_sets_preserve_multiple_manual_peaks_and_unknowns_are_not_false():
    pool, scores, evidence, current = fixture()
    moves, _ = blind_moves(
        pool,
        scores,
        evidence,
        current,
        current,
        pd.Series(True, index=pool.index),
        pd.Series(False, index=pool.index),
        {1},
    )
    manual = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                species="PE(36:2)",
                adduct="[M+H]+",
                feature_id=fid,
                internal_standard=False,
                mapped=True,
            )
            for fid in ["a", "b"]
        ]
    )
    assert add_truth(moves, manual).diagnostic_change.tolist() == [
        "correct_to_other_manual_peak"
    ]
    assert add_truth(moves, manual.iloc[[0]]).diagnostic_change.tolist() == [
        "correct_to_no_manual_match"
    ]
    assert add_truth(moves, manual.iloc[[1]]).diagnostic_change.tolist() == [
        "no_manual_match_to_correct"
    ]
    assert add_truth(moves, manual.iloc[:0]).diagnostic_change.tolist() == [
        "unconfirmed_identity_move"
    ]
    assert add_truth(
        moves, manual.iloc[[0]].assign(mapped=False, feature_id="")
    ).diagnostic_change.tolist() == ["unresolved_identity_move"]
    assert "diagnostic_change" not in moves


def test_physical_screen_keeps_original_strict_mass_RT_and_standard_protection():
    pool, _, _, _ = fixture()
    protection = dict(protected_features={"b"}, declared_masses=np.array([1000.0]))
    pool.loc[2, "feat_mz"] = 700.0 * (1 + 3.01e-6)
    physical, protected = physical_masks(pool, "RP_Pos", protection)
    assert physical.tolist() == [True, False, False] and protected.tolist() == [
        False,
        True,
        False,
    ]
    pool.loc[0, "ref_rt"] = 5.3
    assert not physical_masks(pool, "RP_Pos", protection)[0].any()
