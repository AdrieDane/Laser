"""Synthetic diagnostic safeguards; no annotation experiments."""

import numpy as np
import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources
from diagnose_qe_combined_label_ranks import blind_moves, add_truth

configure_frozen_resources(REPO)


def fixture():
    pool = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                name=name,
                **{"class": cls},
                adduct="[M+H]+",
                feat_feature_id=fid,
                feat_intensity=10 - i,
                is_is=False,
            )
            for i, (name, cls, fid) in enumerate(
                [
                    ("PE(36:2)", "PE", "1"),
                    ("PC(34:2)", "PC", "1"),
                    ("PC(36:3)", "PC", "1"),
                    ("PC(34:2)", "PC", "2"),
                ]
            )
        ]
    )
    scores = pd.Series([0.4, 0.9, 0.8, 0.5])
    evidence = pd.DataFrame(
        dict(q_raw_counterpart_profile_measurement_available=np.ones(4))
    )
    current = pool.iloc[[0]].copy()
    return pool, scores, evidence, current


def moves(
    scores=None, current=None, occupied=None, physical=None, protected=None, clear=None
):
    p, s, e, c = fixture()
    return blind_moves(
        p,
        s if scores is None else scores,
        e,
        c if current is None else current,
        c if occupied is None else occupied,
        pd.Series(True, index=p.index) if physical is None else physical,
        pd.Series(False, index=p.index) if protected is None else protected,
        {1} if clear is None else clear,
    )


def test_same_original_feature_and_fixed_best_label_no_weaker_fallback():
    result, _ = moves(physical=pd.Series([True, False, True, True]), clear={2})
    assert result.feature_id.tolist() == ["1"]
    assert result.new_species.tolist() == ["PC(34:2)"]
    assert not result.clear_physical_unused_identity.any()
    assert result.feature_only.all()
    assert not {"diagnostic_change", "automatic_accepted", "training_positive"} & set(
        result
    )


def test_guarded_screen_protects_old_peak_and_requires_unused_absolute_identity_best():
    result, _ = moves()
    assert result.clear_physical_profile.all()
    result, _ = moves(protected=pd.Series([True, False, False, False]))
    assert not result.clear_physical_unused_identity.any()
    p, _, _, c = fixture()
    result, _ = moves(occupied=pd.concat([c, p.iloc[[3]]]))
    assert not result.new_identity_unused.any()
    result, _ = moves(scores=pd.Series([0.4, 0.9, 0.8, 0.95]))
    assert result.new_species.tolist() == ["PC(34:2)"]
    assert not result.new_identity_best_here.any()


def test_old_score_required_no_supervision_and_no_nonfinite_rank():
    p, s, e, c = fixture()
    result, counts = moves(current=c.assign(name="PE(40:2)"))
    assert result.empty and counts["old_identity_score_unknown"] == 1
    with pytest.raises(ValueError, match="supervised"):
        blind_moves(
            p.assign(training_positive=False), s, e, c, c, s.gt(0), s.lt(0), {1}
        )
    with pytest.raises(ValueError, match="Finite"):
        moves(scores=s * float("nan"))
    result, counts = moves(current=pd.concat([c, c], ignore_index=True))
    assert result.empty and counts["standard_or_ambiguous_old_call"] == 2


def test_numeric_and_string_original_feature_identifiers_match():
    _, _, _, c = fixture()
    result, _ = moves(current=c.assign(feat_feature_id=1))
    assert result.new_species.tolist() == ["PC(34:2)"]


def manual(species, fids, mapped=True):
    return pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                species=species,
                adduct="[M+H]+",
                feature_id=fid,
                internal_standard=False,
                mapped=mapped,
            )
            for fid in fids
        ]
    )


def test_gold_sets_and_unconfirmed_extra_delta_are_separate_from_wrong_peak():
    result, _ = moves()
    gold = manual("PC(34:2)", ["1", "2"])
    labeled = add_truth(result, gold)
    assert labeled.diagnostic_change.tolist() == ["no_manual_match_to_correct"]
    assert labeled.unconfirmed_extra_delta.tolist() == [-1]
    assert labeled.different_peak_delta.tolist() == [0]
    labeled = add_truth(result, manual("PE(36:2)", ["1", "2"]))
    assert labeled.diagnostic_change.tolist() == ["correct_to_no_manual_match"]
    assert labeled.unconfirmed_extra_delta.tolist() == [1]
    labeled = add_truth(result, manual("PC(34:2)", ["2"]))
    assert labeled.diagnostic_change.tolist() == ["neither_label_matches_manual_peak"]
    assert labeled.different_peak_delta.tolist() == [1]
    labeled = add_truth(result, manual("PC(34:2)", [""], mapped=False))
    assert labeled.diagnostic_change.tolist() == ["unresolved_comparison"]
    assert labeled.unconfirmed_extra_delta.tolist() == [-1]
    assert labeled.different_peak_delta.tolist() == [0]
    assert "diagnostic_change" not in result
