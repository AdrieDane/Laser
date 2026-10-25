"""Synthetic orchestration/denominator/extra accounting safeguards."""

import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
import run_qe_bmp_context_experiment as runner
from verify_qe_bmp_context_experiment import independent_counts
from test_bmp_context_tables import fixture as scope_fixture


def test_authorization_precedes_any_label_read(monkeypatch):
    called = []
    monkeypatch.setattr(
        runner.original.inherited.common,
        "label_allowed_query",
        lambda *a, **k: called.append(True),
    )
    raw = scope_fixture()
    with pytest.raises(ValueError):
        runner.authorize_labels(raw, {"B", "C"}, {"A", "D"}, partial=True)
    assert called == []


def manual():
    return pd.DataFrame(
        [
            dict(
                stream="NP_Neg",
                species=n,
                adduct="[M-H]-",
                feature_id=fid,
                mapped=mapped,
                internal_standard=False,
                lipid_class="BMP",
            )
            for n, fid, mapped in [
                ("BMP(36:2)", "n1", True),
                ("BMP(36:2)", "n_other", True),
                ("BMP(40:4)", "", False),
            ]
        ]
    )


def test_opposite_support_never_hides_primary_extras_and_wrong_peak_is_separate():
    chosen = pd.DataFrame(
        [
            dict(
                stream=s,
                name=n,
                adduct=a,
                feat_feature_id=fid,
                training_positive=tp,
                training_label_known=known,
                calibration_identity_in_manual=inside,
            )
            for s, n, a, fid, tp, known, inside in [
                ("NP_Neg", "BMP(36:2)", "[M-H]-", "n1", True, True, True),
                ("NP_Pos", "BMP(36:2)", "[M+NH4]+", "p1", False, True, False),
                ("NP_Pos", "BMP(38:3)", "[M+NH4]+", "p2", False, True, False),
                ("NP_Neg", "BMP(36:2)", "[M-H]-", "n2", False, True, True),
                ("NP_Neg", "BMP(40:4)", "[M-H]-", "n3", False, False, True),
            ]
        ]
    )
    expected = dict(
        added_correct=1,
        extras=2,
        extras_without_opposite_composition=1,
        different_peak=1,
        unresolved_calls=1,
    )
    assert runner.addition_counts(chosen, manual()) == expected
    assert independent_counts(chosen, manual()) == expected
    assert runner.addition_counts(chosen.iloc[:0], manual()) == dict.fromkeys(
        expected, 0
    )


def test_full_manual_denominator_preserves_unresolved_and_multiple_original_peaks():
    occupied = pd.DataFrame(
        [
            dict(
                stream="NP_Neg",
                name="BMP(36:2)",
                **{"class": "BMP"},
                adduct="[M-H]-",
                feat_feature_id="n1",
                is_is=False,
            )
        ]
    )
    counts = runner.prior_counts(occupied, manual())
    assert counts["NP_Neg"] == dict(manual=3, endogenous_manual=3, previous_correct=1)
    assert counts["NP_Pos"] == dict(manual=0, endogenous_manual=0, previous_correct=0)


def test_disable_is_not_score_at_least_one_and_unknown_empty_schema_is_safe():
    pool = pd.DataFrame(dict(class_selection_score=[0.5, 0.9, 1.0]))
    assert len(runner.threshold_subset(pool, 0.9)) == 2
    assert runner.threshold_subset(pool, 1.0).empty
    assert runner.threshold_subset(pd.DataFrame(), 0.5).empty
    with pytest.raises(ValueError):
        runner.threshold_subset(pool, 0.7 + 0.00001)


def test_verifier_empty_disable_schema_is_not_a_changed_decision():
    from verify_qe_bmp_context_experiment import check_cached_threshold

    empty = pd.DataFrame(columns=["original"])
    ranked = empty.assign(class_selection_score=pd.Series(dtype=float))
    check_cached_threshold(empty, ranked, ["original"])
    with pytest.raises(AssertionError):
        check_cached_threshold(empty, pd.DataFrame(dict(original=[1])), ["original"])
    with pytest.raises(AssertionError):
        check_cached_threshold(
            pd.DataFrame(dict(original=[1])),
            pd.DataFrame(dict(original=[2])),
            ["original"],
        )
    with pytest.raises(AssertionError):
        check_cached_threshold(empty, pd.DataFrame(columns=["other"]), ["original"])
