import numpy as np
import pandas as pd
import pytest

import matlab_annotation.conditional_peak_refinement as module
from matlab_annotation.mode_evidence_experts import MODES


def fixture(monkeypatch, scores=None):
    accepted = pd.DataFrame(
        [
            (
                "RP_Pos",
                "PE(36:2)",
                "H",
                "PE",
                "1",
                500.0,
                500.0001,
                5.0,
                1000.0,
                False,
                0.8,
            ),
            (
                "RP_Pos",
                "PC(34:2)",
                "H",
                "PC",
                "4",
                500.0,
                500.0001,
                5.005,
                1000.0,
                False,
                0.9,
            ),
            (
                "RP_Pos",
                "PE_IS",
                "H",
                "PE",
                "9",
                700.0,
                700.0,
                5.0,
                2000.0,
                True,
                np.nan,
            ),
        ],
        columns=[
            *module.KEYS,
            "class",
            "feat_feature_id",
            "ref_mz",
            "feat_mz",
            "feat_rt",
            "feat_intensity",
            "is_is",
            "class_selection_score",
        ],
    )
    rows = []
    for name, cls, fid, mz, rt, intensity in [
        ("PE(36:2)", "PE", "1", 500.0001, 5.0, 1000.0),
        ("PE(36:2)", "PE", "2", 500.0002, 5.01, 1500.0),
        ("PE(36:2)", "PE", "3", 500.0003, 5.02, 1800.0),
        ("PC(34:2)", "PC", "4", 500.0001, 5.005, 1000.0),
        ("PC(34:2)", "PC", "2", 500.0002, 5.01, 1500.0),
        ("PE(36:2)", "PE", "4", 500.0001, 5.005, 1000.0),
        ("PE(36:2)", "PE", "8", 500.0004, 5.01, 1900.0),
        ("PE(36:2)", "PE", "7", 500.003, 5.01, 1900.0),
        ("PE(36:2)", "PE", "10", 500.0004, 5.7, 1900.0),
    ]:
        rows.append(
            dict(
                stream="RP_Pos",
                name=name,
                adduct="H",
                **{"class": cls},
                feat_feature_id=fid,
                ref_mz=500.0,
                feat_mz=mz,
                feat_rt=rt,
                feat_intensity=intensity,
                is_is=False,
                automatic_accepted=False,
                automatic_candidate=True,
                test_quality=False,
            )
        )
    pool = pd.DataFrame(rows)
    evidence = pd.DataFrame(
        {
            "absolute_RT_error_seconds": [1.0] * 8 + [40.0],
            "q_declared_standard_spike_feature": [0.0] * 6 + [1.0, 0.0, 0.0],
        }
    )
    data = dict(
        query_group="z",
        reference_training_groups=["a", "b"],
        reference_excluded_groups=["z"],
        standard_reference_training_groups=["a", "b"],
        standard_reference_excluded_groups=["z"],
        pool=pool,
        evidence=evidence,
    )
    model = dict(training_cohorts=["a", "b"], excluded_groups=["z"])
    values = {1: 0.97, 2: 0.75, 4: 0.84} if scores is None else scores
    monkeypatch.setattr(
        module,
        "predict_peak_preferences",
        lambda data, model, left, right: np.asarray([values[i] for i in left]),
    )
    features = {s: set() for s in MODES}
    masses = {s: [] for s in MODES}
    masses["RP_Pos"] = [700.0]
    accepted["automatic_accepted"] = True
    accepted["test_quality"] = (
        np.nan
    )  # mixed CSV dtype must not corrupt identity fields
    accepted["stale_peak_quality"] = [0.99, 0.88, 0.77]
    accepted["operating_threshold"] = 0.4
    return data, accepted, model, features, masses


def preferences(monkeypatch, scores=None):
    data, accepted, model, features, masses = fixture(monkeypatch, scores)
    p = module.peak_refinement_preferences(
        data,
        accepted,
        model,
        protected_features_by_stream=features,
        protected_masses_by_stream=masses,
    )
    return data, accepted, p


def thresholds(value=0.9):
    return {s: value for s in MODES}


def originals(data):
    columns = ["feat_feature_id", "feat_mz", "feat_rt", "feat_intensity"]
    raw = (
        data["pool"][columns]
        .drop_duplicates("feat_feature_id")
        .rename(columns={c: c.removeprefix("feat_") for c in columns})
    )
    raw["sample_A"] = raw.intensity * 2
    raw["mzmin"] = raw.mz - 0.0001
    return {s: raw.copy() for s in MODES}


def test_only_strict_free_nonspike_candidates_scored_and_originals_not_mutated(
    monkeypatch,
):
    data, accepted, p = preferences(monkeypatch)
    assert p.candidate_index.tolist() == [1, 2, 4]
    moves, audit = module.select_peak_refinements(p, thresholds())
    assert moves.candidate_index.tolist() == [1]
    assert audit.decision.eq("selected_peak_move").sum() == 1
    assert accepted.feat_feature_id.tolist() == ["1", "4", "9"]
    assert data["pool"].feat_feature_id.tolist() == [
        "1",
        "2",
        "3",
        "4",
        "2",
        "4",
        "8",
        "7",
        "10",
    ]


def test_close_same_free_feature_preferences_withheld_not_identity_displacement(
    monkeypatch,
):
    _, _, p = preferences(monkeypatch, {1: 0.97, 2: 0.75, 4: 0.95})
    moves, audit = module.select_peak_refinements(p, thresholds())
    assert moves.empty
    assert audit.decision.eq("ambiguous_free_feature").sum() == 2


def test_close_challengers_and_disabled_modes_keep_originals(monkeypatch):
    _, _, p = preferences(monkeypatch, {1: 0.97, 2: 0.95, 4: 0.84})
    moves, audit = module.select_peak_refinements(p, thresholds())
    assert moves.empty and audit.decision.eq("ambiguous_challenger_peaks").any()
    moves, _ = module.select_peak_refinements(p, thresholds(1.0))
    assert moves.empty


def test_independent_feature_and_mass_protections_mandatory(monkeypatch):
    data, accepted, model, features, masses = fixture(monkeypatch)
    features["RP_Pos"] = {"2", "3"}
    p = module.peak_refinement_preferences(
        data,
        accepted,
        model,
        protected_features_by_stream=features,
        protected_masses_by_stream=masses,
    )
    assert p.empty
    with pytest.raises(ValueError, match="ALL four"):
        module.peak_refinement_preferences(
            data,
            accepted,
            model,
            protected_features_by_stream={},
            protected_masses_by_stream=masses,
        )
    features = {s: set() for s in MODES}
    masses["RP_Pos"] = [500.0]
    p = module.peak_refinement_preferences(
        data,
        accepted,
        model,
        protected_features_by_stream=features,
        protected_masses_by_stream=masses,
    )
    assert p.empty


def test_applied_moves_preserve_identities_standards_counts_and_original_raw_coordinates(
    monkeypatch,
):
    data, accepted, p = preferences(monkeypatch)
    moves, _ = module.select_peak_refinements(p, thresholds())
    out = module.apply_peak_refinements(
        accepted, data["pool"], moves, features_by_stream=originals(data)
    )
    pd.testing.assert_frame_equal(
        accepted[[*module.KEYS, "class", "ref_mz", "is_is"]],
        out[[*module.KEYS, "class", "ref_mz", "is_is"]],
        check_exact=True,
    )
    assert len(out) == len(accepted) and out.feat_feature_id.tolist() == ["2", "4", "9"]
    assert out.loc[0, "feat_mz"] == data["pool"].loc[1, "feat_mz"]
    assert out.loc[0, "feat_rt"] == data["pool"].loc[1, "feat_rt"]
    assert out.loc[0, "feat_intensity"] == data["pool"].loc[1, "feat_intensity"]
    assert out.loc[0, "feat_sample_A"] == 3000.0
    assert out.loc[0, "feat_mzmin"] == data["pool"].loc[1, "feat_mz"] - 0.0001
    assert np.isnan(out.loc[0, "class_selection_score"])
    assert out.loc[0, "base_identity_acceptance_score"] == 0.8
    assert np.isnan(out.loc[0, "stale_peak_quality"])
    assert np.isnan(out.loc[0, "operating_threshold"])
    assert out.loc[0, "base_identity_operating_threshold"] == 0.4
    assert out.loc[0, "peak_refinement_score"] == 0.97
    assert out.loc[0, "automatic_accepted"] and not out.loc[0, "automatic_candidate"]
    pd.testing.assert_series_equal(accepted.loc[2], out.loc[2, accepted.columns])
    assert accepted.feat_feature_id.tolist() == ["1", "4", "9"]


def test_identity_mismatch_or_occupied_target_fails_closed(monkeypatch):
    data, accepted, p = preferences(monkeypatch)
    moves, _ = module.select_peak_refinements(p, thresholds())
    wrong = moves.copy()
    wrong.loc[wrong.index[0], "candidate_index"] = 3
    wrong.loc[wrong.index[0], "new_feature_id"] = "4"
    with pytest.raises(ValueError, match="identity"):
        module.apply_peak_refinements(
            accepted, data["pool"], wrong, features_by_stream=originals(data)
        )
    wrong = moves.copy()
    wrong.loc[wrong.index[0], "candidate_index"] = 5
    wrong.loc[wrong.index[0], "new_feature_id"] = "4"
    with pytest.raises(ValueError, match="occupied"):
        module.apply_peak_refinements(
            accepted, data["pool"], wrong, features_by_stream=originals(data)
        )
