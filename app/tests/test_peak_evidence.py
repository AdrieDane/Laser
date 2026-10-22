from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from annotator.peak_evidence import (
    C13_SPACING,
    PeakEvidenceSettings,
    peak_evidence,
    sample_roles,
)


def features():
    f = pd.DataFrame(
        {
            "feature_id": ["001", "iso", "sodium", "far"],
            "mz": [
                750.0,
                750.0 + C13_SPACING,
                750.0 + 21.981941533379,
                750.0 + C13_SPACING,
            ],
            "rt": [10.0, 10.0 + 0.2 / 60, 10.0 + 0.5 / 60, 11.0],
            "s1": [100.0, 40.0, 30.0, 400.0],
            "s2": [200.0, 80.0, 60.0, 800.0],
            "s3": [300.0, 120.0, 90.0, 1200.0],
            "s4": [400.0, 160.0, 120.0, 1600.0],
            "blank": [1000.0, 0.0, 0.0, 0.0],
            "qc": [200.0, 80.0, 60.0, 800.0],
        }
    )
    f.attrs = {
        "sample_columns": ["s1", "s2", "s3", "s4", "blank", "qc"],
        "sample_group_mapping_status": "validated",
        "sample_group_map": {
            **dict.fromkeys(["s1", "s2", "s3", "s4"], "Group_01_brain"),
            "blank": "Group_08_BlankIS",
            "qc": "Group_04_Quality_Control",
        },
    }
    return f


def test_roles_exclude_blankis_and_recognize_quality_control():
    columns, roles, source = sample_roles(features())
    assert len(columns) == 6 and roles["blank"] == "blank" and roles["qc"] == "qc"
    assert roles["s1"] == "biological" and source == "validated_sample_group_labels"


def test_median_and_detection_use_biological_samples_not_bright_blank():
    row = peak_evidence(features(), ["001"], polarity="Pos").iloc[0]
    assert row.q_log_median == pytest.approx(np.log1p(250))
    assert row.q_detection_fraction == 1 and row.q_biological_samples == 4
    assert row.q_log_sample_blank_ratio == pytest.approx(np.log1p(250) - np.log1p(1000))


def test_isotope_mass_coelution_and_profile_support_are_all_required():
    result = peak_evidence(features(), ["001", "iso"], polarity="Pos").set_index(
        "feature_id"
    )
    assert result.loc["001", "isotope_upper_feature_id"] == "iso"
    assert result.loc["001", "q_isotope_upper_correlation"] == pytest.approx(
        1, abs=1e-5
    )
    assert result.loc["001", "q_isotope_upper_nearby_count"] == 1
    assert result.loc["iso", "isotope_lower_feature_id"] == "001"
    assert result.loc["001", "q_isotope_upper_log_ratio"] < 0


def test_mass_neighbours_are_not_sufficient_when_profiles_are_constant():
    f = features()
    f.loc[:, ["s1", "s2", "s3", "s4"]] = 100
    row = peak_evidence(f, ["001"], polarity="Neg").iloc[0]
    assert row.q_isotope_upper_nearby_count == 1
    assert (
        pd.isna(row.q_isotope_upper_correlation) and row.isotope_upper_feature_id == ""
    )


def test_positive_sodium_adduct_evidence_does_not_cross_polarity():
    pos = peak_evidence(features(), ["001"], polarity="Pos").iloc[0]
    assert pos.sodium_feature_id == "sodium"
    neg = peak_evidence(features(), ["001"], polarity="Neg")
    assert "sodium_feature_id" not in neg


def test_missing_metadata_remains_unknown_not_a_qc_or_blank_pass():
    f = features()
    f.attrs.pop("sample_group_mapping_status")
    row = peak_evidence(f, ["001"], polarity="Pos").iloc[0]
    assert row.evidence_role_source == "unknown" and row.q_biological_samples == 0
    assert row.q_blank_samples == row.q_qc_samples == 0
    assert pd.isna(row.q_log_sample_blank_ratio) and pd.isna(row.q_qc_relative_mad)


def test_no_sample_columns_is_valid_but_has_no_profile_evidence():
    f = features()
    f.attrs = {}
    row = peak_evidence(f, ["001"], polarity="Pos").iloc[0]
    assert row.q_profile_samples == 0 and pd.isna(row.q_detection_fraction)
    assert pd.isna(row.q_log_median) and pd.isna(row.q_isotope_upper_correlation)


def test_explicit_roles_take_precedence():
    f = features()
    f.attrs["sample_roles"] = {"s1": "qc", "s2": "qc", "s3": "qc", "s4": "qc"}
    _, roles, source = sample_roles(f)
    assert roles["s1"] == "qc" and roles["blank"] == "unknown"
    assert source == "explicit_sample_roles"


def test_input_and_feature_identifiers_are_preserved():
    f = features()
    old = f.copy(deep=True)
    out = peak_evidence(f, ["001"], polarity="Pos")
    assert out.feature_id.tolist() == ["001"]
    assert_frame_equal(f, old)
    assert f.attrs == old.attrs


@pytest.mark.parametrize(
    "setting",
    [
        PeakEvidenceSettings(ppm=4),
        PeakEvidenceSettings(coelution_seconds=6),
        PeakEvidenceSettings(minimum_profile_pairs=3),
    ],
)
def test_unbounded_evidence_tolerances_are_rejected(setting):
    with pytest.raises(ValueError):
        peak_evidence(features(), ["001"], polarity="Pos", settings=setting)


def test_feature_or_sample_role_errors_are_not_silently_ignored():
    with pytest.raises(ValueError, match="unavailable"):
        peak_evidence(features(), ["missing"], polarity="Pos")
    f = features()
    f.attrs["sample_roles"] = {"s1": "guess"}
    with pytest.raises(ValueError, match="sample role"):
        sample_roles(f)


@pytest.mark.parametrize("charge", [2, 4])
def test_multiply_charged_isotopes_use_charge_scaled_spacing(charge):
    f = features()
    f.loc[f.feature_id.eq("iso"), "mz"] = 750 + C13_SPACING / charge
    correct = peak_evidence(f, ["001"], polarity="Neg", charge=charge).iloc[0]
    assert correct.isotope_upper_feature_id == "iso"
    assert (
        peak_evidence(
            f, ["001"], polarity="Neg", charge=1
        ).isotope_upper_feature_id.iloc[0]
        == ""
    )


@pytest.mark.parametrize("charge", [0, 5, True, 1.0])
def test_charge_hypotheses_are_explicit_bounded_integers(charge):
    with pytest.raises(ValueError, match="charge"):
        peak_evidence(features(), ["001"], polarity="Pos", charge=charge)


def test_default_charge_one_keeps_existing_evidence_identical():
    assert_frame_equal(
        peak_evidence(features(), ["001", "iso"], polarity="Pos"),
        peak_evidence(features(), ["001", "iso"], polarity="Pos", charge=1),
    )
