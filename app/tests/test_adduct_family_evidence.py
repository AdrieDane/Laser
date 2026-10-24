from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from matlab_annotation.adduct_family_evidence import PREFIX, adduct_family_evidence
from matlab_annotation.vendor.compound_model import PROTON_MASS

ROOT = Path(__file__).resolve().parents[2] / "reference_data/matlab_pipeline"


def fixture():
    source_mz = 900.0
    sodium_mz = source_mz + 21.981945 - 17.026549
    potassium_mz = source_mz + 37.955882 - 17.026549
    features = pd.DataFrame(
        dict(
            feature_id=["source", "sodium", "potassium"],
            mz=[source_mz, sodium_mz, potassium_mz],
            rt=[10.0, 10.01, 10.02],
            s1=[10.0, 20.0, 30.0],
            s2=[100.0, 200.0, 300.0],
            s3=[1000.0, 2000.0, 3000.0],
            s4=[10000.0, 20000.0, 30000.0],
        )
    )
    features.attrs.update(
        sample_columns=["s1", "s2", "s3", "s4"],
        sample_roles={c: "biological" for c in ["s1", "s2", "s3", "s4"]},
    )
    pool = pd.DataFrame(
        dict(
            stream=["RP_Pos"],
            **{"class": ["TG"]},
            adduct=["[M+NH4]+"],
            ref_mz=[source_mz],
            feat_feature_id=["source"],
        ),
        index=[17],
    )
    return pool, features


def test_ammonium_source_finds_actual_sodium_and_potassium_companions_not_fixed_H_offsets():
    pool, features = fixture()
    before = features.copy(deep=True)
    values = adduct_family_evidence(pool, features, ROOT, stream="RP_Pos").iloc[0]
    assert values[PREFIX + "expected_other_ions"] == 2
    assert values[PREFIX + "nearby_other_ions"] == 2
    assert values[PREFIX + "profile_supported_ions"] == 2
    assert values[PREFIX + "best_correlation"] > 0.99
    assert values[PREFIX + "best_abs_ppm"] < 1e-6
    assert values[PREFIX + "best_profile_pairs"] == 4
    assert values[PREFIX + "source_abs_charge"] == 1
    pd.testing.assert_frame_equal(features, before)
    assert not any("feature_id" in c for c in values.index)


def test_reverse_sodium_source_uses_correct_neutral_mass_without_promoting_that_ion():
    pool, features = fixture()
    query = pool.assign(
        adduct="[M+Na]+", ref_mz=features.mz.iloc[1], feat_feature_id="sodium"
    )
    values = adduct_family_evidence(query, features, ROOT, stream="RP_Pos").iloc[0]
    assert values[PREFIX + "nearby_other_ions"] == 2
    assert values[PREFIX + "profile_supported_ions"] == 2
    assert "automatic_accepted" not in query


def test_profiles_need_four_usable_non_qc_samples_and_wrong_mass_rt_cannot_support():
    pool, features = fixture()
    for altered in (
        features.assign(rt=[10.0, 11.0, 11.0]),
        features.assign(mz=features.mz + [0.0, 0.1, 0.1]),
    ):
        assert (
            adduct_family_evidence(pool, altered, ROOT, stream="RP_Pos").iloc[0][
                PREFIX + "nearby_other_ions"
            ]
            == 0
        )
    qc_only = features.copy()
    qc_only.attrs["sample_roles"] = {c: "qc" for c in ["s1", "s2", "s3", "s4"]}
    values = adduct_family_evidence(pool, qc_only, ROOT, stream="RP_Pos").iloc[0]
    assert values[PREFIX + "profile_supported_ions"] == 0
    assert np.isnan(values[PREFIX + "best_correlation"])


def test_no_cross_mode_or_label_inputs_and_unknown_class_stays_unknown():
    pool, features = fixture()
    with pytest.raises(ValueError, match="combine acquisition"):
        adduct_family_evidence(pool, features, ROOT, stream="NP_Pos")
    with pytest.raises(ValueError, match="label-free"):
        adduct_family_evidence(
            pool.assign(training_positive=True), features, ROOT, stream="RP_Pos"
        )
    values = adduct_family_evidence(
        pool.assign(**{"class": "UNKNOWN"}), features, ROOT, stream="RP_Pos"
    ).iloc[0]
    assert values[PREFIX + "source_configured"] == 0
    assert np.isnan(values[PREFIX + "best_correlation"])


@pytest.mark.parametrize("source_charge", [1, 2])
def test_negative_single_and_double_charge_use_native_neutral_mass(source_charge):
    pool, features = fixture()
    neutral = 1400.0
    features["mz"] = [
        neutral - PROTON_MASS,
        neutral / 2 - PROTON_MASS,
        neutral + 21.981945 - PROTON_MASS,
    ]
    source = source_charge - 1
    pool["stream"] = "RP_Neg"
    pool["class"] = "CL"
    pool["adduct"] = "[M-H]-" if source_charge == 1 else "[M-2H]2-"
    pool["ref_mz"] = features.mz.iloc[source]
    pool["feat_feature_id"] = features.feature_id.iloc[source]
    values = adduct_family_evidence(pool, features, ROOT, stream="RP_Neg").iloc[0]
    assert values[PREFIX + "source_abs_charge"] == source_charge
    assert values[PREFIX + "expected_other_ions"] == 2
    assert values[PREFIX + "profile_supported_ions"] == 2
    assert values[PREFIX + "best_abs_ppm"] < 1e-6


def test_duplicate_partner_peaks_count_as_ambiguity_not_multiple_support():
    pool, features = fixture()
    duplicate = features.iloc[[1]].assign(feature_id="second_sodium")
    features = pd.concat([features, duplicate], ignore_index=True)
    values = adduct_family_evidence(pool, features, ROOT, stream="RP_Pos").iloc[0]
    assert values[PREFIX + "ambiguous_other_ions"] == 1
    assert values[PREFIX + "profile_supported_ions"] == 2
