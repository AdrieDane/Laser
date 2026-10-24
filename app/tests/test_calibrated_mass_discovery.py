import numpy as np
import pandas as pd
import pytest

from matlab_annotation.calibrated_mass_discovery import (
    calibrated_mass_candidates,
    independently_measured_mass_anchors,
    select_calibrated_additions,
)
from test_declared_standard_context import fixture as standard_fixture


def fixtures():
    targets = pd.DataFrame(
        [
            dict(
                name="PE(36:2)",
                **{"class": "PE"},
                carbon=36,
                db=2,
                adduct="[M+H]+",
                ref_mz=800.0,
                atlas_rt_seconds=300.0,
                atlas_training_cohorts=5,
                atlas_training_mad_seconds=1.0,
                atlas_in_domain=True,
            )
        ]
    )
    features = pd.DataFrame(
        dict(feature_id=["new"], mz=[800 * (1 + 3.4e-6)], rt=[5.01], intensity=[1000.0])
    )
    anchors = pd.DataFrame(
        dict(
            stream=["NP_Pos"] * 7,
            neutral_key=[str(i) for i in range(7)],
            feature_id=[f"s{i}" for i in range(7)],
            reference_mz=np.arange(100.0, 800.0, 100.0),
            ppm=[1.0] * 7,
        )
    )
    old = pd.DataFrame(columns=["name", "adduct", "feat_feature_id"])
    auto = pd.DataFrame(
        [
            dict(
                name="PE(36:2)",
                **{"class": "PE"},
                carbon=36,
                db=2,
                stream="RP_Pos",
                adduct="[M+H]+",
                feat_rt=6.0,
                feat_intensity=1000.0,
                feat_feature_id="other",
                ppm_error=0.1,
                post_regression_confidence=0.99,
                is_is=False,
            )
        ]
    )
    return targets, features, anchors, old, auto


def candidates(targets, features, anchors, old, **kwargs):
    return calibrated_mass_candidates(
        targets, features, anchors, old, stream="NP_Pos", **kwargs
    )


def test_recovers_mass_window_pair_without_changing_raw_coordinates():
    t, f, a, old, auto = fixtures()
    before = f.copy(deep=True)
    pool = candidates(t, f, a, old)
    assert len(pool) == 1 and pool.discovery_novel_mass_pair.all()
    assert pool.iloc[0].feat_mz == f.iloc[0].mz
    assert pool.iloc[0].ppm_error == pytest.approx(3.4)
    assert pool.iloc[0].calibrated_mass_error_ppm == pytest.approx(2.4, abs=1e-5)
    assert len(select_calibrated_additions(pool, auto, old, stream="NP_Pos")) == 1
    pd.testing.assert_frame_equal(f, before)


@pytest.mark.parametrize(
    "case", ["few", "spread", "large_center", "rt_support", "isotope"]
)
def test_uncertain_calibration_or_RT_and_isotope_targets_withheld(case):
    t, f, a, old, _ = fixtures()
    if case == "few":
        a = a.iloc[:5]
    elif case == "spread":
        a["ppm"] = [-2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 2.8]
    elif case == "large_center":
        a["ppm"] = 2.5
    elif case == "rt_support":
        t["atlas_training_cohorts"] = 2
    else:
        t["isotopologue"] = "13C1"
    assert candidates(t, f, a, old).empty


def test_target_excludes_its_own_neutral_calibrator_and_spike_alias_whole_target():
    t, f, a, old, _ = fixtures()
    a = a.iloc[:6].copy()
    a.loc[a.index[0], "reference_mz"] = 800.0
    assert candidates(t, f, a, old).empty  # only5OTHERmassgroups
    _, _, a, _, _ = fixtures()
    assert candidates(t, f, a, old, declared_standard_masses=[800 * (1 + 4e-6)]).empty


def test_ambiguity_competes_before_fine_RT_occupancy_and_other_mode_gates():
    t, f, a, old, auto = fixtures()
    pool = candidates(t, f, a, old)
    occupied = pd.DataFrame(
        [dict(name="different", adduct="[M+H]+", feat_feature_id="new")]
    )
    assert select_calibrated_additions(pool, auto, occupied, stream="NP_Pos").empty
    assert select_calibrated_additions(
        pool, auto.assign(stream="NP_Pos"), old, stream="NP_Pos"
    ).empty
    f = pd.concat([f, f.assign(feature_id="otherpeak", rt=5.12)], ignore_index=True)
    ambiguous = candidates(t, f, a, old)
    assert len(ambiguous) == 2
    assert select_calibrated_additions(ambiguous, auto, occupied, stream="NP_Pos").empty
    competing = pd.concat(
        [pool, pool.assign(name="PG(36:2)", **{"class": "PG"})], ignore_index=True
    )
    assert select_calibrated_additions(competing, auto, old, stream="NP_Pos").empty


def test_existing_raw_window_pair_is_only_a_competitor_never_a_new_call():
    t, f, a, old, auto = fixtures()
    f["mz"] = 800 * (1 + 2e-6)
    pool = candidates(t, f, a, old)
    assert len(pool) == 1 and not pool.discovery_novel_mass_pair.any()
    assert select_calibrated_additions(pool, auto, old, stream="NP_Pos").empty


def test_anchor_mapping_keeps_actual_labelled_neutral_mass_and_rejects_supervision():
    _, spike = standard_fixture()
    spike["internal_standard_neutral_mass"] = 482.123456
    anchors = independently_measured_mass_anchors(spike)
    assert len(anchors) == 1
    assert anchors.iloc[0].neutral_key == "482.123456"
    assert anchors.iloc[0].ppm == pytest.approx(0.2)
    with pytest.raises(ValueError, match="supervision"):
        independently_measured_mass_anchors(spike.assign(training_positive=True))
