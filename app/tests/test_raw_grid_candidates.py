import pandas as pd
import pytest

from matlab_annotation.raw_grid_candidates import match_raw_grid_candidates


def fixture():
    target = pd.DataFrame(
        [
            dict(
                name="PE(36:2)",
                **{"class": "PE"},
                carbon=36,
                db=2,
                adduct="[M-H]-",
                ref_mz=700.0,
                ref_rt=5.0,
                atlas_rt_seconds=300.0,
                raw_grid_rt_seconds=335.0,
                raw_grid_status="independent_mass_series_shift",
                atlas_local_support_count=0,
            )
        ]
    )
    features = pd.DataFrame(
        [dict(feature_id="original", mz=700 * (1 + 1e-6), rt=337 / 60, intensity=10000)]
    )
    return target, features


def test_preserves_original_coordinates_and_only_creates_candidate():
    target, features = fixture()
    original = features.copy(deep=True)
    r = match_raw_grid_candidates(
        target, features, pd.DataFrame(), stream="NP_Neg"
    ).iloc[0]
    assert r.feat_feature_id == "original"
    assert r.feat_mz == features.iloc[0].mz
    assert r.feat_rt == features.iloc[0].rt
    assert r.rt_error * 60 == pytest.approx(2)
    assert r.automatic_candidate and not r.automatic_accepted
    pd.testing.assert_frame_equal(features, original)


def test_well_aligned_class_or_ambiguous_grid_is_not_cold_started():
    target, features = fixture()
    target["atlas_local_support_count"] = 4
    assert match_raw_grid_candidates(
        target, features, pd.DataFrame(), stream="NP_Neg"
    ).empty
    target["atlas_local_support_count"] = 0
    target["raw_grid_status"] = "ambiguous_competing_grid"
    assert match_raw_grid_candidates(
        target, features, pd.DataFrame(), stream="NP_Neg"
    ).empty


def test_cannot_borrow_standard_features_or_declared_spike_mass():
    target, features = fixture()
    automatic = pd.DataFrame([dict(is_is=True, feat_feature_id="original")])
    assert match_raw_grid_candidates(target, features, automatic, stream="NP_Neg").empty
    assert match_raw_grid_candidates(
        target,
        features,
        pd.DataFrame(),
        stream="NP_Neg",
        declared_standard_masses=[700],
    ).empty
    assert (
        len(
            match_raw_grid_candidates(
                target,
                features,
                pd.DataFrame(),
                stream="NP_Neg",
                declared_standard_masses=[707],
            )
        )
        == 1
    )


def test_physical_limits_remain_strict_and_cannot_be_widened():
    target, features = fixture()
    features["rt"] += 10 / 60
    assert match_raw_grid_candidates(
        target, features, pd.DataFrame(), stream="NP_Neg"
    ).empty
    features["rt"] -= 10 / 60
    features["mz"] = 700 * (1 + 4e-6)
    assert match_raw_grid_candidates(
        target, features, pd.DataFrame(), stream="NP_Neg"
    ).empty
    with pytest.raises(ValueError):
        match_raw_grid_candidates(
            target, features, pd.DataFrame(), stream="NP_Neg", rt_seconds=20
        )
