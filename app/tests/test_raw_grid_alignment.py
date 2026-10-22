import numpy as np
import pandas as pd
import pytest

from matlab_annotation.raw_grid_alignment import (
    infer_raw_grid_alignment,
    _candidate_centers,
    _cluster,
)


def fixture(n=12):
    targets = pd.DataFrame(
        [
            dict(
                name=f"PE({30 + 2 * i}:{i % 3})",
                **{"class": "PE"},
                carbon=30 + 2 * i,
                db=i % 3,
                adduct="[M-H]-",
                ref_mz=500 + 14 * i,
                atlas_rt_seconds=300 + 3 * i,
            )
            for i in range(n)
        ]
    )
    features = pd.DataFrame(
        [
            dict(
                feature_id=str(i),
                mz=(500 + 14 * i) * (1 + 0.2e-6),
                rt=(335 + 3 * i) / 60,
                intensity=10000 + i,
            )
            for i in range(n)
        ]
    )
    priors = {
        "NP_Neg|PE": dict(
            cohorts=20,
            joint={f"{int(r.carbon)}:{int(r.db)}": 15 for r in targets.itertuples()},
        )
    }
    return targets, features, priors


def test_finds_shift_without_manual_rt_and_excludes_own_composition():
    targets, features, priors = fixture()
    before = features.copy(deep=True)
    result = infer_raw_grid_alignment(targets, features, priors, stream="NP_Neg")
    assert result.raw_grid_shift_seconds.tolist() == pytest.approx([35] * 12)
    assert result.raw_grid_support.tolist() == [11] * 12
    assert result.raw_grid_rt_seconds.tolist() == pytest.approx(
        (targets.atlas_rt_seconds + 35).tolist()
    )
    pd.testing.assert_frame_equal(features, before)


def test_two_competing_mass_series_are_not_a_clear_alignment():
    targets, features, priors = fixture()
    alternative = features.copy()
    alternative["rt"] -= 1
    alternative["feature_id"] = "other" + alternative.feature_id
    result = infer_raw_grid_alignment(
        targets, pd.concat([features, alternative]), priors, stream="NP_Neg"
    )
    assert result.raw_grid_rt_seconds.isna().all()
    assert result.raw_grid_status.eq("ambiguous_competing_grid").all()


def test_repeated_adducts_do_not_multiply_composition_support():
    targets, features, priors = fixture(4)
    targets = pd.concat([targets] * 10, ignore_index=True)
    assert (
        infer_raw_grid_alignment(targets, features, priors, stream="NP_Neg")
        .raw_grid_rt_seconds.isna()
        .all()
    )


def test_weak_training_support_and_declared_spike_mass_do_not_anchor():
    targets, features, priors = fixture(9)
    result = infer_raw_grid_alignment(
        targets, features, priors, stream="NP_Neg", declared_standard_masses=[500]
    )
    assert result.raw_grid_rt_seconds.isna().all()
    priors["NP_Neg|PE"]["cohorts"] = 2
    assert (
        infer_raw_grid_alignment(targets, features, priors, stream="NP_Neg")
        .raw_grid_rt_seconds.isna()
        .all()
    )


def test_near_zero_shift_is_not_a_new_rt_source():
    targets, features, priors = fixture()
    features["rt"] -= 34 / 60
    result = infer_raw_grid_alignment(targets, features, priors, stream="NP_Neg")
    assert result.raw_grid_rt_seconds.isna().all()
    assert result.raw_grid_status.eq("near_original_no_extra_shift").all()


def test_outside_anchor_mass_window_cannot_vote():
    targets, features, priors = fixture()
    features["mz"] = targets.ref_mz * (1 + 2e-6)
    assert (
        infer_raw_grid_alignment(targets, features, priors, stream="NP_Neg")
        .raw_grid_rt_seconds.isna()
        .all()
    )


def test_vectorized_prefilter_is_identical_to_brute_force_cluster_support():
    rng = np.random.default_rng(20261022)
    for tolerance in [3.0, 5.0]:
        votes = pd.DataFrame(
            dict(
                composition=[(int(i), 1) for i in rng.integers(0, 25, 300)],
                delta=rng.uniform(-90, 90, 300),
                relative_intensity=rng.uniform(0.1, 1, 300),
                absolute_ppm=rng.uniform(0, 1.5, 300),
            )
        )
        expected = [
            v
            for v in np.arange(-90.0, 90.001, 2.0)
            if len(_cluster(votes, v, tolerance)) >= 9
        ]
        assert _candidate_centers(votes, tolerance).tolist() == expected


def test_one_mass_degenerate_raw_feature_cannot_supply_multiple_compositions():
    targets, features, priors = fixture()
    targets["ref_mz"] = 500
    targets["atlas_rt_seconds"] = 300
    assert (
        infer_raw_grid_alignment(targets, features, priors, stream="NP_Neg")
        .raw_grid_rt_seconds.isna()
        .all()
    )
