import numpy as np
import pandas as pd
import pytest

from matlab_annotation.retention_atlas import fit_retention_atlas
from matlab_annotation.standard_rt_transfer import (
    exclude_declared_standard_masses,
    fit_standard_class_transfer,
    predict_standard_class_transfer,
    standard_class_targets,
    combine_standard_rt_sources,
)


def inputs():
    train = pd.DataFrame(
        [
            dict(
                cohort=f"g{g}",
                stream="RP_Pos",
                **{"class": "1-acylLPC"},
                carbon=c,
                db=d,
                rt_seconds=300 + 10 * (c - 16) - 20 * d,
            )
            for g in range(4)
            for c in [16, 18, 20]
            for d in [0, 1, 2]
        ]
    )
    atlas = fit_retention_atlas(train)
    residuals = pd.DataFrame(
        [
            dict(
                cohort=f"g{g}",
                stream="RP_Pos",
                standard_compound_id="LPC14|100",
                **{"class": "1-acylLPC"},
                residual_seconds=3.0,
            )
            for g in range(4)
        ]
    )
    targets = pd.DataFrame(
        {
            "standard_compound_id": ["LPC14|100"],
            "class": ["LPC"],
            "carbon": [14],
            "db": [0],
            "ref_mz": [500.0],
        }
    )
    auto = pd.DataFrame(columns=["feat_feature_id", "is_is"])
    return atlas, residuals, targets, auto


def test_standard_specific_residual_and_phase_reporting_class_transfer():
    atlas, residuals, targets, auto = inputs()
    transfer = fit_standard_class_transfer(residuals)
    result = predict_standard_class_transfer(
        targets, auto, atlas, transfer, stream="RP_Pos"
    )
    assert result.standard_class_transfer_rt_seconds.iloc[0] == pytest.approx(
        283.0, abs=0.2
    )
    assert result.standard_class_transfer_cohorts.iloc[0] == 4
    labels = {"models": {"RP_Pos|LPC14|100": {"reporting_class": "1-acylLPC"}}}
    assert (
        standard_class_targets(targets, labels, stream="RP_Pos")["class"].iloc[0]
        == "1-acylLPC"
    )


def test_all_possible_spike_masses_are_excluded_from_automatic_calibration():
    _, _, targets, _ = inputs()
    auto = pd.DataFrame(
        {"feat_feature_id": ["001", "other"], "feat_mz": [500.0001, 700.0]}
    )
    old = auto.copy(deep=True)
    clean = exclude_declared_standard_masses(auto, targets)
    assert clean.feat_feature_id.tolist() == ["other"]
    pd.testing.assert_frame_equal(old, auto)


def test_sparse_unstable_and_unknown_transfer_stays_unavailable():
    atlas, residuals, targets, auto = inputs()
    sparse = fit_standard_class_transfer(residuals.iloc[:2])
    assert not sparse["models"]
    result = predict_standard_class_transfer(
        targets, auto, atlas, sparse, stream="RP_Pos"
    )
    assert np.isnan(result.standard_class_transfer_rt_seconds.iloc[0])
    unstable = fit_standard_class_transfer(
        residuals.assign(residual_seconds=[-50.0, -20.0, 20.0, 50.0])
    )
    assert not unstable["models"]


def test_independent_RT_references_do_not_force_a_conflicting_completion():
    primary = pd.DataFrame({"standard_rt_seconds": [100.0, np.nan, 200.0]})
    secondary = pd.DataFrame(
        {"standard_class_transfer_rt_seconds": [102.0, 150.0, 230.0]}
    )
    combined = combine_standard_rt_sources(primary, secondary)
    assert combined.standard_rt_seconds.iloc[0] == 100
    assert combined.standard_rt_seconds.iloc[1] == 150
    assert np.isnan(combined.standard_rt_seconds.iloc[2])
