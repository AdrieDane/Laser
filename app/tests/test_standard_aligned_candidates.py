import numpy as np
import pandas as pd
import pytest

from test_standard_anchored_retention import fixture
from matlab_annotation.standard_aligned_candidates import (
    prepare_standard_aligned_candidates,
    aligned_retention_evidence,
)


def inputs():
    targets, automatic, declared, measured, model = fixture()
    targets = targets.assign(
        name=["PE(34:2)", "PE(32:0)", "PE(O-34:2)"],
        adduct="[M+H]+",
        ref_rt=198 / 60,
    )
    features = pd.DataFrame(
        dict(
            feature_id=["correct", "late", "mass_bad", "spike_0", "competitor"],
            mz=[800.0001, 800.0001, 800.004, 800.0001, 800.0002],
            rt=np.array([206, 217, 206, 206, 208]) / 60,
            intensity=[1000, 2000, 3000, 10000, 900],
        )
    )
    return targets, features, automatic, declared, measured, model


def call(args):
    return prepare_standard_aligned_candidates(
        *args, stream="RP_Pos", query_group="query"
    )


def test_strict_join_original_coordinates_protected_spikes_and_no_occupancy_pruning():
    args = inputs()
    before = args[1].copy(deep=True)
    pool, targets, predictions = call(args)
    assert set(pool.feat_feature_id) == {"correct", "competitor"}
    assert pool.ppm_error.abs().lt(3).all()
    assert (pool.rt_error.abs() * 60).le(10).all()
    assert not pool.automatic_accepted.any()
    assert pool.isotopologue.eq("").all()
    raw = before.set_index("feature_id").loc[pool.feat_feature_id]
    np.testing.assert_array_equal(pool.feat_mz, raw.mz)
    np.testing.assert_array_equal(pool.feat_rt, raw.rt)
    evidence = aligned_retention_evidence(pool, targets, predictions)
    np.testing.assert_allclose(
        evidence.q_standard_aligned_RT_error_seconds, [0, 2], atol=0.01
    )
    assert evidence.q_standard_aligned_available.eq(1).all()
    pd.testing.assert_frame_equal(args[1], before)
    assert not {"name", "feat_feature_id"} & set(evidence)


def test_no_unsupported_polarity_isotopes_or_own_spike_mass_targets():
    for column, value in [
        ("adduct", "[M-H]-"),
        ("isotopologue", "M+1"),
        ("carbon", 100),
    ]:
        args = list(inputs())
        args[0] = args[0].assign(**{column: value})
        assert call(args)[0].empty
    args = list(inputs())
    args[1] = args[1].assign(mz=500.0001)
    assert call(args)[0].empty


def test_missing_alignment_is_unknown_for_old_queries_and_invalid_scopes_fail():
    args = list(inputs())
    args[4] = args[4].iloc[:0]
    pool, targets, predictions = call(args)
    assert pool.empty
    old = pd.DataFrame(dict(name=["PE(34:2)"], adduct=["[M+H]+"], feat_rt=[206 / 60]))
    evidence = aligned_retention_evidence(old, targets, predictions)
    assert evidence.q_standard_aligned_RT_error_seconds.isna().all()
    assert evidence.q_standard_aligned_available.eq(0).all()
    args = list(inputs())
    args[5]["training_groups"].append("query")
    with pytest.raises(ValueError, match="excluded"):
        call(args)
