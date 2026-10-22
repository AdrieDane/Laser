import pandas as pd

from matlab_annotation.np_series_completion import complete_np_series


def fixture():
    before = pd.DataFrame(
        [dict(name="PC(34:1)", adduct="H", feat_feature_id="occupied", is_is=False)]
    )
    targets = pd.DataFrame(
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
                raw_grid_support=12,
                raw_grid_mad_seconds=0.2,
            )
        ]
    )
    features = pd.DataFrame(
        [dict(feature_id="free", mz=700 * (1 + 0.5e-6), rt=337 / 60, intensity=10000)]
    )
    priors = {"NP_Neg|PE": dict(cohorts=20, joint={"36:2": 15})}
    return before, features, targets, priors


def test_adds_unique_supported_hit_without_touching_any_old_annotation():
    before, features, targets, priors = fixture()
    result, _, audit = complete_np_series(
        before, features, targets, priors, stream="NP_Neg"
    )
    assert result.feat_feature_id.tolist() == ["occupied", "free"]
    assert audit.action.eq("add_unique_independent_NP_series_hit").any()
    pd.testing.assert_frame_equal(
        result.iloc[:1][before.columns], before, check_dtype=False
    )


def test_occupancy_filter_cannot_turn_two_raw_hits_into_one():
    before, features, targets, priors = fixture()
    other = features.assign(feature_id="occupied")
    result, _, audit = complete_np_series(
        before, pd.concat([features, other]), targets, priors, stream="NP_Neg"
    )
    assert len(result) == len(before)
    assert audit.action.eq("withhold_multiple_raw_hits").any()


def test_existing_identity_and_declared_spike_masses_are_protected():
    before, features, targets, priors = fixture()
    result, _, _ = complete_np_series(
        before.assign(name="PE(36:2)", adduct="[M-H]-"),
        features,
        targets,
        priors,
        stream="NP_Neg",
    )
    assert len(result) == 1
    result, _, _ = complete_np_series(
        before,
        features,
        targets,
        priors,
        stream="NP_Neg",
        declared_standard_masses=[700],
    )
    assert len(result) == 1
    result, _, _ = complete_np_series(
        before,
        features,
        targets,
        priors,
        stream="NP_Neg",
        declared_standard_masses=[707],
    )
    assert len(result) == 2


def test_unknown_training_composition_or_RP_stream_cannot_be_completed():
    before, features, targets, priors = fixture()
    assert (
        len(complete_np_series(before, features, targets, {}, stream="NP_Neg")[0]) == 1
    )
    assert (
        len(complete_np_series(before, features, targets, priors, stream="RP_Neg")[0])
        == 1
    )


def test_multiple_identity_claims_on_one_free_feature_are_all_withheld():
    before, features, targets, priors = fixture()
    second = targets.assign(name="PG(36:2)", **{"class": "PG"})
    targets = pd.concat([targets, second], ignore_index=True)
    priors["NP_Neg|PG"] = priors["NP_Neg|PE"]
    result, _, audit = complete_np_series(
        before, features, targets, priors, stream="NP_Neg"
    )
    assert len(result) == 1
    assert audit.action.eq("withhold_competing_identity_claims").all()
