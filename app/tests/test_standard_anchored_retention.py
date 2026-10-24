import numpy as np
import pandas as pd
import pytest

from matlab_annotation.retention_atlas import fit_retention_atlas
from matlab_annotation.standard_rt_transfer import fit_standard_class_transfer
from matlab_annotation.standard_anchored_retention import standard_anchored_retention


def fixture(cohorts=12, shifts=(8.0,)):
    observations = pd.DataFrame(
        [
            dict(
                cohort=f"g{g}",
                stream="RP_Pos",
                **{"class": "PE"},
                carbon=c,
                db=d,
                rt_seconds=200 + 2 * (c - 32) - 3 * d,
            )
            for g in range(cohorts)
            for c in (28, 32, 36, 40)
            for d in (0, 2, 4)
        ]
    )
    atlas = fit_retention_atlas(observations)
    measured, residuals = [], []
    for i, shift in enumerate(shifts):
        compound = f"PE_std_{i}"
        c = 32 + i * 4
        expected = 200 + 2 * (c - 32) + 3.0
        measured.append(
            dict(
                stream="RP_Pos",
                **{"class": "PE"},
                standard_compound_id=compound,
                name=f"PE({c}:0) (IS)",
                carbon=c,
                db=0,
                adduct="[M+H]+",
                feat_feature_id=f"spike_{i}",
                feat_mz=500.0 + i * 10 + 0.0001,
                ref_mz=500.0 + i * 10,
                feat_rt=(expected + shift) / 60,
                feat_intensity=10000.0,
                is_is=True,
                internal_standard_quantity=100.0,
                internal_standard_unit="nM",
                rt_expected=(expected + shift) / 60,
                standard_training_cohorts=cohorts,
                standard_rt_reference_source="other_declared_standards",
            )
        )
        residuals.extend(
            dict(
                cohort=f"g{g}",
                stream="RP_Pos",
                standard_compound_id=compound,
                **{"class": "PE"},
                residual_seconds=3.0,
            )
            for g in range(cohorts)
        )
    measurements = pd.DataFrame(measured)
    declared = measurements[
        ["stream", "class", "standard_compound_id", "carbon", "db", "adduct", "ref_mz"]
    ].copy()
    targets = pd.DataFrame(
        dict(
            stream=["RP_Pos"] * 3,
            **{"class": ["PE", "PE", "PE-O"]},
            carbon=[34, 32, 34],
            db=[2, 0, 2],
            ref_mz=[800.0, 500.0, 786.0],
        ),
        index=[7, 11, 15],
    )
    automatic = pd.DataFrame(columns=["feat_feature_id", "is_is"])
    model = dict(
        format="laser-nested-standard-context-models-v1",
        training_groups=[f"g{g}" for g in range(cohorts)],
        excluded_groups=["query"],
        atlas=atlas,
        library={"models": {}},
        transfer=fit_standard_class_transfer(pd.DataFrame(residuals)),
    )
    return targets, automatic, declared, measurements, model


def call(inputs):
    return standard_anchored_retention(*inputs, stream="RP_Pos", query_group="query")


def test_independent_spike_aligns_exact_class_not_ether_or_own_mass_and_changes_no_inputs():
    inputs = fixture()
    before = [d.copy(deep=True) for d in inputs[:4]]
    result = call(inputs)
    assert result.loc[7, "standard_clean_atlas_rt_seconds"] == pytest.approx(
        198.0, abs=0.01
    )
    assert result.loc[7, "standard_alignment_shift_seconds"] == pytest.approx(
        8.0, abs=0.01
    )
    assert result.loc[7, "standard_aligned_rt_seconds"] == pytest.approx(
        206.0, abs=0.01
    )
    assert result.loc[7, "standard_alignment_anchor_compounds"] == 1
    assert result.loc[11, "standard_alignment_protected_mass"] == 1
    assert result.loc[[11, 15], "standard_aligned_rt_seconds"].isna().all()
    assert all(pd.api.types.is_numeric_dtype(result[c]) for c in result)
    assert not any("feature_id" in c for c in result)
    for actual, original in zip(inputs[:4], before, strict=True):
        pd.testing.assert_frame_equal(actual, original)


def test_transfer_only_spike_cannot_feed_back_and_wrong_declared_chemical_mass_is_rejected():
    for alteration in (
        dict(standard_rt_reference_source="independent_endogenous_class_transfer"),
        dict(ref_mz=499.0, feat_mz=499.0001),
    ):
        inputs = list(fixture())
        inputs[3] = inputs[3].assign(**alteration)
        assert call(inputs).standard_aligned_rt_seconds.isna().all()


def test_single_weak_large_or_conflicting_offsets_stay_unknown():
    for inputs in (
        fixture(cohorts=3),
        fixture(shifts=(31.0,)),
        fixture(shifts=(8.0, 14.0)),
    ):
        assert call(inputs).standard_aligned_rt_seconds.isna().all()
    inputs = fixture()
    inputs[4]["transfer"]["models"]["RP_Pos|PE_std_0"]["training_MAD_seconds"] = 2.5
    assert call(inputs).standard_aligned_rt_seconds.isna().all()


def test_two_independent_compounds_can_support_with_three_cohorts_and_adducts_are_not_votes():
    inputs = list(fixture(cohorts=3, shifts=(8.0, 9.0)))
    result = call(inputs)
    assert result.loc[7, "standard_alignment_anchor_compounds"] == 2
    assert result.loc[7, "standard_alignment_shift_seconds"] == pytest.approx(
        8.5, abs=0.01
    )
    # Repeated detections of one chemical cannot satisfy the two-compound rule.
    inputs = list(fixture(cohorts=3))
    inputs[3] = pd.concat(
        [inputs[3], inputs[3].assign(feat_feature_id="duplicate")], ignore_index=True
    )
    assert call(inputs).standard_aligned_rt_seconds.isna().all()


def test_query_leakage_cross_stream_and_manual_flags_are_rejected():
    inputs = fixture()
    with pytest.raises(ValueError, match="excluded"):
        standard_anchored_retention(*inputs, stream="RP_Pos", query_group="g0")
    for column in ("same_manual_peak", "manual_confirmed"):
        inputs = list(fixture())
        inputs[1] = pd.DataFrame({column: [True], "feat_feature_id": ["curated"]})
        with pytest.raises(ValueError, match="label-free"):
            call(inputs)
    inputs = list(fixture())
    inputs[3] = inputs[3].assign(stream="NP_Pos")
    with pytest.raises(ValueError, match="streams"):
        call(inputs)


def test_missing_measurements_and_out_of_domain_compositions_do_not_invent_RT():
    inputs = list(fixture())
    inputs[3] = inputs[3].iloc[:0]
    assert call(inputs).standard_aligned_rt_seconds.isna().all()
    inputs = list(fixture())
    inputs[0] = inputs[0].assign(carbon=100)
    assert call(inputs).standard_aligned_rt_seconds.isna().all()
    assert np.isnan(call(inputs).standard_clean_atlas_rt_seconds).all()


@pytest.mark.parametrize("cohorts", [np.nan, np.inf, 0])
def test_unknown_transfer_scope_counts_cannot_qualify_an_anchor(cohorts):
    inputs = fixture()
    inputs[4]["transfer"]["models"]["RP_Pos|PE_std_0"]["training_cohorts"] = cohorts
    assert call(inputs).standard_aligned_rt_seconds.isna().all()
