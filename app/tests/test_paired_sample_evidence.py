import numpy as np
import pandas as pd
import pytest

from matlab_annotation.paired_sample_evidence import paired_sample_evidence


def fixture():
    features, maps = {}, {}
    samples = [f"sample{i}" for i in range(12)]
    profile = np.exp(np.array([1, 2, 1.5, 3, 1.7, 2.3, 3.5, 2.1, 4, 2.8, 3.2, 4.2]))
    for stream, scale in [("RP_Pos", 1), ("NP_Neg", 5)]:
        features[stream] = pd.DataFrame(dict(
            feature_id=["right", "wrong", "background"],
            **{s: [scale * profile[i], scale * profile[::-1][i], 100]
               for i, s in enumerate(samples)}))
        features[stream].attrs = dict(
            sample_columns=samples,
            sample_group_mapping_status="validated",
            sample_group_map={s: "Group_1_Tissue" for s in samples})
        maps[stream] = {s: str(i) for i, s in enumerate(samples)}
    automatic = pd.DataFrame([dict(
        stream="NP_Neg", name="PE(36:2)", **{"class": "PE"},
        carbon=36, db=2, feat_feature_id="right", feat_rt=5,
        feat_intensity=100, ppm_error=.1, post_regression_confidence=.99, is_is=False)])
    queries = pd.DataFrame([dict(
        stream="RP_Pos", **{"class": "PE"}, carbon=36, db=2,
        feat_feature_id=fid) for fid in ["right", "wrong"]])
    return queries, features, automatic, maps


def test_exact_paired_profiles_distinguish_original_peaks_without_rt_comparison():
    args = fixture()
    result = paired_sample_evidence(*args)
    assert result.iloc[0].q_paired_best_log_correlation == pytest.approx(1)
    assert result.iloc[0].q_paired_best_within_group_correlation == pytest.approx(1)
    assert result.iloc[0].q_paired_best_log_ratio_mad == pytest.approx(0)
    assert result.iloc[0].q_paired_profile_specimens == 12
    assert result.iloc[0].q_paired_within_group_dof == 11
    assert result.iloc[1].q_paired_best_log_correlation < .8


def test_column_order_is_irrelevant_and_inputs_unchanged():
    q, features, auto, maps = fixture()
    original = features["RP_Pos"].copy(deep=True)
    expected = paired_sample_evidence(q, features, auto, maps)
    features["NP_Neg"] = features["NP_Neg"][list(reversed(features["NP_Neg"].columns))]
    pd.testing.assert_frame_equal(expected, paired_sample_evidence(q, features, auto, maps))
    pd.testing.assert_frame_equal(original, features["RP_Pos"])


def test_technical_replicates_do_not_multiply_specimens():
    q, features, auto, maps = fixture()
    expected = paired_sample_evidence(q, features, auto, maps)
    for stream, f in features.items():
        f["duplicate"] = f["sample0"]
        f.attrs["sample_columns"] = [*f.attrs["sample_columns"], "duplicate"]
        f.attrs["sample_group_map"]["duplicate"] = "Group_1_Tissue"
        maps[stream]["duplicate"] = "0"
    pd.testing.assert_frame_equal(expected, paired_sample_evidence(q, features, auto, maps))


def test_missing_pairing_and_insufficient_biological_samples_stay_unknown():
    q, features, auto, maps = fixture()
    assert paired_sample_evidence(q, features, auto, {}).isna().all().all()
    for f in features.values():
        f.attrs["sample_roles"] = {s: "biological" if i < 7 else "qc"
                                    for i, s in enumerate(f.attrs["sample_columns"])}
    result = paired_sample_evidence(q, features, auto, maps)
    assert result.q_paired_best_log_correlation.isna().all()


def test_no_ether_positional_or_standard_substitution_or_self_stream_support():
    q, features, auto, maps = fixture()
    auto["class"] = "PE[P]"
    assert paired_sample_evidence(q, features, auto, maps).q_paired_best_log_correlation.isna().all()
    auto["class"] = "PE"
    auto["is_is"] = True
    assert paired_sample_evidence(q, features, auto, maps).q_paired_best_log_correlation.isna().all()
    auto["is_is"] = False
    auto["stream"] = "RP_Pos"
    assert paired_sample_evidence(q, features, auto, maps).q_paired_best_log_correlation.isna().all()


def test_flat_profiles_and_conflicting_specimen_groups_cannot_supply_evidence():
    q, features, auto, maps = fixture()
    features["NP_Neg"].loc[0, features["NP_Neg"].attrs["sample_columns"]] = 100
    assert paired_sample_evidence(q, features, auto, maps).q_paired_best_log_correlation.isna().all()
    features["NP_Neg"].attrs["sample_group_map"]["sample0"] = "Other tissue"
    with pytest.raises(ValueError, match="groups disagree"):
        paired_sample_evidence(q, features, auto, maps)
