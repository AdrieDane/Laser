from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from matlab_annotation.adduct_family_evidence import native_adduct_families
from matlab_annotation.raw_counterpart_evidence import (
    FIELDS,
    PREFIX,
    _family,
    _minimum_leave_one_out,
    raw_counterpart_evidence,
)
from matlab_annotation.raw_counterpart_rt import template_mass
from matlab_annotation.vendor.compound_model import PROTON_MASS

ROOT = Path(__file__).resolve().parents[2] / "reference_data/matlab_pipeline"


def fixture():
    samples = [f"sample{i}" for i in range(12)]
    log_profile = np.array([1, 2, 1.5, 3, 1.7, 2.3, 3.5, 2.1, 4, 2.8, 3.2, 4.2])
    features, maps = {}, {}
    for stream, scale, mz, rt in (
        ("RP_Pos", 1.0, 500.0, 5.0),
        ("RP_Neg", 5.0, 543.990928, 5.3),
    ):
        frame = pd.DataFrame(
            dict(
                feature_id=["right", "wrong", *[f"background{j}" for j in range(5)]],
                mz=[mz, mz * (1 + 1e-6), *range(2000, 2005)],
                rt=[rt, rt + 0.03, *range(1, 6)],
                intensity=100.0,
                **{
                    s: [
                        scale * np.exp(log_profile[i]),
                        scale * np.exp(log_profile[::-1][i]),
                        *[100] * 5,
                    ]
                    for i, s in enumerate(samples)
                },
            )
        )
        frame.attrs = dict(
            sample_columns=list(samples),
            sample_group_mapping_status="validated",
            sample_group_map={s: "Group_1_Tissue" for s in samples},
        )
        features[stream] = frame
        maps[stream] = {s: str(i) for i, s in enumerate(samples)}
    pool = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                **{"class": "PC"},
                adduct="[M+H]+",
                ref_mz=500.0,
                feat_feature_id=fid,
                feat_rt=features["RP_Pos"].rt.iloc[i],
                feat_mz=features["RP_Pos"].mz.iloc[i],
            )
            for i, fid in enumerate(["right", "wrong"])
        ],
        index=[7, 13],
    )
    return pool, features, maps


def measure(pool, features, maps, **kwargs):
    return raw_counterpart_evidence(
        pool, features, ROOT, maps, stream="RP_Pos", **kwargs
    )


def test_unassigned_original_partner_supplies_measurement_not_annotation():
    pool, features, maps = fixture()
    original = deepcopy((pool, features, maps))
    result = measure(pool, features, maps)
    r = result.loc[7]
    assert len(FIELDS) == 32 and len(result.columns) == 32
    assert r[PREFIX + "distinct_mass_hits"] == 2
    assert r[PREFIX + "best_raw_correlation"] == pytest.approx(1)
    assert r[PREFIX + "best_within_group_correlation"] == pytest.approx(1)
    assert r[PREFIX + "best_spearman_correlation"] == pytest.approx(1)
    assert r[PREFIX + "best_leave_one_out_correlation"] == pytest.approx(1)
    assert r[PREFIX + "best_log_ratio_location"] == pytest.approx(np.log(5))
    assert r[PREFIX + "best_log_ratio_mad"] == pytest.approx(0)
    assert r[PREFIX + "best_profile_pairs"] == 12
    assert r[PREFIX + "best_within_group_dof"] == 11
    assert r[PREFIX + "best_raw_RT_delta_seconds"] == pytest.approx(18)
    assert r[PREFIX + "RT_map_available"] == 0
    assert np.isnan(r[PREFIX + "best_RT_residual_seconds"])
    assert np.isnan(r[PREFIX + "RT_supported_profile_hits"])
    assert all(c.startswith(PREFIX) for c in result)
    pd.testing.assert_frame_equal(pool, original[0], check_exact=True)
    for s in features:
        pd.testing.assert_frame_equal(features[s], original[1][s], check_exact=True)
    assert maps == original[2]


def test_duplicate_candidate_versions_and_column_order_do_not_multiply_peaks():
    pool, features, maps = fixture()
    repeated = pd.concat([pool, pool.iloc[[0]].set_axis([99])])
    expected = measure(repeated, features, maps)
    pd.testing.assert_series_equal(expected.loc[7], expected.loc[99], check_names=False)
    for s in features:
        features[s] = features[s][list(reversed(features[s].columns))]
    pd.testing.assert_frame_equal(expected, measure(repeated, features, maps))


def test_strict_mass_and_same_phase_other_polarity_only():
    pool, features, maps = fixture()
    features["RP_Neg"].loc[:1, "mz"] = 543.990928 * (1 + 3.01e-6)
    r = measure(pool, features, maps)
    assert r[PREFIX + "distinct_mass_hits"].eq(0).all()
    assert r[PREFIX + "best_raw_correlation"].isna().all()
    features["NP_Neg"] = features.pop("RP_Neg")
    r = measure(pool, features, maps)
    assert r[PREFIX + "distinct_mass_hits"].isna().all()
    assert r[PREFIX + "pairing_available"].eq(0).all()


def test_explicit_biological_pairing_required_and_technical_replicates_count_once():
    pool, features, maps = fixture()
    expected = measure(pool, features, maps)
    for stream, f in features.items():
        f["duplicate"] = f["sample0"]
        f.attrs["sample_columns"].append("duplicate")
        f.attrs["sample_group_map"]["duplicate"] = "Group_1_Tissue"
        maps[stream]["duplicate"] = "0"
    pd.testing.assert_frame_equal(expected, measure(pool, features, maps))
    assert measure(pool, features, {})[PREFIX + "best_raw_correlation"].isna().all()
    for f in features.values():
        f.attrs["sample_roles"] = {
            s: "biological" if s in {f"sample{i}" for i in range(7)} else "qc"
            for s in f.attrs["sample_columns"]
        }
    r = measure(pool, features, maps)
    assert r[PREFIX + "paired_bio_specimens"].eq(7).all()
    assert r[PREFIX + "best_raw_correlation"].isna().all()
    features["RP_Neg"].attrs["sample_group_map"]["sample0"] = "Other_tissue"
    with pytest.raises(ValueError, match="groups disagree"):
        measure(pool, features, maps)


def test_technical_replicate_intensities_are_averaged_before_logarithm():
    pool, features, maps = fixture()
    expected = measure(pool, features, maps)
    for stream, f in features.items():
        original = f["sample0"].copy()
        f["sample0"] = original * 0.5
        f["duplicate"] = original * 1.5
        f.attrs["sample_columns"].append("duplicate")
        f.attrs["sample_group_map"]["duplicate"] = "Group_1_Tissue"
        maps[stream]["duplicate"] = "0"
    pd.testing.assert_frame_equal(expected, measure(pool, features, maps))


def test_source_coordinates_supervision_and_duplicates_are_hard_guards():
    pool, features, maps = fixture()
    with pytest.raises(AssertionError):
        measure(pool.assign(feat_rt=9), features, maps)
    with pytest.raises(ValueError, match="supervision"):
        measure(pool.assign(training_positive=True), features, maps)
    with pytest.raises(ValueError, match="feature IDs"):
        measure(pool.assign(feat_feature_id="absent"), features, maps)
    features["RP_Neg"].loc[1, "feature_id"] = "right"
    with pytest.raises(ValueError, match="Unique original"):
        measure(pool, features, maps)


def test_independent_RT_link_not_raw_RT_delta_controls_supported_hits():
    pool, features, maps = fixture()
    links = dict(
        source_stream="RP_Pos",
        target_stream="RP_Neg",
        class_maps={},
        global_map=dict(
            slope=1,
            intercept_seconds=18,
            anchor_compounds=6,
            mad_seconds=0.5,
            minimum_source_seconds=200,
            maximum_source_seconds=600,
        ),
    )
    r = measure(pool, features, maps, rt_links=links).loc[7]
    assert r[PREFIX + "RT_map_available"] == 1
    assert r[PREFIX + "best_RT_residual_seconds"] == pytest.approx(0)
    assert r[PREFIX + "RT_supported_profile_hits"] == 2
    links["global_map"]["intercept_seconds"] = -20
    r = measure(pool, features, maps, rt_links=links).loc[7]
    assert r[PREFIX + "RT_supported_profile_hits"] == 0
    assert np.isnan(r[PREFIX + "best_RT_supported_correlation"])
    links["global_map"]["minimum_source_seconds"] = 500
    assert (
        measure(pool, features, maps, rt_links=links)[PREFIX + "RT_map_available"]
        .eq(0)
        .all()
    )
    links["target_stream"] = "NP_Neg"
    with pytest.raises(ValueError, match="cannot cross"):
        measure(pool, features, maps, rt_links=links)


def test_native_charge_math_and_unresolved_NP_branch_templates():
    families = native_adduct_families(str(ROOT))
    family = _family(families, "RP", "CL")
    mass = 619.416238
    backbone = template_mass(mass, "[M-2H](2-)", family, positive=False)
    assert backbone is not None
    offset, charge = family["[M+NH4]+"]
    partner = abs((backbone + offset) / charge + PROTON_MASS)
    assert partner == pytest.approx(1258.880854, abs=2e-5)
    pool, features, maps = fixture()
    pool = pool.iloc[[0]].assign(
        stream="RP_Neg",
        **{"class": "CL"},
        adduct="[M-2H](2-)",
        ref_mz=mass,
        feat_rt=5.3,
        feat_mz=mass,
    )
    features["RP_Neg"].loc[0, "mz"] = mass
    features["RP_Pos"].loc[0, "mz"] = partner
    result = raw_counterpart_evidence(pool, features, ROOT, maps, stream="RP_Neg")
    assert result.iloc[0][PREFIX + "distinct_mass_hits"] == 1
    assert _family(families, "NP", "PC-O+P") == families[("NP", "PC[O]")]
    mismatched = {("NP", "PC[O]"): {"x": (1, 1)}, ("NP", "PC[P]"): {"x": (2, 1)}}
    assert _family(mismatched, "NP", "PC-O+P") == {}


def test_leave_one_out_is_actual_minimum_and_degenerate_is_unknown():
    a = np.arange(12.0)
    b = a.copy()
    b[0] = 8
    expected = min(
        np.corrcoef(np.delete(a, i), np.delete(b, i))[0, 1] for i in range(12)
    )
    assert _minimum_leave_one_out(a, b) == pytest.approx(expected)
    assert np.isnan(_minimum_leave_one_out(np.r_[np.ones(7), 2], np.arange(8.0)))
    assert np.isnan(_minimum_leave_one_out(a[:7], b[:7]))
