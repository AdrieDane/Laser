"""Synthetic measurement tests; not annotation validation."""

import numpy as np
import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)

from bmp_context_measurements import measure, PREFIX
from matlab_annotation.vendor.compound_model import PROTON_MASS


def fixture():
    families = {
        ("NP", "BMP"): {"[M-H]-": (0.0, -1), "[M+NH4]+": (17.026549, 1)},
        ("NP", "PA"): {"[M-H]-": (0.0, -1), "[M+CHO2]-": (46.00548, -1)},
    }
    pm = 800 + 2 * PROTON_MASS + 17.026549
    rows = [
        ("NP_Pos", "BMP(36:2)", "BMP", "[M+NH4]+", "p", pm, 6.0),
        ("NP_Neg", "BMP(36:2)", "BMP", "[M-H]-", "n", 800.0, 7.0),
        ("NP_Neg", "PG(36:2)", "PG", "[M-H]-", "g", 800.0, 8.0),
        ("NP_Neg", "PA(38:2)", "PA", "[M-H]-", "a", 753.99452, 7.01),
    ]
    pool = pd.DataFrame(
        [
            dict(
                stream=s,
                name=n,
                **{"class": c},
                adduct=a,
                feat_feature_id=f,
                ref_mz=m,
                feat_mz=m,
                feat_rt=t,
                feat_intensity=100.0,
                is_is=False,
            )
            for s, n, c, a, f, m, t in rows
        ]
    )
    features, maps = {}, {}
    for stream in ("NP_Pos", "NP_Neg"):
        part = pool.loc[pool.stream.eq(stream)]
        frame = (
            part[["feat_feature_id", "feat_mz", "feat_rt", "feat_intensity"]]
            .rename(
                columns={
                    "feat_feature_id": "feature_id",
                    "feat_mz": "mz",
                    "feat_rt": "rt",
                    "feat_intensity": "intensity",
                }
            )
            .reset_index(drop=True)
        )
        samples = ["S" + str(i) for i in range(8)]
        for i, col in enumerate(samples):
            frame[col] = [
                float(np.exp(i / 4) * (10 if f == "a" else 1)) for f in frame.feature_id
            ]
        frame.attrs.update(
            sample_columns=samples,
            sample_roles={s: "biological" for s in samples},
            sample_group_map={s: "tissue" for s in samples},
        )
        features[stream] = frame
        maps[stream] = {s: s for s in samples}
    return pool, features, maps, families, pd.Series(True, index=pool.index)


def test_only_same_class_own_RT_candidate_not_unrestricted_isobar_or_RT_transfer():
    p, f, m, fam, physical = fixture()
    result = measure(p, f, m, fam, physical)
    assert list(result.index) == [0, 1]
    assert result.loc[0, PREFIX + "candidate_counterparts"] == 1
    assert result.loc[0, PREFIX + "counterpart_raw"] == pytest.approx(1)
    # RP/NP or positive/negative RT equality is not needed: each candidate
    # passes its OWN masked RT window. Removing only BMP target eligibility
    # leaves a perfectly mass-matched PG raw peak, which must NOT replace it.
    physical.loc[1] = False
    result = measure(p, f, m, fam, physical)
    assert result.loc[0, PREFIX + "candidate_counterparts"] == 0
    assert np.isnan(result.loc[0, PREFIX + "counterpart_raw"])


def test_actual_PA_formate_offset_coelution_profile_and_tenfold_ratio():
    p, f, m, fam, physical = fixture()
    original = p.copy(deep=True)
    result = measure(p, f, m, fam, physical)
    assert result.loc[1, PREFIX + "PA_candidates"] == 1
    assert result.loc[1, PREFIX + "PA_raw"] == pytest.approx(1)
    assert result.loc[1, PREFIX + "PA_log_ratio_location"] == pytest.approx(np.log(10))
    assert result.loc[1, PREFIX + "PA_RT_seconds"] == pytest.approx(0.6)
    assert result.loc[0, PREFIX + "counterpart_PA_log_ratio_location"] == pytest.approx(
        np.log(10)
    )
    pd.testing.assert_frame_equal(p, original, check_exact=True)
    physical.loc[3] = False
    result = measure(p, f, m, fam, physical)
    assert result.loc[1, PREFIX + "PA_candidates"] == 0
    assert np.isnan(result.loc[1, PREFIX + "PA_raw"])


def test_PA_outside_two_seconds_or_three_ppm_is_not_support():
    p, f, m, fam, physical = fixture()
    p.loc[3, "feat_rt"] = 7.1
    f["NP_Neg"].loc[f["NP_Neg"].feature_id.eq("a"), "rt"] = 7.1
    assert measure(p, f, m, fam, physical).loc[1, PREFIX + "PA_candidates"] == 0
    p.loc[3, "feat_rt"] = 7.01
    f["NP_Neg"].loc[f["NP_Neg"].feature_id.eq("a"), "rt"] = 7.01
    p.loc[3, "feat_mz"] *= 1 + 3.01e-6
    f["NP_Neg"].loc[f["NP_Neg"].feature_id.eq("a"), "mz"] = p.loc[3, "feat_mz"]
    assert measure(p, f, m, fam, physical).loc[1, PREFIX + "PA_candidates"] == 0


def test_unknown_pairing_and_QC_are_not_fabricated_biological_support():
    p, f, m, fam, physical = fixture()
    result = measure(p, f, {}, fam, physical)
    assert result.loc[0, PREFIX + "candidate_counterparts"] == 1
    assert result.loc[0, PREFIX + "counterpart_profiles"] == 0
    assert np.isnan(result.loc[0, PREFIX + "counterpart_raw"])
    for frame in f.values():
        frame.attrs["sample_roles"]["S0"] = "qc"
    result = measure(p, f, m, fam, physical)
    assert result.loc[0, PREFIX + "counterpart_profiles"] == 0
    assert result.loc[1, PREFIX + "PA_profiles"] == 0


def test_supervision_or_nonoriginal_coordinates_fail():
    p, f, m, fam, physical = fixture()
    with pytest.raises(ValueError, match="supervision"):
        measure(p.assign(training_positive=False), f, m, fam, physical)
    p.loc[0, "feat_rt"] += 0.001
    with pytest.raises(AssertionError):
        measure(p, f, m, fam, physical)


def test_duplicate_partner_hypotheses_do_not_inflate_physical_link_counts():
    p, f, m, fam, physical = fixture()
    p = pd.concat([p, p.iloc[[1, 3]]], ignore_index=True)
    result = measure(p, f, m, fam, pd.Series(True, index=p.index))
    assert result.loc[0, PREFIX + "candidate_counterparts"] == 1
    assert result.loc[0, PREFIX + "counterpart_profiles"] == 1
    assert result.loc[1, PREFIX + "PA_candidates"] == 1
    assert result.loc[1, PREFIX + "PA_profiles"] == 1


def test_independent_scalar_charge_graph_and_numpy_profile_reconstruction():
    from verify_qe_bmp_context_pilot import independent_measure

    p, f, m, fam, physical = fixture()
    expected = independent_measure(p, f, m, fam, physical)
    pd.testing.assert_frame_equal(
        measure(p, f, m, fam, physical),
        expected,
        check_exact=False,
        atol=1e-6,
        rtol=1e-8,
    )
    pd.testing.assert_frame_equal(
        measure(p, f, {}, fam, physical),
        independent_measure(p, f, {}, fam, physical),
        check_exact=False,
        atol=1e-6,
        rtol=1e-8,
    )


def test_technical_replicates_count_once_and_average_areas_before_log():
    p, f, m, fam, physical = fixture()
    for stream, frame in f.items():
        frame["T0"] = frame["S0"] * 3
        frame.attrs["sample_columns"].append("T0")
        frame.attrs["sample_roles"]["T0"] = "biological"
        frame.attrs["sample_group_map"]["T0"] = "tissue"
        m[stream]["T0"] = "S0"
    result = measure(p, f, m, fam, physical)
    assert result.loc[0, PREFIX + "counterpart_n"] == 8
    assert result.loc[1, PREFIX + "PA_n"] == 8
    assert result.loc[1, PREFIX + "PA_log_ratio_location"] == pytest.approx(np.log(10))
    from verify_qe_bmp_context_pilot import independent_measure

    pd.testing.assert_frame_equal(
        result,
        independent_measure(p, f, m, fam, physical),
        check_exact=False,
        atol=1e-6,
        rtol=1e-8,
    )
