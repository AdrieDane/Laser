"""Synthetic cross-phase measurements, not annotation-performance validation."""

import numpy as np
import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
from cross_phase_candidate_measurements import measure, PREFIX, FIELDS


def fixture():
    pool = pd.DataFrame(
        [
            dict(
                stream=s,
                name="FA(18:1)",
                **{"class": "FA"},
                adduct="[M-H]-",
                feat_feature_id="same_id",
                ref_mz=300.0,
                feat_mz=300.0,
                feat_rt=rt,
                ref_rt=rt,
                feat_intensity=100.0,
                is_is=False,
            )
            for s, rt in (("RP_Neg", 10.0), ("NP_Neg", 2.0))
        ]
    )
    families = {(phase, "FA"): {"[M-H]-": (0.0, -1)} for phase in ("RP", "NP")}
    features, maps = {}, {}
    samples = ["S" + str(i) for i in range(8)] + ["S0_repeat"]
    for s in ("RP_Neg", "NP_Neg"):
        rt = 10.0 if s == "RP_Neg" else 2.0
        frame = pd.DataFrame(
            dict(
                feature_id=["same_id", "background"],
                mz=[300.0, 600.0],
                rt=[rt, rt + 1],
                intensity=[100.0, 100.0],
            )
        )
        for i, col in enumerate(samples):
            base = np.exp(i / 4) if i < 8 else 9.0
            frame[col] = [base * (2 if s == "NP_Neg" else 1), 100.0]
        # Mean technical areas5,not geometric mean3: NP is exactly twiceRP.
        frame.loc[0, "S0"] = 1.0 if s == "RP_Neg" else 2.0
        frame.attrs.update(
            sample_columns=samples,
            sample_roles={c: "biological" for c in samples},
            sample_group_map={c: "tissue" for c in samples},
        )
        features[s] = frame
        maps[s] = {
            c: ("B0" if c in ("S0", "S0_repeat") else "B" + c[1:]) for c in samples
        }
    return pool, features, maps, families, pd.Series(True, index=pool.index)


def test_different_phase_RT_and_same_feature_name_are_not_a_self_match():
    p, f, m, fam, physical = fixture()
    result, links = measure(p, f, m, fam, physical)
    assert list(result) == [PREFIX + c for c in FIELDS] and len(links) == 2
    assert result[PREFIX + "candidate_counterparts"].tolist() == [1.0, 1.0]
    assert result[PREFIX + "best_raw"].tolist() == pytest.approx([1.0, 1.0])
    assert result.loc[0, PREFIX + "best_n"] == 8
    assert result.loc[0, PREFIX + "best_log_ratio_location"] == pytest.approx(np.log(2))
    assert result.loc[1, PREFIX + "best_log_ratio_location"] == pytest.approx(
        -np.log(2)
    )
    assert links.source_feature.eq(links.target_feature).all()
    assert links.source_stream.ne(links.target_stream).all()


def test_source_and_partner_need_own_physical_RT_mass_standard_eligibility():
    p, f, m, fam, physical = fixture()
    physical.loc[1] = False
    result, links = measure(p, f, m, fam, physical)
    assert result.loc[0, PREFIX + "candidate_counterparts"] == 0
    assert np.isnan(result.loc[0, PREFIX + "best_raw"]) and links.empty
    assert result.loc[1, PREFIX + "source_physical"] == 0
    p.loc[1, "is_is"] = True
    result, links = measure(p, f, m, fam, pd.Series(True, index=p.index))
    assert result.index.tolist() == [0] and links.empty


def test_duplicate_hypotheses_do_not_inflate_physical_partner_support():
    p, f, m, fam, physical = fixture()
    p = pd.concat([p, p.iloc[[1]]], ignore_index=True)
    result, links = measure(p, f, m, fam, pd.Series(True, index=p.index))
    assert result.loc[0, PREFIX + "candidate_counterparts"] == 1
    assert result.loc[0, PREFIX + "counterpart_profiles"] == 1
    assert len(links.loc[links.source_index.eq(0)]) == 1


def test_unknown_or_flat_profiles_remain_unknown_not_fabricated_support():
    p, f, m, fam, physical = fixture()
    result, _ = measure(p, f, {}, fam, physical)
    assert result[PREFIX + "candidate_counterparts"].eq(1).all()
    assert result[PREFIX + "counterpart_profiles"].eq(0).all()
    assert result[PREFIX + "best_raw"].isna().all()
    for frame in f.values():
        frame.loc[0, frame.attrs["sample_columns"]] = 10.0
    result, _ = measure(p, f, m, fam, physical)
    assert result[PREFIX + "counterpart_profiles"].eq(0).all()


def test_wrong_class_or_different_theoretical_formula_not_a_partner():
    p, f, m, fam, physical = fixture()
    p.loc[1, "class"] = "FA-OH"
    fam[("NP", "FA-OH")] = fam[("RP", "FA-OH")] = {"[M-H]-": (0.0, -1)}
    result, links = measure(p, f, m, fam, physical)
    assert links.empty
    p, f, m, fam, physical = fixture()
    p.loc[1, "ref_mz"] *= 1 + 0.051e-6
    result, links = measure(p, f, m, fam, physical)
    assert links.empty and result[PREFIX + "candidate_counterparts"].eq(0).all()


def test_explicit_groups_and_original_coordinates_and_no_supervision_required():
    p, f, m, fam, physical = fixture()
    with pytest.raises(ValueError, match="supervision"):
        measure(p.assign(training_positive=False), f, m, fam, physical)
    changed = p.copy()
    changed.loc[0, "feat_rt"] += 0.001
    with pytest.raises(AssertionError):
        measure(changed, f, m, fam, physical)
    f["NP_Neg"].attrs["sample_group_map"] = {
        c: "other" for c in f["NP_Neg"].attrs["sample_columns"]
    }
    with pytest.raises(ValueError, match="group"):
        measure(p, f, m, fam, physical)


def test_independent_scalar_graph_and_numpy_profiles_including_unknown_and_empty():
    from cross_phase_candidate_independent import independent_measure

    for kind in (
        "ordinary",
        "no_mapping",
        "blocked_partner",
        "wrong_formula",
        "duplicates",
    ):
        p, f, m, fam, physical = fixture()
        if kind == "no_mapping":
            m = {}
        elif kind == "blocked_partner":
            physical.loc[1] = False
        elif kind == "wrong_formula":
            p.loc[1, "ref_mz"] *= 1 + 0.051e-6
        elif kind == "duplicates":
            p = pd.concat([p, p.iloc[[1]]], ignore_index=True)
            physical = pd.Series(True, index=p.index)
        actual, links = measure(p, f, m, fam, physical)
        expected, wanted = independent_measure(p, f, m, fam, physical)
        pd.testing.assert_frame_equal(
            actual, expected, check_exact=False, rtol=1e-8, atol=1e-6
        )
        pd.testing.assert_frame_equal(
            links, wanted, check_exact=False, rtol=1e-8, atol=1e-6
        )


def test_actual_signed_charge_inversion_not_a_fixed_deprotonated_mass_assumption():
    from cross_phase_candidate_independent import independent_measure
    from matlab_annotation.vendor.compound_model import PROTON_MASS

    p, f, m, fam, physical = fixture()
    p.loc[0, "adduct"] = "[M-2H]2-"
    mz = (300.0 - PROTON_MASS) / 2
    p.loc[0, ["ref_mz", "feat_mz"]] = mz
    f["RP_Neg"].loc[0, "mz"] = mz
    fam[("RP", "FA")] = {"[M-2H]2-": (0.0, -2)}
    actual, links = measure(p, f, m, fam, physical)
    expected, wanted = independent_measure(p, f, m, fam, physical)
    assert actual[PREFIX + "candidate_counterparts"].eq(1).all()
    pd.testing.assert_frame_equal(
        actual, expected, check_exact=False, rtol=1e-8, atol=1e-6
    )
    pd.testing.assert_frame_equal(
        links, wanted, check_exact=False, rtol=1e-8, atol=1e-6
    )


def test_hydroxy_fatty_acid_native_class_is_measured_in_both_negative_phases():
    from cross_phase_candidate_independent import independent_measure

    p, f, m, _fam, physical = fixture()
    p["class"] = "FA-OH"
    p["name"] = "FA-OH(18:1)"
    families = {(phase, "FA-OH"): {"[M-H]-": (0.0, -1)} for phase in ("RP", "NP")}
    actual, links = measure(p, f, m, families, physical)
    expected, wanted = independent_measure(p, f, m, families, physical)
    assert actual[PREFIX + "counterphase_configured"].eq(1).all()
    assert actual[PREFIX + "candidate_counterparts"].eq(1).all()
    pd.testing.assert_frame_equal(
        actual, expected, check_exact=False, rtol=1e-8, atol=1e-6
    )
    pd.testing.assert_frame_equal(
        links, wanted, check_exact=False, rtol=1e-8, atol=1e-6
    )
