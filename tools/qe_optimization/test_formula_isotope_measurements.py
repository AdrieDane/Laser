"""Native formula/charge/isotope synthetic and catalogue checks; no gold."""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
import formula_isotope_measurements as m
import formula_isotope_independent as independent
from matlab_annotation.vendor.compound_model import PROTON_MASS

ROOT = REPO / "reference_data/matlab_pipeline"


def test_formula_groups_and_reject_syntax():
    assert m.atom_counts("C10H18NO8PR2", "FA", 2, 34, 1) == dict(
        C=42, H=82, N=1, O=8, P=1
    )
    assert m.atom_counts("C2H7NO2R", "ST", 1, 18, 1) == dict(C=20, H=35, N=1, O=2)
    assert m.atom_counts("H2R", "isoprene", 1, 10, 0) == dict(C=50, H=82)
    for text in ("C(OH)2", "C13H-1", "13C2H4"):
        with pytest.raises(ValueError):
            m.parse_formula(text)


@pytest.mark.parametrize("ion", list(m.ADDUCT_ATOMS))
def test_all_native_adduct_atom_deltas_and_signed_mass(ion):
    from matlab_annotation.vendor.composition_initializer import _ADDUCTS

    atoms, charge = m.ion_formula(dict(C=40, H=80, O=10, N=1, P=1), ion)
    delta, z = m.ADDUCT_ATOMS[ion]
    assert z == charge
    assert dict(independent.ion_delta(ion)) == delta
    isotopes = m.isotope_table(ROOT)
    neutral = sum(
        isotopes[a]["mass"] * n for a, n in dict(C=40, H=80, O=10, N=1, P=1).items()
    )
    _, addition, native_charge = next(v for v in _ADDUCTS.values() if v[0] == ion)
    native = abs((neutral + addition) / native_charge + PROTON_MASS)
    electron = isotopes["H"]["mass"] - PROTON_MASS
    computed = (
        sum(isotopes[a]["mass"] * n for a, n in atoms.items()) - charge * electron
    ) / abs(charge)
    assert computed == pytest.approx(native, abs=1e-10)


def test_moments_match_independent_polynomial_and_same_formula_isomers():
    iso = m.isotope_table(ROOT)
    _, other = independent.resources(ROOT)
    for atoms in (
        dict(C=42, H=83, N=1, O=8, P=1),
        dict(C=70, H=120, O=25, S=2, Cl=1, K=1),
    ):
        c1, n1, c2, n2 = m.moments(atoms, iso)
        assert independent.polynomial(atoms, other) == pytest.approx((n1, n2))
        assert independent.polynomial(atoms, other, True) == pytest.approx((c1, c2))
    # Native O-n:d == P-n:(d-1), including the ether's additional H2.
    ether = m.atom_counts("C10H20NO7PR2", "FA", 2, 36, 2)
    plasmalogen = m.atom_counts("C10H18NO7PR2", "FA", 2, 36, 1)
    assert ether == plasmalogen


@pytest.mark.parametrize(
    "source,partner", [(0.01, 0.003), (100, 30), (0.1, 4), (1e20, 2e19)]
)
def test_exact_log1p_inversion_not_log_difference(source, partner):
    origin = np.log1p(source)
    difference = np.log1p(partner) - origin
    assert m.true_log_ratio(origin, difference) == pytest.approx(
        math.log(partner / source), abs=1e-12
    )
    assert m.true_log_ratio(origin, difference) == pytest.approx(
        independent.inverse_ratio(origin, difference)
    )


def test_missing_zero_and_standards_unknown_not_zero():
    assert np.isnan(m.true_log_ratio(0, 1))
    assert np.isnan(m.true_log_ratio(1, -1))
    assert np.isnan(m.true_log_ratio(np.nan, 0))
    row = dict(is_is=True)
    assert m.hypothesis_formula(row, {}, {})[1] == "standard_or_labelled"


def test_native_dense_catalogue_parity_all_definitions():
    from matlab_annotation.decode import CodeDecoder
    from matlab_annotation.xcms_project import load_bundled_compound_catalog
    from matlab_annotation.vendor.compound_model import (
        theoretical_mz,
        phase_composition,
    )
    from annotator.np_rt import combine_np_ether_plasmalogen_labels

    catalog = load_bundled_compound_catalog(ROOT)
    lookup, iso = m.definitions(ROOT), m.isotope_table(ROOT)
    definitions, iso2 = independent.resources(ROOT)
    rows = []
    for phase in ("RP", "NP"):
        for idx, attr in enumerate(catalog.composition_attr):
            if phase not in tuple(attr.get("column", ())):
                continue
            if not phase_composition(attr, phase).charges:
                continue
            decoder = CodeDecoder(SimpleNamespace(compounds=catalog), idx + 1, phase)
            # Use low-DB/middle-C native axes, EACH phase-specific adduct.
            for adduct in range(len(decoder.axis(0))):
                code = decoder.code_for({0: adduct, 3: len(decoder.axis(3)) // 2})
                decoded = decoder.decode(code)
                row = dict(
                    stream=phase
                    + (
                        "_Pos"
                        if decoded.adduct.endswith("+") or "+)" in decoded.adduct
                        else "_Neg"
                    ),
                    name=decoded.species,
                    **{"class": decoded.lipid_class},
                    adduct=decoded.adduct,
                    carbon=decoded.carbon or 0,
                    db=decoded.double_bonds or 0,
                    ref_mz=abs(theoretical_mz(attr, phase, code)),
                    is_is=False,
                    isotopologue="",
                )
                frame = pd.DataFrame([row])
                if phase == "NP":
                    frame = combine_np_ether_plasmalogen_labels(frame)
                row = frame.iloc[0]
                produced = m.hypothesis_formula(row, lookup, iso)
                checked = independent.hypothesis(row, definitions, iso2)
                assert produced[1] == checked[1] == "available", (
                    row.to_dict(),
                    produced,
                    checked,
                )
                assert produced[0] == checked[0]
                rows.append(row.to_dict())
    pool = pd.DataFrame(rows)
    pool["q_raw_log_median"] = np.log1p(10.0)
    for channel, value in (("upper", 3), ("lower", 30), ("lower2", 100)):
        pool["q_raw_isotope_" + channel + "_log_ratio"] = np.log1p(value) - np.log1p(10)
    measured, audit = m.measure(pool, lookup, iso)
    checked, audit2 = independent.independent_measure(
        pool, definitions, iso2, m.FIELDS, m.PREFIX
    )
    pd.testing.assert_frame_equal(
        measured, checked, check_exact=False, atol=1e-9, rtol=1e-9
    )
    pd.testing.assert_frame_equal(audit, audit2)
    assert len(rows) > 600
