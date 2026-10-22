import numpy as np
import pandas as pd
import pytest

from matlab_annotation.ion_footprints import (
    neutral_ion_mass,
    filter_new_np_positive_isobars,
)
from matlab_annotation.vendor.compound_model import PROTON_MASS
from matlab_annotation.vendor.composition_initializer import _ADDUCTS


def row(cls, adduct, mz, fid, rt=5.0, stream="NP_Pos"):
    return dict(
        stream=stream,
        name=cls + "(36:2)",
        **{"class": cls},
        adduct=adduct,
        feat_feature_id=fid,
        feat_mz=mz,
        feat_rt=rt,
    )


def test_native_catalogue_neutral_inverse_for_all_adducts_and_charges():
    for label, addition, charge in _ADDUCTS.values():
        observed = (800 + addition) / abs(charge) + np.sign(charge) * PROTON_MASS
        assert neutral_ion_mass(observed, label) == pytest.approx(800, abs=1e-10)
    assert np.isnan(neutral_ion_mass(800, "[unknown]"))


def test_new_coeluting_cross_class_adduct_footprint_is_withheld_old_call_unchanged():
    existing = pd.DataFrame([row("PG", "[M+H]+", 800 + PROTON_MASS, "old")])
    before = existing.copy(deep=True)
    additions = pd.DataFrame(
        [row("BMP", "[M+NH4]+", 800 + PROTON_MASS + 17.026549, "new", rt=5.01)]
    )
    kept, audit = filter_new_np_positive_isobars(existing, additions)
    assert kept.empty
    assert audit.action.eq("withhold_existing_cross_class_neutral_footprint").all()
    pd.testing.assert_frame_equal(existing, before)


def test_retention_separation_and_different_neutral_mass_do_not_conflict():
    existing = pd.DataFrame([row("PG", "[M+H]+", 800 + PROTON_MASS, "old")])
    additions = pd.DataFrame(
        [
            row("BMP", "[M+NH4]+", 800 + PROTON_MASS + 17.026549, "separated", rt=5.1),
            row("BMP", "[M+NH4]+", 800.02 + PROTON_MASS + 17.026549, "different"),
        ]
    )
    kept, _ = filter_new_np_positive_isobars(existing, additions)
    pd.testing.assert_frame_equal(kept, additions)


def test_competing_new_classes_are_all_withheld_not_first_winner():
    empty = pd.DataFrame(columns=list(row("PG", "[M+H]+", 800, "old")))
    additions = pd.DataFrame(
        [
            row("PG", "[M+H]+", 800 + PROTON_MASS, "one"),
            row("BMP", "[M+NH4]+", 800 + PROTON_MASS + 17.026549, "two"),
        ]
    )
    assert filter_new_np_positive_isobars(empty, additions)[0].empty
    assert filter_new_np_positive_isobars(empty, additions.iloc[::-1])[0].empty


def test_same_class_channels_and_negative_mode_are_outside_this_guard():
    existing = pd.DataFrame([row("PG", "[M+H]+", 800 + PROTON_MASS, "old")])
    additions = pd.DataFrame(
        [
            row("PG", "[M+NH4]+", 800 + PROTON_MASS + 17.026549, "sameclass"),
            row("BMP", "[M-H]-", 800 - PROTON_MASS, "negative", stream="NP_Neg"),
        ]
    )
    kept, _ = filter_new_np_positive_isobars(existing, additions)
    pd.testing.assert_frame_equal(kept, additions)


def test_unsupported_new_ion_is_review_only_and_window_is_bounded():
    existing = pd.DataFrame([row("PG", "[M+H]+", 800, "old")])
    additions = pd.DataFrame([row("BMP", "[unknown]", 800, "unknown")])
    assert filter_new_np_positive_isobars(existing, additions)[0].empty
    with pytest.raises(ValueError):
        filter_new_np_positive_isobars(existing, additions, coelution_seconds=30)
