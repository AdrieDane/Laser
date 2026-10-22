from pathlib import Path

import numpy as np
import pytest

from matlab_annotation.declared_standards import declared_standard_targets

ROOT = Path(__file__).resolve().parents[2] / "reference_data/matlab_pipeline"


def declarations():
    return [
        dict(name=n, quantity=100.0, unit="nM")
        for n in ["SPH(d17:1)", "S1P(d17:0)", "CE(16:0)-D7"]
    ]


def test_declared_targets_include_missing_standards_without_any_raw_features_or_RT():
    targets = declared_standard_targets(ROOT, declarations())
    row = targets[
        targets.stream.eq("RP_Neg")
        & targets.name.eq("S1P(d17:0) (IS)")
        & targets.adduct.eq("[M-H]-")
    ]
    assert len(row) == 1 and row.ref_mz.iloc[0] == pytest.approx(366.241485, abs=1e-6)
    sph = targets[
        targets.stream.eq("NP_Pos")
        & targets.name.eq("SPH(d17:1) (IS)")
        & targets.adduct.eq("[M+H]+")
    ]
    assert len(sph) == 1 and sph.ref_mz.iloc[0] == pytest.approx(286.274055, abs=1e-6)
    assert (
        targets.is_is.all()
        and targets.ref_rt.isna().all()
        and (targets.matlab_code < 0).all()
    )
    assert targets.isotopologue.eq("").all()


def test_labelled_standard_retains_its_labelled_mass():
    target = declared_standard_targets(ROOT, declarations())
    ce = target[
        target.stream.eq("RP_Pos")
        & target.name.eq("CE(16:0) (IS)")
        & target.adduct.eq("[M+NH4]+")
    ].iloc[0]
    assert ce.internal_standard_name == "CE(16:0)-D7"
    assert ce.ref_mz == pytest.approx(649.662294, abs=1e-6)
    assert ce.internal_standard_quantity == 100


def test_no_declaration_or_zero_spike_never_invents_a_standard():
    assert declared_standard_targets(ROOT, []).empty
    assert declared_standard_targets(
        ROOT, [dict(name="SPH(d17:1)", quantity=0.0, unit="nM")]
    ).empty
    with pytest.raises(ValueError):
        declared_standard_targets(ROOT, declarations() * 2)
    with pytest.raises(ValueError):
        declared_standard_targets(
            ROOT, [dict(name="SPH(d17:1)", quantity=np.nan, unit="nM")]
        )
