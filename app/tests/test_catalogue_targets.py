from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from annotator.composition_ranges import canonical_lipid_class
from matlab_annotation.catalogue_targets import catalogue_missing_targets
from matlab_annotation.chemical_targets import fit_composition_envelopes
from matlab_annotation.vendor.compound_model import theoretical_mz
from matlab_annotation.xcms_project import load_bundled_compound_catalog

ROOT = Path(__file__).resolve().parents[2] / "reference_data/matlab_pipeline"


def inputs(cls="GM2[t]", stream="NP_Neg", c=36, db=1):
    envelope = fit_composition_envelopes(
        pd.DataFrame(
            [
                dict(cohort=str(g), stream=stream, **{"class": cls}, carbon=c, db=db)
                for g in range(4)
            ]
        )
    )
    reference = pd.DataFrame(columns=["name", "adduct"])
    spikes = pd.DataFrame(columns=["adduct", "ref_mz", "internal_standard_quantity"])
    return reference, envelope, spikes


def test_whole_missing_class_and_single_surviving_node_do_not_require_RT_success():
    ref, env, std = inputs()
    new, _ = catalogue_missing_targets(
        ref, ROOT, env, stream="NP_Neg", declared_standards=std
    )
    query = new[new.name.eq("GM2(t36:1)") & new.adduct.eq("[M-H]-")]
    assert len(query) == 1 and new.ref_rt.isna().all() and new.is_is.eq(False).all()
    one = query[["name", "adduct"]].copy()
    more, _ = catalogue_missing_targets(
        one, ROOT, env, stream="NP_Neg", declared_standards=std
    )
    assert not (
        (more.name == query.iloc[0]["name"]) & (more.adduct == query.iloc[0].adduct)
    ).any()
    assert new["class"].map(canonical_lipid_class).eq("GM2[t]").all()


@pytest.mark.parametrize(
    ("cls", "stream", "name", "ion"),
    [
        ("GA2[d]", "RP_Pos", "GA2(d36:1)", "[M+H]+"),
        ("GD1[d]", "NP_Neg", "GD1(d36:1)", "[M-2H](2-)"),
        ("GM2[t]", "RP_Neg", "GM2(t36:1)", "[M-H]-"),
    ],
)
def test_mass_and_codes_are_exact_native_chemistry_not_guessed_lattice(
    cls, stream, name, ion
):
    ref, env, std = inputs(cls, stream)
    new, _ = catalogue_missing_targets(
        ref, ROOT, env, stream=stream, declared_standards=std
    )
    row = new[new.name.eq(name) & new.adduct.eq(ion)].iloc[0]
    catalog = load_bundled_compound_catalog(ROOT)
    expected = abs(
        theoretical_mz(
            catalog.composition_attr[int(row.matlab_compound) - 1],
            stream[:2],
            int(row.matlab_code),
        )
    )
    assert row.ref_mz == pytest.approx(expected, abs=1e-10)
    assert row.matlab_code > 0 and not row.is_is


def test_mass_identical_declared_spike_is_protected_but_labelled_mass_is_not_unlabelled():
    ref, env, std = inputs("CE", "RP_Pos", c=16, db=0)
    new, _ = catalogue_missing_targets(
        ref, ROOT, env, stream="RP_Pos", declared_standards=std
    )
    row = new[new.name.eq("CE(16:0)") & new.adduct.eq("[M+NH4]+")].iloc[0]
    labelled = pd.DataFrame(
        [
            dict(
                adduct=row.adduct,
                ref_mz=row.ref_mz + 7.0439,
                internal_standard_quantity=100,
            )
        ]
    )
    kept, _ = catalogue_missing_targets(
        ref, ROOT, env, stream="RP_Pos", declared_standards=labelled
    )
    assert kept.name.eq("CE(16:0)").any()
    unlabelled = labelled.assign(ref_mz=row.ref_mz)
    guarded, audit = catalogue_missing_targets(
        ref, ROOT, env, stream="RP_Pos", declared_standards=unlabelled
    )
    assert not (guarded.name.eq(row["name"]) & guarded.adduct.eq(row.adduct)).any()
    assert audit.blocked_declared_spike_masses.sum() >= 1


def test_no_training_support_unknown_formula_and_wrong_mode_do_not_invent_targets():
    ref, env, std = inputs()
    env["models"] = {}
    assert catalogue_missing_targets(
        ref, ROOT, env, stream="NP_Neg", declared_standards=std
    )[0].empty
    ref, env, std = inputs("X4", "NP_Neg")
    assert catalogue_missing_targets(
        ref, ROOT, env, stream="NP_Neg", declared_standards=std
    )[0].empty
    with pytest.raises(ValueError):
        catalogue_missing_targets(
            ref, ROOT, env, stream="unknown", declared_standards=std
        )
    with pytest.raises(ValueError):
        catalogue_missing_targets(
            ref, ROOT, env, stream="NP_Neg", declared_standards=pd.DataFrame()
        )


def test_equivalent_charge_notations_cannot_overwrite_declared_spike_protection():
    ref, env, std = inputs("GD1[d]", "NP_Neg", c=36, db=1)
    new, _ = catalogue_missing_targets(
        ref, ROOT, env, stream="NP_Neg", declared_standards=std
    )
    pair = new[new.adduct.eq("[M-2H](2-)")].head(2)
    assert len(pair) == 2
    spikes = pair[["adduct", "ref_mz"]].copy().reset_index(drop=True)
    spikes["internal_standard_quantity"] = 100
    spikes.loc[1, "adduct"] = "[M-2H]2-"
    guarded, _ = catalogue_missing_targets(
        ref, ROOT, env, stream="NP_Neg", declared_standards=spikes
    )
    assert not guarded.name.isin(pair.name).any()


def test_native_carbon_and_double_bond_bounds_and_original_inputs_preserved():
    ref, env, std = inputs("PE-P", "RP_Pos", c=38, db=4)
    old = ref.copy(deep=True)
    new, _ = catalogue_missing_targets(
        ref, ROOT, env, stream="RP_Pos", declared_standards=std
    )
    assert not new.name.str.contains("O-").any()
    for row in new.itertuples():
        bounds = env["models"]["RP_Pos|" + canonical_lipid_class("PE-P")]
        assert bounds["carbon_min"] <= row.carbon <= bounds["carbon_max"]
        assert bounds["db_min"] <= row.db <= min(bounds["db_max"], row.carbon // 2)
    assert len(new) > 0 and np.isfinite(new.ref_mz).all()
    pd.testing.assert_frame_equal(old, ref)
