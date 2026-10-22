import numpy as np
import pandas as pd
import pytest

from matlab_annotation.chemical_targets import (
    CH2,
    H2,
    compile_additional_targets,
    fit_composition_envelopes,
)


def training(cls="PE-O+P", stream="NP_Neg", carbon=40, db=10):
    return pd.DataFrame(
        [
            dict(
                cohort=f"cohort-{i}",
                stream=stream,
                **{"class": cls},
                carbon=carbon,
                db=db,
            )
            for i in range(4)
        ]
    )


def reference(cls="PE-O+P", ion="[M-H]-", head="PE(O+P-", charge=1):
    return pd.DataFrame(
        [
            dict(
                name=f"{head}{c}:{d})",
                **{"class": cls},
                adduct=ion,
                carbon=c,
                db=d,
                ref_mz=200 + (c * CH2 - d * H2) / charge,
                ref_rt=8.0,
            )
            for c, d in [(34, 0), (34, 1), (36, 0), (36, 1)]
        ]
    )


def test_high_db_combined_np_targets_and_no_input_mutation():
    data = training()
    envelope = fit_composition_envelopes(data)
    ref = reference()
    old = ref.copy(deep=True)
    new, settings, audit = compile_additional_targets(ref, envelope, stream="NP_Neg")
    row = new[new.name.eq("PE(O+P-40:10)")].iloc[0]
    assert row.ref_mz == pytest.approx(200 + 40 * CH2 - 10 * H2)
    assert np.isnan(row.ref_rt) and np.isnan(row.matlab_code)
    assert settings.decide("PE-O+P", 40, 10).allowed
    assert audit.added.sum() == len(new)
    pd.testing.assert_frame_equal(old, ref)
    assert not new.name.str.contains(r"\(P-").any()


@pytest.mark.parametrize(
    ("ion", "charge"), [("[M-2H]2-", 2), ("[M-2H](2-)", 2), ("[M+H]+", 1)]
)
def test_charge_aware_exact_mass_increments(ion, charge):
    envelope = fit_composition_envelopes(
        training(cls="CL", stream="RP_Neg", carbon=80, db=12)
    )
    ref = reference(cls="CL", ion=ion, head="CL(", charge=charge)
    new, _, _ = compile_additional_targets(ref, envelope, stream="RP_Neg")
    row = new[new.name.eq("CL(80:12)")].iloc[0]
    assert row.ref_mz == pytest.approx(200 + (80 * CH2 - 12 * H2) / charge)


def test_labelled_standard_does_not_suppress_unlabelled_species_but_identical_spike_does():
    envelope = fit_composition_envelopes(
        training(cls="CE", stream="RP_Pos", carbon=16, db=0)
    )
    ref = reference(cls="CE", ion="[M+NH4]+", head="CE(")
    target = 200 + 16 * CH2
    spike = pd.DataFrame(
        [
            dict(
                name="CE(16:0) (IS)",
                **{"class": "CE"},
                adduct="[M+NH4]+",
                carbon=16,
                db=0,
                ref_mz=target + 7.0439,
                ref_rt=8.0,
            )
        ]
    )
    new, _, _ = compile_additional_targets(
        pd.concat([ref, spike], ignore_index=True), envelope, stream="RP_Pos"
    )
    assert new.name.eq("CE(16:0)").any()
    spike["ref_mz"] = target
    new, _, audit = compile_additional_targets(
        pd.concat([ref, spike], ignore_index=True), envelope, stream="RP_Pos"
    )
    assert not new.name.eq("CE(16:0)").any()
    assert audit.mass_identical_standards_blocked.sum() >= 1


def test_sparse_classes_inconsistent_masses_and_wrong_stream_do_not_expand():
    ref = reference()
    data = training().iloc[:2]
    assert not fit_composition_envelopes(data)["models"]
    envelope = fit_composition_envelopes(training())
    assert compile_additional_targets(ref, envelope, stream="RP_Pos")[0].empty
    ref.loc[0, "ref_mz"] += 0.1
    new, _, audit = compile_additional_targets(ref, envelope, stream="NP_Neg")
    assert new.empty and audit.status.eq("inconsistent_mass_lattice").all()
    with pytest.raises(ValueError):
        fit_composition_envelopes(training().assign(carbon=40.5))


def test_separate_declared_catalog_protects_spikes_missing_from_the_reference_grid():
    envelope = fit_composition_envelopes(
        training(cls="CE", stream="RP_Pos", carbon=16, db=0)
    )
    ref = reference(cls="CE", ion="[M+NH4]+", head="CE(")
    spike = pd.DataFrame(
        [dict(name="CE(16:0) (IS)", adduct="[M+NH4]+", ref_mz=200 + 16 * CH2)]
    )
    old, _, audit = compile_additional_targets(ref, envelope, stream="RP_Pos")
    assert (
        old.name.eq("CE(16:0)").any()
        and audit.standard_protection_input.eq("unavailable").all()
    )
    fixed, _, audit = compile_additional_targets(
        ref, envelope, stream="RP_Pos", declared_standards=spike
    )
    assert not fixed.name.eq("CE(16:0)").any()
    assert audit.standard_protection_input.eq("explicit_declared_catalog").all()
