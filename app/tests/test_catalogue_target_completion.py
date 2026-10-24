from pathlib import Path

import pandas as pd
import pytest

from matlab_annotation.catalogue_target_completion import compile_catalogue_completion
from matlab_annotation.chemical_targets import FORMAT

ROOT = Path(__file__).resolve().parents[2] / "reference_data/matlab_pipeline"


def fixture():
    reference = pd.DataFrame(
        columns=["name", "class", "adduct", "carbon", "db", "ref_mz"]
    )
    standards = pd.DataFrame(columns=["stream", "ref_mz", "adduct", "is_is"])
    envelope = dict(
        format=FORMAT,
        models={
            "RP_Pos|GA2[d]": dict(
                carbon_min=34, carbon_max=36, db_min=1, db_max=2, training_cohorts=4
            ),
        },
    )
    return reference, standards, envelope


def compile(reference, standards, envelope, **kwargs):
    return compile_catalogue_completion(
        ROOT,
        reference,
        envelope,
        stream=kwargs.pop("stream", "RP_Pos"),
        declared_standards=standards,
        **kwargs,
    )


def test_missing_grid_has_native_phase_specific_targets_without_invented_rt_or_legacy_code():
    reference, standards, envelope = fixture()
    before = reference.copy(deep=True)
    targets, ranges, audit = compile(reference, standards, envelope)
    assert "GA2(d34:1)" in set(targets.name)
    assert targets["class"].eq("GA2-d").all()
    assert targets.adduct.eq("[M+H]+").all()
    assert targets.ref_mz.gt(0).all() and targets.ref_rt.isna().all()
    assert targets.matlab_code.isna().all()
    assert targets.isotopologue.eq("").all() and not targets.is_is.any()
    assert audit.original_seed_compositions.eq(0).all()
    assert all(
        ranges.decide(r["class"], r["carbon"], r["db"]).allowed
        for r in targets.to_dict("records")
    )
    pd.testing.assert_frame_equal(reference, before)
    assert compile(reference, standards, envelope, stream="NP_Neg")[0].empty


def test_existing_two_seed_compositions_are_not_regenerated_or_mutated():
    reference, standards, envelope = fixture()
    targets, _, _ = compile(reference, standards, envelope)
    seeded = targets.iloc[[0, -1]].copy()
    assert len(seeded[["carbon", "db"]].drop_duplicates()) == 2
    assert compile(seeded, standards, envelope)[0].empty


def test_declared_actual_mass_alias_is_blocked_but_labelled_mass_does_not_block_unlabelled_target():
    reference, standards, envelope = fixture()
    targets, _, _ = compile(reference, standards, envelope)
    row = targets.iloc[0]
    standard = pd.DataFrame(
        [dict(stream="RP_Pos", ref_mz=row.ref_mz, adduct=row.adduct, is_is=True)]
    )
    result, _, audit = compile(reference, standard, envelope)
    assert row["name"] not in set(result.name)
    assert audit.declared_mass_aliases_blocked.sum() == 1
    labelled = standard.assign(ref_mz=row.ref_mz + 9.05649)
    assert row["name"] in set(compile(reference, labelled, envelope)[0].name)


def test_mass_disagreement_fails_closed_and_unsupported_training_class_is_not_guessed():
    reference, standards, envelope = fixture()
    targets, _, _ = compile(reference, standards, envelope)
    seed = targets.iloc[[0]].copy()
    result, _, _ = compile(seed, standards, envelope)
    assert seed.name.iloc[0] not in set(result.name)
    seed["ref_mz"] += 1
    with pytest.raises(ValueError, match="masses disagree"):
        compile(seed, standards, envelope)
    envelope["models"]["RP_Pos|GA2[d]"]["training_cohorts"] = 2
    assert compile(reference, standards, envelope)[0].empty
