import numpy as np
import pandas as pd
import pytest

import matlab_annotation.declared_standard_context as module


def fixture():
    query = pd.DataFrame(
        [
            dict(
                stream="NP_Pos",
                **{"class": "BMP"},
                carbon=36,
                db=4,
                adduct="[M+NH4]+",
                feat_feature_id="query",
                feat_rt=5.1,
                feat_intensity=100000.0,
                ppm_error=1.0,
            )
        ]
    )
    standard = pd.DataFrame(
        [
            dict(
                stream="NP_Pos",
                **{"class": "BMP"},
                name="BMP(28:0) (IS)",
                standard_compound_id="BMP28|labelledmass",
                carbon=28,
                db=0,
                adduct="[M+NH4]+",
                feat_feature_id="spike",
                feat_mz=500.0001,
                ref_mz=500.0,
                feat_rt=5.0,
                feat_intensity=10000.0,
                is_is=True,
                internal_standard_quantity=100.0,
                internal_standard_unit="nM",
                rt_expected=5.0,
                standard_training_cohorts=8,
                standard_rt_reference_source="other_declared_standards",
            )
        ]
    )
    return query, standard


def test_RT_CDB_response_and_labelled_actual_mass_are_measured_without_source_edits():
    query, spike = fixture()
    before = query.copy(deep=True)
    values = module.declared_standard_context_evidence(query, spike).iloc[0]
    assert values[module.PREFIX + "RT_difference_seconds"] == pytest.approx(6.0)
    assert values[module.PREFIX + "carbon_difference"] == 8
    assert values[module.PREFIX + "db_difference"] == 4
    assert values[module.PREFIX + "same_class_compounds"] == 1
    assert values[module.PREFIX + "log_over_nM_response"] == pytest.approx(
        np.log1p(100000) - np.log(100)
    )
    assert values[module.PREFIX + "relative_ppm_error"] == pytest.approx(0.8)
    pd.testing.assert_frame_equal(query, before)
    bad_mass = spike.assign(ref_mz=490.0)
    assert (
        module.declared_standard_context_evidence(query, bad_mass)[
            module.PREFIX + "same_class_compounds"
        ].iloc[0]
        == 0
    )


def test_own_spike_and_companion_ion_cannot_supply_self_anchors_or_duplicate_votes():
    query, spike = fixture()
    other = spike.assign(feat_feature_id="companion", adduct="[M+H]+", feat_rt=5.01)
    measured = pd.concat([spike, other, spike], ignore_index=True)
    values = module.declared_standard_context_evidence(query, measured).iloc[0]
    assert values[module.PREFIX + "same_class_compounds"] == 1
    own = module.declared_standard_context_evidence(
        query.assign(feat_feature_id="spike"), measured
    ).iloc[0]
    assert own[module.PREFIX + "spike_feature"] == 1
    assert own[module.PREFIX + "same_class_compounds"] == 0
    assert np.isnan(own[module.PREFIX + "RT_difference_seconds"])


def test_ether_positional_and_stream_classes_are_not_merged():
    query, spike = fixture()
    for cls in ("PE-P", "2-acylLPC"):
        values = module.declared_standard_context_evidence(
            query.assign(**{"class": cls}),
            spike.assign(**{"class": "PE" if cls == "PE-P" else "1-acylLPC"}),
        ).iloc[0]
        assert values[module.PREFIX + "same_class_compounds"] == 0
        assert np.isnan(values[module.PREFIX + "RT_difference_seconds"])
    assert (
        module.declared_standard_context_evidence(query.assign(stream="NP_Neg"), spike)[
            module.PREFIX + "same_class_compounds"
        ].iloc[0]
        == 0
    )


def test_unknown_units_incoherent_adducts_and_conflicting_std_identities_withhold_context():
    query, spike = fixture()
    values = module.declared_standard_context_evidence(
        query, spike.assign(internal_standard_unit="unknown")
    ).iloc[0]
    assert np.isnan(values[module.PREFIX + "log_over_nM_response"])
    assert np.isfinite(values[module.PREFIX + "RT_difference_seconds"])
    for rival in (
        spike.assign(feat_feature_id="rival", adduct="[M+H]+", feat_rt=5.1),
        spike.assign(standard_compound_id="different chemical"),
    ):
        values = module.declared_standard_context_evidence(
            query, pd.concat([spike, rival], ignore_index=True)
        ).iloc[0]
        assert values[module.PREFIX + "same_class_compounds"] == 0


def test_detector_always_uses_empty_proposals_and_frozen_automatic_alignment(
    monkeypatch,
):
    query, spike = fixture()
    automatic = query.assign(is_is=False, name="BMP(36:4)", feat_mz=600.0)
    calls = []

    def spy(annotations, features, declared, *args, **kwargs):
        assert annotations.empty
        pd.testing.assert_frame_equal(kwargs["alignment_annotations"], automatic)
        calls.append(True)
        return spike.copy(), pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(module, "complete_declared_standards", spy)
    result = dict(
        streams={"NP_Pos": dict(annotations=automatic, features=pd.DataFrame())}
    )
    model = {k: {} for k in ("library", "atlas", "transfer", "channels")}
    measured, _ = module.independent_standard_measurements(result, spike, model)
    assert calls and measured.feat_feature_id.tolist() == ["spike"]
    result["streams"]["NP_Pos"]["annotations"] = automatic.assign(
        training_positive=True
    )
    with pytest.raises(ValueError, match="curation"):
        module.independent_standard_measurements(result, spike, model)
