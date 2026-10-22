import numpy as np
import pandas as pd
import pytest

import matlab_annotation.standard_completion as module


@pytest.fixture
def setup(monkeypatch):
    def primary(targets, *args, **kwargs):
        return pd.DataFrame(
            {
                "standard_rt_seconds": 300.0,
                "standard_adduct_supported": True,
                "standard_report_name": targets.name,
                "standard_report_class": targets["class"],
            },
            index=targets.index,
        )

    def secondary(targets, *args, **kwargs):
        return pd.DataFrame(
            {"standard_class_transfer_rt_seconds": np.nan}, index=targets.index
        )

    monkeypatch.setattr(module, "predict_standard_retention", primary)
    monkeypatch.setattr(module, "predict_standard_class_transfer", secondary)
    targets = pd.DataFrame(
        [
            dict(
                name="PE(28:0) (IS)",
                **{"class": "PE"},
                carbon=28,
                db=0,
                adduct="[M+H]+",
                ref_mz=500.0,
                ref_rt=np.nan,
                is_is=True,
                standard_compound_id="PE28|100",
                internal_standard_name="PE(14:0)2",
                internal_standard_quantity=100.0,
                internal_standard_unit="nM",
            )
        ]
    )
    features = pd.DataFrame(
        {
            "feature_id": ["001", "background"],
            "mz": [500.0001, 600.0],
            "rt": [5.005, 5.0],
            "intensity": [1e6, 1e5],
        }
    )
    columns = [
        "name",
        "class",
        "adduct",
        "feat_feature_id",
        "feat_mz",
        "feat_rt",
        "ref_mz",
        "is_is",
    ]
    empty = pd.DataFrame(columns=columns)
    obs = pd.DataFrame(
        [
            dict(cohort=f"g{i}", stream="RP_Pos", name="PE(28:0) (IS)", adduct="[M+H]+")
            for i in range(4)
        ]
    )
    channels = module.fit_standard_channel_prior(obs)

    def run(annotations=empty, raw=features, catalog=targets):
        return module.complete_declared_standards(
            annotations,
            raw,
            catalog,
            {},
            {},
            {},
            channels,
            stream="RP_Pos",
            alignment_annotations=empty,
        )

    return targets, features, empty, run


def test_unique_original_feature_standard_completion_and_source_preservation(setup):
    targets, features, empty, run = setup
    old = features.copy(deep=True)
    selected, candidates, audit = run()
    assert selected.feat_feature_id.tolist() == ["001"]
    assert selected.is_is.all() and selected.automatic_accepted.all()
    assert selected.name.iloc[0] == "PE(28:0) (IS)"
    assert len(candidates) == 1 and audit.action.eq("complete_declared_standard").any()
    pd.testing.assert_frame_equal(old, features)


def test_ambiguity_is_not_erased_by_occupied_feature_exclusion(setup):
    targets, features, empty, run = setup
    raw = pd.concat(
        [features, features.iloc[[0]].assign(feature_id="rival", rt=5.01)],
        ignore_index=True,
    )
    occupied = pd.DataFrame(
        [
            dict(
                name="TG(52:2)",
                **{"class": "TG"},
                adduct="[M+H]+",
                feat_feature_id="001",
                feat_mz=500.0001,
                feat_rt=5.005,
                ref_mz=500.0,
                is_is=False,
            )
        ]
    )
    selected, _, audit = run(occupied, raw)
    assert selected.name.tolist() == ["TG(52:2)"]
    assert audit.action.eq("standard_ambiguous_or_absent").any()


def test_mass_identical_endogenous_spike_alias_becomes_a_standard_not_two_calls(setup):
    _, _, _, run = setup
    alias = pd.DataFrame(
        [
            dict(
                name="PE(28:0)",
                **{"class": "PE"},
                adduct="[M+H]+",
                feat_feature_id="001",
                feat_mz=500.0001,
                feat_rt=5.005,
                ref_mz=500.0,
                is_is=False,
            )
        ]
    )
    selected, _, audit = run(alias)
    assert len(selected) == 1 and selected.is_is.all()
    assert audit.action.eq("demote_mass_identical_spike_alias").any()
    protected = alias.assign(manual_confirmed=True)
    selected, _, audit = run(protected)
    assert selected.name.tolist() == ["PE(28:0)"]
    assert not audit.action.eq("demote_mass_identical_spike_alias").any()


def test_standard_channels_without_training_support_remain_companions(setup):
    targets, features, _, run = setup
    extra = pd.DataFrame(
        [
            dict(
                name="PE(28:0) (IS)",
                **{"class": "PE"},
                adduct="[M+Na]+",
                feat_feature_id="Na",
                feat_mz=522.0,
                feat_rt=5.0,
                ref_mz=522.0,
                is_is=True,
            )
        ]
    )
    raw = pd.concat(
        [
            features,
            pd.DataFrame([dict(feature_id="Na", mz=522.0, rt=5.0, intensity=1e6)]),
        ],
        ignore_index=True,
    )
    selected, candidates, audit = run(extra, raw)
    assert selected.adduct.tolist() == ["[M+H]+"]
    assert candidates.feat_feature_id.eq("Na").any()
    assert audit.action.eq("demote_unvalidated_standard_channel").any()


def test_two_standard_identities_cannot_claim_one_feature(setup):
    targets, _, _, run = setup
    other = targets.assign(
        name="PG(28:0) (IS)", **{"class": "PG"}, standard_compound_id="PG28|100"
    )
    selected, _, audit = run(catalog=pd.concat([targets, other], ignore_index=True))
    assert selected.empty
    assert audit.action.eq("conflicting_declared_standard_identities").all()


def test_standard_positional_reporting_needs_RT_and_never_overrides_an_explicit_other_position(
    setup, monkeypatch
):
    targets, _, _, run = setup
    catalog = targets.assign(name="LPC(14:0) (IS)", **{"class": "LPC"})

    def primary(targets, *args, **kwargs):
        return pd.DataFrame(
            {
                "standard_rt_seconds": 300.0,
                "standard_adduct_supported": True,
                "standard_report_name": "1-acylLPC(14:0) (IS)",
                "standard_report_class": "1-acylLPC",
            },
            index=targets.index,
        )

    monkeypatch.setattr(module, "predict_standard_retention", primary)
    old = pd.DataFrame(
        [
            dict(
                name="LPC(14:0) (IS)",
                **{"class": "LPC"},
                adduct="[M+H]+",
                feat_feature_id="001",
                feat_mz=500.0001,
                feat_rt=5.005,
                ref_mz=500.0,
                is_is=True,
            )
        ]
    )
    selected, _, _ = run(old, catalog=catalog)
    assert selected.name.tolist() == ["1-acylLPC(14:0) (IS)"]
    explicit = old.assign(name="2-acylLPC(14:0) (IS)", **{"class": "2-acylLPC"})
    selected, _, _ = run(explicit, catalog=catalog)
    assert selected.name.tolist() == ["2-acylLPC(14:0) (IS)"]
    far = old.assign(feat_rt=9.0)
    selected, _, _ = run(far, catalog=catalog)
    assert selected.name.tolist() == ["LPC(14:0) (IS)"]
