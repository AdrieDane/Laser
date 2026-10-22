import numpy as np
import pandas as pd
import pytest

from matlab_annotation.catalogue_completion import complete_catalogue_targets


def fixture():
    a = pd.DataFrame(
        [dict(name="PC(34:1)", adduct="[M+H]+", feat_feature_id="old", is_is=False)]
    )
    f = pd.DataFrame([dict(feature_id="free", mz=700.0, rt=5.0, intensity=1000.0)])
    t = pd.DataFrame(
        [
            dict(
                name="GM2(t36:1)",
                **{"class": "GM2-t"},
                adduct="[M-H]-",
                carbon=36,
                db=1,
                ref_mz=700.0,
                ref_rt=np.nan,
                atlas_rt_seconds=300.0,
                atlas_training_cohorts=8,
                atlas_training_mad_seconds=1.0,
                atlas_global_status="automatic_anchor_alignment",
                chemical_target_origin="native_catalogue_without_RT_dependency",
            )
        ]
    )
    p = {"NP_Neg|GM2[t]": dict(cohorts=8, joint={"36:1": 5})}
    c = dict(
        format="laser-class-channel-support-v1",
        models={"NP_Neg|GM2[t]": dict(cohorts=8, adducts={"[M-H]-": 8})},
    )
    return a, f, t, p, c


def test_unique_supported_catalogue_hit_is_added_without_old_changes_or_fake_confidence():
    a, f, t, p, c = fixture()
    result, raw, audit = complete_catalogue_targets(a, f, t, p, c, stream="NP_Neg")
    assert result.feat_feature_id.tolist() == ["old", "free"]
    assert np.isnan(result.class_selection_score.iloc[-1])
    assert raw.automatic_accepted.all()
    assert audit.action.eq("add_unique_restored_catalogue_hit").all()
    pd.testing.assert_frame_equal(result.iloc[:1][a.columns], a, check_dtype=False)
    again, _, _ = complete_catalogue_targets(result, f, t, p, c, stream="NP_Neg")
    pd.testing.assert_frame_equal(result, again)


def test_existing_occupancy_cannot_erase_raw_feature_ambiguity_and_spikes_protected():
    a, f, t, p, c = fixture()
    both = pd.concat([f, f.assign(feature_id="old")], ignore_index=True)
    assert len(complete_catalogue_targets(a, both, t, p, c, stream="NP_Neg")[0]) == 1
    assert (
        len(
            complete_catalogue_targets(
                a, f, t, p, c, stream="NP_Neg", declared_standard_masses=[700.0]
            )[0]
        )
        == 1
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(atlas_training_cohorts=2),
        dict(atlas_training_mad_seconds=6),
        dict(atlas_global_status="insufficient_automatic_anchors"),
        dict(atlas_rt_seconds=np.nan),
    ],
)
def test_missing_or_weak_independent_RT_never_becomes_full(change):
    a, f, t, p, c = fixture()
    assert (
        len(
            complete_catalogue_targets(a, f, t.assign(**change), p, c, stream="NP_Neg")[
                0
            ]
        )
        == 1
    )


def test_unseen_compositions_and_channels_are_not_promoted():
    a, f, t, p, c = fixture()
    assert len(complete_catalogue_targets(a, f, t, {}, c, stream="NP_Neg")[0]) == 1
    assert (
        len(
            complete_catalogue_targets(
                a, f, t.assign(adduct="[M+Cl]-"), p, c, stream="NP_Neg"
            )[0]
        )
        == 1
    )


def test_competing_new_identity_claims_are_all_withheld_independent_of_order():
    a, f, t, p, c = fixture()
    t = pd.concat(
        [t, t.assign(name="GA2(d36:1)", **{"class": "GA2-d"})], ignore_index=True
    )
    p["NP_Neg|GA2[d]"] = p["NP_Neg|GM2[t]"]
    c["models"]["NP_Neg|GA2[d]"] = c["models"]["NP_Neg|GM2[t]"]
    assert len(complete_catalogue_targets(a, f, t, p, c, stream="NP_Neg")[0]) == 1
    assert (
        len(complete_catalogue_targets(a, f, t.iloc[::-1], p, c, stream="NP_Neg")[0])
        == 1
    )


def test_truth_columns_and_unproven_chemical_origin_rejected():
    a, f, t, p, c = fixture()
    with pytest.raises(ValueError, match="Curated"):
        complete_catalogue_targets(
            a, f, t.assign(same_manual_peak=True), p, c, stream="NP_Neg"
        )
    with pytest.raises(ValueError, match="Only independently"):
        complete_catalogue_targets(
            a, f, t.assign(chemical_target_origin="guessed"), p, c, stream="NP_Neg"
        )
