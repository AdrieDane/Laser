from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from matlab_annotation.rp_isomer_anchors import (
    RPIsomerAnchorSettings,
    _reference_groups,
    propose_rp_isomer_assignments,
    refine_rp_isomer_annotations,
)
from adapters.standard_engines import build_automatic_anchor_settings


def lyso_fixture(groups=5, shift=12.0):
    reference, features = [], []
    for k, carbon in enumerate(range(14, 14 + 2 * groups, 2)):
        mz = 500.0 + carbon * 10.0
        for position in (1, 2):
            predicted = 5 + carbon * 0.1 + (0.25 if position == 1 else 0)
            name = f"{position}-acylLPC({carbon}:0)"
            reference.append(
                dict(
                    name=name,
                    **{"class": f"{position}-acylLPC"},
                    carbon=carbon,
                    db=0,
                    adduct="[M+H]+",
                    ref_mz=mz,
                    ref_rt=predicted,
                    matlab_code=carbon,
                    matlab_compound=position,
                )
            )
            features.append(
                dict(
                    feature_id=f"{carbon}-{position}",
                    mz=mz,
                    rt=predicted + shift / 60,
                    intensity=1e7 / (k + position),
                )
            )
    return pd.DataFrame(reference), pd.DataFrame(features)


def baseline_fixture(count=3):
    return pd.DataFrame(
        [
            dict(
                feat_feature_id=f"baseline-{i}",
                name=f"PC({30 + 2 * i}:1)",
                **{"class": "PC"},
                carbon=30 + 2 * i,
                db=1,
                adduct="[M+H]+",
                matlab_code=10 + i,
                matlab_compound=25,
                is_is=False,
                isotopologue="",
                feat_mz=900.0 + i,
                feat_rt=10.0 + i,
                rt_expected=10.0 + i,
                ref_mz=900.0 + i,
                ref_rt=10.0 + i,
                feat_intensity=1e5,
                ppm_error=0.0,
                rt_error=0.0,
                grid_coherence_score=1.0,
                grid_coherence_pass=True,
                post_regression_confidence=0.55,
                auto_annotation_status="baseline_count_retained",
                automatic_accepted=True,
            )
            for i in range(count)
        ]
    )


MODEL = {
    "format": "laser-fitted-grid-confidence-v1",
    "intercept": 0.0,
    "numeric": {},
    "categorical": {},
    "candidate_limits": {
        "ppm": 3.0,
        "rt_seconds": 10.0,
        "require_grid_coherence_pass": True,
    },
}


def test_learns_shift_from_four_other_ordered_abundant_pairs():
    reference, features = lyso_fixture()
    before = reference.copy(deep=True)
    proposed, anchors, corrected = propose_rp_isomer_assignments(reference, features)
    assert len(anchors) == 10
    assert len(proposed) == 10
    assert proposed.rp_isomer_anchor_count.eq(4).all()
    assert np.allclose(proposed.rt_error, 0)
    assert np.allclose(corrected.rp_isomer_shift_seconds, 12)
    for _, pair in proposed.groupby("carbon"):
        early = pair[pair["class"].eq("2-acylLPC")].feat_rt.iloc[0]
        late = pair[pair["class"].eq("1-acylLPC")].feat_rt.iloc[0]
        assert early < late
    assert_frame_equal(reference, before)


def test_composition_cannot_support_its_own_prediction():
    reference, features = lyso_fixture(groups=4)
    proposed, anchors, corrected = propose_rp_isomer_assignments(reference, features)
    assert len(anchors) == 8
    assert proposed.empty
    assert corrected.rp_isomer_anchor_count.eq(0).all()


def test_o_p_pairing_accounts_for_the_vinyl_ether_double_bond():
    reference = pd.DataFrame(
        [
            dict(
                name="PE(O-36:5)",
                **{"class": "PE-O"},
                carbon=36,
                db=5,
                adduct="[M-H]-",
                ref_mz=722.5,
                ref_rt=11.5,
            ),
            dict(
                name="PE(P-36:4)",
                **{"class": "PE-P"},
                carbon=36,
                db=4,
                adduct="[M-H]-",
                ref_mz=722.5,
                ref_rt=11.8,
            ),
        ]
    )
    paired = _reference_groups(reference)
    assert len(paired) == 2
    assert paired._rp_group.nunique() == 1
    assert paired._rp_total_db.eq(5).all()
    # The order is method/reference-specific, not universally P then O.
    reference.loc[0, "ref_rt"] = 12.0
    flipped = _reference_groups(reference)
    assert flipped.loc[flipped["class"].eq("PE-P"), "_rp_member"].iloc[0] == "early"
    reference.loc[1, "db"] = 5
    assert _reference_groups(reference).empty


def test_wrong_formula_and_unvalidated_channel_do_not_seed():
    reference, _ = lyso_fixture()
    reference.loc[reference["class"].eq("1-acylLPC"), "ref_mz"] += 2.0
    assert _reference_groups(reference).empty
    reference, _ = lyso_fixture()
    reference["adduct"] = "[M+Na]+"
    assert _reference_groups(reference).empty


def test_ambiguous_split_peak_does_not_become_an_anchor_or_assignment():
    reference, features = lyso_fixture(groups=6)
    extra = features.iloc[[0]].assign(feature_id="split")
    features = pd.concat([features, extra], ignore_index=True)
    proposed, anchors, _ = propose_rp_isomer_assignments(reference, features)
    assert "14-1" not in set(anchors.feat_feature_id)
    assert "14-1" not in set(proposed.feat_feature_id)
    assert "split" not in set(proposed.feat_feature_id)


def test_order_does_not_assume_one_acyl_is_the_stronger_peak():
    reference, features = lyso_fixture()
    features.loc[features.feature_id.str.endswith("-2"), "intensity"] *= 100
    proposed, _, _ = propose_rp_isomer_assignments(reference, features)
    assert set(proposed.loc[proposed["class"].eq("1-acylLPC"), "feat_feature_id"]) == {
        f"{c}-1" for c in range(14, 24, 2)
    }


def test_no_sparse_or_long_range_propagation():
    reference, features = lyso_fixture()
    isolated = reference.iloc[:2].copy()
    isolated["name"] = ["1-acylLPC(40:0)", "2-acylLPC(40:0)"]
    isolated["carbon"] = 40
    isolated["ref_mz"] = 1200.0
    reference = pd.concat([reference, isolated], ignore_index=True)
    proposed, _, corrected = propose_rp_isomer_assignments(reference, features)
    assert not proposed.carbon.eq(40).any()
    assert corrected.loc[corrected.carbon.eq(40), "rp_isomer_anchor_count"].eq(0).all()


def test_preserves_count_and_never_confuses_ranking_with_probability():
    reference, features = lyso_fixture()
    baseline = baseline_fixture()
    with patch("matlab_annotation.rp_isomer_anchors._load_model", return_value=MODEL):
        accepted, candidates, audit, _, _ = refine_rp_isomer_annotations(
            baseline,
            baseline,
            reference,
            features,
            stream="RP_Pos",
        )
    assert len(accepted) == len(baseline)
    assert not accepted.feat_feature_id.duplicated().any()
    supported = accepted[
        accepted.auto_annotation_status.eq("rp_isomer_anchor_supported")
    ]
    assert not supported.empty
    assert supported.post_regression_confidence.eq(0.5).all()
    assert supported.rp_isomer_selection_score.gt(0.8).all()
    assert audit.status.eq("completed").all()
    assert candidates.automatic_accepted.sum() == len(accepted)
    # Count truncation must not leave a newly introduced orphan 2-acyl.
    for row in supported[supported["class"].eq("2-acylLPC")].itertuples():
        assert (
            (accepted["class"].eq("1-acylLPC")) & accepted.carbon.eq(row.carbon)
        ).any()


def test_declared_standard_is_unchanged_and_cannot_seed_an_endogenous_pair():
    reference, features = lyso_fixture()
    baseline = baseline_fixture()
    baseline.loc[0, "is_is"] = True
    baseline.loc[0, "feat_feature_id"] = "14-1"
    protected = baseline.iloc[0].copy()
    with patch("matlab_annotation.rp_isomer_anchors._load_model", return_value=MODEL):
        accepted, _, _, anchors, _ = refine_rp_isomer_annotations(
            baseline,
            baseline,
            reference,
            features,
            stream="RP_Pos",
        )
    actual = accepted[accepted.feat_feature_id.eq("14-1")].iloc[0]
    for column, value in protected.items():
        assert actual[column] == value
    if not anchors.empty:
        assert not anchors.feat_feature_id.eq("14-1").any()


def test_np_and_disabled_are_exact_no_ops():
    reference, features = lyso_fixture()
    baseline = baseline_fixture()
    for stream, settings in [
        ("NP_Pos", RPIsomerAnchorSettings()),
        ("RP_Pos", RPIsomerAnchorSettings(enabled=False)),
    ]:
        accepted, candidates, audit, _, _ = refine_rp_isomer_annotations(
            baseline,
            baseline,
            reference,
            features,
            stream=stream,
            settings=settings,
        )
        assert accepted is baseline and candidates is baseline
        assert audit.status.iloc[0] in {"not_applicable", "disabled"}


def test_missing_model_falls_back_with_an_audited_reason():
    reference, features = lyso_fixture()
    baseline = baseline_fixture()
    with patch(
        "matlab_annotation.rp_isomer_anchors._load_model",
        side_effect=OSError("missing"),
    ):
        accepted, _, audit, _, _ = refine_rp_isomer_annotations(
            baseline,
            baseline,
            reference,
            features,
            stream="RP_Pos",
        )
    assert accepted is baseline
    assert audit.status.iloc[0] == "model_unavailable"


def test_public_settings_allow_explicit_baseline_comparison():
    assert build_automatic_anchor_settings({}).use_rp_isomer_anchor_refinement
    assert not build_automatic_anchor_settings(
        {
            "use_rp_isomer_anchor_refinement": False,
        }
    ).use_rp_isomer_anchor_refinement


def test_audits_and_candidates_survive_saved_project_export(tmp_path):
    from completed_project import load_completed_project
    from saved_project import load_project_table, save_standard_project

    reference, features = lyso_fixture()
    baseline = baseline_fixture()
    with patch("matlab_annotation.rp_isomer_anchors._load_model", return_value=MODEL):
        accepted, candidates, audit, anchors, models = refine_rp_isomer_annotations(
            baseline,
            baseline,
            reference,
            features,
            stream="RP_Pos",
        )
    result = {
        "engine": "matlab",
        "automatic_annotations": accepted,
        "identified_lipids": accepted,
        "equivalent_lipids": pd.DataFrame(),
        "all_candidates": candidates,
        "matlab_rp_isomer_anchor_audit": audit,
        "matlab_rp_isomer_anchors": anchors,
        "matlab_rp_isomer_rt_models": models,
        "streams": {
            "RP_Pos": {
                "annotations": accepted,
                "all_pairs": candidates,
                "rp_isomer_rt_models": models,
            }
        },
    }
    root = tmp_path / "project"
    save_standard_project(root, result)
    restored = load_completed_project(root)
    assert len(load_project_table(restored, "matlab_rp_isomer_anchors")) == len(anchors)
    assert len(load_project_table(restored, "matlab_rp_isomer_rt_models")) == len(
        models
    )
    assert len(load_project_table(restored, "automatic_annotations")) == len(accepted)


@pytest.mark.parametrize("empty", ["reference", "features"])
def test_empty_inputs_fall_back_without_changes(empty):
    reference, features = lyso_fixture()
    baseline = baseline_fixture()
    with patch("matlab_annotation.rp_isomer_anchors._load_model", return_value=MODEL):
        accepted, _, audit, _, _ = refine_rp_isomer_annotations(
            baseline,
            baseline,
            pd.DataFrame() if empty == "reference" else reference,
            pd.DataFrame() if empty == "features" else features,
            stream="RP_Pos",
        )
    assert accepted is baseline
    assert audit.status.iloc[0] == "no_isomer_reference_or_features"


@pytest.mark.parametrize(
    "settings",
    [
        replace(RPIsomerAnchorSettings(), final_rt_seconds=11),
        replace(RPIsomerAnchorSettings(), anchor_rt_seconds=31),
        replace(RPIsomerAnchorSettings(), minimum_support_groups=1),
        replace(RPIsomerAnchorSettings(), pair_score_margin=float("nan")),
    ],
)
def test_rejects_unsafe_settings(settings):
    reference, features = lyso_fixture()
    with pytest.raises(ValueError):
        propose_rp_isomer_assignments(reference, features, settings)
