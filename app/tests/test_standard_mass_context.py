import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from matlab_annotation.standard_mass_context import (
    validated_mass_anchors,
    standard_mass_evidence,
    summarize_mass_anchors,
)
from matlab_annotation.declared_standards import declared_standard_targets


def fixtures(count=7):
    targets = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                name=f"PE({i}:0) (IS)",
                adduct="[M+H]+",
                ref_mz=500 + i * 20,
                internal_standard_neutral_mass=499 + i * 20,
                internal_standard_quantity=100,
            )
            for i in range(count)
        ]
    )
    automatic = targets[["stream", "name", "adduct", "ref_mz"]].copy()
    automatic["feat_mz"] = automatic.ref_mz * (1 + 1e-6)
    automatic["feat_feature_id"] = [str(i) for i in range(count)]
    automatic["is_is"] = True
    pool = pd.DataFrame(
        [dict(stream="RP_Pos", ref_mz=755.0, ppm_error=1.3, feat_feature_id="query")]
    )
    return automatic, targets, pool


def test_real_labelled_catalogue_and_positional_reporting_aliases():
    root = Path(__file__).resolve().parents[2] / "reference_data/matlab_pipeline"
    targets = declared_standard_targets(
        root,
        [
            dict(name=n, quantity=100.0, unit="nM")
            for n in ("CE(16:0)-D7", "SPH(d17:1)", "S1P(d17:0)")
        ],
    )
    automatic = targets[["stream", "name", "adduct", "ref_mz"]].copy()
    automatic["feat_mz"] = automatic.ref_mz * (1 + 0.5e-6)
    automatic["feat_feature_id"] = [str(i) for i in range(len(automatic))]
    automatic["is_is"] = True
    anchors = validated_mass_anchors(automatic, targets)
    assert len(anchors) == len(automatic)
    assert anchors.ppm.to_numpy() == pytest.approx(np.full(len(anchors), 0.5), abs=1e-8)
    a, t, _ = fixtures()
    t["name"] = "LPC(17:1) (IS)"
    a["name"] = "1-acyl LPC(17:1) (IS)"
    assert len(validated_mass_anchors(a, t)) == len(a)


def test_declared_chemical_anchors_and_centered_evidence_preserve_input():
    a, t, p = fixtures()
    original = p.copy(deep=True)
    anchors = validated_mass_anchors(a, t)
    e = standard_mass_evidence(p, anchors)
    assert len(e.columns) == 8 and e.q_standard_mass_anchor_count.iloc[0] == 7
    assert e.q_standard_mass_center_ppm.iloc[0] == pytest.approx(1)
    assert e.q_centered_mass_error_ppm.iloc[0] == pytest.approx(0.3, abs=1e-6)
    assert e.q_squared_centered_mass_error_ppm.iloc[0] == pytest.approx(0.09, abs=1e-6)
    pd.testing.assert_frame_equal(original, p)


def test_zero_quantity_undeclared_wrong_labelled_mass_and_nonstandards_are_excluded():
    a, t, _ = fixtures()
    assert len(validated_mass_anchors(a, t.assign(internal_standard_quantity=0))) == 0
    assert len(validated_mass_anchors(a.assign(ref_mz=a.ref_mz - 7), t)) == 0
    assert len(validated_mass_anchors(a.assign(is_is=False), t)) == 0
    assert len(validated_mass_anchors(a.assign(name="PE(999:0) (IS)"), t)) == 0


def test_other_adducts_and_split_features_do_not_multiply_chemical_support():
    a, t, p = fixtures(count=5)
    a = pd.concat([a, a.assign(feat_feature_id="split")], ignore_index=True)
    sodium = t.assign(adduct="[M+Na]+", ref_mz=t.ref_mz + 21.981945)
    extra = a.iloc[:5].copy()
    extra["adduct"] = "[M+Na]+"
    extra["ref_mz"] = sodium.ref_mz.to_numpy()
    extra["feat_mz"] = extra.ref_mz * (1 + 1e-6)
    a = pd.concat([a, extra], ignore_index=True)
    t = pd.concat([t, sodium], ignore_index=True)
    e = standard_mass_evidence(p, validated_mass_anchors(a, t))
    assert e.q_standard_mass_anchor_count.iloc[0] == 5
    assert e.q_standard_mass_enabled.iloc[0] == 0
    assert e.q_centered_mass_error_ppm.isna().all()


def test_query_and_its_neutral_companion_channels_cannot_calibrate_themselves():
    a, t, p = fixtures(count=6)
    p = p.assign(feat_feature_id="0")
    e = standard_mass_evidence(p, validated_mass_anchors(a, t))
    assert (
        e.q_standard_mass_anchor_count.iloc[0] == 5
        and e.q_standard_mass_enabled.iloc[0] == 0
    )
    # A second feature with the same theoretical standard mass is an alias too.
    p = p.assign(feat_feature_id="other", ref_mz=500.0)
    assert (
        standard_mass_evidence(
            p, validated_mass_anchors(a, t)
        ).q_standard_mass_anchor_count.iloc[0]
        == 5
    )


def test_dispersed_or_excessive_offsets_are_unknown_and_streams_never_mix():
    a, t, p = fixtures()
    a["feat_mz"] = a.ref_mz * (
        1 + np.array([-2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5]) * 1e-6
    )
    e = standard_mass_evidence(p, validated_mass_anchors(a, t))
    assert (
        e.q_standard_mass_enabled.iloc[0] == 0
        and e.q_standard_mass_center_ppm.isna().all()
    )
    a, t, p = fixtures()
    a["feat_mz"] = a.ref_mz * (1 + 2.5e-6)
    assert (
        standard_mass_evidence(
            p, validated_mass_anchors(a, t)
        ).q_standard_mass_enabled.iloc[0]
        == 0
    )
    a, t, p = fixtures()
    assert (
        standard_mass_evidence(
            p.assign(stream="NP_Pos"), validated_mass_anchors(a, t)
        ).q_standard_mass_enabled.iloc[0]
        == 0
    )


def test_mass_context_does_not_read_labels_or_identifiers_as_predictors():
    a, t, p = fixtures()
    p = p.assign(
        training_positive=True, manual_feature_id="not allowed", dataset="not used"
    )
    e = standard_mass_evidence(p, validated_mass_anchors(a, t))
    e2 = standard_mass_evidence(
        p.assign(training_positive=False, dataset="different"),
        validated_mass_anchors(a, t),
    )
    pd.testing.assert_frame_equal(e, e2)
    assert not {
        "training_positive",
        "manual_feature_id",
        "dataset",
        "feat_feature_id",
    } & set(e)


def test_conflicting_feature_ids_cannot_vote_as_several_chemicals_and_bad_schema_fails():
    a, t, p = fixtures()
    a["feat_feature_id"] = "same"
    assert validated_mass_anchors(a, t).empty
    with pytest.raises(ValueError):
        standard_mass_evidence(p, pd.DataFrame())
    a, t, p = fixtures()
    assert summarize_mass_anchors(validated_mass_anchors(a, t))["RP_Pos"]["enabled"]
