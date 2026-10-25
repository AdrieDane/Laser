import hashlib
import copy

import numpy as np
import pandas as pd
import pytest

from annotator.composition_ranges import DEFAULT_COMPOSITION_RANGE_SETTINGS
from matlab_annotation.coherent_retention import (
    coherent_retention_predictions,
    composition_block,
    alignment_doublebonds,
    retention_family,
)


def inputs(monkeypatch, classes=("PC",)):
    targets, features = [], []
    for j, cls in enumerate(classes):
        for c in range(30, 42, 2):
            for d in range(6):
                name = f"{cls}({c}:{d})"
                mass = 500 + 3 * c + 0.123 * d + 100 * j
                targets.append(
                    dict(
                        name=name,
                        **{"class": cls},
                        carbon=c,
                        db=d,
                        adduct="[M+H]+",
                        ref_mz=mass,
                    )
                )
                features.append(
                    dict(
                        feature_id=name,
                        mz=mass * (1 + 0.2e-6),
                        rt=(200 + 5 * c - 3 * d + 4 * j + 25) / 60,
                        intensity=1000.0,
                    )
                )
    data = pd.DataFrame(targets)

    def base(targets, automatic, atlas, *, stream):
        branch = targets["class"].map({cls: 4 * j for j, cls in enumerate(classes)})
        return pd.DataFrame(
            dict(
                atlas_rt_seconds=200 + 5 * targets.carbon - 3 * targets.db + branch,
                atlas_in_domain=True,
                atlas_training_cohorts=4,
                atlas_training_mad_seconds=1.0,
            ),
            index=targets.index,
        )

    monkeypatch.setattr(
        "matlab_annotation.coherent_retention.predict_aligned_retention_atlas", base
    )
    bundle = dict(
        format="laser-nested-reference-bundle-v1",
        training_groups=["a", "b", "c", "d"],
        excluded_groups=["query", "outer"],
        atlas=dict(
            training_cohorts=4,
            training_scope_sha256=hashlib.sha256(b"a\nb\nc\nd").hexdigest(),
        ),
    )
    return [
        data,
        pd.DataFrame(features),
        pd.DataFrame(),
        pd.DataFrame(columns=["stream", "ref_mz"]),
        pd.DataFrame(),
        bundle,
    ]


def call(args, **kwargs):
    return coherent_retention_predictions(
        *args,
        stream="RP_Pos",
        query_group="query",
        expected_excluded_groups={"query", "outer"},
        ranges=DEFAULT_COMPOSITION_RANGE_SETTINGS,
        **kwargs,
    )


def test_blind_physical_shift_and_composition_crossfit_without_measurement_mutation(
    monkeypatch,
):
    args = inputs(monkeypatch)
    before = [f.copy(deep=True) for f in args[:5]]
    result, audit = call(args)
    assert result.coherent_available.all()
    np.testing.assert_allclose(result.coherent_shift_seconds, 25, atol=1e-9)
    assert len(audit) == 5 and audit.supported.all()
    for frame, original in zip(args[:5], before):
        pd.testing.assert_frame_equal(frame, original, check_exact=True)
    own = args[0].iloc[0]
    own_block = composition_block([own.carbon], [own.db])[0]
    same_block = composition_block(args[0].carbon, args[0].db) == own_block
    args[1].loc[same_block, "rt"] += 50 / 60
    changed, _ = call(args)
    np.testing.assert_array_equal(
        changed.loc[same_block, "coherent_rt_seconds"],
        result.loc[same_block, "coherent_rt_seconds"],
    )


def test_rp_isomer_branches_keep_relative_rt_and_np_does_not_inherit_rp_grouping(
    monkeypatch,
):
    args = inputs(monkeypatch, classes=("PC[O]", "PC[P]"))
    result, _ = call(args)
    assert result.coherent_available.all()
    np.testing.assert_allclose(
        result.coherent_rt_seconds.iloc[36:].to_numpy()
        - result.coherent_rt_seconds.iloc[:36].to_numpy(),
        4,
    )
    assert retention_family("1-acyl LPC", "RP_Pos") == retention_family(
        "2-acyl LPC", "RP_Pos"
    )
    assert retention_family("1-acyl LPC", "NP_Pos") != retention_family(
        "2-acyl LPC", "NP_Pos"
    )
    assert retention_family("PC[O]", "RP_Pos") == retention_family("PC[P]", "RP_Pos")


def test_formula_equivalent_ether_plasmalogen_hypotheses_hold_out_each_others_peaks(
    monkeypatch,
):
    args = inputs(monkeypatch, classes=("PC[O]", "PC[P]"))
    plasmalogen = args[0]["class"].eq("PC[P]")
    # P(C:D) is isobaric with O(C:D+1), including across the holdout boundary.
    args[0].loc[plasmalogen, "ref_mz"] = (
        500
        + 3 * args[0].loc[plasmalogen, "carbon"]
        + 0.123 * (args[0].loc[plasmalogen, "db"] + 1)
    )
    args[1].loc[plasmalogen, "mz"] = args[0].loc[plasmalogen, "ref_mz"] * (1 + 0.2e-6)
    effective = [
        alignment_doublebonds(c, d, "RP_Pos")
        for c, d in args[0][["class", "db"]].itertuples(index=False, name=None)
    ]
    blocks = composition_block(args[0].carbon, effective)
    before, _ = call(args)
    own = args[0].index[
        args[0]["class"].eq("PC[O]") & args[0].carbon.eq(30) & args[0].db.eq(1)
    ][0]
    alternate = args[0].index[plasmalogen & args[0].carbon.eq(30) & args[0].db.eq(0)][0]
    assert blocks[own] == blocks[alternate]
    assert args[0].loc[own, "ref_mz"] == args[0].loc[alternate, "ref_mz"]
    args[1].loc[blocks == blocks[own], "rt"] += 50 / 60
    after, _ = call(args)
    pd.testing.assert_frame_equal(
        before.loc[blocks == blocks[own]],
        after.loc[blocks == blocks[own]],
        check_exact=True,
    )


def test_odd_carbon_prediction_is_allowed_but_odd_anchors_do_not_vote(monkeypatch):
    args = inputs(monkeypatch)
    odd = args[0].iloc[[0]].assign(name="PC(33:1)", carbon=33, db=1, ref_mz=999.0)
    args[0] = pd.concat([args[0], odd], ignore_index=True)
    result, _ = call(args)
    assert result.coherent_available.iloc[-1]
    args[0] = args[0].assign(carbon=33)
    result, audit = call(args)
    assert not result.coherent_available.any() and not audit.supported.any()


def test_spike_mass_and_feature_protection_and_bad_mass_hits_fail_closed(monkeypatch):
    for protection in ("mass", "feature", "bad_mass"):
        args = inputs(monkeypatch)
        if protection == "mass":
            args[3] = pd.DataFrame(dict(stream="RP_Pos", ref_mz=args[1].mz))
        elif protection == "feature":
            args[4] = pd.DataFrame(
                dict(stream="RP_Pos", feat_feature_id=args[1].feature_id)
            )
        else:
            args[1]["mz"] *= 1 + 3e-6
        result, audit = call(args)
        assert not result.coherent_available.any()
        np.testing.assert_array_equal(
            result.coherent_rt_seconds, result.baseline_rt_seconds
        )


def test_supervision_reference_scope_and_original_feature_conflicts_rejected(
    monkeypatch,
):
    args = inputs(monkeypatch)
    for mutate in (
        "own_training",
        "missing_outer",
        "scope_hash",
        "supervision",
        "duplicate_feature",
    ):
        case = copy.deepcopy(args)
        if mutate == "own_training":
            case[5]["training_groups"].append("query")
        elif mutate == "missing_outer":
            case[5]["excluded_groups"] = ["query"]
        elif mutate == "scope_hash":
            case[5]["atlas"]["training_scope_sha256"] = "wrong"
        elif mutate == "supervision":
            case[1]["same_manual_peak"] = False
        else:
            case[1]["feature_id"] = "same"
        with pytest.raises(ValueError):
            call(case)


def headgroup_inputs(monkeypatch):
    args = inputs(monkeypatch, classes=("PC", "PE"))
    pe = args[0]["class"].eq("PE")
    args[0].loc[pe, "ref_mz"] -= 100
    args[1].loc[pe, "mz"] = args[0].loc[pe, "ref_mz"] * (1 + 0.2e-6)
    args[1].loc[pe, "rt"] += 56 / 60

    def base(targets, automatic, atlas, *, stream):
        return pd.DataFrame(
            dict(
                atlas_rt_seconds=200
                + 5 * targets.carbon
                - 3 * targets.db
                + targets["class"].eq("PE").astype(int) * 60,
                atlas_in_domain=True,
                atlas_training_cohorts=4,
                atlas_training_mad_seconds=1.0,
            ),
            index=targets.index,
        )

    monkeypatch.setattr(
        "matlab_annotation.coherent_retention.predict_aligned_retention_atlas", base
    )
    return args, pe


def test_competing_class_mass_hypotheses_veto_a_strong_wrong_headgroup_shift(
    monkeypatch,
):
    args, pe = headgroup_inputs(monkeypatch)
    # Only PC is abundant enough to anchor. Its identical masses must not drag
    # PE predictions 35 seconds toward the wrong headgroup's raw peaks.
    args[1].loc[pe, "intensity"] = 1.0
    unguarded, _ = call(args)
    assert unguarded.loc[pe, "coherent_shift_seconds"].lt(-30).all()
    guarded, audit = call(args, competing_class_anchor_guard=True)
    assert not guarded.loc[pe, "coherent_available"].any()
    assert audit.competing_class_vetoed_hypotheses.sum() > 0
    np.testing.assert_array_equal(
        guarded.loc[pe, "coherent_rt_seconds"], guarded.loc[pe, "baseline_rt_seconds"]
    )


def test_competing_class_guard_can_separate_two_chemically_isobaric_raw_rt_series(
    monkeypatch,
):
    args, _ = headgroup_inputs(monkeypatch)
    guarded, _ = call(args, competing_class_anchor_guard=True)
    assert guarded.coherent_available.all()
    np.testing.assert_allclose(guarded.coherent_shift_seconds, 25, atol=1e-8)
