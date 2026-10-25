"""Synthetic-only163-field fits/gates/calibration; no example-data fits."""

import copy

import numpy as np
import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
from bmp_context_completion import (
    profile_gate,
    choose_joint_grid,
    prepare_training,
    fit_context,
    predict_context,
    LEVELS,
    MODES,
)
from bmp_context_measurements import PREFIX


def grid():
    return pd.DataFrame(
        [
            dict(
                stream=s,
                threshold=t,
                manual=500,
                endogenous_manual=490,
                previous_correct=200,
                added_correct=0,
                extras=0,
                extras_without_opposite_composition=0,
                different_peak=0,
                positive_projects=0,
            )
            for s in MODES
            for t in LEVELS
        ]
    )


def test_profile_gate_requires_all_scoped_measurements_no_nan_support():
    evidence = pd.DataFrame(
        {
            PREFIX + f: [v, v]
            for f, v in dict(
                source_physical=1.0,
                candidate_counterparts=1.0,
                counterpart_profiles=1.0,
                counterpart_n=8.0,
                counterpart_raw=0.8,
                counterpart_normalized=0.8,
                counterpart_leave_one_out=0.6,
            ).items()
        }
    )
    assert profile_gate(evidence).all()
    for field in evidence:
        corrupted = evidence.copy()
        corrupted.loc[0, field] = np.nan
        assert profile_gate(corrupted).tolist() == [False, True]
    evidence.loc[0, PREFIX + "counterpart_n"] = 7
    assert profile_gate(evidence).tolist() == [False, True]


def test_joint_grid_disable_and_fixed_budgets():
    table = grid()
    cuts, full = choose_joint_grid(table)
    assert cuts == dict.fromkeys(MODES, 1.0)
    assert len(full) == len(LEVELS) ** 2
    for s in MODES:
        mask = table.stream.eq(s) & table.threshold.eq(0.9)
        table.loc[mask, ["added_correct", "extras", "positive_projects"]] = [40, 30, 3]
    cuts, full = choose_joint_grid(table)
    assert cuts == dict.fromkeys(MODES, 0.9)
    # A better nominal coverage point with too many TOTAL extras is not valid.
    bad = table.stream.eq("NP_Pos") & table.threshold.eq(0.5)
    table.loc[bad, ["added_correct", "extras", "positive_projects"]] = [200, 300, 3]
    assert choose_joint_grid(table)[0] == dict.fromkeys(MODES, 0.9)
    # Other-polarity support never removes extras from the primary budget.
    table.loc[table.threshold.eq(0.9), "extras"] = 41
    assert choose_joint_grid(table)[0] == dict.fromkeys(MODES, 1.0)


def test_independent_mode_support_and_wrong_peak_budget():
    table = grid()
    good = table.threshold.eq(0.9)
    table.loc[good, ["added_correct", "extras", "positive_projects"]] = [30, 20, 3]
    table.loc[table.stream.eq("NP_Pos") & good, "different_peak"] = 3
    cuts, _ = choose_joint_grid(table)
    assert cuts == {"NP_Pos": 1.0, "NP_Neg": 0.9}
    table.loc[table.stream.eq("NP_Neg") & good, "positive_projects"] = 2
    assert choose_joint_grid(table)[0] == dict.fromkeys(MODES, 1.0)


def test_incomplete_or_invalid_counts_cannot_select_thresholds():
    table = grid()
    with pytest.raises(ValueError, match="Every declared"):
        choose_joint_grid(table.iloc[1:])
    table.loc[0, "extras"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        choose_joint_grid(table)


def training_record(project):
    n = 40
    stream = [MODES[i // 20] for i in range(n)]
    cls = ["BMP" if i % 20 < 12 else "PG" for i in range(n)]
    pool = pd.DataFrame(
        dict(
            stream=stream,
            **{"class": cls},
            is_is=False,
            training_positive=[i % 20 < 8 for i in range(n)],
            training_label_known=True,
        )
    )
    evidence = pd.DataFrame(
        dict(
            stream=stream,
            **{"class": cls},
            adduct=["[M+NH4]+" if s.endswith("Pos") else "[M-H]-" for s in stream],
            stream_class=[s + "|" + c for s, c in zip(stream, cls)],
        )
    )
    numeric = pd.DataFrame(
        {"f" + str(j): [(i % 20) + 0.01 * j for i in range(n)] for j in range(159)}
    )
    evidence = pd.concat([evidence, numeric], axis=1)
    refs = sorted({"A", "B", "C"} - {project})
    return dict(
        query_dataset=project + "_sample",
        query_group=project,
        reference_training_groups=refs,
        reference_excluded_groups=[project, "D"],
        excluded_folds=[0, 4],
        pool=pool,
        evidence=evidence,
        bmp_context_provenance=dict(
            query_dataset=project + "_sample",
            mask=[0, 4],
            source131_sha256="a",
            sidecar_sha256="b",
            reference_training_groups=refs,
            reference_excluded_groups=[project, "D"],
            new_numeric_fields=32,
            curation_runtime_input=False,
            graph_membership_unioned=False,
        ),
        bmp_partial_supervision_audit=dict(
            evaluation_labels_changed=False,
            positive_rows_before=16,
            positive_rows_after=16,
            known_rows_after=n,
        ),
    )


def test_exposed_training_preserves_original_cohort_weights_and_actual_support():
    data = {p: training_record(p) for p in ("A", "B", "C")}
    before = copy.deepcopy(data)
    prepared = prepare_training(data, training_groups=set(data), excluded_groups={"D"})
    for part in prepared.values():
        assert part["supported"] and part["support"]["known_BMP_positive_pairs"] == 24
        assert part["support"]["BMP_positive_projects"] == ["A", "B", "C"]
        assert part["x"].shape == (60, 163)
        for p in data:
            assert part["weights"][
                np.array(part["projects"]) == p
            ].sum() == pytest.approx(20)
    for p in data:
        pd.testing.assert_frame_equal(
            data[p]["evidence"], before[p]["evidence"], check_exact=True
        )
    data["A"]["bmp_context_provenance"]["reference_training_groups"] = ["D"]
    with pytest.raises(ValueError):
        prepare_training(data, training_groups=set(data), excluded_groups={"D"})


def test_synthetic_fit_is_bitexact_old_NP_context_construction():
    from matlab_annotation.np_context_bmp_completion import (
        fit_np_context,
        predict_np_context,
    )

    data = {p: training_record(p) for p in ("A", "B", "C")}
    parameters = dict(
        max_iter=3,
        max_leaf_nodes=7,
        min_samples_leaf=2,
        l2_regularization=2.0,
        learning_rate=0.08,
        early_stopping=False,
        random_state=20261022,
    )
    new = fit_context(
        data, training_groups=set(data), excluded_groups={"D"}, parameters=parameters
    )
    old = fit_np_context(
        data, training_groups=set(data), excluded_groups={"D"}, parameters=parameters
    )
    query = training_record("A")
    query.update(
        query_dataset="D_sample",
        query_group="D",
        excluded_folds=[4],
        reference_training_groups=sorted(data),
        reference_excluded_groups=["D"],
    )
    query["pool"] = query["pool"].drop(
        columns=["training_positive", "training_label_known"]
    )
    query["bmp_context_provenance"].update(
        query_dataset="D_sample",
        mask=[4],
        reference_training_groups=sorted(data),
        reference_excluded_groups=["D"],
    )
    a, b = predict_context(query, new), predict_np_context(query, old)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    assert new["fresh_mode_classifier_fits"] == 2
    query["pool"]["training_positive"] = False
    with pytest.raises(ValueError, match="supervision"):
        predict_context(query, new)


def test_physical_profile_gate_cannot_promote_weaker_identity_alternative():
    from bmp_context_completion import select_additions

    pool = pd.DataFrame(
        [
            dict(
                stream="NP_Pos",
                name="BMP(36:2)",
                **{"class": "BMP"},
                adduct="[M+NH4]+",
                is_is=False,
                feat_feature_id=fid,
                feat_intensity=100.0,
                ref_mz=800.0,
                feat_mz=800.0001,
                feat_rt=5.0,
                ref_rt=5.0,
            )
            for fid in ["a", "b"]
        ]
    )
    scores = pd.Series([0.95, 0.9])
    evidence = pd.DataFrame(
        {
            PREFIX + f: [v, v]
            for f, v in dict(
                source_physical=1.0,
                candidate_counterparts=1.0,
                counterpart_profiles=1.0,
                counterpart_n=8.0,
                counterpart_raw=0.9,
                counterpart_normalized=0.9,
                counterpart_leave_one_out=0.8,
            ).items()
        }
    )
    evidence.loc[0, PREFIX + "counterpart_raw"] = 0.1
    before = pool.copy(deep=True)
    selected = select_additions(
        pool,
        scores,
        evidence,
        pool.iloc[:0],
        stream="NP_Pos",
        threshold=0.5,
        protected_features=set(),
        declared_masses=np.array([]),
    )
    assert selected.empty
    pd.testing.assert_frame_equal(pool, before, check_exact=True)
    evidence.loc[0, PREFIX + "counterpart_raw"] = 0.9
    selected = select_additions(
        pool,
        scores,
        evidence,
        pool.iloc[:0],
        stream="NP_Pos",
        threshold=0.5,
        protected_features=set(),
        declared_masses=np.array([]),
    )
    assert selected.feat_feature_id.tolist() == ["a"]
