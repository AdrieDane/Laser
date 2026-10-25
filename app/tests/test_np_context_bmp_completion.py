import copy

import numpy as np
import pandas as pd
import pytest

from test_evidence_training import fixture, data
from test_bmp_rescue import fixture as candidate_fixture
from matlab_annotation.np_context_bmp_completion import (
    NP_MODES,
    mask_training_bmp,
    fit_np_context,
    predict_np_context,
    select_context_bmp_additions,
)


def records():
    examples, groups, parameters = fixture()
    manual = pd.DataFrame(
        dict(stream=list(NP_MODES), lipid_class="BMP", internal_standard=False)
    )
    for key, record in list(examples.items()):
        positive = [True] * 10 + [False] * 10 + [True] * 10 + [False] * 10
        classes = ["BMP"] * 10 + ["PG"] * 10 + ["PC"] * 20
        record["pool"] = pd.concat(
            [
                pd.DataFrame(
                    dict(
                        stream=m,
                        **{"class": classes},
                        is_is=False,
                        training_positive=positive,
                        training_label_known=True,
                    )
                )
                for m in NP_MODES
            ],
            ignore_index=True,
        )
        record["evidence"] = pd.DataFrame(
            dict(
                stream=record["pool"].stream,
                **{"class": record["pool"]["class"]},
                mass_error=[0.1 if x else 2 for x in positive] * 2,
            )
        )
        examples[key] = mask_training_bmp(
            record, manual, training_groups=groups, excluded_groups={"z"}
        )
    query = data("z", sorted(groups), ["z"])
    query["pool"] = (
        examples["a"]["pool"]
        .drop(columns=["training_positive", "training_label_known"])
        .copy()
    )
    query["evidence"] = examples["a"]["evidence"].copy()
    return examples, groups, parameters, query


def test_two_np_experts_use_all_class_contrast_not_just_bmp_and_keep_query_order():
    examples, groups, parameters, query = records()
    model = fit_np_context(
        examples, training_groups=groups, excluded_groups={"z"}, parameters=parameters
    )
    assert model["fresh_mode_classifier_fits"] == 2
    assert model["training_pairs"] == 240 and model["positives"] == 120
    for mode, expert in model["experts"].items():
        assert set(expert["categories"]["class"]) == {"BMP", "PC", "PG"}
        assert model["support"][mode]["other_class_known_context_rows"] == 90
        assert model["support"][mode]["BMP_positive_projects"] == ["a", "b", "c"]
    before = query["pool"].copy(deep=True)
    result = predict_np_context(query, model)
    assert result.index.equals(query["pool"].index) and np.isfinite(result).all()
    pd.testing.assert_frame_equal(query["pool"], before)


def test_partial_labels_are_training_only_unknown_not_positive_or_query_evidence():
    examples, groups, parameters, query = records()
    record = copy.deepcopy(examples["a"])
    negative = record["pool"].stream.eq("NP_Neg") & record["pool"]["class"].eq("BMP")
    record["pool"].loc[negative, "training_positive"] = False
    manual = pd.DataFrame(
        dict(stream=["NP_Pos"], lipid_class=["BMP"], internal_standard=[False])
    )
    masked = mask_training_bmp(
        record, manual, training_groups=groups, excluded_groups={"z"}
    )
    assert not masked["pool"].loc[negative, "training_label_known"].any()
    pd.testing.assert_series_equal(
        masked["pool"].training_positive, record["pool"].training_positive
    )
    with pytest.raises(ValueError):
        mask_training_bmp(query, manual, training_groups=groups, excluded_groups={"z"})
    model = fit_np_context(
        examples, training_groups=groups, excluded_groups={"z"}, parameters=parameters
    )
    query["pool"]["training_positive"] = False
    with pytest.raises(ValueError, match="refuses supervision"):
        predict_np_context(query, model)


def test_no_bmp_only_training_or_invented_sparse_support_or_bad_expert_scope():
    examples, groups, parameters, query = records()
    for e in examples.values():
        bmp = e["pool"]["class"].eq("BMP")
        e["pool"].loc[bmp, "training_positive"] = False
        audit = e["bmp_partial_supervision_audit"]
        audit["positive_rows_before"] = audit["positive_rows_after"] = int(
            e["pool"].training_positive.sum()
        )
    model = fit_np_context(
        examples, training_groups=groups, excluded_groups={"z"}, parameters=parameters
    )
    assert model["fresh_mode_classifier_fits"] == 0
    assert predict_np_context(query, model).isna().all()
    examples, groups, parameters, query = records()
    for e in examples.values():
        rows = e["pool"]["class"].eq("BMP")
        e["pool"] = e["pool"].loc[rows]
        e["evidence"] = e["evidence"].loc[rows]
        e["bmp_partial_supervision_audit"].update(
            positive_rows_before=20, positive_rows_after=20, known_rows_after=20
        )
    with pytest.raises(ValueError, match="not a BMP-only"):
        fit_np_context(
            examples,
            training_groups=groups,
            excluded_groups={"z"},
            parameters=parameters,
        )
    examples, groups, parameters, query = records()
    model = fit_np_context(
        examples, training_groups=groups, excluded_groups={"z"}, parameters=parameters
    )
    model["experts"]["NP_Pos"]["excluded_groups"] = ["wrong"]
    with pytest.raises(ValueError, match="scopes disagree"):
        predict_np_context(query, model)


def candidates():
    pool, scores, previous = candidate_fixture()
    pool = pool.assign(
        ref_mz=[700.0, 700.0, 800.0],
        feat_mz=[700.0001, 700.0001, 800.0001],
        feat_rt=5.0,
        ref_rt=5.0,
    )
    return pool, scores, pool.iloc[:0].copy()


def choose(pool, scores, previous, **kwargs):
    return select_context_bmp_additions(
        pool,
        scores,
        previous,
        stream="NP_Pos",
        threshold=kwargs.pop("threshold", 0.6),
        protected_features=kwargs.pop("protected_features", set()),
        declared_masses=kwargs.pop("declared_masses", []),
        **kwargs,
    )


def test_global_other_class_winner_or_ambiguity_blocks_bmp_before_occupancy():
    pool, scores, previous = candidates()
    rival = pool.iloc[[0]].assign(name="PG(36:2)", **{"class": "PG"})
    pool = pd.concat([pool, rival], ignore_index=True)
    for other in (0.99, 0.94):
        chosen = choose(pool, pd.Series([0.95, 0.7, 0.9, other]), previous)
        assert chosen.feat_feature_id.tolist() == ["other"]
    # Even when PG is on an already occupied feature,no weaker BMP peak is
    # promoted for the blocked BMP identity.
    occupied = rival.copy()
    assert choose(
        pool, pd.Series([0.95, 0.7, 0.9, 0.99]), occupied
    ).feat_feature_id.tolist() == ["other"]


def test_all_old_identities_features_and_spike_measurements_stay_protected():
    pool, scores, previous = candidates()
    old = pool.iloc[[0]].copy()
    before = old.copy(deep=True)
    assert choose(pool, scores, old).feat_feature_id.tolist() == ["other"]
    pd.testing.assert_frame_equal(old, before, check_exact=True)
    assert choose(
        pool, scores, previous, protected_features={"best"}
    ).feat_feature_id.tolist() == ["other"]
    assert choose(
        pool, scores, previous, declared_masses=[700.0]
    ).feat_feature_id.tolist() == ["other"]
    assert choose(pool, scores, previous, threshold=1.0).empty
    assert choose(pool, pd.Series([1.0, 1.0, 1.0]), previous, threshold=1.0).empty


def test_raw_mass_rt_guards_original_coordinates_and_label_invariance():
    pool, scores, previous = candidates()
    before = pool.copy(deep=True)
    chosen = choose(pool, scores, previous)
    assert chosen.feat_feature_id.tolist() == ["best", "other"]
    pd.testing.assert_frame_equal(pool, before, check_exact=True)
    pool["training_positive"] = [False, True, False]
    pool["calibration_identity_in_manual"] = False
    again = choose(pool, scores, previous)
    assert again.feat_feature_id.tolist() == chosen.feat_feature_id.tolist()
    for c, value in (("feat_mz", 900.0), ("feat_rt", 6.0), ("feat_intensity", 0.0)):
        bad = before.copy()
        bad[c] = value
        assert choose(bad, scores, previous).empty
    bad = before.assign(**{"class": "PC"})
    empty = choose(bad, scores, previous)
    assert empty.empty and "feat_feature_id" in empty
