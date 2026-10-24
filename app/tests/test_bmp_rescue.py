import pandas as pd
import pytest

from matlab_annotation.bmp_rescue import bmp_queries, select_bmp_rescue


def fixture():
    pool = pd.DataFrame(
        dict(
            name=["BMP(36:2)", "BMP(36:2)", "BMP(38:4)"],
            adduct=["[M+H]+"] * 3,
            stream=["NP_Pos"] * 3,
            **{"class": ["BMP"] * 3},
            is_is=[False] * 3,
            feat_feature_id=["best", "loser", "other"],
            feat_intensity=[100, 10, 20],
        )
    )
    return pool, pd.Series([0.95, 0.7, 0.9]), pool.iloc[:0].copy()


def test_occupied_identity_and_feature_are_never_replaced_or_promote_fallback():
    pool, scores, previous = fixture()
    previous = pool.iloc[[0]].copy()
    before = previous.copy(deep=True)
    selected = select_bmp_rescue(pool, scores, previous, stream="NP_Pos", threshold=0.6)
    assert selected.feat_feature_id.tolist() == ["other"]
    pd.testing.assert_frame_equal(previous, before)
    occupied_by_other_class = previous.assign(name="PG(36:2)", **{"class": "PG"})
    selected = select_bmp_rescue(
        pool, scores, occupied_by_other_class, stream="NP_Pos", threshold=0.6
    )
    assert selected.feat_feature_id.tolist() == ["other"]


def test_ambiguity_is_checked_before_occupancy_and_no_manual_columns_are_used():
    pool, scores, previous = fixture()
    rival = pool.iloc[[0]].assign(name="BMP(34:1)")
    pool = pd.concat([pool, rival], ignore_index=True)
    scores = pd.Series([0.95, 0.7, 0.9, 0.94])
    selected = select_bmp_rescue(pool, scores, previous, stream="NP_Pos", threshold=0.6)
    assert selected.feat_feature_id.tolist() == ["other"]
    pool["training_positive"] = True
    again = select_bmp_rescue(pool, scores, previous, stream="NP_Pos", threshold=0.6)
    assert again.feat_feature_id.tolist() == ["other"]


def test_np_subset_preserves_indices_and_does_not_merge_other_classes():
    pool, _, _ = fixture()
    more = pd.concat(
        [pool, pool.assign(stream="RP_Pos"), pool.assign(**{"class": "PG"})],
        ignore_index=True,
    )
    data = dict(
        pool=more,
        evidence=pd.DataFrame({"value": range(len(more))}),
        query_group="q",
        reference_training_groups=["t"],
    )
    subset = bmp_queries(data)
    assert subset["pool"].index.tolist() == [0, 1, 2]
    assert subset["query_group"] == "q"
    assert len(data["pool"]) == 9
    with pytest.raises(ValueError, match="only to NP"):
        select_bmp_rescue(
            pool,
            pd.Series([1.0, 1.0, 1.0]),
            pool.iloc[:0],
            stream="RP_Pos",
            threshold=0.9,
        )
