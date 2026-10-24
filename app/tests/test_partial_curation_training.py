import pandas as pd
import pytest

from matlab_annotation.partial_curation_training import mask_uncurated_opposite_np_bmp


def fixtures():
    pool = pd.DataFrame(
        dict(
            stream=["NP_Pos", "NP_Neg", "NP_Neg", "NP_Neg", "RP_Neg", "NP_Neg"],
            **{"class": ["BMP", "BMP", "BMP", "PE", "BMP", "BMP"]},
            is_is=[False, False, False, False, False, True],
            training_positive=[True, False, False, False, False, False],
            training_label_known=[True, True, False, True, True, True],
            unchanged_predictor=[0, 1, 2, 3, 4, 5],
        ),
        index=[11, 13, 17, 19, 23, 29],
    )
    manual = pd.DataFrame(
        dict(
            stream=["NP_Pos"],
            lipid_class=["BMP"],
            internal_standard=[False],
            mapped=[False],
        )
    )
    return pool, manual


def apply(pool, manual, **kwargs):
    return mask_uncurated_opposite_np_bmp(
        pool,
        manual,
        query_group="a",
        permitted_groups={"a", "b"},
        excluded_groups={"z"},
        **kwargs,
    )


def test_only_known_negative_in_uncurated_opposite_mode_withheld_and_inputs_immutable():
    pool, manual = fixtures()
    before = pool.copy(deep=True)
    result, audit = apply(pool, manual)
    assert result.training_label_known.tolist() == [
        True,
        False,
        False,
        True,
        True,
        True,
    ]
    pd.testing.assert_frame_equal(pool, before)
    pd.testing.assert_frame_equal(
        result.drop(columns="training_label_known"),
        before.drop(columns="training_label_known"),
    )
    assert audit["withheld_known_negatives"] == 1
    assert audit["positive_rows_before"] == audit["positive_rows_after"] == 1
    assert audit["uncurated_opposite_np_modes"] == ["NP_Neg"]


@pytest.mark.parametrize("case", ["both", "none", "standard_only", "rp_only"])
def test_no_policy_for_fully_curated_or_absent_class_or_standards(case):
    pool, manual = fixtures()
    pool["training_positive"] = False
    if case == "both":
        manual = pd.concat([manual, manual.assign(stream="NP_Neg")], ignore_index=True)
    elif case == "none":
        manual = manual.iloc[:0]
    elif case == "standard_only":
        manual["internal_standard"] = True
    else:
        manual["stream"] = "RP_Pos"
    result, audit = apply(pool, manual)
    pd.testing.assert_frame_equal(result, pool)
    assert audit["withheld_known_negatives"] == 0


def test_reverse_polarity_and_unresolved_manual_presence():
    pool, manual = fixtures()
    pool["training_positive"] = False
    manual["stream"] = "NP_Neg"
    result, audit = apply(pool, manual)
    assert audit["uncurated_opposite_np_modes"] == ["NP_Pos"]
    assert not result.loc[11, "training_label_known"]
    assert result.loc[13, "training_label_known"]


def test_conflicting_positive_and_forbidden_scope_fail_closed():
    pool, manual = fixtures()
    pool.loc[13, "training_positive"] = True
    with pytest.raises(ValueError, match="positive conflicts"):
        apply(pool, manual)
    with pytest.raises(ValueError, match="allowed training"):
        mask_uncurated_opposite_np_bmp(
            pool, manual, query_group="z", permitted_groups={"a"}, excluded_groups={"z"}
        )
    pool["training_positive"] = pool.training_positive.astype(object)
    pool.loc[13, "training_positive"] = None
    with pytest.raises(ValueError, match="booleans"):
        apply(pool, manual)
