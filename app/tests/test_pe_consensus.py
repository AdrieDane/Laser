from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from matlab_annotation.pe_consensus import add_rp_consensus_evidence


def source():
    return pd.DataFrame(
        [
            dict(
                name="PE(36:2)",
                **{"class": "PE"},
                adduct="[M+H]+",
                feat_feature_id="001",
                is_is=False,
                internal_standard_name="",
                isotopologue="",
            )
        ]
    )


def test_frozen_primary_identity_evidence_preserves_original_source_ids():
    pool = pd.DataFrame({"name": ["PE(36:2)", "PE(38:2)"]})
    positive = source()
    negative = source().assign(adduct="[M-H]-", feat_feature_id="NEG-002")
    before = positive.copy(deep=True)
    out = add_rp_consensus_evidence(pool, {"NP_Pos": positive, "NP_Neg": negative})
    assert out.pe_rp_np_positive_feature_id.tolist() == ["001", ""]
    assert out.pe_rp_np_negative_feature_id.tolist() == ["NEG-002", ""]
    assert out.pe_rp_np_positive_confirmed.tolist() == [True, False]
    assert out.pe_rp_np_negative_confirmed.tolist() == [True, False]
    assert_frame_equal(positive, before)
    assert pool.columns.tolist() == ["name"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("class", "PC"),
        ("adduct", "[M+Na]+"),
        ("is_is", True),
        ("internal_standard_name", "declared spike"),
        ("isotopologue", "M+1"),
        ("feat_feature_id", None),
        ("feat_feature_id", ""),
    ],
)
def test_other_classes_adducts_standards_and_isotopes_cannot_confirm(field, value):
    p = source()
    p.loc[0, field] = value
    result = add_rp_consensus_evidence(
        pd.DataFrame({"name": ["PE(36:2)"]}), {"NP_Pos": p}
    )
    assert not result.pe_rp_np_positive_confirmed.any()
    assert not result.pe_rp_np_negative_confirmed.any()


def test_missing_or_multiple_primary_choices_are_not_claimed_as_confirmation():
    pool = pd.DataFrame({"name": ["PE(36:2)"]})
    multiple = pd.concat([source(), source().assign(feat_feature_id="other")])
    for sources in (
        {},
        {"NP_Pos": multiple},
        {"NP_Pos": pd.DataFrame({"name": ["PE(36:2)"]})},
    ):
        result = add_rp_consensus_evidence(pool, sources)
        assert not result.pe_rp_np_positive_confirmed.any()


def test_text_false_standard_flags_do_not_discard_ordinary_PE():
    result = add_rp_consensus_evidence(
        pd.DataFrame({"name": ["PE(36:2)"]}),
        {"NP_Pos": source().assign(is_is="False")},
    )
    assert result.pe_rp_np_positive_confirmed.all()
