import numpy as np
import pandas as pd
import pytest

from matlab_annotation.related_standard_context import (
    PREFIX,
    related_standard_context_evidence,
)
from matlab_annotation.declared_standard_context import (
    PREFIX as EXACT_PREFIX,
    declared_standard_context_evidence,
)
from test_declared_standard_context import fixture


@pytest.mark.parametrize(
    "query_class,parent,role",
    [
        ("PE-P", "PE", 1),
        ("PC-O", "PC", 1),
        ("PE[O+P]", "PE", 1),
        ("2-acylLPC", "1-acyl LPC", 2),
        ("1-acyl LPE", "LPE", 2),
    ],
)
def test_related_context_is_distinct_from_exact_identity(query_class, parent, role):
    query, spike = fixture()
    query["class"], spike["class"] = query_class, parent
    original = query.copy(deep=True)
    before_spike = spike.copy(deep=True)
    values = related_standard_context_evidence(query, spike).iloc[0]
    assert values[PREFIX + "related_compounds"] == 1
    assert values[PREFIX + "relation"] == role
    assert values[PREFIX + "RT_difference_seconds"] == pytest.approx(6)
    assert (
        declared_standard_context_evidence(query, spike).iloc[0][
            EXACT_PREFIX + "same_class_compounds"
        ]
        == 0
    )
    pd.testing.assert_frame_equal(query, original)
    pd.testing.assert_frame_equal(spike, before_spike)


def test_same_class_and_other_modes_or_unrelated_heads_cannot_be_related_anchors():
    query, spike = fixture()
    query["class"] = "PE-P"
    for cls, stream in (("PE-P", "NP_Pos"), ("PC", "NP_Pos"), ("PE", "NP_Neg")):
        result = related_standard_context_evidence(
            query, spike.assign(**{"class": cls, "stream": stream})
        ).iloc[0]
        assert result[PREFIX + "related_compounds"] == 0
        assert np.isnan(result[PREFIX + "RT_difference_seconds"])


def test_any_spike_feature_is_blocked_and_companion_ions_count_once():
    query, spike = fixture()
    query["class"], spike["class"] = "PE-P", "PE"
    measured = pd.concat(
        [
            spike,
            spike.assign(adduct="[M+H]+", feat_feature_id="companion", feat_rt=5.01),
        ],
        ignore_index=True,
    )
    result = related_standard_context_evidence(query, measured).iloc[0]
    assert result[PREFIX + "related_compounds"] == 1
    for fid in ("spike", "companion"):
        own = related_standard_context_evidence(
            query.assign(feat_feature_id=fid), measured
        ).iloc[0]
        assert own[PREFIX + "related_compounds"] == 0
        assert own[PREFIX + "measured_spike_feature"] == 1
        assert np.isnan(own[PREFIX + "RT_difference_seconds"])


def test_empty_and_invalid_standards_remain_unknown_not_zero_RT():
    query, spike = fixture()
    query["class"], spike["class"] = "PC-P", "PC"
    for measured in (spike.iloc[:0], spike.assign(internal_standard_quantity=0)):
        result = related_standard_context_evidence(query, measured)
        assert len(result.columns) == 16
        assert result[PREFIX + "related_compounds"].iloc[0] == 0
        assert np.isnan(result[PREFIX + "RT_difference_seconds"].iloc[0])
