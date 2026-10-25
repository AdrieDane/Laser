"""Scope failures in the NEW reference-dependent measurement namespace."""

import copy

import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
from prepare_qe_bmp_context_tables import validate_scope, masks_for


def fixture():
    return dict(
        query_dataset="A_sample",
        query_group="A",
        excluded_folds=[0, 1],
        reference_training_groups=["B", "C"],
        reference_excluded_groups=["A", "D"],
        pool=pd.DataFrame(index=[0]),
        evidence=pd.DataFrame({"q_BMP_context_source_physical": [1.0]}, index=[0]),
        bmp_context_provenance=dict(
            query_dataset="A_sample",
            mask=[0, 1],
            source131_sha256="a",
            sidecar_sha256="b",
            reference_training_groups=["B", "C"],
            reference_excluded_groups=["A", "D"],
            new_numeric_fields=32,
            curation_runtime_input=False,
            graph_membership_unioned=False,
        ),
    )


def test_new_scope_guard_accepts_only_matching_reference_context():
    raw = fixture()
    before = copy.deepcopy(raw)
    validate_scope(raw, {"B", "C"}, {"A", "D"}, query_is_training=False)
    pd.testing.assert_frame_equal(raw["evidence"], before["evidence"], check_exact=True)
    assert raw["bmp_context_provenance"] == before["bmp_context_provenance"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("mask", [0]),
        ("reference_training_groups", ["A", "B", "C"]),
        ("reference_excluded_groups", ["D"]),
        ("curation_runtime_input", True),
        ("graph_membership_unioned", True),
        ("query_dataset", "other"),
        ("new_numeric_fields", 31),
    ],
)
def test_mismatched_mask_or_new_reference_leak_cannot_hide_behind_old_scope(key, value):
    raw = fixture()
    raw["bmp_context_provenance"][key] = value
    with pytest.raises(ValueError):
        validate_scope(raw, {"B", "C"}, {"A", "D"}, query_is_training=False)


def test_all_eleven_masks_keep_query_fold_and_at_most_two_other_blocks():
    for own in range(5):
        masks = masks_for(own)
        assert len(set(masks)) == 11
        assert all(own in m and len(m) <= 3 for m in masks)
