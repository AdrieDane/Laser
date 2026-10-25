"""Synthetic category guards; no example annotations or model fits."""

import pandas as pd
import numpy as np
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources
from diagnose_qe_raw_counterpart_pilot import option_categories
from verify_qe_raw_counterpart_pilot import comparison, profiles

configure_frozen_resources(REPO)


def test_categories_keep_unknown_and_unconfirmed_separate_from_wrong_peak():
    pool = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                name=name,
                **{"class": name.split("(")[0]},
                adduct="[M+H]+",
                feat_feature_id=fid,
                is_is=std,
            )
            for name, fid, std in [
                ("PE(36:2)", "1", False),
                ("PE(36:2)", "2", False),
                ("PE(38:4)", "3", False),
                ("PC(36:2)", "1", False),
                ("PC(38:4)", "8", False),
                ("PE(40:6)", "9", False),
                ("PE(36:2)", "7", True),
            ]
        ]
    )
    manual = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                species=name,
                adduct="[M+H]+",
                feature_id=fid,
                lipid_class="PE",
                internal_standard=False,
                mapped=mapped,
            )
            for name, fid, mapped in [
                ("PE(36:2)", "1", True),
                ("PE(38:4)", "3", True),
                ("PE(40:6)", "", False),
            ]
        ]
    )
    accepted = pool.iloc[[0]].copy()
    pool.index = [4, 8, 12, 16, 20, 24, 28]
    got = option_categories(pool, manual, accepted)
    assert got.index.equals(pool.index)
    assert got.diagnostic_category.tolist() == [
        "manual_correct_current",
        "different_peak_for_manually_curated_identity",
        "manual_correct_currently_missing",
        "alternative_label_on_manual_peak_unconfirmed",
        "manual_absent_identity_unconfirmed",
        "unresolved_manual_identity_unknown",
        "internal_standard_excluded",
    ]
    assert not {"training_positive", "training_label_known"} & set(got)


@pytest.mark.parametrize(
    "n,missing,groups", [(12, False, 1), (12, True, 2), (8, False, 8), (7, False, 1)]
)
def test_independent_comparisons_reconstruct_all_statistics(n, missing, groups):
    from matlab_annotation.raw_counterpart_evidence import _profile_result

    rng = np.random.default_rng(21025)
    a, b, ca, cb = [rng.normal(size=n) for _ in range(4)]
    if missing:
        a[0], cb[3] = np.nan, np.nan
    labels = np.array([str(i % groups) for i in range(n)])
    old, new = _profile_result(a, b, ca, cb, labels), comparison(a, b, ca, cb, labels)
    if old is None:
        assert new is None
    else:
        assert old.keys() == new.keys()
        np.testing.assert_allclose(
            list(old.values()),
            [new[k] for k in old],
            rtol=1e-8,
            atol=1e-8,
            equal_nan=True,
        )


def test_independent_profiles_reconstruct_biological_technical_replicates_only():
    from matlab_annotation.paired_sample_evidence import _profiles

    frame = pd.DataFrame(
        dict(
            feature_id=["a", "b"],
            bio1=[2, 0],
            bio1_repeat=[6, 4],
            bio2=[8, 16],
            qc=[999, 999],
            blank=[999, 999],
        )
    )
    frame.attrs = dict(
        sample_columns=["bio1", "bio1_repeat", "bio2", "qc", "blank"],
        sample_group_mapping_status="validated",
        sample_group_map={c: "Group_1_Tissue" for c in frame if c != "feature_id"},
        sample_roles=dict(
            bio1="biological",
            bio1_repeat="biological",
            bio2="biological",
            qc="qc",
            blank="blank",
        ),
    )
    mapping = dict(bio1="one", bio1_repeat="one", bio2="two", qc="qc", blank="blank")
    expected, centers, labels = _profiles(frame, {"a", "b"}, mapping)
    actual, other_centers, other_labels = profiles(frame, mapping)
    pd.testing.assert_frame_equal(expected, actual)
    assert centers.to_dict() == other_centers and labels == other_labels
    assert actual.loc["a", "one"] == pytest.approx(np.log(4))
    assert actual.loc["b", "one"] == pytest.approx(np.log(4))
