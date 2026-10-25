"""Synthetic score-free ceiling masks: occupancy, unknown and nested gates."""

import numpy as np
import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
from bmp_context_measurements import PREFIX
from diagnose_qe_bmp_admission_ceiling import freeze_masks


def fixture():
    pool = pd.DataFrame(
        [
            dict(
                stream="NP_Neg",
                name=f"BMP({32 + 2 * i}:2)",
                adduct="[M-H]-",
                feat_feature_id=f"n{i}",
                is_is=False,
                **{"class": "BMP"},
            )
            for i in range(7)
        ]
    )
    evidence = pd.DataFrame(
        {
            PREFIX + k: [v] * 7
            for k, v in dict(
                source_physical=1,
                candidate_counterparts=1,
                counterpart_profiles=1,
                counterpart_n=8,
                counterpart_raw=0.8,
                counterpart_normalized=0.8,
                counterpart_leave_one_out=0.6,
            ).items()
        }
    )
    evidence.loc[1, PREFIX + "counterpart_raw"] = np.nan
    evidence.loc[2, PREFIX + "counterpart_n"] = 7
    evidence.loc[3, PREFIX + "candidate_counterparts"] = 0
    evidence.loc[4, PREFIX + "source_physical"] = 0
    return dict(pool=pool, evidence=evidence)


def test_nested_masks_and_both_identity_and_feature_occupancy():
    raw = fixture()
    old = raw["pool"].iloc[[5, 6]].copy()
    old.loc[5, "feat_feature_id"] = "different_peak"
    old.loc[6, "name"] = "PA(44:2)"
    old.loc[6, "class"] = "PA"
    masks = freeze_masks(raw, old)["masks"]
    assert masks.physical.tolist() == [True, True, True, True, False, False, False]
    assert masks.counterpart.tolist() == [True, True, True, False, False, False, False]
    assert masks.measurable_profile.tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert masks.strong_profile.tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_supervision_and_misalignment_refused():
    raw = fixture()
    old = raw["pool"].iloc[:0].copy()
    raw["pool"]["training_positive"] = False
    with pytest.raises(ValueError):
        freeze_masks(raw, old)
    raw = fixture()
    raw["evidence"] = raw["evidence"].iloc[::-1]
    with pytest.raises(ValueError):
        freeze_masks(raw, old)
