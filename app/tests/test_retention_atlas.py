"""Synthetic tests only; no real benchmark labels or feature IDs in runtime."""
import json

import numpy as np
import pandas as pd
import pytest

from matlab_annotation.retention_atlas import (
    align_retention_atlas, fit_retention_atlas, predict_retention_atlas,
    predict_aligned_retention_atlas,
)


def training():
    rows = []
    for cohort in range(5):
        for stream in ("RP_Pos", "NP_Neg"):
            for cls, offset in (("PC", 0), ("PE", 130), ("SM[d]", 300)):
                for c in range(30, 43, 2):
                    for db in range(4):
                        rt = 200 + offset + 8*(c-30) - 12*db + .2*(c-30)**2
                        rows.append(dict(cohort=f"training_{cohort}", stream=stream,
                                         **{"class": cls}, carbon=c, db=db,
                                         rt_seconds=rt + 2*(cohort-2)))
    return pd.DataFrame(rows)


def test_polynomial_reference_and_portable_json():
    data = training()
    atlas = json.loads(json.dumps(fit_retention_atlas(data)))
    targets = data[data.cohort.eq("training_2")].drop(columns=["cohort", "rt_seconds"])
    predictions = predict_retention_atlas(targets, atlas)
    expected = data.loc[targets.index, "rt_seconds"]
    assert np.median(np.abs(predictions.atlas_rt_seconds - expected)) < 1
    assert predictions.atlas_in_domain.all()
    assert not any("training_0" in str(value) for value in atlas.values())


def test_unknown_and_far_outside_domain_are_unavailable():
    atlas = fit_retention_atlas(training())
    targets = pd.DataFrame({"stream": ["RP_Pos"]*3, "class": ["unknown", "PC", "PC"],
                            "carbon": [36, 200, np.nan], "db": [2, 2, 2]})
    result = predict_retention_atlas(targets, atlas)
    assert result.atlas_rt_seconds.isna().all()
    assert not result.atlas_in_domain.any()


def test_reference_needs_multiple_cohorts_and_valid_units():
    data = training()
    atlas = fit_retention_atlas(data[data.cohort.eq("training_0")])
    assert not atlas["models"]
    with pytest.raises(ValueError, match="Invalid"):
        fit_retention_atlas(data.assign(rt_seconds=-1))
    with pytest.raises(ValueError, match="format or RT unit"):
        predict_retention_atlas(data, {**fit_retention_atlas(data), "rt_unit": "minutes"})


def test_aliases_and_input_preservation():
    data = training().replace({"class": {"SM[d]": "SM-d"}})
    before = data.copy(deep=True)
    atlas = fit_retention_atlas(data)
    pd.testing.assert_frame_equal(before, data)
    targets = data.iloc[:1].assign(**{"class": "SM[d]"})
    assert predict_retention_atlas(targets, atlas).atlas_rt_seconds.notna().all()


def test_alignment_uses_only_strong_other_class_automatic_anchors():
    data = training()
    atlas = fit_retention_atlas(data)
    anchors = data[data.cohort.eq("training_2") & data.stream.eq("RP_Pos")].copy()
    anchors["feat_rt"] = (anchors.rt_seconds*1.02 + 8)/60
    anchors["post_regression_confidence"] = .99
    anchors["ppm_error"] = .2
    anchors["is_is"] = False
    result = align_retention_atlas(anchors, atlas, stream="RP_Pos")
    assert result["status"] == "automatic_anchor_alignment"
    assert abs(result["slope"] - 1.02) < .01
    assert result["median_absolute_after_seconds"] < 1
    result = align_retention_atlas(anchors, atlas, stream="RP_Pos", exclude_class="PE")
    assert result["status"] == "insufficient_automatic_anchors"
    assert result["anchor_classes"] == 2
    weak = align_retention_atlas(anchors.assign(post_regression_confidence=np.nan), atlas, stream="RP_Pos")
    assert weak["anchor_count"] == 0


def test_outlier_resistance():
    data = training()
    data.loc[data.index[::40], "rt_seconds"] += 1000
    atlas = fit_retention_atlas(data)
    clean = training()
    target = clean[clean.cohort.eq("training_2")]
    predicted = predict_retention_atlas(target, atlas).atlas_rt_seconds
    assert np.median(np.abs(predicted-target.rt_seconds)) < 3


def test_class_local_alignment_excludes_target_composition():
    data = training()
    atlas = fit_retention_atlas(data)
    local = data[data.cohort.eq("training_2") & data.stream.eq("RP_Pos") & data["class"].eq("PC")].copy()
    local["feat_rt"] = (local.rt_seconds + 25)/60
    local["post_regression_confidence"] = .99
    local["ppm_error"] = .2
    local["is_is"] = False
    # Four anchors include the target itself: only three remain, so no correction.
    anchors = local.head(4)
    target = local.head(1)
    result = predict_aligned_retention_atlas(target, anchors, atlas, stream="RP_Pos")
    assert result.atlas_local_support_count.iloc[0] == 0
    # A fifth distinct composition makes four independent local anchors available.
    anchors = local.head(5)
    result = predict_aligned_retention_atlas(target, anchors, atlas, stream="RP_Pos")
    assert result.atlas_local_support_count.iloc[0] == 4
    assert abs(result.atlas_local_shift_seconds.iloc[0] - 25) < 1
    # Repeating adduct rows must not turn four compositions into five.
    duplicate = pd.concat([local.head(4), local.head(4)], ignore_index=True)
    result = predict_aligned_retention_atlas(target, duplicate, atlas, stream="RP_Pos")
    assert result.atlas_local_support_count.iloc[0] == 0
