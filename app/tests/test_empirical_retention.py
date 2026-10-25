import copy

import numpy as np
import pandas as pd
import pytest

from matlab_annotation.empirical_retention import (
    fit_empirical_retention,
    predict_empirical_reference,
    predict_aligned_empirical_retention,
)
from matlab_annotation.retention_atlas import (
    fit_retention_atlas,
    predict_retention_atlas,
    predict_aligned_retention_atlas,
)


def fixture(n=6):
    observations = pd.DataFrame(
        [
            {
                "cohort": f"g{g}",
                "stream": "RP_Pos",
                "class": cls,
                "carbon": c,
                "db": d,
                "rt_seconds": base
                + 15 * (c - 30)
                - 10 * d
                + (8 if cls == "PE" and c == 34 and d == 2 else 0),
            }
            for g in range(n)
            for cls, base in [("PC", 200), ("PE", 300), ("PS", 400)]
            for c in (28, 30, 32, 34, 36, 38)
            for d in (0, 2, 4)
        ]
    )
    groups = {f"g{g}" for g in range(n)}
    atlas = fit_retention_atlas(observations, minimum_cohorts=2)
    model = fit_empirical_retention(
        observations, atlas, training_groups=groups, excluded_groups={"query"}
    )
    targets = pd.DataFrame(
        {
            "stream": ["RP_Pos"] * 3,
            "class": ["PE", "PE-O", "PE"],
            "carbon": [34, 34, 35],
            "db": [2, 2, 2],
        },
        index=[3, 7, 9],
    )
    return observations, groups, atlas, model, targets


def test_other_project_exact_composition_reference_changes_only_supported_predictions():
    observations, groups, atlas, model, targets = fixture()
    original = copy.deepcopy((observations, atlas, model, targets))
    prediction = predict_empirical_reference(targets, atlas, model, query_group="query")
    item = model["models"]["RP_Pos|PE|34|2"]
    assert item["training_projects"] == 6
    assert item["offset_seconds"] == pytest.approx(
        item["unshrunk_offset_seconds"] * 6 / 9
    )
    old = predict_retention_atlas(targets, atlas)
    assert prediction.loc[3, "atlas_rt_seconds"] == pytest.approx(
        old.loc[3, "atlas_rt_seconds"] + item["offset_seconds"]
    )
    assert abs(prediction.loc[3, "atlas_rt_seconds"] - 348) < abs(
        old.loc[3, "atlas_rt_seconds"] - 348
    )
    assert prediction.loc[[7, 9], "empirical_offset_seconds"].isna().all()
    assert prediction.loc[9, "atlas_rt_seconds"] == old.loc[9, "atlas_rt_seconds"]
    pd.testing.assert_frame_equal(observations, original[0])
    assert atlas == original[1] and model == original[2]
    pd.testing.assert_frame_equal(targets, original[3])


def test_repeated_adducts_cannot_multiply_training_project_support():
    observations, groups, atlas, model, _ = fixture()
    duplicate = fit_empirical_retention(
        pd.concat([observations] * 3, ignore_index=True),
        atlas,
        training_groups=groups,
        excluded_groups={"query"},
    )
    assert duplicate == model
    _, _, _, small, _ = fixture(2)
    assert small["models"] == {}


def test_query_and_parent_scope_mismatches_are_rejected():
    observations, groups, atlas, model, targets = fixture()
    with pytest.raises(ValueError, match="excluded"):
        predict_empirical_reference(targets, atlas, model, query_group="g0")
    bad = copy.deepcopy(atlas)
    bad["training_scope_sha256"] = "wrong"
    with pytest.raises(ValueError, match="scopes differ"):
        fit_empirical_retention(
            observations, bad, training_groups=groups, excluded_groups={"query"}
        )
    with pytest.raises(ValueError, match="unauthorized"):
        fit_empirical_retention(
            observations.assign(cohort="query"),
            atlas,
            training_groups=groups,
            excluded_groups={"query"},
        )


def test_empty_residual_model_reproduces_entire_old_alignment_without_self_vote():
    observations, _, atlas, model, targets = fixture()
    automatic = (
        observations[observations.cohort.eq("g0")]
        .drop(columns=["cohort", "rt_seconds"])
        .copy()
    )
    rt = observations[observations.cohort.eq("g0")].rt_seconds
    automatic["feat_rt"] = (rt.to_numpy() + 4) / 60
    automatic["feat_intensity"] = 10000
    automatic["ppm_error"] = 0.1
    automatic["post_regression_confidence"] = 0.99
    automatic["is_is"] = False
    empty = dict(model, models={})
    old = predict_aligned_retention_atlas(targets, automatic, atlas, stream="RP_Pos")
    new = predict_aligned_empirical_retention(
        targets, automatic, atlas, empty, stream="RP_Pos", query_group="query"
    )
    pd.testing.assert_frame_equal(old, new[old.columns], check_exact=True)
    # Query-composition anchors cannot influence their own local offset.
    before = predict_aligned_empirical_retention(
        targets, automatic, atlas, model, stream="RP_Pos", query_group="query"
    )
    automatic.loc[
        automatic["class"].eq("PE") & automatic.carbon.eq(34) & automatic.db.eq(2),
        "feat_rt",
    ] += 100
    after = predict_aligned_empirical_retention(
        targets, automatic, atlas, model, stream="RP_Pos", query_group="query"
    )
    assert before.loc[3, "atlas_rt_seconds"] == after.loc[3, "atlas_rt_seconds"]
    assert np.isfinite(before.loc[3, "atlas_rt_seconds"])


def test_curated_anchors_and_supervised_query_flags_rejected():
    _, _, atlas, model, targets = fixture()
    for automatic in (
        pd.DataFrame({"manual_annotation": [True]}),
        pd.DataFrame({"training_positive": [True]}),
    ):
        with pytest.raises(ValueError, match="query|curated"):
            predict_aligned_empirical_retention(
                targets, automatic, atlas, model, stream="RP_Pos", query_group="query"
            )
