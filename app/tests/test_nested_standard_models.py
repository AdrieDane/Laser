import copy

import pandas as pd
import pytest

from matlab_annotation.nested_reference_models import fit_nested_reference_bundle
from matlab_annotation.nested_standard_models import fit_nested_standard_bundle
from test_nested_reference_models import fixture as endogenous_fixture


def fixture():
    endogenous, folds, groups = endogenous_fixture()
    standard_rows, channel_rows, automatic, declared = [], [], {}, {}
    for dataset in folds:
        automatic[dataset] = {
            "RP_Pos": pd.DataFrame(
                columns=[
                    "class",
                    "carbon",
                    "db",
                    "feat_rt",
                    "ppm_error",
                    "post_regression_confidence",
                ]
            )
        }
        declared[dataset] = pd.DataFrame(
            [
                dict(
                    stream="RP_Pos",
                    name="PE(34:0) (IS)",
                    **{"class": "PE"},
                    standard_compound_id="PE34|chemicalmass",
                    carbon=34,
                    db=0,
                    ref_mz=700.0,
                    adduct="[M+H]+",
                )
            ]
        )
        standard_rows.append(
            dict(
                dataset=dataset,
                cohort=groups[dataset],
                stream="RP_Pos",
                mapped=True,
                standard_compound_id="PE34|chemicalmass",
                adduct="[M+H]+",
                rt_seconds=301.0,
                reporting_name="PE(34:0) (IS)",
                reporting_class="PE",
            )
        )
        channel_rows.append(
            dict(
                dataset=dataset,
                cohort=groups[dataset],
                stream="RP_Pos",
                mapped=True,
                name="PE(34:0) (IS)",
                adduct="[M+H]+",
            )
        )
    references = fit_nested_reference_bundle(endogenous, folds, groups, (0, 1))
    return (
        pd.DataFrame(standard_rows),
        pd.DataFrame(channel_rows),
        automatic,
        declared,
        references,
        folds,
        groups,
    )


def test_query_outer_and_inner_standard_truth_and_automatic_cannot_change_models():
    obs, channels, automatic, declared, refs, folds, groups = fixture()
    expected = fit_nested_standard_bundle(
        obs, channels, automatic, declared, refs, folds, groups, (0, 1)
    )
    changed = obs.copy()
    excluded = changed.dataset.map(folds).isin([0, 1])
    changed.loc[excluded, "rt_seconds"] += 500
    changed.loc[excluded, "reporting_name"] = "wrong positional isomer"
    other = channels.copy()
    other.loc[other.dataset.map(folds).isin([0, 1]), "adduct"] = "wrong channel"
    auto = copy.deepcopy(automatic)
    for d in folds:
        if folds[d] in {0, 1}:
            auto[d] = {"malformed_excluded": object()}
    actual = fit_nested_standard_bundle(
        changed, other, auto, declared, refs, folds, groups, (0, 1)
    )
    assert actual == expected
    assert len(actual["training_groups"]) == 9
    assert actual["library"]["models"] and actual["transfer"]["models"]


def test_bad_group_scope_and_mapping_metadata_are_rejected():
    obs, channels, auto, declared, refs, folds, groups = fixture()
    with pytest.raises(ValueError, match="cohort metadata"):
        fit_nested_standard_bundle(
            obs.assign(cohort="wrong"),
            channels,
            auto,
            declared,
            refs,
            folds,
            groups,
            (0, 1),
        )
    with pytest.raises(ValueError, match="scopes"):
        fit_nested_standard_bundle(
            obs, channels, auto, declared, refs, folds, groups, (0,)
        )
    with pytest.raises(ValueError, match="Unassigned"):
        fit_nested_standard_bundle(
            obs.assign(dataset="unknown"),
            channels,
            auto,
            declared,
            refs,
            folds,
            groups,
            (0, 1),
        )


def test_feature_ids_and_baseline_comparison_flags_do_not_enter_any_fit():
    obs, channels, auto, declared, refs, folds, groups = fixture()
    expected = fit_nested_standard_bundle(
        obs, channels, auto, declared, refs, folds, groups, (0, 1)
    )
    actual = fit_nested_standard_bundle(
        obs.assign(original_feature_id="never_fit", before_same_peak=False),
        channels.assign(original_feature_id="never_fit"),
        auto,
        declared,
        refs,
        folds,
        groups,
        (0, 1),
    )
    assert actual == expected
