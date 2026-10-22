import pandas as pd
import pytest

from matlab_annotation.nested_reference_models import fit_nested_reference_bundle


def fixture():
    folds, groups, rows = {}, {}, []
    for fold in range(5):
        for project in range(3):
            name = f"dataset_{fold}_{project}"
            folds[name] = fold
            groups[name] = name
            for carbon in (34, 36, 38):
                for db in (0, 1, 2):
                    rows.append(
                        dict(
                            dataset=name,
                            cohort=name,
                            stream="RP_Pos",
                            **{"class": "PE"},
                            carbon=carbon,
                            db=db,
                            rt_seconds=300 + 10 * (carbon - 34) - 20 * db,
                            mapped=True,
                        )
                    )
    return pd.DataFrame(rows), folds, groups


def test_query_and_validation_observations_cannot_change_any_fitted_reference():
    data, folds, groups = fixture()
    expected = fit_nested_reference_bundle(data, folds, groups, (0, 1))
    changed = data.copy()
    excluded = changed.dataset.map(folds).isin([0, 1])
    changed.loc[excluded, "rt_seconds"] += 500
    changed.loc[excluded, "carbon"] += 20
    changed.loc[excluded, "class"] = "PC"
    actual = fit_nested_reference_bundle(changed, folds, groups, (0, 1))
    assert actual == expected
    assert len(actual["training_groups"]) == 9
    assert not set(actual["training_groups"]) & set(actual["excluded_groups"])


def test_bad_cohort_mapping_unassigned_dataset_and_unknown_mask_rejected():
    data, folds, groups = fixture()
    with pytest.raises(ValueError, match="cohort metadata"):
        fit_nested_reference_bundle(data.assign(cohort="wrong"), folds, groups, (0,))
    with pytest.raises(ValueError, match="unassigned"):
        fit_nested_reference_bundle(data.assign(dataset="unknown"), folds, groups, (0,))
    with pytest.raises(ValueError, match="mask"):
        fit_nested_reference_bundle(data, folds, groups, (0, 0))


def test_original_feature_ids_are_not_exported_or_used_by_fitter():
    data, folds, groups = fixture()
    expected = fit_nested_reference_bundle(data, folds, groups, (0, 1, 2))
    actual = fit_nested_reference_bundle(
        data.assign(original_feature_id="irrelevant"), folds, groups, (0, 1, 2)
    )
    assert actual == expected
    assert len(actual["training_groups"]) == 6
