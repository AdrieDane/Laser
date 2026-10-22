import pytest

from matlab_annotation.nested_feature_folds import plan_nested_feature_folds


def fixture():
    assignments = {f"dataset{i}": i for i in range(5)}
    groups = {f"dataset{i}": f"project{i}" for i in range(5)}
    assignments["related"] = 2
    groups["related"] = "project2"
    return assignments, groups


def test_five_blocks_need25_models_and_four_inner_blocks():
    plan = plan_nested_feature_folds(*fixture())
    assert len(plan["models"]) == 25
    assert all(len(outer["inner"]) == 4 for outer in plan["outer"].values())


def test_every_query_evaluation_and_inner_validation_project_excluded_from_features():
    assignments, groups = fixture()
    plan = plan_nested_feature_folds(assignments, groups)
    for outer in plan["outer"].values():
        test = set(outer["heldout_datasets"])
        for d, mask in outer["training_feature_masks"].items():
            model = plan["models"][mask]
            assert d not in model["training_datasets"]
            assert groups[d] not in model["training_groups"]
            assert not test & set(model["training_datasets"])
        assert not test & set(
            plan["models"][outer["heldout_feature_mask"]]["training_datasets"]
        )
        for inner in outer["inner"].values():
            val = set(inner["validation_datasets"])
            assert not (test | val) & set(inner["score_training_datasets"])
            for d, mask in inner["training_feature_masks"].items():
                model = plan["models"][mask]
                assert not (test | val | {d}) & set(model["training_datasets"])
                assert groups[d] not in model["training_groups"]
            assert not (test | val) & set(
                plan["models"][inner["validation_feature_mask"]]["training_datasets"]
            )


def test_related_acquisitions_cannot_split_and_invalid_metadata_rejected():
    assignments, groups = fixture()
    assignments["related"] = 1
    with pytest.raises(ValueError, match="Related acquisitions"):
        plan_nested_feature_folds(assignments, groups)
    with pytest.raises(ValueError, match="Every dataset"):
        plan_nested_feature_folds(assignments, {})
    with pytest.raises(ValueError, match="integers"):
        plan_nested_feature_folds({"a": True}, {"a": "group"})


def test_input_order_invariance_and_no_mutation():
    assignments, groups = fixture()
    original = assignments.copy()
    a = plan_nested_feature_folds(assignments, groups)
    b = plan_nested_feature_folds(dict(reversed(list(assignments.items()))), groups)
    assert a == b
    assert assignments == original
