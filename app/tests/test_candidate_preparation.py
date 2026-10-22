import pandas as pd
import pytest

from matlab_annotation.candidate_preparation import (
    prepare_candidate_pool,
    enrich_candidate_pools,
)
from matlab_annotation.chemical_targets import FORMAT
from test_class_candidates import inputs


def fixture():
    auto, reference, features, atlas = inputs()
    auto = pd.DataFrame(columns=["name", "adduct", "feat_feature_id", "is_is"])
    fitted = reference.assign(
        feat_feature_id="0017",
        feat_mz=700.0001,
        feat_rt=5.01,
        feat_intensity=1e5,
        ppm_error=0.14,
        rt_error=0.01,
        is_is=False,
    )
    for i, col in enumerate("abcd"):
        features[col] = [10 + i, 20 + i, 30 + i]
    features.attrs = dict(
        sample_columns=list("abcd"),
        sample_group_mapping_status="validated",
        sample_group_map={c: "Group_1_Tissue" for c in "abcd"},
    )
    result = dict(
        streams={
            "RP_Pos": dict(
                annotations=auto,
                reference=reference,
                features=features,
                all_pairs=fitted,
            )
        }
    )
    return result, atlas, dict(format=FORMAT, models={})


def test_combined_preparation_preserves_original_feature_and_is_not_acceptance():
    result, atlas, envelope = fixture()
    original = result["streams"]["RP_Pos"]["features"].copy(deep=True)
    pool = prepare_candidate_pool(result, atlas, envelope)
    assert pool.feat_feature_id.tolist() == ["0017"]
    assert pool.feat_rt.iloc[0] == pytest.approx(5.01)
    assert not pool.automatic_accepted.any()
    pd.testing.assert_frame_equal(original, result["streams"]["RP_Pos"]["features"])


def test_evidence_cache_replay_is_identical_and_never_contains_supervised_identifiers():
    result, atlas, envelope = fixture()
    pool = prepare_candidate_pool(result, atlas, envelope)
    before = pool.copy(deep=True)
    first, cache = enrich_candidate_pools({"a": pool}, result, {"a": {}})
    second, _ = enrich_candidate_pools(
        {"a": pool}, result, {"a": {}}, raw_quality_cache=cache
    )
    pd.testing.assert_frame_equal(first["a"]["evidence"], second["a"]["evidence"])
    assert not {
        "name",
        "feat_feature_id",
        "training_positive",
        "training_label_known",
        "dataset",
        "cohort",
    } & set(first["a"]["evidence"])
    pd.testing.assert_frame_equal(pool, before)


def test_training_labels_or_non_raw_quality_caches_are_rejected():
    result, atlas, envelope = fixture()
    pool = prepare_candidate_pool(result, atlas, envelope)
    with pytest.raises(ValueError, match="Raw quality cache"):
        enrich_candidate_pools(
            {"a": pool},
            result,
            {"a": {}},
            raw_quality_cache={
                ("RP_Pos", 1): pd.DataFrame(
                    {"training_positive": [True]}, index=["0017"]
                )
            },
        )
    result["streams"]["RP_Pos"]["reference"]["training_positive"] = True
    with pytest.raises(ValueError, match="Supervised labels"):
        prepare_candidate_pool(result, atlas, envelope)
