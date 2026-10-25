"""Synthetic export/authorization tests ONLY;never open an example dataset."""

import sys

import numpy as np
import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources
import run_qe_np_context_bmp_experiment as runner

configure_frozen_resources(REPO)
sys.path.insert(0, str(REPO / "app/tests"))
from test_bmp_rescue import fixture  # noqa:E402


def inputs(tmp_path, monkeypatch):
    pool, scores, _ = fixture()
    pool = pool.assign(
        ref_mz=[700.0, 700.0, 800.0],
        feat_mz=[700.0001, 700.0001, 800.0001],
        feat_rt=5.0,
        ref_rt=5.0,
        class_selection_score=scores,
        feat_sample_A=[1.25, np.nan, np.nan],
        feat_mobility_unit=["NA", "", ""],
        automatic_accepted=[True, False, False],
        automatic_candidate=[False, True, True],
        auto_annotation_status=["old_annotation", "candidate", "candidate"],
    )
    previous = pool.iloc[[0]].astype(str).copy()
    previous["empty_note"] = ""
    old = tmp_path / "old/runs/example"
    old.mkdir(parents=True)
    previous.to_csv(old / "automatic_annotations.csv", index=False)
    pool.to_pickle(old / "candidates.pkl")
    monkeypatch.setattr(runner, "PREVIOUS", tmp_path / "old")
    base = tmp_path / "base/runs/example"
    base.mkdir(parents=True)
    features = pd.DataFrame(
        dict(
            feature_id=["other"],
            mz=[800.0001],
            rt=[5.0],
            intensity=[20.0],
            sample_A=[123.4],
            mzmin=[800.0],
            mzmax=[800.0002],
            rtmin=[4.9],
            rtmax=[5.1],
            mobility=[np.nan],
            mobility_unit=[""],
            quantification_measure=["original peak area"],
        )
    )
    pd.to_pickle(
        dict(streams={"NP_Pos": dict(features=features)}), base / "full_annotation.pkl"
    )
    monkeypatch.setattr(runner, "BASE", tmp_path / "base")
    added = pool.iloc[[2]].copy()
    added["class_selection_score"] = 0.97
    return previous, pool, added, features


def test_disabled_assembly_preserves_entire_old_export_and_candidate_frame(
    tmp_path, monkeypatch
):
    previous, pool, added, _ = inputs(tmp_path, monkeypatch)
    read = runner.read_previous("example")
    pd.testing.assert_frame_equal(read, previous)
    empty = added.iloc[:0].assign(new_unused_column=1.0)
    annotations, candidate = runner.assemble_exports("example", read, empty)
    pd.testing.assert_frame_equal(annotations, previous, check_exact=True)
    pd.testing.assert_frame_equal(candidate, pool, check_exact=True)


def test_added_annotation_and_promoted_candidate_use_full_original_quantification(
    tmp_path, monkeypatch
):
    previous, pool, added, features = inputs(tmp_path, monkeypatch)
    filled = runner.fill_original_features("example", added)
    for c in features:
        assert filled["feat_" + c].iloc[0] == features[c].iloc[0] or (
            pd.isna(filled["feat_" + c].iloc[0]) and pd.isna(features[c].iloc[0])
        )
    annotations, candidate = runner.assemble_exports("example", previous, filled)
    pd.testing.assert_frame_equal(
        annotations.iloc[:1][list(previous)],
        previous,
        check_dtype=False,
        check_exact=True,
    )
    promoted = candidate.loc[candidate.feat_feature_id.eq("other")].iloc[0]
    assert promoted.feat_sample_A == 123.4 and promoted.feat_rtmin == 4.9
    assert promoted.class_selection_score == promoted.context_completion_score == 0.97
    assert promoted.previous_029_class_selection_score == 0.9
    assert set(candidate.loc[candidate.automatic_accepted, "feat_feature_id"]) == {
        "best",
        "other",
    }
    assert set(runner.accepted_keys(annotations)) == set(
        runner.accepted_keys(candidate.loc[candidate.automatic_accepted])
    )
    assert (
        previous.feat_mobility_unit.iloc[0] == "NA"
        and previous.empty_note.iloc[0] == ""
    )


def test_identity_alias_or_original_feature_cannot_replace_any_old_call(
    tmp_path, monkeypatch
):
    previous, _, added, _ = inputs(tmp_path, monkeypatch)
    for replacement in (
        added.assign(name=" BMP(36:2) "),
        added.assign(feat_feature_id="best"),
    ):
        with pytest.raises(AssertionError):
            runner.assemble_exports("example", previous, replacement)


def test_runner_requires_preflight_and_authorization_before_any_truth_or_fit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "QE", tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.initialize()
    accesses = []
    monkeypatch.setattr(runner.pd, "read_pickle", lambda p: accesses.append(p))
    with pytest.raises(AssertionError):
        runner.label_allowed_query(
            dict(query_group="heldout"), {"training"}, {"heldout"}, partial=True
        )
    assert accesses == []


def test_descriptive_classifier_kind_is_not_an_estimator_keyword_or_parameter_change():
    from sklearn.ensemble import HistGradientBoostingClassifier

    config = dict(
        kind="HistGradientBoostingClassifier",
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=2,
        learning_rate=0.08,
        early_stopping=False,
        random_state=20261022,
    )
    before = config.copy()
    params = runner.estimator_parameters(config)
    assert config == before and params == {
        k: v for k, v in config.items() if k != "kind"
    }
    constructor = HistGradientBoostingClassifier(**params)
    for k, v in params.items():
        assert constructor.get_params()[k] == v
    with pytest.raises(ValueError):
        runner.estimator_parameters(dict(kind="unapproved_model", max_iter=180))


def test_audit_preserves_legacy_candidate_flags_but_forbids_them_on_new_acceptances(
    tmp_path, monkeypatch
):
    from verify_qe_np_context_bmp_experiment import verify_candidate_flags

    previous, pool, added, _ = inputs(tmp_path, monkeypatch)
    pool.loc[0, "automatic_candidate"] = True
    old = runner.PREVIOUS / "runs/example/candidates.pkl"
    pool.to_pickle(old)
    annotations, candidate = runner.assemble_exports(
        "example", previous, runner.fill_original_features("example", added)
    )
    assert len(annotations) == 2
    assert verify_candidate_flags(candidate, pool) == 1
    bad = candidate.copy()
    bad.loc[bad.feat_feature_id.eq("other"), "automatic_candidate"] = True
    with pytest.raises(AssertionError):
        verify_candidate_flags(bad, pool)
    bad = candidate.copy()
    bad.loc[bad.feat_feature_id.eq("best"), "automatic_candidate"] = False
    with pytest.raises(AssertionError):
        verify_candidate_flags(bad, pool)
