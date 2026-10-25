import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources
from immutable_exports import assemble_immutable
from test_np_context_bmp_experiment import inputs
import run_qe_np_context_bmp_experiment as old

configure_frozen_resources(REPO)


def test_disabled_explicit_checkpoint_is_exact_and_not_implicitly029(
    tmp_path, monkeypatch
):
    previous, candidate, added, _ = inputs(tmp_path, monkeypatch)
    previous["earlier_stage_note"] = "retained041"
    candidate["earlier_stage_note"] = "retained041"
    annotations, actual = assemble_immutable(
        previous, candidate, added.iloc[:0], method="new"
    )
    pd.testing.assert_frame_equal(annotations, previous, check_exact=True)
    pd.testing.assert_frame_equal(actual, candidate, check_exact=True)


def test_complete_original_measurements_and_all_old_cells_survive_new_provenance(
    tmp_path, monkeypatch
):
    previous, candidate, added, features = inputs(tmp_path, monkeypatch)
    candidate.loc[0, "automatic_candidate"] = True
    filled = old.fill_original_features("example", added)
    filled["auto_annotation_status"] = "factorized"
    annotations, actual = assemble_immutable(
        previous, candidate, filled, method="factorized"
    )
    assert len(annotations) == 2
    assert actual.loc[0, "automatic_candidate"] is True
    chosen = actual.loc[actual.feat_feature_id.eq("other")].iloc[0]
    assert chosen.feat_sample_A == features.sample_A.iloc[0]
    assert chosen.auto_annotation_status == "factorized"
    assert chosen.previous_041_class_selection_score == 0.9
    assert not chosen.automatic_candidate
    with pytest.raises(AssertionError):
        assemble_immutable(
            previous,
            candidate,
            filled.assign(feat_feature_id="best"),
            method="factorized",
        )
