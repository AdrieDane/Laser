"""Synthetic orchestration/quantification guards only, no example classifier fit."""

import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources
from test_np_context_bmp_experiment import inputs
import run_qe_feature_label_experiment as runner
from verify_immutable_exports import check_export
from immutable_exports import assemble_immutable

configure_frozen_resources(REPO)


def test_runner_refuses_no_prefit_before_any_output(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PREFLIGHT", tmp_path / "missing")
    monkeypatch.setattr(runner, "OUT", tmp_path / "outputs")
    with pytest.raises(FileNotFoundError):
        runner.initialize()
    assert not runner.OUT.exists()


def test_query_labels_cannot_be_opened_before_scope_authorization(monkeypatch):
    reads = []
    monkeypatch.setattr(
        runner.inherited.common.pd, "read_pickle", lambda p: reads.append(p)
    )
    with pytest.raises(ValueError):
        runner.authorize_labels(dict(query_group="test"), {"train"}, {"test"})
    assert reads == []


def test_raw_export_check_validates_every_new_original_field_and_catches_changed_area(
    tmp_path, monkeypatch
):
    _, candidate, added, features = inputs(tmp_path, monkeypatch)
    filled = runner.inherited.common.fill_original_features("example", added)
    candidate["automatic_accepted"] = False
    candidate["automatic_candidate"] = True
    previous = filled.iloc[:0].astype(str)
    annotations, actual = assemble_immutable(
        previous, candidate, filled, method="factorized"
    )
    path = tmp_path / "annotations.csv"
    annotations.to_csv(path, index=False)
    baseline = dict(streams={"NP_Pos": dict(features=features)})
    receipt = check_export(path, previous, actual, candidate, baseline)
    assert (
        receipt["raw_stream_checks"] == 1
        and receipt["inherited_empty_old_measurement_cells"] == 0
    )
    assert receipt["available_measurement_cells"] >= 10
    broken = pd.read_csv(path)
    broken["feat_sample_A"] = 0
    broken.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        check_export(path, previous, actual, candidate, baseline)
