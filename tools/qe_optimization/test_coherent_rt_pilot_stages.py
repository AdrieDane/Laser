"""Synthetic orchestration guards; never read any example or manual dataset."""

import json

import pytest

import diagnose_qe_coherent_rt as pilot


class ManualReadBoundary(Exception):
    pass


def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "OUT", tmp_path)
    monkeypatch.setattr(pilot, "initialize", lambda: ({}, [], {}))
    observed = []
    monkeypatch.setattr(
        pilot, "configure_frozen_resources", lambda _: observed.append("resources")
    )

    def manual_read(_):
        observed.append("manual_read")
        raise ManualReadBoundary

    monkeypatch.setattr(pilot.pd, "read_pickle", manual_read)
    monkeypatch.setattr(pilot, "sha", lambda _: "expected")
    return observed


def test_manual_evaluation_cannot_start_without_all_blind_predictions(
    monkeypatch, tmp_path
):
    observed = setup(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        pilot.evaluate()
    assert observed == []
    (tmp_path / "blind_predictions_complete.json").write_text(
        json.dumps(dict(all_33_complete=False))
    )
    with pytest.raises(AssertionError):
        pilot.evaluate()
    assert observed == []


def test_changed_protocol_or_prediction_blocks_manual_read(monkeypatch, tmp_path):
    observed = setup(monkeypatch, tmp_path)
    for protocol, artifact in (("changed", "expected"), ("expected", "changed")):
        (tmp_path / "blind_predictions_complete.json").write_text(
            json.dumps(
                dict(
                    all_33_complete=True,
                    manual_evaluation_loaded=False,
                    protocol_sha256=protocol,
                    prediction_sha256={"fake_prediction": artifact},
                )
            )
        )
        with pytest.raises(AssertionError):
            pilot.evaluate()
        assert observed == []


def test_manual_read_is_after_frozen_complete_receipt_and_all_hash_checks(
    monkeypatch, tmp_path
):
    observed = setup(monkeypatch, tmp_path)
    (tmp_path / "blind_predictions_complete.json").write_text(
        json.dumps(
            dict(
                all_33_complete=True,
                manual_evaluation_loaded=False,
                protocol_sha256="expected",
                prediction_sha256={"fake_prediction": "expected"},
            )
        )
    )
    # No actual manual data is read: an explicit sentinel stops at that boundary.
    monkeypatch.setattr(pilot, "configure_frozen_resources", lambda _: None)
    # Import only code, not dataset files, as the production stage does.
    from context import REPO
    from frozen_resources import configure_frozen_resources

    configure_frozen_resources(REPO)
    with pytest.raises(ManualReadBoundary):
        pilot.evaluate()
    assert observed == ["manual_read"]
