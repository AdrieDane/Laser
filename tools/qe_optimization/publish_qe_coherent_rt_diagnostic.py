"""Publish ONLY all33 aggregate numerical RT metrics after full replay passes."""

import json
import argparse
from pathlib import Path

import pandas as pd

from context import REPO
from run_qe_standard_mode_experiment import sha, dump
import diagnose_qe_coherent_rt as pilot

DESTINATION = REPO / "docs/validation/qe_20261025_coherent_RT"
ALLOWED = {
    "dataset",
    "stream",
    "class",
    "assessed_original_RTs",
    "blind_lookup_available",
    "coherent_RT_available",
    "within_10_seconds_before",
    "within_10_seconds_after",
    "newly_within_10_seconds",
    "no_longer_within_10_seconds",
    "median_absolute_RT_error_before",
    "median_absolute_RT_error_after",
}


def main():
    protocol, _, _ = pilot.initialize()
    source = pilot.OUT
    summary = json.loads((source / "summary.json").read_text())
    integrity = json.loads((source / "integrity_receipt.json").read_text())
    assert (
        integrity["all_passed"]
        and integrity["full_blind_stream_prediction_reproductions"] == 132
    )
    assert integrity["freshly_reproduced_independent_standard_inputs"] == 33
    assert summary["datasets"] == 33 and summary["dataset_modes"] == 132
    assert not summary["annotation_changes"] and summary["classifier_fits"] == 0
    assert integrity["protocol_sha256"] == sha(source / "protocol.json")
    assert (
        summary["blind_all33_receipt_sha256"]
        == integrity["blind_all33_receipt_sha256"]
        == sha(source / "blind_predictions_complete.json")
    )
    assert summary["evaluated_manual_observations_sha256"] == sha(pilot.OBS)
    artifacts = json.loads((source / "blind_predictions_complete.json").read_text())[
        "prediction_sha256"
    ]
    for p, h in artifacts.items():
        assert sha(Path(p)) == h
    assert (DESTINATION / "interpretation.txt").is_file(), (
        "Write the actual interpretation/decision first"
    )
    tables = {}
    for name in (
        "by_mode",
        "by_mode_class",
        "by_dataset_mode",
        "by_dataset_mode_class",
    ):
        table = pd.read_csv(source / (name + ".csv"))
        assert set(table).issubset(ALLOWED)
        if name == "by_dataset_mode":
            assert len(table) == 132 and table.dataset.nunique() == 33
        assert (
            int(table.assessed_original_RTs.sum()) == summary["assessed_original_RTs"]
        )
        assert (
            int(table.within_10_seconds_before.sum())
            == summary["within_10_seconds_before"]
        )
        assert (
            int(table.within_10_seconds_after.sum())
            == summary["within_10_seconds_after"]
        )
        tables[name] = table
    for name, table in tables.items():
        table.to_csv(DESTINATION / (name + ".csv"), index=False)
    dump(DESTINATION / "summary.json", summary)
    dump(DESTINATION / "integrity_receipt.json", integrity)
    dump(
        DESTINATION / "protocol_summary.json",
        {
            k: protocol[k]
            for k in (
                "purpose",
                "policy",
                "stages",
                "supersedes",
                "annotation_changes",
                "classifier_fits",
                "numerical_reference_fits",
                "reused_numerical_reference_bundles",
                "datasets",
                "modes",
                "groups",
            )
        },
    )
    print("PUBLISHED AGGREGATE RT DIAGNOSTIC ONLY", DESTINATION)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--competing-class", action="store_true")
    args = parser.parse_args()
    if args.competing_class:
        import diagnose_qe_competing_class_rt as pilot

        DESTINATION = REPO / "docs/validation/qe_20261025_competing_class_RT"
    main()
