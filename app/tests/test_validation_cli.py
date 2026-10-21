from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[2] / "tools/compare_manual_annotations.py"


def inputs(tmp_path):
    run = tmp_path / "runs" / "example"
    run.mkdir(parents=True)
    truth = tmp_path / "curation" / "example"
    truth.mkdir(parents=True)
    pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                name="PE(36:2)",
                **{"class": "PE"},
                adduct="[M+H]+",
                feat_feature_id="0017",
            )
        ]
    ).to_csv(run / "final_automatic_annotations.csv", index=False)
    pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                matlab_species="PE(36:2)",
                matlab_class="PE",
                matlab_adduct="[M+H]+",
                xcms_feature_id="0017",
                crosswalk_status="profile_match",
            )
        ]
    ).to_csv(truth / "curated_feature_crosswalk.csv", index=False)
    return [
        sys.executable,
        str(SCRIPT),
        "--runs-root",
        str(run.parent),
        "--curation-root",
        str(truth.parent),
        "--output",
        str(tmp_path / "report"),
    ]


def test_standalone_csv_only_cli_preserves_ids_and_does_not_touch_inputs(tmp_path):
    args = inputs(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.csv")}
    result = subprocess.run(args, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "report/validation_manifest.json").read_text())
    assert manifest["complete"] and manifest["selection_source"] == "annotations_CSV"
    assert manifest["annotation_revision"] is None
    details = pd.read_csv(
        tmp_path / "report/manual_comparison.csv", dtype={"feature_id": str}
    )
    assert details.feature_id.tolist() == ["0017"]
    assert (
        details.same_manual_peak.all()
        and details.candidate_evidence.iloc[0] == "diagnostics_not_supplied"
    )
    for path, contents in before.items():
        assert path.read_bytes() == contents
    second = subprocess.run(args, text=True, capture_output=True)
    assert second.returncode != 0 and "new or empty" in second.stderr


def test_pickle_requires_explicit_trust_before_any_read(tmp_path):
    result = subprocess.run(
        [*inputs(tmp_path), "--result-name", "untrusted.pkl"],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0 and "requires --trust-local-pickles" in result.stderr
    assert not (tmp_path / "report").exists()


def test_unknown_pooled_exclusion_fails_instead_of_silent_misspelling(tmp_path):
    result = subprocess.run(
        [*inputs(tmp_path), "--exclude-from-pooled", "misspelled"],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0 and "must name a selected dataset" in result.stderr


def test_empty_pool_still_writes_report(tmp_path):
    result = subprocess.run(
        [*inputs(tmp_path), "--exclude-from-pooled", "example"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "report/validation_manifest.json").read_text())
    assert manifest["excluded_from_pooled"] == ["example"] and manifest["complete"]
    pooled = pd.read_csv(tmp_path / "report/pooled_by_stream.csv")
    assert pooled.empty and "manual_same_peak" in pooled
