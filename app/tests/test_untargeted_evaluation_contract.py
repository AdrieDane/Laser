"""Synthetic tests for denominator and frozen-snapshot integrity."""
import json
from pathlib import Path
import runpy

import pandas as pd
import pytest

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[2] / "tools/freeze_untargeted_baseline.py"))


def reference(rows):
    base = dict(stream="RP_Pos", feature_id="1", mapped=True, name="synthetic",
                lipid_class="PC", adduct="[M+H]+", formula="C10H22O2",
                source_sheet="test", source_excel_row="1")
    return TOOL["reference_ledger"](pd.DataFrame([{**base, **r} for r in rows]))


def test_all_manual_rows_preserved_but_ion_queries_deduplicated():
    ledger = reference([{}, {"source_excel_row": "2"},
                        {"source_excel_row": "3", "feature_id": "", "mapped": False},
                        {"source_excel_row": "4", "feature_id": "2", "formula": ""}])
    assert len(ledger) == 4  # Unmapped and missing-formula rows stay visible.
    assert ledger.ion_metric_eligible.sum() == 2
    assert ledger.mapping_reason.value_counts()["unresolved_original_feature"] == 1
    assert ledger.formula_reason.value_counts()["reference_formula_missing"] == 1
    assert set(ledger.split) == {"development"}


def test_conflicting_gold_is_not_scored_as_verified():
    ledger = reference([{}, {"source_excel_row": "2", "adduct": "[M+Na]+"}])
    assert ledger.ambiguous_reference.all()
    assert not ledger.ion_metric_eligible.any()


def test_duplicate_source_row_is_rejected_instead_of_silently_dropped():
    with pytest.raises(ValueError, match="Repeated source worksheet row"):
        reference([{}, {}])


def test_counterpart_score_excludes_source_and_retains_missing_target():
    ledger = reference([{}, {"source_excel_row": "2", "stream": "NP_Pos", "feature_id": "2"}])
    q = [{"stream": "RP_Pos", "feature_id": "1", "manual_PC": "synthetic"}]
    replay = [{"stream": "RP_Pos", "feature_id": "1", "representatives": [
        {"mode": "RP_Pos", "feature_id": "1", "adduct": "[M+H]+"}]}]
    rows, metric = TOOL["score_representatives"](q, replay, ledger)
    assert len(rows) == 4
    assert metric == {"n": 0, "N": 1, "percent": 0.0}
    assert next(r for r in rows if r["target_mode"] == "NP_Pos")["status"] == "missing"
    assert TOOL["ratio"](0, 0)["percent"] is None


@pytest.mark.parametrize("fid,adduct,status", [
    ("2", "[M+H]+", "correct_feature_and_adduct"),
    ("2", "[M+Na]+", "wrong_adduct"),
    ("3", "[M+H]+", "different_peak_unconfirmed_identity"),
])
def test_exact_original_peak_and_adduct_are_both_required(fid, adduct, status):
    ledger = reference([{"stream": "NP_Pos", "feature_id": "2"}])
    q = [{"stream": "RP_Pos", "feature_id": "1", "manual_PC": "synthetic"}]
    replay = [{"stream": "RP_Pos", "feature_id": "1", "representatives": [
        {"mode": "NP_Pos", "feature_id": fid, "adduct": adduct}]}]
    rows, metric = TOOL["score_representatives"](q, replay, ledger)
    assert next(r for r in rows if r["target_mode"] == "NP_Pos")["status"] == status
    assert metric["n"] == int(status == "correct_feature_and_adduct")


def test_producing_job_paths_relocate_into_frozen_project(tmp_path):
    project = tmp_path / "original"
    snapshot = tmp_path / "snapshot"
    for root in (project, snapshot):
        (root / "Untargeted").mkdir(parents=True)
        (root / "Untargeted/run_manifest.json").write_text(json.dumps({"inputs": {
            "input_snapshot": str(project / ".laser/jobs/untargeted/run/input_snapshot")}}))
        job = root / ".laser/jobs/untargeted/run"
        job.mkdir(parents=True)
        (job / "request.json").write_text("{}")
    first = TOOL["fingerprint"](list(TOOL["project_files"](project)), project)
    second = TOOL["fingerprint"](list(TOOL["project_files"](snapshot)), snapshot)
    assert first == second
    assert ".laser/jobs/untargeted/run/request.json" in first


def test_canonical_inputs_and_raw_scans_are_frozen_but_old_archives_are_not(tmp_path):
    for relative in ("XCMS/peakgrouplist_RP_Pos.txt", "Raw/RP_Pos/sample.mzXML",
                     "Archive/LASER/old/table.csv", "evaluation/reference/gold.xlsx"):
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("synthetic")
    paths = {p.relative_to(tmp_path).as_posix() for p in TOOL["project_files"](tmp_path)}
    assert paths == {"XCMS/peakgrouplist_RP_Pos.txt", "Raw/RP_Pos/sample.mzXML"}


def test_query_order_mismatch_cannot_produce_silent_scores():
    with pytest.raises(ValueError, match="order/key"):
        TOOL["score_representatives"](
            [{"stream": "RP_Pos", "feature_id": "1", "manual_PC": "synthetic"}],
            [{"stream": "RP_Pos", "feature_id": "2", "representatives": []}], reference([{}]))


def test_conditional_agreement_is_distinct_from_coverage_and_full_manual_recovery(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
    from audit_untargeted_checkpoint import metric_rows
    ledger = reference([{}, {"source_excel_row": "2", "feature_id": "2"},
                        {"source_excel_row": "3", "feature_id": "3"},
                        {"source_excel_row": "4", "feature_id": "", "mapped": False}])
    ledger["assigned_adduct"] = ["[M+H]+", "[M+Na]+", "", ""]
    ledger["correct_adduct"] = [True, False, False, False]
    predictions = pd.DataFrame({"stream": ["RP_Pos"] * 3, "feature_id": ["1", "2", "extra"]})
    rows = metric_rows(ledger, predictions)
    rp = next(r for r in rows if r["scope"] == "all_classes" and r["mode"] == "RP_Pos")
    assert rp["conditional_adduct_agreement_percent"] == 50  # 1/2 assigned
    assert rp["assignment_coverage_percent"] == pytest.approx(200/3)  # 2/3 eligible
    assert rp["all_manual_recovery_percent"] == 25  # 1/4 including unresolved
    assert rp["abstentions"] == 1
    assert rp["unconfirmed_extra_ions"] == 1
    negative = next(r for r in rows if r["scope"] == "PC" and r["mode"] == "RP_Neg")
    assert negative["all_manual_recovery_percent"] is None


def test_unresolved_target_reference_stays_in_counterpart_denominator():
    ledger = reference([{"stream": "NP_Pos", "mapped": False, "feature_id": ""}])
    rows, metric = TOOL["score_representatives"](
        [{"stream": "RP_Pos", "feature_id": "1", "manual_PC": "synthetic"}],
        [{"stream": "RP_Pos", "feature_id": "1", "representatives": []}], ledger)
    assert metric == {"n": 0, "N": 1, "percent": 0.0}
    assert next(r for r in rows if r["target_mode"] == "NP_Pos")["status"] == "unresolved_reference_mapping"


def test_evidence_identity_allows_text_feature_keys_but_rejects_numeric_drift():
    before = pd.DataFrame({'feature_id': [1, 2], 'p_value': [.001, .03]})
    after = before.copy()
    after['feature_id'] = after.feature_id.astype(str)
    TOOL['verify_unchanged_feature_evidence'](before, after)
    after.loc[0, 'p_value'] += 1e-15
    with pytest.raises(AssertionError):
        TOOL['verify_unchanged_feature_evidence'](before, after)
    after = before.copy()
    after['feature_id'] = ['001', '2']
    with pytest.raises(AssertionError):
        TOOL['verify_unchanged_feature_evidence'](before, after)
