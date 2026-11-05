"""Audit a private frozen checkpoint without importing prediction code or refitting.

Detailed study outputs stay alongside the checkpoint, outside the repository.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from freeze_untargeted_baseline import MODES, ROOT, digest, fingerprint, project_files, ratio, write_json


def metric_rows(ledger, predictions):
    reference_keys = set(zip(ledger.stream, ledger.feature_id.astype(str)))
    metrics = []
    for scope, frame in [("all_classes", ledger), ("PC", ledger[ledger.lipid_class.eq("PC")])]:
        for mode in MODES:
            rows = frame[frame.stream.eq(mode)]
            eligible = rows[rows.ion_metric_eligible & rows.adduct.ne("")]
            chosen = eligible[eligible.assigned_adduct.ne("")]
            correct = int(chosen.correct_adduct.sum())
            metrics.append({"scope": scope, "mode": mode, "manual_rows": len(rows),
                "unresolved_mappings": int((~rows.mapped).sum()), "eligible_unique_ions": len(eligible),
                "assigned_reference_ions": len(chosen), "correct_assigned_adducts": correct,
                "incorrect_assigned_adducts": len(chosen) - correct,
                "abstentions": len(eligible) - len(chosen),
                "conditional_adduct_agreement_percent": ratio(correct, len(chosen))["percent"],
                "assignment_coverage_percent": ratio(len(chosen), len(eligible))["percent"],
                "all_manual_recovery_n": int(rows.correct_adduct.sum()),
                "all_manual_recovery_percent": ratio(int(rows.correct_adduct.sum()), len(rows))["percent"],
                "unconfirmed_extra_ions": sum((r.stream, str(r.feature_id)) not in reference_keys
                    for r in predictions[predictions.stream.eq(mode)].itertuples()) if scope == "all_classes" else None})
    return metrics


def audit(checkpoint, duplicate_project=None):
    checkpoint = checkpoint.resolve()
    if checkpoint.is_relative_to(ROOT):
        raise ValueError("Private checkpoint must be outside the repository")
    out = checkpoint / "audit"
    out.mkdir()  # Refuse to overwrite an earlier audit.
    provenance = json.loads((checkpoint / "provenance.json").read_text())
    expected = json.loads((checkpoint / "input_fingerprints.json").read_text())
    project = checkpoint / "project"
    assert fingerprint(list(project_files(project)), project) == expected, "Frozen project changed"
    assert digest(checkpoint / "source.tar") == provenance["source_archive_sha256"], "Source archive changed"
    assert digest(checkpoint / "reference/manual_mapping.csv") == provenance["manual_mapping_sha256"]
    assert digest(checkpoint / "reference/regression_queries.json") == provenance["regression_queries_sha256"]
    for number in (1, 2):
        for name, value in provenance["replay_hashes"][number - 1].items():
            assert digest(checkpoint / f"replay_{number}" / name) == value, f"Replay {number} changed: {name}"
    payload = json.loads((checkpoint / "replay_1/payload.json").read_text())
    coords = {}
    for mode, stream in payload["streams"].items():
        mz = np.frombuffer(base64.b64decode(stream["mz"]), dtype="<f8")
        rt = np.frombuffer(base64.b64decode(stream["rt"]), dtype="<f8")
        assert len(stream["ids"]) == len(set(stream["ids"])), "Duplicate original feature keys"
        coords.update({(mode, str(fid)): (m, r) for fid, m, r in zip(stream["ids"], mz, rt, strict=True)})
    ledger = pd.read_csv(checkpoint / "reference/query_ledger.csv", dtype={"feature_id": str}).fillna("")
    ledger["original_feature_present"] = [(r.stream, str(r.feature_id)) in coords for r in ledger.itertuples()]
    assert ledger.loc[ledger.mapped, "original_feature_present"].all(), "Mapped reference absent from frozen original data"
    ledger["mapping_mass_delta_ppm"] = [
        (coords[(r.stream, str(r.feature_id))][0] - float(r.manual_mz)) / float(r.manual_mz) * 1e6
        if r.mapped else np.nan for r in ledger.itertuples()]
    ledger["mapping_rt_delta_seconds"] = [
        coords[(r.stream, str(r.feature_id))][1] * 60 - float(r.manual_rt_seconds)
        if r.mapped else np.nan for r in ledger.itertuples()]
    predictions = pd.read_csv(checkpoint / "replay_1/compound_features.csv", dtype={"feature_id": str}).fillna("")
    assert not predictions.duplicated(["stream", "feature_id"]).any(), "Multiple assignments to one original ion"
    assigned = {(r.stream, str(r.feature_id)): str(r.adduct) for r in predictions.itertuples() if r.adduct}
    ledger["assigned_adduct"] = [assigned.get((r.stream, str(r.feature_id)), "") for r in ledger.itertuples()]
    ledger["correct_adduct"] = ledger.mapped & ledger.assigned_adduct.ne("") & ledger.assigned_adduct.eq(ledger.adduct)
    def keys_for_table(name):
        frame = pd.read_csv(project / f"Untargeted/tables/{name}.csv", dtype=str).fillna("")
        stream = next((c for c in ("stream", "standard_stream") if c in frame), None)
        feature = next((c for c in ("feature_id", "feat_feature_id") if c in frame), None)
        return set(zip(frame[stream], frame[feature])) if stream and feature else set()
    accepted = keys_for_table("automatic_annotations")
    tested = keys_for_table("untargeted_features")
    seeds = keys_for_table("untargeted_raw_p_features")
    reasons = []
    for row in ledger.itertuples():
        key = (row.stream, str(row.feature_id))
        reasons.append("unresolved_original_feature" if not row.mapped else
            "assigned" if row.assigned_adduct else
            "accepted_lipid_excluded_from_biological_testing" if key in accepted else
            "not_in_saved_untargeted_test" if key not in tested else
            "not_selected_by_raw_p_threshold" if key not in seeds else "no_supported_adduct_assignment")
    ledger["runtime_assignment_reason"] = reasons
    ledger.to_csv(out / "reference_queries_checked.csv", index=False)
    metrics = metric_rows(ledger, predictions)
    pd.DataFrame(metrics).to_csv(out / "baseline_adduct_metrics.csv", index=False)
    pd.DataFrame([{"class": cls, "mode": mode, "manual_rows": len(group[group.stream.eq(mode)]),
                   "mapped_rows": int(group[group.stream.eq(mode)].mapped.sum()),
                   "formula_present": int(group[group.stream.eq(mode)].formula.ne("").sum())}
                  for cls, group in ledger.groupby("lipid_class") for mode in MODES]).to_csv(out / "all_class_four_mode_scope.csv", index=False)
    raw_suffixes = {".mzml", ".mzxml", ".raw", ".mgf", ".msp", ".cdf"}
    raw = [str(p.relative_to(project)) for p in project.rglob("*") if p.suffix.lower() in raw_suffixes]
    standards = pd.read_csv(project / "metadata/internal_standards.csv")
    calibration = project / "LASER/tables/matlab_mass_calibration_audit.csv"
    calibration_rows = pd.read_csv(calibration).fillna("").to_dict("records") if calibration.exists() else []
    inv = json.loads((checkpoint / "replay_1/runtime_inventory.json").read_text())
    specimen_pairs = []
    for source in MODES:
        for target in MODES:
            maps = []
            for mode in (source, target):
                attrs = inv["streams"][mode]["attrs"]
                records = attrs.get("sample_metadata_records", {})
                groups = attrs.get("sample_group_map", {})
                allowed = json.loads((checkpoint / "replay_1/saved_settings.json").read_text())["criteria"]
                keys = [(str(row.get("Sample ID", "")), groups.get(sample)) for sample, row in records.items()
                        if str(row.get("Sample ID", "")) and groups.get(sample) in (allowed["group_a"], allowed["group_b"])]
                counts = Counter(keys)
                maps.append({key for key in keys if counts[key] == 1})
            specimen_pairs.append({"source": source, "target": target,
                                   "uniquely_paired_biological_specimens": len(maps[0] & maps[1])})
    pd.DataFrame(specimen_pairs).to_csv(out / "specimen_pairing.csv", index=False)
    duplicates = {}
    if duplicate_project:
        for path in sorted(project.glob("peakgrouplist_*.txt")):
            other = duplicate_project / path.name
            duplicates[path.name] = {"same_sha256": other.is_file() and digest(path) == digest(other),
                                     "sha256": digest(path)}
    inventory = {"data_scope": "one development acquisition, all reference class labels, all four modes",
        "original_features_total": sum(s["original_features"] for s in inv["streams"].values()),
        "raw_or_spectral_files_in_frozen_current_project": raw,
        "raw_search_scope": "Frozen current project only; no assertion about files elsewhere or historical collections",
        "isotope_evidence": "Table-derived M+1/M+2 candidates available; no independently verified isotope envelopes",
        "msms": "No raw scan or spectrum files found in frozen project" if not raw else "Inspect listed raw/spectral files",
        "standards_catalog_rows": len(standards), "standards_catalog_columns": list(standards),
        "standards_limitation": "Catalog names/concentrations do not verify formula, isotope envelope, adduct or purity",
        "mass_calibration_audit": calibration_rows,
        "rt_calibration_provenance": "No standalone calibration export/model found in current project; upstream corrections unknown",
        "duplicate_project_raw_inputs": duplicates,
        "reference_adduct_counts": ledger.adduct.value_counts().to_dict(),
        "independent_isotope_reference_count": 0, "independent_final_validation_count": 0,
        "reference_formula_count": int(ledger.formula.ne("").sum()),
        "mapped_rows_verified_against_original_features": int(ledger.mapped.sum()),
        "fully_mapped_unique_reference_ions": int(ledger.ion_metric_eligible.sum()),
        "runtime_assignment_reasons": dict(Counter(reasons)),
        "source_and_replay_integrity": "passed"}
    write_json(out / "inventory.json", inventory)
    print(json.dumps({"metrics": metrics, "inventory": inventory}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--duplicate-project", type=Path)
    args = parser.parse_args()
    audit(args.checkpoint, args.duplicate_project)
