"""Evaluate fresh runs against a previously frozen read-only manual crosswalk."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd

from context import WORKSPACE as ROOT, SOURCE

AUDIT = ROOT / "outputs/qe_dataset_audit_20261022"


def adduct(value):
    text = re.sub(r"\]\((\d+)([+-])\)$", r"]\1\2", str(value).strip())
    return {"[M+HCOO]-": "[M+CHO2]-"}.get(text, text)


def manual_crosswalk(dataset):
    checkpoint = AUDIT / "checkpoints" / (dataset + ".json")
    record = json.loads(checkpoint.read_text())
    fingerprints = {r["name"]: r["sha256"] for r in record["files"]}
    for name in (
        "MS_data.xlsx",
        *(f"peakgrouplist_{s}.txt" for s in ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")),
    ):
        h = hashlib.file_digest(
            (SOURCE / dataset / name).open("rb"), "sha256"
        ).hexdigest()
        assert h == fingerprints[name], (dataset, name, "stale manual crosswalk")
    return pd.DataFrame(
        [
            dict(
                stream=p["stream"],
                matlab_species=p["name"],
                matlab_class=p["lipid_class"],
                matlab_adduct=adduct(p["adduct"]),
                xcms_feature_id=p["original_feature_id"],
                crosswalk_status="profile_match"
                if p["matched_original_feature"]
                else "manual_peak_unresolved",
                matlab_is_internal_standard=p["name"].endswith(" (IS)")
                or p["name"].startswith("IS: "),
                source_sheet=p["source_sheet"],
                source_excel_row=p["source_excel_row"],
            )
            for p in record["proofs"]
        ]
    )


def verified_reference(dataset, reference):
    original = manual_crosswalk(dataset)
    if reference is None:
        return original
    repaired = pd.read_pickle(reference / dataset / "curated_feature_crosswalk.pkl")
    identity = [
        "stream",
        "matlab_species",
        "matlab_class",
        "matlab_adduct",
        "matlab_is_internal_standard",
        "source_sheet",
        "source_excel_row",
    ]
    pd.testing.assert_frame_equal(original[identity], repaired[identity])
    known = original.crosswalk_status.eq("profile_match")
    pd.testing.assert_series_equal(
        original.loc[known, "xcms_feature_id"], repaired.loc[known, "xcms_feature_id"]
    )
    assert repaired.loc[known, "crosswalk_status"].eq("profile_match").all()
    changed = ~known & repaired.crosswalk_status.eq("profile_match")
    assert repaired.loc[changed, "matlab_is_internal_standard"].all()
    unchanged = ~changed
    assert (
        original.loc[unchanged, "xcms_feature_id"]
        .fillna("")
        .astype(str)
        .equals(repaired.loc[unchanged, "xcms_feature_id"].fillna("").astype(str))
    )
    assert original.loc[unchanged, "crosswalk_status"].equals(
        repaired.loc[unchanged, "crosswalk_status"]
    )
    audit = pd.read_csv(
        reference / "mapping_corrections.csv", dtype={"feature_id": str}
    )
    audit = audit[audit.dataset.eq(dataset)]
    assert len(audit) == int(changed.sum())
    for row in repaired.loc[changed].itertuples():
        proof = audit[
            audit.source_sheet.eq(row.source_sheet)
            & audit.source_excel_row.eq(row.source_excel_row)
        ]
        assert len(proof) == 1 and str(proof.iloc[0].feature_id) == str(
            row.xcms_feature_id
        )
        assert (
            proof.iloc[0].profile_correlation >= 0.999999
            and proof.iloc[0].profile_samples >= 4
        )
        assert (
            abs(proof.iloc[0].mz_difference_Da) <= 1e-7
            and abs(proof.iloc[0].RT_difference_seconds) <= 1e-5
        )
    return repaired


def evaluate(out, partial=False, reference=None):
    out = Path(out)
    sys.path.insert(0, str(out / "source_snapshot/app"))
    from standard_lipidomics.validation import (
        AnnotationComparison,
        compare_manual_annotations,
        summarize_comparison,
    )

    if (out / "run_manifest.json").exists():
        manifest = json.loads((out / "run_manifest.json").read_text())
    else:
        protocol = json.loads((out / "protocol.json").read_text())
        baseline = json.loads(
            (Path(protocol["baseline"]) / "run_manifest.json").read_text()
        )
        manifest = dict(
            datasets=baseline["datasets"],
            fresh_raw_input_runs=False,
            evaluation_note=protocol["run_kind"] + " " + protocol["validation"],
        )
    comparisons = []
    for dataset in manifest["datasets"]:
        run = out / "runs" / dataset
        if not (run / "complete.json").exists():
            if partial:
                continue
            raise RuntimeError("Dataset did not complete: " + dataset)
        suffix = "_v2" if reference is not None else ""
        dest = out / ("evaluation" + suffix) / dataset
        manual = verified_reference(dataset, reference)

        def digest(path):
            with path.open("rb") as handle:
                return hashlib.file_digest(handle, "sha256").hexdigest()

        identity = dict(
            selected_sha256=digest(run / "automatic_annotations.csv"),
            candidates_sha256=digest(run / "candidates.pkl"),
            manual_sha256=hashlib.sha256(
                manual.to_csv(index=False).encode()
            ).hexdigest(),
            evaluator_sha256=digest(Path(__file__)),
            validation_sha256=digest(
                out / "source_snapshot/app/standard_lipidomics/validation.py"
            ),
        )
        cache = dest / "evaluation_identity.json"
        if cache.exists() and json.loads(cache.read_text()) == identity:
            comparison = AnnotationComparison(
                pd.read_pickle(dest / "manual_comparison.pkl"),
                pd.read_pickle(dest / "laser_comparison.pkl"),
            )
        else:
            dest.mkdir(parents=True, exist_ok=True)
            manual.to_csv(dest / "curated_feature_crosswalk.csv", index=False)
            selected = pd.read_csv(
                run / "automatic_annotations.csv",
                dtype={"feat_feature_id": str},
                low_memory=False,
            )
            selected["adduct"] = selected["adduct"].map(adduct)
            candidates = pd.read_pickle(run / "candidates.pkl")
            candidates["adduct"] = candidates["adduct"].map(adduct)
            comparison = compare_manual_annotations(
                manual, selected, candidates=candidates, dataset=dataset
            )
            comparison.manual.to_csv(dest / "manual_comparison.csv", index=False)
            comparison.laser.to_csv(dest / "laser_comparison.csv", index=False)
            comparison.manual.to_pickle(dest / "manual_comparison.pkl")
            comparison.laser.to_pickle(dest / "laser_comparison.pkl")
            cache.write_text(json.dumps(identity, indent=2), encoding="utf-8")
        comparisons.append(comparison)
        print(
            "EVALUATED",
            dataset,
            "matched",
            int(comparison.manual.same_manual_peak.sum()),
            "/",
            len(comparison.manual),
            "extras",
            int(comparison.laser.comparison.eq("not_in_manual_list").sum()),
            flush=True,
        )
    if not comparisons:
        print("No completed runs yet", flush=True)
        return
    combined = AnnotationComparison(
        pd.concat([c.manual for c in comparisons], ignore_index=True),
        pd.concat([c.laser for c in comparisons], ignore_index=True),
    )
    reports = out / ("reports" + suffix)
    reports.mkdir(exist_ok=True)
    tables = {}
    for name, groups in [
        ("by_dataset_mode", ("dataset", "stream")),
        ("by_dataset_mode_class", ("dataset", "stream", "lipid_class")),
        ("by_mode_class", ("stream", "lipid_class")),
        ("by_mode", ("stream",)),
    ]:
        table = summarize_comparison(combined, group_by=groups)
        # Primary numerator is unique correct manual entries; unresolved rows
        # stay in the ALL-manual denominator. Duplicates are exposed separately.
        table["correct_manual_percent"] = (
            100 * table.manual_same_peak / table.manual_rows.replace(0, float("nan"))
        )
        table["extras_per_100_manual"] = (
            100
            * table.laser_not_in_manual_list
            / table.manual_rows.replace(0, float("nan"))
        )
        table.to_csv(reports / (name + ".csv"), index=False)
        tables[name] = table
    combined.manual.to_pickle(reports / "all_manual_comparison.pkl")
    combined.laser.to_pickle(reports / "all_laser_comparison.pkl")
    cls = tables["by_mode_class"]
    cls[(cls.manual_rows > 0) & (cls.correct_manual_percent < 90)].sort_values(
        ["manual_rows", "correct_manual_percent"], ascending=[False, True]
    ).to_csv(reports / "classes_below_90.csv", index=False)
    unresolved = combined.manual[~combined.manual.mapped]
    unresolved.to_csv(reports / "unresolved_manual_features.csv", index=False)
    summary = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        datasets_evaluated=len(comparisons),
        expected_datasets=len(manifest["datasets"]),
        complete=len(comparisons) == len(manifest["datasets"]),
        scope="All manually curated lipid classes; same original feature required; all manual rows in denominator",
        fresh_raw_input_annotation_runs=bool(
            manifest.get("fresh_raw_input_runs", False)
        ),
        manual_crosswalk_source="Frozen read-only audit, checked against current workbook and all four original peak-list SHA256 hashes",
        reference_version="v2: standard raw-profile normalization repair applied identically before/after; not an algorithm gain"
        if reference
        else "v1 audit profile mapping",
        manual_rows=int(tables["by_mode"].manual_rows.sum()),
        correct=int(tables["by_mode"].manual_same_peak.sum()),
        extras=int(tables["by_mode"].laser_not_in_manual_list.sum()),
        different_peak=int(
            tables["by_mode"].laser_same_manual_lipid_different_peak.sum()
        ),
        unresolved_manual_rows=int(tables["by_mode"].unresolved_manual_rows.sum()),
        duplicate_mapped_manual_rows=int(
            tables["by_mode"].duplicate_mapped_manual_rows.sum()
        ),
        class_mode_combinations_with_manual=int((cls.manual_rows > 0).sum()),
        class_mode_combinations_below_90=int(
            ((cls.manual_rows > 0) & (cls.correct_manual_percent < 90)).sum()
        ),
        note=manifest.get(
            "evaluation_note",
            "Baseline is current default annotation, experimental PE selector OFF. No curation was exposed to annotation. Known prior development exposure prevents an untouched-external-validation claim.",
        ),
    )
    (reports / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(
        tables["by_mode"][
            [
                "stream",
                "manual_rows",
                "manual_same_peak",
                "correct_manual_percent",
                "laser_not_in_manual_list",
                "laser_same_manual_lipid_different_peak",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--manual-reference", type=Path)
    args = parser.parse_args()
    evaluate(args.output, args.partial, args.manual_reference)
