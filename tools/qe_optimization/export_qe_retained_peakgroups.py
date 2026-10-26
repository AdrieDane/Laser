"""Export accepted, verified 041 calls without rerunning/changing annotations.

Private outputs only. Use the existing app byte-preserving writer, independently
undo its insertion, and freshly score the actual exported identities against
the original verified manual crosswalk. Never read curation before export.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd

from context import REPO, SOURCE, WORKSPACE
from frozen_resources import configure_frozen_resources

MODES = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")
MODE_NAMES = dict(zip(MODES, ("RP positive", "RP negative", "NP positive", "NP negative")))
QE = WORKSPACE / "outputs/qe_optimization_20261022"
EXPERIMENT = QE / "experiment_041"
FIELDS = (
    "manual_rows", "manual_same_peak", "unresolved_manual_rows",
    "laser_not_in_manual_list", "laser_same_manual_lipid_different_peak",
    "unique_laser_annotations",
)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_export(original, exported, expected, stream):
    """Independent literal byte/identity check; no writer-private helpers."""
    restored, records = [], []
    header_ids = None
    expected = {str(k): (str(n), str(a)) for k, (n, a) in expected.items()}
    seen = set()
    for index, line in enumerate(exported.splitlines(keepends=True)):
        fields = line.split(b"\t")
        if len(fields) < 4:
            raise ValueError("Export does not have exactly two inserted columns")
        restored.append(b"\t".join([fields[0], *fields[3:]]))
        if index == 0:
            if fields[1:3] != [b"laser_annotation", b"adduct"]:
                raise ValueError("Annotation columns must be second and third")
            header = b"\t".join([fields[0], *fields[3:]]).decode("utf-8-sig")
            names = header.rstrip("\r\n").split("\t")
            if "laser_annotation" in names or "adduct" in names:
                raise ValueError("Original already contains output columns")
            candidates = [i for i, n in enumerate(names)
                          if n.strip().casefold() in {"name", "feature", "feature_id", "id"}]
            if len(candidates) != 1:
                raise ValueError("An unambiguous original feature-ID column is required")
            header_ids = candidates[0]
            continue
        original_fields = [fields[0], *fields[3:]]
        fid = original_fields[header_ids].decode("utf-8").rstrip("\r\n")
        if fid in seen:
            raise ValueError("Duplicate original feature ID")
        seen.add(fid)
        pair = (fields[1].decode("utf-8"), fields[2].decode("utf-8"))
        if pair != expected.get(fid, ("", "")):
            raise ValueError("An exported identity/adduct differs from the accepted call")
        if pair[0]:
            records.append(dict(stream=stream, feat_feature_id=fid, name=pair[0], adduct=pair[1]))
    if b"".join(restored) != original:
        raise ValueError("Removing inserted columns does not reproduce original bytes")
    if not set(expected).issubset(seen) or len(records) != len(expected):
        raise ValueError("An accepted original feature is missing from the export")
    return records, len(seen)


def scores_text(frame, title, *, by_class=False, by_dataset=False):
    lines = [title, "", "Correct = SAME lipid, adduct, mode AND original peak as manual curation.",
             "Correct percentage = correct / ALL manual entries, including unresolved peaks.",
             "Standards included; only X4 excluded. Extras are TOTAL manual-absent calls,",
             "not proven false chemical identifications. Wrong-peak choices are separate.",
             "Saved validated research result 041; NOT a fresh app/deployment benchmark.", ""]
    for row in frame.to_dict("records"):
        label = MODE_NAMES[row["stream"]]
        if by_dataset:
            label = row["dataset"] + " / " + label
        if by_class:
            label += " / " + row["lipid_class"]
        n, correct = int(row["manual_rows"]), int(row["manual_same_peak"])
        percentage = f"{100 * correct / n:.1f}% ({correct}/{n})" if n else "n/a (0 manual annotations)"
        lines += [label, f"  Correct compared with manual: {percentage}",
                  f"  Total extra annotations: {int(row['laser_not_in_manual_list'])}",
                  f"  Same manual lipid on a different peak: {int(row['laser_same_manual_lipid_different_peak'])}",
                  f"  Manual annotations not correctly matched: {n-correct}",
                  f"  Unresolved manual peaks (included above): {int(row['unresolved_manual_rows'])}",
                  f"  Total LASER annotations: {int(row['unique_laser_annotations'])}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", action="store_true", dest="make_zip")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists() or destination == SOURCE or SOURCE in destination.parents:
        raise ValueError("Choose a NEW output folder outside the active source collection")
    zip_path = destination.with_suffix(".zip")
    if args.make_zip and zip_path.exists():
        raise FileExistsError(zip_path)
    manifest_path = EXPERIMENT / "run_manifest.json"
    datasets = json.loads(manifest_path.read_text())["datasets"]
    if len(datasets) != 33 or set(datasets) != {p.name for p in SOURCE.iterdir() if p.is_dir()}:
        raise ValueError("The active 33-dataset inventory changed")
    integrity = EXPERIMENT / "integrity_receipt.json"
    if not json.loads(integrity.read_text())["all_passed"]:
        raise ValueError("A verified retained experiment is required")
    configure_frozen_resources(EXPERIMENT / "source_snapshot")
    from annotator.standard_annotated_inputs import append_standard_annotations
    from output_io import atomic_write_bytes, atomic_write_json, atomic_write_text
    from standard_lipidomics.validation import (
        AnnotationComparison, compare_manual_annotations, summarize_comparison,
    )
    import evaluate_qe_baseline as evaluation

    destination.mkdir(parents=True, exist_ok=False)
    receipts, summaries, class_summaries, comparisons = [], [], [], []
    for dataset in datasets:
        selected_path = EXPERIMENT / "runs" / dataset / "automatic_annotations.csv"
        identity = json.loads((EXPERIMENT / "evaluation_v2" / dataset / "evaluation_identity.json").read_text())
        selected_sha = sha(selected_path)
        if selected_sha != identity["selected_sha256"]:
            raise ValueError("Retained selected annotations changed since verification")
        accepted = pd.read_csv(selected_path, dtype={"feat_feature_id": str}, low_memory=False)
        if not accepted.automatic_accepted.eq(True).all():
            raise ValueError("Selected table contains unaccepted calls")
        if accepted.duplicated(["stream", "feat_feature_id"]).any():
            raise ValueError("Expected one accepted identity per original feature/mode")
        if set(accepted.stream) != set(MODES):
            raise ValueError("Missing measurement mode")
        for column in ("name", "adduct", "feat_feature_id"):
            if accepted[column].isna().any() or accepted[column].astype(str).str.contains(r"[\t\r\n]", regex=True).any():
                raise ValueError("Invalid identity/adduct/feature text")
        folder = destination / dataset
        folder.mkdir()
        exported_records = []
        for stream in MODES:
            source = SOURCE / dataset / f"peakgrouplist_{stream}.txt"
            raw = source.read_bytes()
            original_sha = hashlib.sha256(raw).hexdigest()
            audit = json.loads((evaluation.AUDIT / "checkpoints" / (dataset + ".json")).read_text())
            fingerprint = {r["name"]: r["sha256"] for r in audit["files"]}[source.name]
            if original_sha != fingerprint:
                raise ValueError("Original peak list changed since the benchmark")
            part = accepted.loc[accepted.stream.eq(stream)]
            # Explicit columns avoid accidental precedence from sparse display columns.
            calls = part[["stream", "feat_feature_id", "name", "adduct"]]
            expected = {fid: (name, ion) for fid, name, ion in
                        calls[["feat_feature_id", "name", "adduct"]].itertuples(index=False, name=None)}
            output = append_standard_annotations(raw, calls, stream=stream)
            records, count = check_export(raw, output, expected, stream)
            target = folder / stream / f"peakgrouplist_{stream}_annotated.txt"
            atomic_write_bytes(target, output)
            # Verify actual saved bytes too, not just the in-memory producer result.
            if target.read_bytes() != output or sha(source) != original_sha:
                raise ValueError("Saved output/source changed during export")
            exported_records.extend(records)
            receipts.append(dict(dataset=dataset, stream=stream,
                                 output=str(target.relative_to(destination)),
                                 original_sha256=original_sha, output_sha256=sha(target),
                                 original_rows=count, annotated_rows=len(records),
                                 original_bytes_exact_after_dropping_new_columns=True,
                                 selected_annotations_sha256=selected_sha))
        # Curation is opened ONLY AFTER identities are exported and independently checked.
        reconstructed = pd.DataFrame(exported_records).merge(
            accepted[["stream", "feat_feature_id", "class", "is_is"]],
            on=["stream", "feat_feature_id"], how="left", validate="one_to_one",
        )
        manual = evaluation.verified_reference(dataset, QE / "manual_reference_v2")
        if hashlib.sha256(manual.to_csv(index=False).encode()).hexdigest() != identity["manual_sha256"]:
            raise ValueError("Manual reference changed since evaluation")
        comparison = compare_manual_annotations(
            manual, reconstructed.assign(adduct=reconstructed.adduct.map(evaluation.adduct)), dataset=dataset,
        )
        target_comparison = AnnotationComparison(
            comparison.manual.loc[comparison.manual.lipid_class.ne("X4")].copy(),
            comparison.laser.loc[comparison.laser.lipid_class.ne("X4")].copy(),
        )
        summary = summarize_comparison(target_comparison)
        classes = summarize_comparison(target_comparison, group_by=("dataset", "stream", "lipid_class"))
        reference = pd.read_csv(EXPERIMENT / "comparison_with_experiment_029/target_scope_without_X4/by_dataset_mode_before_after.csv")
        expected_counts = reference.loc[reference.dataset.eq(dataset)].set_index("stream")
        for row in summary.to_dict("records"):
            if any(int(row[f]) != int(expected_counts.loc[row["stream"], f + "_after"]) for f in FIELDS):
                raise ValueError("Exported annotation scores differ from retained 041")
        atomic_write_text(folder / "annotation_scores.txt", scores_text(summary, dataset))
        atomic_write_text(folder / "annotation_scores_by_class.txt", scores_text(classes, dataset, by_class=True))
        summaries.append(summary)
        class_summaries.append(classes)
        comparisons.append(target_comparison)
        print("EXPORTED AND FRESHLY SCORED", dataset, len(exported_records), flush=True)
    all_comparison = AnnotationComparison(
        pd.concat([c.manual for c in comparisons], ignore_index=True),
        pd.concat([c.laser for c in comparisons], ignore_index=True),
    )
    pooled = summarize_comparison(all_comparison, group_by=("stream",))
    pooled_classes = summarize_comparison(all_comparison, group_by=("stream", "lipid_class"))
    totals = {f: int(pooled[f].sum()) for f in FIELDS}
    if (totals["manual_rows"], totals["manual_same_peak"], totals["laser_not_in_manual_list"],
        totals["laser_same_manual_lipid_different_peak"]) != (92100, 82740, 9270, 1604):
        raise ValueError("Pooled exported benchmark does not reproduce retained results")
    if len(receipts) != 132 or Counter(r["dataset"] for r in receipts) != Counter(dict.fromkeys(datasets, 4)):
        raise ValueError("Incomplete 132-file export")
    atomic_write_text(destination / "annotation_scores_by_mode.txt", scores_text(pooled, "All 33 datasets pooled by mode"))
    atomic_write_text(destination / "annotation_scores_all_datasets.txt", scores_text(pd.concat(summaries), "All 132 dataset/mode scores", by_dataset=True))
    atomic_write_text(destination / "annotation_scores_by_class_and_mode.txt", scores_text(pooled_classes, "All 33 datasets pooled by class and mode", by_class=True))
    atomic_write_json(destination / "export_verification.json", dict(
        complete=True, retained_method="experiment_041", datasets=33, dataset_modes=132,
        annotation_changes=False, fresh_annotation_run=False, fresh_export_scores=True,
        integrated_app_deployment_benchmark=False, optimization_remains_paused=True,
        exact_original_byte_preservation_all_files=True, original_inputs_changed=False,
        selected_and_exported_identities_exact=True, totals=totals, files=receipts,
        retained_integrity_receipt_sha256=sha(integrity),
        exporter_sha256=sha(Path(__file__)),
        writer_sha256=sha(EXPERIMENT / "source_snapshot/app/annotator/standard_annotated_inputs.py"),
        scores_by_dataset_and_mode=pd.concat(summaries)[["dataset", "stream", *FIELDS]].to_dict("records"),
    ))
    atomic_write_text(destination / "README.txt", (
        "LASER annotated QE peakgroup lists — retained research method 041\n\n"
        "33 datasets, four mode subfolders each, 132 *_annotated.txt files.\n"
        "ONLY columns 2/3 laser_annotation/adduct were added. All original bytes\n"
        "are recovered exactly by removing those columns; originals unchanged.\n"
        "Accepted LASER calls only; unselected peaks have empty added columns.\n"
        "Each dataset has scores by mode and by class/mode; pooled and combined\n"
        "scores are at this root. Correct means the SAME original manual peak.\n"
        "Manual denominator 92,100 includes standards and unresolved entries; X4\n"
        "only excluded. Extras are unconfirmed, not established false positives.\n"
        "Correct 82,740/92,100=89.8371%; extras 9,270; wrong peaks 1,604.\n"
        "These export saved validated RESEARCH calls, not a fresh installed-app\n"
        "or deployment benchmark. The 90% target in every class/mode is NOT met.\n"
        "Experiment 049 was rejected; no rejected-method calls were substituted.\n"
        "Private raw peakgroup files must not be uploaded to a public Git repo.\n"
    ))
    if args.make_zip:
        with ZipFile(zip_path, mode="x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(destination.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=str(Path(destination.name) / path.relative_to(destination)))
        with ZipFile(zip_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP integrity check failed")
        print("ZIP VERIFIED", zip_path, zip_path.stat().st_size, flush=True)
    print("COMPLETE: 33 datasets /132 lossless exports /fresh matching scores", totals, flush=True)


if __name__ == "__main__":
    main()
