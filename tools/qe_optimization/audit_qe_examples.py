"""Read-only collection audit. Writes evidence outside the source collection/repo."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import statistics
import sys
import time

from context import WORKSPACE as ROOT, REPO, SOURCE

OUT = ROOT / "outputs/qe_dataset_audit_20261022"
sys.path.insert(0, str(REPO))
from tools.dataset_scanner.core import (
    Limits,
    STREAMS,
    curated_rows,
    inspect_dataset,
    quantile,
    reference_masses,
    relevant,
    sha256,
    write_reports,
)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def csv_dump(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stats(values, prefix):
    return {
        prefix + "_n": len(values),
        prefix + "_signed_median": statistics.median(values) if values else None,
        prefix + "_p95_absolute": quantile([abs(x) for x in values], 0.95),
        prefix + "_p99_absolute": quantile([abs(x) for x in values], 0.99),
        prefix + "_max_absolute": max(map(abs, values), default=None),
    }


def main():
    OUT.mkdir(exist_ok=False, parents=True)
    checkpoint = OUT / "checkpoints"
    checkpoint.mkdir()
    masses = reference_masses(REPO / "reference_data/4d_lipid_reference.csv")
    datasets = sorted(p for p in SOURCE.iterdir() if p.is_dir())
    records, evidence, mode_details, class_details = [], [], [], []
    started = time.monotonic()
    for index, project in enumerate(datasets, 1):
        print(f"[{index}/{len(datasets)}] CHECKING {project.name}", flush=True)
        manifest_path = project / "collection_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if manifest_path.exists()
            else {}
        )
        files = sorted(p for p in project.iterdir() if p.is_file() and relevant(p))
        before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
        record = inspect_dataset(project, files, project, SOURCE, masses, Limits())
        observed = {f["name"]: f for f in record["files"]}
        integrity = []
        for expected in manifest.get("files", []):
            local = project / expected["name"]
            actual = observed.get(expected["name"])
            digest = (
                actual["sha256"]
                if actual
                else sha256(local)
                if local.is_file()
                else None
            )
            integrity.append(
                dict(
                    name=expected["name"],
                    exists=local.is_file(),
                    size_matches=local.is_file()
                    and local.stat().st_size == expected["size"],
                    sha256_matches=digest == expected["sha256"],
                    sha256=digest,
                )
            )
        unchanged = all(
            (project / name).stat().st_size == size
            and (project / name).stat().st_mtime_ns == mtime
            for name, (size, mtime) in before.items()
        )
        rows, _ = curated_rows(project / "MS_data.xlsx")
        lookup = {(r["sheet"], r["excel_row"]): r for r in rows}
        groups = defaultdict(list)
        for p in record["proofs"]:
            row = lookup[(p["source_sheet"], p["source_excel_row"])]
            p["lipid_class"] = row["lipid_class"]
            groups[(p["stream"], row["lipid_class"])].append(p)
        for (stream, lipid_class), proofs in sorted(groups.items()):
            mass = [
                p["raw_mass_error_ppm"]
                for p in proofs
                if p["raw_mass_error_ppm"] is not None
            ]
            rt = [
                p["stored_fit_RT_error_seconds"]
                for p in proofs
                if p["stored_fit_RT_error_seconds"] is not None
            ]
            class_details.append(
                dict(
                    dataset=project.name,
                    stream=stream,
                    lipid_class=lipid_class,
                    manual_entries=len(proofs),
                    matched=sum(p["matched_original_feature"] for p in proofs),
                    unmapped=sum(not p["matched_original_feature"] for p in proofs),
                    mass_outside_3ppm=sum(abs(x) >= 3 for x in mass),
                    rt_outside_10seconds=sum(abs(x) > 10 for x in rt),
                    **stats(mass, "ppm"),
                    **stats(rt, "rt_seconds"),
                )
            )
        previous_modes = {m["stream"]: m for m in manifest.get("mode_checks", [])}
        differences = []
        for mode in record["modes"]:
            stream = mode["stream"]
            if stream in previous_modes:
                for key, value in mode.items():
                    old = previous_modes[stream].get(key)
                    if isinstance(value, (int, float)) and isinstance(
                        old, (int, float)
                    ):
                        same = abs(value - old) < 1e-7
                    else:
                        same = value == old
                    if not same:
                        differences.append(
                            dict(stream=stream, metric=key, old=old, current=value)
                        )
            proofs = [p for p in record["proofs"] if p["stream"] == stream]
            mass = [
                p["raw_mass_error_ppm"]
                for p in proofs
                if p["raw_mass_error_ppm"] is not None
            ]
            rt = [
                p["stored_fit_RT_error_seconds"]
                for p in proofs
                if p["stored_fit_RT_error_seconds"] is not None
            ]
            mode_details.append(
                dict(
                    dataset=project.name,
                    **mode,
                    unmapped=sum(not p["matched_original_feature"] for p in proofs),
                    mass_outside_3ppm=sum(abs(x) >= 3 for x in mass),
                    rt_outside_10seconds=sum(abs(x) > 10 for x in rt),
                    **stats(mass, "ppm"),
                    **stats(rt, "rt_seconds"),
                )
            )
        evidence.append(
            dict(
                dataset=project.name,
                manifest_source=manifest.get("source"),
                manifest_limits=manifest.get("limits"),
                files_verified=integrity,
                metadata_unchanged_during_audit=unchanged,
                rescanned_mode_differences=differences,
                file_count=len(files),
                bytes=sum(x[0] for x in before.values()),
            )
        )
        dump(checkpoint / (project.name + ".json"), record)
        records.append(record)
        print(
            f"  {record['status']}; "
            + "; ".join(
                f"{m['stream']}: n={m.get('curated_entries', '?')}, ppm={m.get('median_absolute_ppm', '?')}, RT={m.get('median_absolute_rt_seconds', '?')}"
                for m in record["modes"]
            )
            + f"; elapsed {time.monotonic() - started:.0f}s",
            flush=True,
        )
        dump(OUT / "integrity_progress.json", evidence)

    peak_groups = defaultdict(list)
    mode_peak_groups = {s: defaultdict(list) for s in STREAMS}
    all_groups = defaultdict(list)
    for record in records:
        fs = {f["name"]: f["sha256"] for f in record["files"]}
        peaks = tuple(fs.get(f"peakgrouplist_{s}.txt") for s in STREAMS)
        if all(peaks):
            peak_groups[peaks].append(record["project"])
        for s in STREAMS:
            if fs.get(f"peakgrouplist_{s}.txt"):
                mode_peak_groups[s][fs[f"peakgrouplist_{s}.txt"]].append(
                    record["project"]
                )
        fingerprint = tuple(sorted((f["name"], f["sha256"]) for f in record["files"]))
        all_groups[fingerprint].append(record["project"])
    duplicate_peaks = [v for v in peak_groups.values() if len(v) > 1]
    exact_duplicates = [v for v in all_groups.values() if len(v) > 1]
    mode_duplicates = {
        s: [v for v in groups.values() if len(v) > 1]
        for s, groups in mode_peak_groups.items()
    }

    archive_overlap = []
    archive = SOURCE.parent / "archive"
    if archive.is_dir():
        for rp in sorted(archive.rglob("peakgrouplist_RP_Pos.txt")):
            paths = [rp.parent / f"peakgrouplist_{s}.txt" for s in STREAMS]
            if not all(p.is_file() for p in paths):
                continue
            hashes = tuple(sha256(p) for p in paths)
            if hashes in peak_groups:
                archive_overlap.append(
                    dict(archive_path=str(rp.parent), new_datasets=peak_groups[hashes])
                )

    integrity_failed = [
        e["dataset"]
        for e in evidence
        if not e["files_verified"]
        or not all(
            f["exists"] and f["size_matches"] and f["sha256_matches"]
            for f in e["files_verified"]
        )
    ]
    plan = dict(
        schema="laser-dataset-scan-v1",
        scan_id="qe-read-only-audit-20261022",
        created_at=datetime.now(timezone.utc).isoformat(),
        source_root=str(SOURCE),
        allowed_years=["2024", "2025", "2026"],
        limits=asdict(Limits()),
        datasets=records,
        discovery_issues=[],
        note="Explicit per-folder reinspection of the supplied flat collected dataset folder, not year-folder discovery. No annotation run or workflow changes. RT residuals use saved in-sample predictions, not independent validation.",
    )
    write_reports(plan, OUT / "scanner_evidence")
    csv_dump(OUT / "mode_details.csv", mode_details)
    csv_dump(OUT / "class_details.csv", class_details)
    summary = dict(
        folders_checked=len(datasets),
        mode_checks=len(mode_details),
        statuses=dict(Counter(r["status"] for r in records)),
        files_verified=sum(len(e["files_verified"]) for e in evidence),
        bytes=sum(e["bytes"] for e in evidence),
        integrity_failed=integrity_failed,
        sources_stat_unchanged=all(
            e["metadata_unchanged_during_audit"] for e in evidence
        ),
        exact_collection_duplicates=exact_duplicates,
        identical_four_peak_list_groups=duplicate_peaks,
        unique_four_peak_list_sets=len(peak_groups),
        identical_per_mode_peak_lists=mode_duplicates,
        archive_overlap=archive_overlap,
        current_limits=asdict(Limits()),
        elapsed_seconds=time.monotonic() - started,
    )
    dump(OUT / "summary.json", summary)
    dump(OUT / "integrity.json", evidence)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
