"""Evaluation-only correction for differently normalized internal standards.

Requires exact exported coordinates AND near-perfect raw sample-profile
agreement, never a LASER annotation choice. Existing profile matches are fixed.
All manual entries remain in the denominator; no original files are changed.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import sys

import pandas as pd

from context import WORKSPACE as ROOT, REPO, SOURCE

AUDIT = ROOT / "outputs/qe_dataset_audit_20261022/checkpoints"
OUT = ROOT / "outputs/qe_optimization_20261022/manual_reference_v2"
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
from tools.dataset_scanner.core import curated_rows, metadata_maps, correlation, number
from evaluate_qe_baseline import manual_crosswalk


def coordinate(mz, rt):
    return (round(float(mz), 7), round(float(rt), 5))


def eligible_raw_standard_match(row, candidates, samples, used):
    """A standard-only, exact-coordinate and raw-profile match, or no match."""
    if not (row["name"].endswith(" (IS)") or row["name"].startswith("IS: ")):
        return None
    matches = []
    for feature in candidates:
        if (
            abs(feature["mz"] - row["mz"]) > 1e-7
            or abs(feature["rt"] - row["rt_seconds"]) > 1e-5
        ):
            continue
        matches.append(feature)
    if len(matches) != 1 or matches[0]["id"] in used:
        return None
    feature = matches[0]
    left = []
    right = []
    for filename, (sample, _factor) in samples.items():
        a = number(feature["raw"].get(filename))
        b = row["profile"].get(sample)
        if a is not None and b is not None:
            left.append(a)
            right.append(b)
    corr = correlation(left, right)
    if len(left) < 4 or corr is None or corr < 0.999999:
        return None
    return dict(
        feature_id=feature["id"],
        profile_correlation=corr,
        profile_samples=len(left),
        mz_difference_Da=feature["mz"] - row["mz"],
        RT_difference_seconds=feature["rt"] - row["rt_seconds"],
        method="exact_export_coordinates_and_uncorrected_standard_profile",
    )


def self_test():
    row = dict(
        name="SPH(d17:0) (IS)",
        mz=300.1,
        rt_seconds=200.0,
        profile={"a": 10.0, "b": 30.0, "c": 80.0, "d": 180.0},
    )
    samples = {k: (k, p) for k, p in zip("abcd", [1, 2, 4, 8])}
    f = dict(
        id="0017", mz=300.1, rt=200.0, raw={"a": "10", "b": "30", "c": "80", "d": "180"}
    )
    result = eligible_raw_standard_match(row, [f], samples, set())
    assert result and result["feature_id"] == "0017"
    assert eligible_raw_standard_match(row, [f], samples, {"0017"}) is None
    assert eligible_raw_standard_match(row, [f, f], samples, set()) is None
    assert (
        eligible_raw_standard_match({**row, "name": "SPH(d18:0)"}, [f], samples, set())
        is None
    )
    assert (
        eligible_raw_standard_match(row, [{**f, "rt": 200.1}], samples, set()) is None
    )
    assert (
        eligible_raw_standard_match(row, [f], dict(list(samples.items())[:3]), set())
        is None
    )
    assert (
        eligible_raw_standard_match(
            row, [{**f, "raw": dict.fromkeys("abcd", "1")}], samples, set()
        )
        is None
    )


def main():
    self_test()
    OUT.mkdir(parents=True, exist_ok=False)
    changes = []
    controls = []
    summaries = []
    fingerprints = []
    for project in sorted(SOURCE.iterdir()):
        if not project.is_dir():
            continue
        cross = manual_crosswalk(
            project.name
        )  # Also verifies workbook + four raw-list hashes.
        record = json.loads((AUDIT / (project.name + ".json")).read_text())
        rows, _ = curated_rows(project / "MS_data.xlsx")
        manual = {(r["sheet"], r["excel_row"]): r for r in rows}
        samples = metadata_maps(
            {phase: next(project.glob(f"*_{phase}.xlsx")) for phase in ("RP", "NP")}
        )
        cross["mapping_method"] = "protein_corrected_profile"
        cross.loc[cross.crosswalk_status.ne("profile_match"), "mapping_method"] = (
            "unresolved"
        )
        for stream in ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"):
            current = cross[cross.stream.eq(stream)]
            targets = {
                coordinate(
                    manual[(r.source_sheet, r.source_excel_row)]["mz"],
                    manual[(r.source_sheet, r.source_excel_row)]["rt_seconds"],
                ): []
                for r in current.itertuples()
                if r.crosswalk_status != "profile_match"
                and r.matlab_is_internal_standard
            }
            known = {
                str(r.xcms_feature_id): manual[(r.source_sheet, r.source_excel_row)]
                for r in current.itertuples()
                if r.crosswalk_status == "profile_match"
            }
            used = set(known)
            unique_coordinates = Counter()
            coords_for_controls = {}
            with (project / f"peakgrouplist_{stream}.txt").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                for feature in csv.DictReader(handle, delimiter="\t"):
                    mz, rt = float(feature["mzmed"]), float(feature["rtmed"]) * 60
                    key = coordinate(mz, rt)
                    unique_coordinates[key] += 1
                    if key in targets:
                        targets[key].append(
                            dict(id=feature["name"], mz=mz, rt=rt, raw=feature)
                        )
                    if feature["name"] in known:
                        row = known[feature["name"]]
                        coords_for_controls[feature["name"]] = coordinate(mz, rt)
                        controls.append(
                            dict(
                                dataset=project.name,
                                stream=stream,
                                feature_id=feature["name"],
                                same_export_coordinates=key
                                == coordinate(row["mz"], row["rt_seconds"]),
                            )
                        )
            for index, r in current.iterrows():
                if (
                    r.crosswalk_status == "profile_match"
                    or not r.matlab_is_internal_standard
                ):
                    continue
                row = manual[(r.source_sheet, r.source_excel_row)]
                candidates = targets[coordinate(row["mz"], row["rt_seconds"])]
                match = eligible_raw_standard_match(
                    row, candidates, samples[stream], used
                )
                if match is None:
                    continue
                used.add(match["feature_id"])
                cross.loc[index, "xcms_feature_id"] = match["feature_id"]
                cross.loc[index, "crosswalk_status"] = "profile_match"
                cross.loc[index, "mapping_method"] = match["method"]
                changes.append(
                    dict(
                        dataset=project.name,
                        stream=stream,
                        name=r.matlab_species,
                        lipid_class=r.matlab_class,
                        source_sheet=r.source_sheet,
                        source_excel_row=int(r.source_excel_row),
                        **match,
                    )
                )
        dest = OUT / project.name
        dest.mkdir()
        cross.to_pickle(dest / "curated_feature_crosswalk.pkl")
        cross.to_csv(dest / "curated_feature_crosswalk.csv", index=False)
        fingerprints.append(dict(dataset=project.name, source_files=record["files"]))
        summaries.append(
            dict(
                dataset=project.name,
                manual_entries=len(cross),
                mapped=int(cross.crosswalk_status.eq("profile_match").sum()),
                unresolved=int(cross.crosswalk_status.ne("profile_match").sum()),
            )
        )
        print(
            "REFERENCE CHECK",
            project.name,
            "additional strictly verified standards",
            sum(c["dataset"] == project.name for c in changes),
            flush=True,
        )
    pd.DataFrame(changes).to_csv(OUT / "mapping_corrections.csv", index=False)
    pd.DataFrame(controls).to_csv(
        OUT / "existing_profile_match_coordinate_controls.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(OUT / "dataset_reference_counts.csv", index=False)
    conflicts = sum(not c["same_export_coordinates"] for c in controls)
    summary = dict(
        datasets=len(summaries),
        corrected_standard_mappings=len(changes),
        all_existing_mappings_preserved=True,
        all_manual_denominators_preserved=True,
        existing_profile_match_controls=len(controls),
        existing_matches_with_different_rounded_coordinates=conflicts,
        thresholds=dict(
            mz_Da=1e-7,
            RT_seconds=1e-5,
            raw_profile_correlation=0.999999,
            minimum_samples=4,
        ),
        note="Evaluation-only standard normalization correction; NOT a change in annotations or an algorithm gain. Apply identical corrected crosswalks to baseline and experiment. No non-standard mappings changed.",
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "source_manifest.json").write_text(
        json.dumps(fingerprints, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
