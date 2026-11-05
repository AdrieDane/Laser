# ruff: noqa: E402
"""Read-only library and original-feature evidence validation; no model fitting.

Private detail tables go only to the explicitly selected output directory.
Curated identities/formulas are used for scoring after predictions, never as
inputs to isotope finding, formula enumeration or relationship searches.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT)]

from standard_lipidomics.project_store import load_untargeted_workflow
from standard_lipidomics.untargeted import (
    UntargetedComparisonCriteria,
    _attach_isotope_evidence,
    _coordinate_sources,
    _elemental_formula_candidates,
)
from standard_lipidomics.untargeted_adducts import search_adduct_evidence
from standard_lipidomics.untargeted_series import attach_lipid_series
from tools.dataset_scanner.core import (
    Limits,
    curated_rows,
    mass_name,
    metadata_maps,
    reference_masses,
    screen_stream,
)
from tools.dataset_scanner.readers import Workbook

STREAMS = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")
REFERENCE = ROOT / "reference_data/4d_lipid_reference.csv"


def atoms(formula):
    text = str(formula).strip()
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", text)
    if not parts or "".join(a + n for a, n in parts) != text:
        return {}
    output = defaultdict(int)
    for atom, number in parts:
        output[atom] += int(number or 1)
    return dict(output)


def canonical_formula(formula):
    return tuple(sorted(atoms(formula).items()))


def canonical_adduct(value):
    return re.sub(r"\]\((\d+)([+-])\)$", r"]\1\2", str(value)).replace("HCOO", "CHO2")


def charge(value):
    match = re.search(r"\](?:\()?([23])", str(value))
    return int(match.group(1)) if match else 1


def candidates(text):
    return [
        (canonical_adduct(adduct), canonical_formula(formula))
        for adduct, formula in re.findall(
            r"(\[[^\]]+\](?:\d)?[+-])\s+([A-Z][A-Za-z0-9]*)\s+\(", str(text)
        )
    ]


def formula_scores(text, expected, adduct):
    options = candidates(text)
    target = canonical_formula(expected)
    correct = [formula == target and bool(target) for _, formula in options]
    joint = [
        formula == target and ion == canonical_adduct(adduct) and bool(target)
        for ion, formula in options
    ]
    return {
        "top1_formula": bool(correct and correct[0]),
        "top5_formula": any(correct),
        "top1_formula_adduct": bool(joint and joint[0]),
        "top5_formula_adduct": any(joint),
    }


def in_scope(formula, ion):
    counts = atoms(formula)
    return (
        bool(counts)
        and set(counts) <= {"C", "H", "N", "O", "P", "S"}
        and counts.get("C", 0) >= 4
        and (
            counts.get("N", 0) <= 3
            and 1 <= counts.get("O", 0) <= 14
            and counts.get("P", 0) <= 2
            and counts.get("S", 0) <= 2
            and charge(ion) <= 2
        )
    )


def library_controls(output):
    frame = (
        pd.read_csv(REFERENCE)
        .rename(columns={"class": "lipid_class"})
        .drop_duplicates(["formula", "adduct", "charge", "mz"])
    )
    records = []
    for index, row in enumerate(frame.itertuples(index=False)):
        composition = atoms(row.formula)
        ion_carbon = composition.get("C", 0) + (
            2
            if "CH3COO" in row.adduct
            else 1
            if "HCOO" in row.adduct or "CHO2" in row.adduct
            else 0
        )
        ratio = 0.0107 / 0.9893
        estimate = ion_carbon * ratio * 100 / 1.07
        text, count = _elemental_formula_candidates(
            float(row.mz),
            "pos" if row.charge > 0 else "neg",
            estimate,
            3.0,
            charge=abs(int(row.charge)),
            m_plus_2_percent=ion_carbon * (ion_carbon - 1) / 2 * ratio**2 * 100,
        )
        records.append(
            {
                "name": row.name,
                "lipid_class": row.lipid_class,
                "formula": row.formula,
                "adduct": row.adduct,
                "mz": row.mz,
                "in_scope": in_scope(row.formula, row.adduct),
                "candidate_count": count,
                **formula_scores(text, row.formula, row.adduct),
                "predictions": text,
            }
        )
        if index % 500 == 0:
            print(f"Library controls {index + 1}/{len(frame)}", flush=True)
    result = pd.DataFrame(records)
    result.to_csv(output / "library_controls.csv", index=False)
    summary = {
        key: int(result[key].sum())
        for key in (
            "in_scope",
            "top1_formula",
            "top5_formula",
            "top1_formula_adduct",
            "top5_formula_adduct",
        )
    }
    summary["total"] = len(result)
    return summary


def synthetic_controls(output):
    features, queries, truth = [], [], []
    for index in range(44):
        cohort = (
            "clean" if index < 20 else "missing" if index < 32 else "implausible_decoy"
        )
        z = 1 if index % 2 == 0 else 2
        carbon = 36 + index % 10
        neutral = 677.49955528 + (carbon - 36) * 14.015650064
        mz, rt = (neutral + z * 1.007276466621) / z, 5.0 + index * 2.0
        fid = f"parent_{index}"

        def peak(name, mass, retention, intensity):
            return {
                "feature_id": name,
                "mz": mass,
                "rt": retention,
                "a1": intensity * 0.9,
                "a2": intensity * 1.1,
                "b1": intensity * 1.2,
                "b2": intensity * 0.8,
            }

        features.append(peak(fid, mz, rt, 1e5))
        query = {
            "stream": "RP_Pos",
            "feature_id": fid,
            "mz": mz,
            "rt": rt,
            "mean_intensity_a": 1e5,
            "mean_intensity_b": 1e5,
            "significant": True,
            "raw_p_significant": True,
        }
        queries.append(query)
        ratio = 0.0107 / 0.9893
        target = mz + 1.00335483507 / z
        if cohort == "clean":
            features.append(peak(f"m1_{index}", target, rt, 1e5 * carbon * ratio))
            features.append(
                peak(
                    f"m2_{index}",
                    mz + 2 * 1.00335483507 / z,
                    rt,
                    1e5 * carbon * (carbon - 1) / 2 * ratio**2,
                )
            )
        elif cohort == "missing":
            features.append(peak(f"wrong_mass_{index}", target * (1 + 30e-6), rt, 1e4))
            features.append(peak(f"wrong_rt_{index}", target, rt + 1.0, 1e4))
        else:
            features.append(peak(f"unrelated_{index}", target, rt, 1e6))
        truth.append(
            {
                "feature_id": fid,
                "cohort": cohort,
                "expected_charge": z,
                "expected_m1": f"m1_{index}" if cohort == "clean" else "",
            }
        )
    frame = pd.DataFrame(features)
    frame.attrs["sample_group_map"] = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    result = {
        "streams": {"RP_Pos": {"features": frame}},
        "automatic_annotations": pd.DataFrame(),
    }
    predicted = _attach_isotope_evidence(
        result,
        {"RP_Pos": pd.DataFrame(queries)},
        UntargetedComparisonCriteria("A", "B"),
    )["RP_Pos"]
    scored = pd.DataFrame(truth).merge(
        predicted, on="feature_id", validate="one_to_one"
    )
    scored["m1_correct"] = scored.expected_m1.eq(scored.isotope_m_plus_1_feature_id)
    scored["charge_correct"] = scored.expected_charge.eq(
        scored.isotope_charge_hypothesis
    )
    scored.to_csv(output / "synthetic_isotope_controls.csv", index=False)
    return [
        {
            "cohort": name,
            "total": len(group),
            "m1_correct": int(group.m1_correct.sum()),
            "charge_correct": int(group.charge_correct.sum()),
        }
        for name, group in scored.groupby("cohort")
    ]


def manual_mapping(source, output):
    curated, masses = curated_rows(source / "MS_data.xlsx")
    sample_maps = metadata_maps(
        {
            phase: source / f"26P0026_ALD rat brain_{phase}.xlsx"
            for phase in ("RP", "NP")
        }
    )
    formulas = defaultdict(set)
    with Workbook(source / "MS_data.xlsx") as book:
        for row in book.records("Primary list (normalized)"):
            formula = str(row.get("chem formula", "")).strip()
            if atoms(formula):
                formulas[mass_name(row.get("name", ""))].add(formula)
    records = []
    for mode in STREAMS:
        rows = [row for row in curated if row["stream"] == mode]
        _, proofs = screen_stream(
            mode,
            rows,
            masses,
            reference_masses(REFERENCE),
            [],
            source / f"peakgrouplist_{mode}.txt",
            sample_maps[mode],
            Limits(),
            progress=lambda message: print(message, flush=True),
        )
        for row, proof in zip(rows, proofs):
            choices = formulas[mass_name(row["name"])]
            records.append(
                {
                    "stream": mode,
                    "feature_id": proof["original_feature_id"],
                    "mapped": proof["matched_original_feature"],
                    "name": row["name"],
                    "lipid_class": row["lipid_class"],
                    "adduct": row["adduct"],
                    "formula": next(iter(choices)) if len(choices) == 1 else "",
                    "profile_correlation": proof["profile_correlation"],
                    "manual_mz": row["mz"],
                    "manual_rt_seconds": row["rt_seconds"],
                    "source_sheet": row["sheet"],
                    "source_excel_row": row["excel_row"],
                }
            )
    frame = pd.DataFrame(records)
    frame.to_csv(output / "curated_original_feature_mapping.csv", index=False)
    return frame


def expected_adduct_pairs(manual, criteria):
    pairs = set()
    for _, group in manual.groupby("name"):
        rows = list(group.itertuples(index=False))
        for a in rows:
            for b in rows:
                if (a.stream, a.feature_id) == (b.stream, b.feature_id):
                    continue
                if a.stream == b.stream and canonical_adduct(
                    a.adduct
                ) == canonical_adduct(b.adduct):
                    continue
                if (
                    a.stream[:2] == b.stream[:2]
                    and abs(a.manual_rt_seconds - b.manual_rt_seconds)
                    > criteria.opposite_polarity_rt_tolerance_min * 60
                ):
                    continue
                pairs.add(
                    (
                        a.stream,
                        str(a.feature_id),
                        b.stream,
                        str(b.feature_id),
                        canonical_adduct(a.adduct),
                        canonical_adduct(b.adduct),
                    )
                )
    return pairs


def expected_series_pairs(manual, criteria):
    buckets = defaultdict(list)
    for row in manual.itertuples(index=False):
        formula = atoms(row.formula)
        if not formula:
            continue
        key = (
            row.stream,
            row.lipid_class,
            canonical_adduct(row.adduct),
            formula.get("H", 0) - 2 * formula.get("C", 0),
            tuple(sorted((a, n) for a, n in formula.items() if a not in {"C", "H"})),
        )
        buckets[key].append((formula.get("C", 0), row))
    pairs = set()
    for entries in buckets.values():
        for c, a in entries:
            for d, b in entries:
                if (
                    1 <= abs(d - c) <= 4
                    and abs(a.manual_rt_seconds - b.manual_rt_seconds)
                    <= criteria.ch2_rt_tolerance_min * 60
                ):
                    pairs.add((a.stream, str(a.feature_id), str(b.feature_id)))
    return pairs


def ald_controls(project, source, output):
    result, _ = load_untargeted_workflow(project)
    criteria = UntargetedComparisonCriteria(**result["untargeted_analysis"]["criteria"])
    manual = manual_mapping(source, output)
    mapped = manual[manual.mapped].copy()
    by_stream = {}
    for mode in STREAMS:
        features = result["streams"][mode]["features"].copy()
        features["feature_id"] = features.feature_id.astype(str)
        selected = (
            features.set_index("feature_id", drop=False)
            .loc[mapped.loc[mapped.stream.eq(mode), "feature_id"]]
            .copy()
        )
        mapping = features.attrs["sample_group_map"]
        a = [
            key
            for key, value in mapping.items()
            if value == criteria.group_a and key in features
        ]
        b = [
            key
            for key, value in mapping.items()
            if value == criteria.group_b and key in features
        ]
        query = selected[["feature_id", "mz", "rt"]].reset_index(drop=True)
        query["stream"] = mode
        query["mean_intensity_a"] = selected[a].mean(axis=1).to_numpy()
        query["mean_intensity_b"] = selected[b].mean(axis=1).to_numpy()
        query["significant"] = True
        query["raw_p_significant"] = True
        by_stream[mode] = query
    print("Blind isotope and formula queries: mapped original ALD peaks", flush=True)
    predictions = _attach_isotope_evidence(result, by_stream, criteria)
    combined = pd.concat(predictions.values(), ignore_index=True)
    combined.to_csv(output / "ald_blind_evidence_predictions.csv", index=False)
    scored = manual.merge(
        combined, on=["stream", "feature_id"], how="left", validate="many_to_one"
    )
    scores = pd.DataFrame(
        [
            formula_scores(
                row.get("elemental_formula_candidates", ""),
                row["formula"],
                row["adduct"],
            )
            for row in scored.to_dict("records")
        ]
    )
    scored = pd.concat([scored, scores], axis=1)
    scored["formula_in_scope"] = [
        in_scope(f, a) for f, a in zip(scored.formula, scored.adduct)
    ]
    scored["m1_found"] = (
        scored.isotope_m_plus_1_feature_id.fillna("").astype(str).ne("")
    )
    scored["charge_correct"] = scored.isotope_charge_hypothesis.eq(
        scored.adduct.map(charge)
    )
    scored.to_csv(output / "ald_formula_and_isotope_scores.csv", index=False)
    print("Blind adduct queries", flush=True)
    coords = _coordinate_sources(result)
    links = search_adduct_evidence(
        coords,
        predictions,
        ppm=criteria.cross_mode_ppm_tolerance,
        rt_tolerance=criteria.opposite_polarity_rt_tolerance_min,
        progress=lambda n, total, message: print(message, flush=True),
    )
    links.to_csv(output / "ald_adduct_candidates.csv", index=False)
    expected = expected_adduct_pairs(mapped, criteria)
    observed = {
        (
            r.source_stream,
            str(r.source_feature_id),
            r.target_stream,
            str(r.target_feature_id),
            canonical_adduct(r.source_adduct),
            canonical_adduct(r.target_adduct),
        )
        for r in links.itertuples(index=False)
    }
    print("Blind homologous-series queries", flush=True)
    _, series = attach_lipid_series(predictions, coords, {}, criteria)
    series.to_csv(output / "ald_series_candidates.csv", index=False)
    expected_series = expected_series_pairs(mapped, criteria)
    observed_series = {
        (r.stream, str(r.source_feature_id), str(r.target_feature_id))
        for r in series.itertuples(index=False)
    }
    lookup = {
        (row.stream, str(row.feature_id)): row for row in mapped.itertuples(index=False)
    }
    assessment = []
    for row in links.itertuples(index=False):
        target = lookup.get((row.target_stream, str(row.target_feature_id)))
        source_row = lookup.get((row.source_stream, str(row.source_feature_id)))
        if target is not None and source_row is not None:
            assessment.append(
                {
                    "stream": row.source_stream,
                    "same_manual_name": source_row.name == target.name,
                    "same_manual_formula": bool(atoms(source_row.formula))
                    and canonical_formula(source_row.formula)
                    == canonical_formula(target.formula),
                    "correct_adduct_pair": canonical_adduct(row.source_adduct)
                    == canonical_adduct(source_row.adduct)
                    and canonical_adduct(row.target_adduct)
                    == canonical_adduct(target.adduct),
                }
            )
    assessed = pd.DataFrame(assessment)
    assessed.to_csv(output / "ald_curated_endpoint_adduct_assessment.csv", index=False)
    summary = []
    for mode in STREAMS:
        group = scored[scored.stream.eq(mode)]
        true_pairs = {pair for pair in expected if pair[0] == mode}
        true_series = {pair for pair in expected_series if pair[0] == mode}
        assessed_mode = (
            assessed[assessed.stream.eq(mode)] if not assessed.empty else assessed
        )
        assessed_series = {
            pair
            for pair in observed_series
            if pair[0] == mode and (mode, pair[2]) in lookup
        }
        summary.append(
            {
                "stream": mode,
                "manual_entries": len(group),
                "mapped_original_peaks": int(group.mapped.sum()),
                "known_formula": int(group.formula.map(lambda f: bool(atoms(f))).sum()),
                "formula_in_scope": int(group.formula_in_scope.sum()),
                **{
                    name: int(group[name].sum())
                    for name in (
                        "m1_found",
                        "charge_correct",
                        "top1_formula",
                        "top5_formula",
                        "top1_formula_adduct",
                        "top5_formula_adduct",
                    )
                },
                "expected_adduct_pairs": len(true_pairs),
                "found_expected_adduct_pairs": len(true_pairs & observed),
                "curated_endpoint_candidates": len(assessed_mode),
                "curated_consistent_adduct_candidates": int(
                    (
                        assessed_mode.same_manual_name
                        & assessed_mode.correct_adduct_pair
                    ).sum()
                )
                if len(assessed_mode)
                else 0,
                "expected_series_pairs": len(true_series),
                "found_expected_series_pairs": len(true_series & observed_series),
                "curated_endpoint_series_candidates": len(assessed_series),
                "curated_consistent_series_candidates": len(
                    assessed_series & expected_series
                ),
            }
        )
    pd.DataFrame(summary).to_csv(output / "ald_summary.csv", index=False)
    return summary


def fingerprint(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--curation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--part", choices=["all", "library", "ald", "synthetic"], default="all"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = [
        REFERENCE,
        args.curation / "MS_data.xlsx",
        *(args.project / f"peakgrouplist_{s}.txt" for s in STREAMS),
    ]
    before = {str(path): fingerprint(path) for path in inputs}
    for mode in STREAMS:
        if fingerprint(args.project / f"peakgrouplist_{mode}.txt") != fingerprint(
            args.curation / f"peakgrouplist_{mode}.txt"
        ):
            raise ValueError(f"{mode}: test and curated original inputs differ")
    sources = {
        str(path.relative_to(ROOT)): fingerprint(path)
        for path in [
            Path(__file__),
            ROOT / "app/standard_lipidomics/untargeted.py",
            ROOT / "app/standard_lipidomics/untargeted_adducts.py",
            ROOT / "app/standard_lipidomics/untargeted_series.py",
        ]
    }
    record = {
        "started": datetime.now(timezone.utc).isoformat(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "inputs": before,
        "sources": sources,
        "part": args.part,
    }
    (args.output / "provenance.json").write_text(json.dumps(record, indent=2))
    start = time.monotonic()
    if args.part in {"all", "library"}:
        record["library"] = library_controls(args.output)
    if args.part in {"all", "synthetic"}:
        record["synthetic"] = synthetic_controls(args.output)
    if args.part in {"all", "ald"}:
        record["ald"] = ald_controls(args.project, args.curation, args.output)
    if before != {str(path): fingerprint(path) for path in inputs}:
        raise RuntimeError("Source inputs changed during validation")
    record["elapsed_seconds"] = time.monotonic() - start
    record["complete"] = True
    (args.output / "summary.json").write_text(json.dumps(record, indent=2))
    print("Validation complete", flush=True)


if __name__ == "__main__":
    main()
