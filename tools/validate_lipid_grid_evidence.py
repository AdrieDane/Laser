# ruff: noqa: E402
"""Blind connected-grid queries followed by manual-class/reference-grid scoring.

This is development evidence, not externally validated chemical identification.
Manual formulas/classes are read only AFTER the blind family predictions.
"""

from pathlib import Path
from collections import defaultdict, deque
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from standard_lipidomics.project_store import load_untargeted_workflow  # noqa: E402
from standard_lipidomics.untargeted import (
    _coordinate_sources,
    UntargetedComparisonCriteria,
)  # noqa: E402
from standard_lipidomics.untargeted_grid import search_lipid_grids  # noqa: E402

MODES = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")


def atoms(text):
    return {a: int(n or 1) for a, n in re.findall(r"([A-Z][a-z]?)(\d*)", str(text))}


def reference_neighbours(mapping, edge_rt):
    """Adjacent occupied manual composition cells, with the same RT-gap gate."""
    grouped = defaultdict(list)
    for row in mapping.itertuples(index=False):
        f = atoms(row.formula)
        if not f.get("C") or not f.get("H"):
            continue
        group = (
            row.stream,
            row.lipid_class,
            row.adduct,
            tuple(sorted((a, n) for a, n in f.items() if a not in {"C", "H"})),
        )
        grouped[group].append(
            (
                str(row.feature_id),
                f["C"],
                f["H"] - 2 * f["C"],
                float(row.manual_rt_seconds) / 60,
            )
        )
    expected = set()
    for group, items in grouped.items():
        cell = defaultdict(list)
        by_id = {fid: (c, h, rt) for fid, c, h, rt in items}
        for item in items:
            cell[item[1:3]].append(item)
        adjacency = defaultdict(set)
        for fid, c, h, rt in items:
            for dc, dh in [(-1, 0), (1, 0), (0, -2), (0, 2)]:
                for other, _, _, other_rt in cell.get((c + dc, h + dh), []):
                    if abs(other_rt - rt) <= edge_rt:
                        adjacency[fid].add(other)
        for fid, c, h, _ in items:
            seen = {fid}
            queue = deque([fid])
            while queue:
                current = queue.popleft()
                for other in adjacency[current] - seen:
                    oc, oh, _ = by_id[other]
                    if abs(oc - c) <= 12 and abs(oh - h) <= 16:
                        seen.add(other)
                        queue.append(other)
            expected.update((group[0], fid, other) for other in seen - {fid})
    return expected


def score_pairs(pairs, lookup, expected):
    records = []
    for stream, source, target in sorted(pairs):
        a, b = lookup.get((stream, source)), lookup.get((stream, target))
        known = a is not None and b is not None
        consistent = known and a.lipid_class == b.lipid_class and a.adduct == b.adduct
        records.append(
            {
                "stream": stream,
                "source_feature_id": source,
                "target_feature_id": target,
                "known_curated_endpoints": known,
                "same_manual_class_and_adduct": consistent,
                "in_manual_connected_grid": (stream, source, target) in expected,
            }
        )
    return pd.DataFrame(records)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__),
        *(
            ROOT / "app/standard_lipidomics" / name
            for name in [
                "untargeted_grid.py",
                "untargeted.py",
                "untargeted_adducts.py",
                "project_store.py",
            ]
        ),
    ]
    input_paths = [
        *(args.project / f"peakgrouplist_{mode}.txt" for mode in MODES),
        *(
            args.validation / name
            for name in [
                "ald_blind_evidence_predictions.csv",
                "curated_original_feature_mapping.csv",
                "ald_series_candidates.csv",
            ]
        ),
    ]
    source_hashes = {str(path.relative_to(ROOT)): fingerprint(path) for path in sources}
    input_hashes = {str(path): fingerprint(path) for path in input_paths}
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    (args.output / "provenance.json").write_text(
        json.dumps(
            {"revision": revision, "sources": source_hashes, "inputs": input_hashes},
            indent=2,
        )
    )
    result, _ = load_untargeted_workflow(args.project)
    criteria = UntargetedComparisonCriteria(**result["untargeted_analysis"]["criteria"])
    predictions = pd.read_csv(
        args.validation / "ald_blind_evidence_predictions.csv",
        dtype={"feature_id": str},
    )
    by_stream = {mode: frame.copy() for mode, frame in predictions.groupby("stream")}
    start = time.monotonic()
    print("Blind connected grid queries", flush=True)
    new = search_lipid_grids(by_stream, _coordinate_sources(result), criteria)
    new.to_csv(args.output / "blind_grid_predictions.csv", index=False)
    print("Blind predictions complete; reading scoring labels", flush=True)
    mapping = pd.read_csv(
        args.validation / "curated_original_feature_mapping.csv",
        dtype={"feature_id": str},
    )
    mapped = mapping[mapping.mapped].copy()
    mapped["feature_id"] = mapped.feature_id.str.replace(r"\.0$", "", regex=True)
    lookup = {
        (row.stream, str(row.feature_id)): row for row in mapped.itertuples(index=False)
    }
    expected = reference_neighbours(mapped, criteria.ch2_rt_tolerance_min)
    old = pd.read_csv(
        args.validation / "ald_series_candidates.csv",
        dtype={"source_feature_id": str, "target_feature_id": str},
    )
    grid_roots = set(zip(new.stream, new.source_feature_id.astype(str)))
    old_pairs = {
        (r.stream, str(r.source_feature_id), str(r.target_feature_id))
        for r in old.itertuples(index=False)
    }
    new_pairs = {
        (r.stream, str(r.source_feature_id), str(r.target_feature_id))
        for r in new.itertuples(index=False)
    }
    effective = new_pairs | {pair for pair in old_pairs if pair[:2] not in grid_roots}
    summaries = []
    class_summaries = []
    for version, pairs in [
        ("before_CH2", old_pairs),
        ("after_connected_grid", effective),
    ]:
        scored = score_pairs(pairs, lookup, expected)
        scored.to_csv(args.output / f"{version}_assessment.csv", index=False)
        for mode in MODES:
            data = scored[scored.stream.eq(mode)]
            true = {pair for pair in expected if pair[0] == mode}
            stats = {
                "version": version,
                "stream": mode,
                "manual_entries": int(mapping.stream.eq(mode).sum()),
                "mapped_original_features": int(mapped.stream.eq(mode).sum()),
                "supported_grid_seeds": sum(s == mode for s, _ in grid_roots)
                if version == "after_connected_grid"
                else 0,
                "expected_manual_connected_pairs": len(true),
                "recovered_manual_connected_pairs": len(true & pairs),
                "returned_pairs": len(data),
                "curated_endpoint_pairs": int(data.known_curated_endpoints.sum()),
                "curated_class_adduct_consistent_pairs": int(
                    data.same_manual_class_and_adduct.sum()
                ),
                "uncurated_endpoint_pairs": int((~data.known_curated_endpoints).sum()),
            }
            summaries.append(stats)
        for (mode, lipid_class), group in mapped.groupby(["stream", "lipid_class"]):
            ids = set(group.feature_id.astype(str))
            data = scored[scored.stream.eq(mode) & scored.source_feature_id.isin(ids)]
            truth = {pair for pair in expected if pair[0] == mode and pair[1] in ids}
            class_summaries.append(
                {
                    "version": version,
                    "stream": mode,
                    "lipid_class": lipid_class,
                    "manual_entries": int(
                        (
                            mapping.stream.eq(mode)
                            & mapping.lipid_class.eq(lipid_class)
                        ).sum()
                    ),
                    "mapped": len(group),
                    "expected_pairs": len(truth),
                    "recovered_pairs": len(truth & pairs),
                    "curated_endpoint_pairs": int(data.known_curated_endpoints.sum()),
                    "curated_class_adduct_consistent_pairs": int(
                        data.same_manual_class_and_adduct.sum()
                    ),
                }
            )
    pd.DataFrame(summaries).to_csv(args.output / "grid_summary.csv", index=False)
    pd.DataFrame(class_summaries).to_csv(
        args.output / "grid_class_mode_summary.csv", index=False
    )
    assert source_hashes == {
        str(path.relative_to(ROOT)): fingerprint(path) for path in sources
    }
    assert input_hashes == {str(path): fingerprint(path) for path in input_paths}
    receipt = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sources": source_hashes,
        "inputs": input_hashes,
        "original_inputs": {
            mode: fingerprint(args.project / f"peakgrouplist_{mode}.txt")
            for mode in MODES
        },
        "elapsed_seconds": time.monotonic() - start,
        "summary": summaries,
        "scope": "Same ALD development input, forced manual-original-feature queries; reference-grid and curated-class agreement, not cluster identity/FDR.",
    }
    (args.output / "summary.json").write_text(json.dumps(receipt, indent=2))
    print("Grid evaluation complete", flush=True)


if __name__ == "__main__":
    main()
