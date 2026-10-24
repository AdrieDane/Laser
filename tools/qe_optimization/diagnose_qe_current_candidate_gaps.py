"""Read-only current candidate/selection gap diagnosis; produces NO annotations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_qe_atlas_experiment import species, adduct


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.experiment / "reports_v2/summary.json").read_text())
    assert summary["complete"] and summary["datasets_evaluated"] == 33
    projects = json.loads((args.experiment / "run_manifest.json").read_text())[
        "datasets"
    ]
    args.output.mkdir(exist_ok=False)
    frames, hashes = [], {}
    for dataset in projects:
        candidates = args.experiment / "runs" / dataset / "candidates.pkl"
        evaluated = (
            args.experiment / "evaluation_v2" / dataset / "manual_comparison.pkl"
        )
        for p in (candidates, evaluated):
            hashes[str(p)] = sha(p)
        pool = pd.read_pickle(candidates)
        manual = pd.read_pickle(evaluated).copy()
        keys = {
            (s, species(n), adduct(a), str(fid), bool(standard))
            for s, n, a, fid, standard in pool[
                ["stream", "name", "adduct", "feat_feature_id", "is_is"]
            ].itertuples(index=False, name=None)
        }
        manual["candidate_available"] = [
            (s, species(n), adduct(a), str(fid), bool(standard)) in keys
            for s, n, a, fid, standard in manual[
                ["stream", "species", "adduct", "feature_id", "internal_standard"]
            ].itertuples(index=False, name=None)
        ]
        # Final005 STD aliases may differ from literal physical target names.
        # Every already verified selected match is necessarily achievable.
        manual["candidate_available"] |= manual.same_manual_peak
        manual["manual_count"] = 1
        manual["selection_gap"] = manual.candidate_available & ~manual.same_manual_peak
        manual["discovery_gap"] = ~manual.candidate_available & manual.mapped
        manual["unresolved"] = ~manual.mapped
        manual["dataset"] = dataset
        assert not (manual.same_manual_peak & ~manual.candidate_available).any()
        assert (
            manual.selection_gap
            | manual.discovery_gap
            | manual.unresolved
            | manual.same_manual_peak
        ).all()
        frames.append(manual)
    rows = pd.concat(frames, ignore_index=True)
    assert len(rows) == 92102 and rows.same_manual_peak.sum() == summary["correct"]
    metrics = [
        "manual_count",
        "same_manual_peak",
        "candidate_available",
        "selection_gap",
        "discovery_gap",
        "unresolved",
    ]
    outputs = {}
    for scope, frame in (
        ("all_manual", rows),
        ("target_without_X4", rows[~rows.lipid_class.eq("X4")]),
        (
            "endogenous_without_X4",
            rows[~rows.internal_standard & ~rows.lipid_class.eq("X4")],
        ),
    ):
        folder = args.output / scope
        folder.mkdir()
        for name, group in (
            ("by_dataset_mode_class", ["dataset", "stream", "lipid_class"]),
            ("by_mode_class", ["stream", "lipid_class"]),
            ("by_mode", ["stream"]),
        ):
            table = frame.groupby(group)[metrics].sum().astype(int).reset_index()
            table["selected_correct_percent"] = (
                100 * table.same_manual_peak / table.manual_count
            )
            table["optimistic_candidate_percent"] = (
                100 * table.candidate_available / table.manual_count
            )
            table["correct_needed_to_90"] = np.maximum(
                np.ceil(0.9 * table.manual_count).astype(int) - table.same_manual_peak,
                0,
            )
            table["discovery_needed_for_90"] = table.optimistic_candidate_percent.lt(90)
            table.to_csv(folder / (name + ".csv"), index=False)
            if name == "by_mode_class":
                outputs[scope] = dict(
                    counts=frame[metrics].sum().astype(int).to_dict(),
                    curated_class_modes=len(table),
                    class_modes_below_90=int(
                        table.selected_correct_percent.lt(90).sum()
                    ),
                    class_modes_needing_discovery_for_90=int(
                        table.discovery_needed_for_90.sum()
                    ),
                )
    for p, digest in hashes.items():
        assert sha(Path(p)) == digest
    receipt = dict(
        datasets=33,
        annotation_changes=False,
        models_fitted=0,
        source_sha256=sha(Path(__file__)),
        input_hash_checks=len(hashes),
        scopes=outputs,
        note="Read-only optimistic rowwise candidate availability,NOT attainable accuracy "
        "or a new annotation result. Includes current full physical pools (mass proposals "
        "and standards); mutually competing identities/features ignored. STD alias "
        "already-matched calls count as available. All raw rows unchanged. X4target "
        "exclusion authorized separately; no curation enters runtime/model predictors.",
    )
    (args.output / "summary.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
