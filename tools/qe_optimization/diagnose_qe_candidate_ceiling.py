"""Read-only row-wise candidate ceiling; NOT new annotation performance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from context import WORKSPACE as ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qe_atlas_experiment import species, adduct

QE = ROOT / "outputs/qe_optimization_20261022"
BASE = QE / "baseline"
POOL = QE / "experiment_007"
SELECTED = QE / "experiment_009"
OUT = QE / "candidate_ceiling_v1"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    frames, receipts = [], []
    for dataset in projects:
        paths = (
            POOL / "prepared" / f"fold_{folds[dataset]}" / (dataset + ".pkl"),
            SELECTED / "evaluation_v2" / dataset / "manual_comparison.pkl",
        )
        before = [sha(p) for p in paths]
        pool = pd.read_pickle(paths[0])["pool"]
        exact = {
            (s, species(n), adduct(a), str(fid))
            for s, n, a, fid in pool[
                ["stream", "name", "adduct", "feat_feature_id"]
            ].itertuples(index=False, name=None)
        }
        manual = pd.read_pickle(paths[1])
        manual = manual[~manual.internal_standard].copy()
        manual["candidate_available"] = [
            (s, n, a, str(fid)) in exact
            for s, n, a, fid in manual[
                ["stream", "species", "adduct", "feature_id"]
            ].itertuples(index=False, name=None)
        ]
        manual["manual_count"] = 1
        manual["selection_gap"] = manual.candidate_available & ~manual.same_manual_peak
        manual["discovery_gap"] = ~manual.candidate_available & manual.mapped
        manual["unresolved"] = ~manual.mapped
        manual["dataset"] = dataset
        frames.append(manual)
        assert before == [sha(p) for p in paths], dataset
        receipts.append(dict(dataset=dataset, input_sha256=before))
    all_rows = pd.concat(frames, ignore_index=True)
    metrics = ["manual_count", "same_manual_peak", "candidate_available",
               "selection_gap", "discovery_gap", "unresolved"]
    for name, keys in (
        ("by_dataset_mode_class", ["dataset", "stream", "lipid_class"]),
        ("by_mode_class", ["stream", "lipid_class"]),
        ("by_mode", ["stream"]),
    ):
        table = all_rows.groupby(keys)[metrics].sum().astype(int).reset_index()
        table["selected_correct_percent"] = 100 * table.same_manual_peak / table.manual_count
        table["rowwise_candidate_ceiling_percent"] = 100 * table.candidate_available / table.manual_count
        table.to_csv(OUT / (name + ".csv"), index=False)
    summary = dict(
        datasets=len(projects), dataset_modes=132, internal_standards_excluded=True,
        purpose="READ-ONLY diagnosis of exact007 candidate presence, not new annotations. Row-wise optimistic upper bound ignores mutually conflicting feature/identity selections; not guaranteed attainable accuracy.",
        denominator="ALL endogenous manual entries, including unresolved mappings. Standards evaluated separately in annotation reports.",
        counts=all_rows[metrics].sum().astype(int).to_dict(),
        source_sha256=sha(Path(__file__)), input_receipts=receipts,
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.read_csv(OUT / "by_mode.csv").to_string(index=False))
    table = pd.read_csv(OUT / "by_mode_class.csv")
    print("Class/modes below90 even in optimistic candidate ceiling:")
    print(table[table.rowwise_candidate_ceiling_percent.lt(90)].to_string(index=False))


if __name__ == "__main__":
    main()
