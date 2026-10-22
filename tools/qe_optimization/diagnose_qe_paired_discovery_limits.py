"""Read-only proposal-limit audit, NOT new annotation performance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from context import WORKSPACE as ROOT

QE = ROOT / "outputs/qe_optimization_20261022"
EXP = QE / "experiment_020"
OUT = QE / "paired_discovery_limits_v1"
sys.path.insert(0, str(EXP / "source_snapshot/app"))
from matlab_annotation.standard_rt_library import ion_key


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    projects = json.loads((EXP / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    frames, receipts = [], []
    for dataset in projects:
        path = EXP / "prepared" / f"fold_{folds[dataset]}" / (dataset + ".pkl")
        before = sha(path)
        pool = pd.read_pickle(path)["pool"].copy()
        old_path = QE / "experiment_009/runs" / dataset / "automatic_annotations.csv"
        old_sha = sha(old_path)
        old = pd.read_csv(old_path, dtype={"feat_feature_id": str}, low_memory=False)
        identities = {(s, str(n), ion_key(a)) for s, n, a in old[
            ["stream", "name", "adduct"]].itertuples(index=False, name=None)}
        occupied = set(old[["stream", "feat_feature_id"]].itertuples(index=False, name=None))
        pool["candidate_rows"] = 1
        pool["novel_rows"] = pool.discovery_novel_to007.astype(bool)
        pool["novel_manual_feature_rows"] = pool.novel_rows & pool.training_positive
        pool["novel_manual_rows_with_pair_gate"] = (
            pool.novel_manual_feature_rows & pool.q_paired_best_log_correlation.ge(.8)
            & pool.q_paired_profile_specimens.ge(8))
        free = [
            (s, str(n), ion_key(a)) not in identities and (s, str(fid)) not in occupied
            for s, n, a, fid in pool[["stream", "name", "adduct", "feat_feature_id"]].itertuples(index=False, name=None)
        ]
        pool["novel_manual_rows_pair_gate_unoccupied_absent_identity"] = (
            pool.novel_manual_rows_with_pair_gate & pd.Series(free, index=pool.index))
        pool["dataset"] = dataset
        frames.append(pool)
        assert before == sha(path) and old_sha == sha(old_path)
        receipts.append(dict(dataset=dataset, prepared_sha256=before, old_calls_sha256=old_sha))
    rows = pd.concat(frames, ignore_index=True)
    metrics = ["candidate_rows", "novel_rows", "novel_manual_feature_rows",
               "novel_manual_rows_with_pair_gate", "novel_manual_rows_pair_gate_unoccupied_absent_identity"]
    for name, keys in (("by_dataset_mode", ["dataset", "stream"]),
                       ("by_mode", ["stream"]), ("by_mode_class", ["stream", "class"])):
        table = rows.groupby(keys)[metrics].sum().astype(int)
        if name == "by_dataset_mode":
            table = table.reindex(pd.MultiIndex.from_product(
                [projects, ["RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"]], names=keys), fill_value=0)
        table.reset_index().to_csv(OUT / (name + ".csv"), index=False)
    summary = dict(
        purpose="READ-ONLY physical proposal/eligibility diagnosis. Counts of manual features considered are NOT correct accepted annotations or promised achievable gains. Score and BOTH identity/feature competition apply after these gates.",
        datasets=33, dataset_modes=132, counts=rows[metrics].sum().astype(int).to_dict(),
        source_sha256=sha(Path(__file__)), input_receipts=receipts,
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["counts"], indent=2))
    print(pd.read_csv(OUT / "by_mode.csv").to_string(index=False))


if __name__ == "__main__":
    main()
