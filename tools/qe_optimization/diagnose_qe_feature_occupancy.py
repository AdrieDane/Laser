"""Read-only009 feature-occupancy diagnosis; never changes annotation choices.

Curated rows are diagnostic labels ONLY. No predicted assignment, model, cutoff
or gain is produced. Only aggregate counts are exported, never original IDs.
"""

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from context import WORKSPACE
from run_qe_atlas_experiment import species, adduct

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "feature_occupancy_diagnostic_v1"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    destination = parser.parse_args().output.resolve()
    projects = json.loads((QE / "baseline/run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    destination.mkdir(exist_ok=False)
    counts, inputs = [], {}
    for dataset in projects:
        manual_path = QE / "baseline/evaluation_v2" / dataset / "manual_comparison.pkl"
        auto_path = QE / "experiment_009/runs" / dataset / "automatic_annotations.csv"
        pool_path = (
            QE
            / "experiment_007/prepared"
            / f"fold_{folds[dataset]}"
            / (dataset + ".pkl")
        )
        for path in (manual_path, auto_path, pool_path):
            inputs[str(path)] = sha(path)
        manual = pd.read_pickle(manual_path)
        manual = manual[~manual.internal_standard].copy()
        auto = pd.read_csv(
            auto_path,
            usecols=["stream", "name", "adduct", "feat_feature_id"],
            dtype={"feat_feature_id": str},
            low_memory=False,
        )
        auto["species"] = auto.name.map(species)
        auto["adduct_key"] = auto.adduct.map(adduct)
        keys = ["stream", "species", "adduct_key", "feat_feature_id"]
        accepted = set(auto[keys].itertuples(index=False, name=None))
        selected = {}
        occupants = {}
        for stream, name, ion, fid in accepted:
            selected.setdefault((stream, name, ion), set()).add(fid)
            occupants.setdefault((stream, fid), set()).add((name, ion))
        pool = pd.read_pickle(pool_path)["pool"]
        available = set(
            (s, species(n), adduct(a), str(fid))
            for s, n, a, fid in pool[
                ["stream", "name", "adduct", "feat_feature_id"]
            ].itertuples(index=False, name=None)
        )
        mapped = manual[manual.mapped].copy()
        # Count distinct curator identities per original feature. Two different
        # identities on one feature cannot both pass a one-label-per-feature
        # selector. This is not proof which curator identity is chemically true.
        identity_counts = mapped.groupby(["stream", "feature_id"]).apply(
            lambda g: len(
                set(g[["species", "adduct"]].itertuples(index=False, name=None))
            ),
            include_groups=False,
        )
        records = {}
        for row in manual.to_dict("records"):
            stream, name, ion = row["stream"], row["species"], row["adduct"]
            group = (stream, row["lipid_class"])
            count = records.setdefault(
                group, Counter(manual_rows_sharing_feature_with_other_manual_identity=0)
            )
            count["manual_endogenous"] += 1
            if not row["mapped"]:
                count["unresolved"] += 1
                continue
            fid = str(row["feature_id"])
            key = (stream, name, ion, fid)
            if identity_counts.loc[(stream, row["feature_id"])] > 1:
                count["manual_rows_sharing_feature_with_other_manual_identity"] += 1
            if key in accepted:
                count["correct_original_feature"] += 1
                continue
            count["not_correct_original_feature"] += 1
            if key not in available:
                count["correct_feature_absent_from007_pool"] += 1
                continue
            count["correct_feature_available_but_not_selected"] += 1
            if (stream, name, ion) in selected:
                count["identity_selected_on_different_feature"] += 1
            competing = occupants.get((stream, fid), set()) - {(name, ion)}
            if competing:
                count["correct_feature_occupied_by_other_label"] += 1
                if (stream, name, ion) in selected:
                    count["wrong_feature_and_correct_feature_occupied"] += 1
            else:
                count["correct_feature_unoccupied"] += 1
        counts.extend(
            dict(dataset=dataset, stream=s, lipid_class=c, **dict(n))
            for (s, c), n in records.items()
        )
    frame = pd.DataFrame(counts).fillna(0)
    values = [c for c in frame if c not in {"dataset", "stream", "lipid_class"}]
    frame[values] = frame[values].astype(int)
    frame.to_csv(destination / "by_dataset_mode_class.csv", index=False)
    frame.groupby(["stream", "lipid_class"])[values].sum().to_csv(
        destination / "by_mode_class.csv"
    )
    frame.groupby("stream")[values].sum().to_csv(destination / "by_mode.csv")
    summary = dict(
        datasets=len(projects),
        counts={c: int(frame[c].sum()) for c in values},
        source_sha256=sha(Path(__file__)),
        input_sha256=inputs,
        annotation_changes=False,
        note="READ-ONLY endogenous009 diagnosis. Occupancy is conditional on existing009 selections; counts are NOT achievable gains or an assignment method. Sharing a mapped feature does not establish curator error or chemical truth. Standard occupants retained when measuring competition. Categories overlap: do not add as independent errors. No new predictions or acceptance decisions.",
    )
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "input_sha256"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
