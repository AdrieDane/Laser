"""Read-only paired training/held-out feature audit; NOT annotation accuracy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import WORKSPACE as ROOT

QE = ROOT / "outputs/qe_optimization_20261022"
POOL = QE / "experiment_007"
OUT = QE / "training_feature_shift_v1"
METRICS = ("q_absolute_atlas_RT_error_seconds", "composition_joint_frequency",
           "atlas_training_cohorts", "atlas_training_mad_seconds")


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def summarize(frame, keys):
    records = []
    for group, part in frame.groupby(keys, sort=True):
        if not isinstance(group, tuple):
            group = (group,)
        for metric in METRICS:
            a = pd.to_numeric(part[metric + "_training"], errors="coerce").to_numpy(float)
            b = pd.to_numeric(part[metric + "_heldout"], errors="coerce").to_numpy(float)
            good = np.isfinite(a) & np.isfinite(b)
            record = dict(zip(keys, group), metric=metric, matched_queries=len(part),
                          training_nonmissing=int(np.isfinite(a).sum()), heldout_nonmissing=int(np.isfinite(b).sum()),
                          paired_nonmissing=int(good.sum()))
            if good.any():
                a, b = a[good], b[good]
                record.update(training_median=float(np.median(a)), heldout_median=float(np.median(b)),
                              training_p90=float(np.quantile(a, .9)), heldout_p90=float(np.quantile(b, .9)),
                              median_paired_heldout_minus_training=float(np.median(b - a)),
                              fraction_heldout_greater=float((b > a + 1e-9).mean()))
            records.append(record)
    return pd.DataFrame(records)


def main():
    OUT.mkdir(exist_ok=False)
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    keys = ["stream", "name", "adduct", "feat_feature_id"]
    pairs, availability, receipts = [], [], []
    for dataset, heldout in folds.items():
        frames = {}
        for fold in range(5):
            path = POOL / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
            digest = sha(path)
            data = pd.read_pickle(path)
            positive = data["pool"].training_positive.astype(bool)
            raw = data["pool"].loc[positive, keys].copy()
            raw["feat_feature_id"] = raw.feat_feature_id.astype(str)
            evidence = data["evidence"].loc[positive]
            raw["lipid_class"] = evidence["class"]
            for metric in METRICS:
                raw[metric] = evidence[metric]
            assert not raw.duplicated(keys).any()
            frames[fold] = raw
            assert sha(path) == digest
            receipts.append(dict(dataset=dataset, fold=fold, prepared_sha256=digest))
        test = frames[heldout]
        for fold, train in frames.items():
            if fold == heldout:
                continue
            merged = train.merge(test, on=keys, how="outer", suffixes=("_training", "_heldout"),
                                 validate="1:1", indicator=True)
            for stream, part in merged.groupby("stream", sort=True):
                availability.append(dict(dataset=dataset, stream=stream, training_fold=fold, heldout_fold=heldout,
                    present_both=int(part._merge.eq("both").sum()), training_only=int(part._merge.eq("left_only").sum()),
                    heldout_only=int(part._merge.eq("right_only").sum())))
            common = merged[merged._merge.eq("both")].copy()
            assert common.lipid_class_training.equals(common.lipid_class_heldout)
            common["lipid_class"] = common.lipid_class_heldout
            common["dataset"] = dataset
            pairs.append(common[["dataset", "stream", "lipid_class", *[m + suffix for m in METRICS for suffix in ("_training", "_heldout")]]])
        print("FEATURE SHIFT AUDITED", dataset, flush=True)
    rows = pd.concat(pairs, ignore_index=True)
    summarize(rows, ["stream"]).to_csv(OUT / "by_mode.csv", index=False)
    summarize(rows, ["stream", "lipid_class"]).to_csv(OUT / "by_mode_class.csv", index=False)
    summarize(rows, ["dataset", "stream"]).to_csv(OUT / "by_dataset_mode.csv", index=False)
    available = pd.DataFrame(availability)
    available.groupby("stream")[["present_both", "training_only", "heldout_only"]].sum().to_csv(OUT / "candidate_availability.csv")
    summary = dict(
        purpose="READ-ONLY diagnostic, not new annotations or proof of a single cause. Same correctly mapped manual lipid/adduct/original-feature queries paired between the one007 atlas excluding their project fold and four atlases that include it (excluding another fold). New score/threshold performance is NOT measured. No per-feature/profile tables written.",
        selection_limit="Feature comparisons condition on candidate presence in BOTH caches. Missing-candidate counts reported separately. Four comparisons per dataset; pairs are NOT independent new datasets or annotations.",
        datasets=33, dataset_modes=132, matched_pair_rows=len(rows), source_sha256=sha(Path(__file__)), input_receipts=receipts)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.read_csv(OUT / "by_mode.csv").to_string(index=False))


if __name__ == "__main__":
    main()
