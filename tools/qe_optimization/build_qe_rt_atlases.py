"""Grouped held-out RT-reference development, NOT an annotation experiment."""

from __future__ import annotations

import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from context import WORKSPACE as ROOT, REPO, SOURCE

BASELINE = ROOT / "outputs/qe_optimization_20261022/baseline"
OUT = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
AUDIT = ROOT / "outputs/qe_dataset_audit_20261022/checkpoints"
sys.path[:0] = [str(REPO), str(REPO / "app")]
from tools.dataset_scanner.core import curated_rows
from matlab_annotation.retention_atlas import (
    fit_retention_atlas,
    predict_retention_atlas,
    predict_aligned_retention_atlas,
)


def observations():
    path = OUT / "observations.pkl"
    if path.exists():
        return pd.read_pickle(path)
    rows = []
    for project in sorted(SOURCE.iterdir()):
        if not project.is_dir():
            continue
        checkpoint = json.loads((AUDIT / (project.name + ".json")).read_text())
        expected = next(
            f["sha256"] for f in checkpoint["files"] if f["name"] == "MS_data.xlsx"
        )
        with (project / "MS_data.xlsx").open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == expected, (
                project
            )
        mapping = {
            (p["source_sheet"], p["source_excel_row"]): p for p in checkpoint["proofs"]
        }
        manual, _ = curated_rows(project / "MS_data.xlsx")
        for row in manual:
            if row["name"].endswith(" (IS)") or row["name"].startswith("IS: "):
                continue
            match = re.search(r"\((?:[dtm]|[OP]-|O\+P-)?(\d+):(\d+)\)", row["name"])
            if not match:
                continue
            c, db = map(int, match.groups())
            proof = mapping[(row["sheet"], row["excel_row"])]
            rows.append(
                dict(
                    dataset=project.name,
                    cohort=project.name[:7],
                    stream=row["stream"],
                    name=row["name"],
                    **{"class": row["lipid_class"]},
                    carbon=c,
                    db=db,
                    rt_seconds=row["rt_seconds"],
                    adduct=row["adduct"],
                    original_feature_id=proof["original_feature_id"],
                    mapped=proof["matched_original_feature"],
                )
            )
        print("READ TRAINING/EVALUATION REFERENCE", project.name, flush=True)
    result = pd.DataFrame(rows)
    result.to_pickle(path)
    result.to_csv(OUT / "observations.csv", index=False)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    diagnostic = OUT / "diagnostics_class_local_v2"
    diagnostic.mkdir(exist_ok=True)
    data = observations()
    projects = data[["dataset", "cohort"]].drop_duplicates().reset_index(drop=True)
    assert len(projects) == 33
    folds = {}
    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=5).split(projects, groups=projects.cohort)
    ):
        train_groups = set(projects.iloc[train_idx].cohort)
        test_groups = set(projects.iloc[test_idx].cohort)
        assert not train_groups & test_groups
        test_datasets = list(projects.iloc[test_idx].dataset)
        path = OUT / f"fold_{fold}_atlas.json"
        if path.exists():
            payload = json.loads(path.read_text())
            assert set(payload["training_groups"]) == train_groups
        else:
            train = data[data.cohort.isin(train_groups) & data.mapped].copy()
            atlas = fit_retention_atlas(train)
            payload = dict(
                fold=fold,
                training_groups=sorted(train_groups),
                test_groups=sorted(test_groups),
                held_out_datasets=test_datasets,
                atlas=atlas,
                source_sha256=hashlib.sha256(
                    (REPO / "app/matlab_annotation/retention_atlas.py").read_bytes()
                ).hexdigest(),
                note="Held-out groups are excluded from all RT model fits; no annotation outputs changed.",
            )
            path.write_text(
                json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
            )
            print(
                "FIT FOLD",
                fold,
                "models",
                len(atlas["models"]),
                "held-out groups",
                len(test_groups),
                flush=True,
            )
        for dataset in test_datasets:
            folds[dataset] = fold
    (OUT / "fold_assignment.json").write_text(
        json.dumps(folds, indent=2), encoding="utf-8"
    )
    predictions = []
    for dataset, fold in sorted(folds.items()):
        run = BASELINE / "runs" / dataset
        if not (run / "complete.json").exists():
            continue
        atlas = json.loads((OUT / f"fold_{fold}_atlas.json").read_text())["atlas"]
        frame = data[data.dataset.eq(dataset)].copy()
        pred = predict_retention_atlas(frame, atlas)
        frame = frame.join(pred)
        anchors = pd.read_csv(run / "automatic_annotations.csv", low_memory=False)
        frame["aligned_prediction_seconds"] = np.nan
        frame["alignment_status"] = "unavailable"
        for stream, group in frame.groupby("stream"):
            aligned = predict_aligned_retention_atlas(
                group[["stream", "class", "carbon", "db"]],
                anchors[anchors.stream.eq(stream)],
                atlas,
                stream=stream,
            )
            frame.loc[group.index, "aligned_prediction_seconds"] = (
                aligned.atlas_rt_seconds
            )
            frame.loc[group.index, "alignment_status"] = aligned.atlas_global_status
        frame["raw_atlas_error_seconds"] = frame.rt_seconds - frame.atlas_rt_seconds
        frame["aligned_atlas_error_seconds"] = (
            frame.rt_seconds - frame.aligned_prediction_seconds
        )
        predictions.append(frame)
        errors = frame.aligned_atlas_error_seconds.dropna().abs()
        print(
            "RT DIAGNOSTIC ONLY",
            dataset,
            "assessed",
            len(errors),
            "median",
            round(float(errors.median()), 2),
            "within10s",
            round(float(errors.le(10).mean() * 100), 1),
            "%",
            flush=True,
        )
    if predictions:
        combined = pd.concat(predictions, ignore_index=True)
        combined.to_csv(diagnostic / "held_out_RT_predictions.csv", index=False)
        rows = []
        for key, group in combined.groupby(["stream", "class"]):
            errors = group.aligned_atlas_error_seconds.dropna().abs()
            rows.append(
                dict(
                    stream=key[0],
                    lipid_class=key[1],
                    manual_rows=len(group),
                    predicted=len(errors),
                    within_10_seconds=int(errors.le(10).sum()),
                    median_absolute_RT_seconds=float(errors.median())
                    if len(errors)
                    else None,
                    percent_within_10_seconds_among_assessed=float(
                        errors.le(10).mean() * 100
                    )
                    if len(errors)
                    else None,
                )
            )
        pd.DataFrame(rows).to_csv(diagnostic / "held_out_RT_by_class.csv", index=False)
    print(
        "RT ATLAS PREPARATION FINISHED; NOT AN ANNOTATION-PERFORMANCE REPORT",
        flush=True,
    )


if __name__ == "__main__":
    main()
