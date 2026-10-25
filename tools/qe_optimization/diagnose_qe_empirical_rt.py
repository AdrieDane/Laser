"""Numerical RT diagnostic ONLY: no candidates, scores, calls or exports changed.

Fixed prospective residual settings; five group-held-out numerical references.
Evaluate known manual RT compositions as diagnostic targets, NOT as runtime
candidate identities. Never interpret within10seconds as annotation accuracy.
"""

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from context import REPO, WORKSPACE
from frozen_resources import configure_frozen_resources
from run_qe_standard_mode_experiment import sha, dump

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "empirical_RT_reference_diagnostic_v1"
OBSERVATIONS = QE / "rt_atlas_development/observations.pkl"
FOLDS = QE / "rt_atlas_development/fold_assignment.json"
BASE = QE / "baseline"
REFS = QE / "nested_reference_development_v2"


def main():
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads(FOLDS.read_text())
    assert len(projects) == 33
    if not (OUT / "protocol.json").exists():
        OUT.mkdir(exist_ok=False)
        snapshot = OUT / "source_snapshot"
        snapshot.mkdir()
        for name in ("app", "methods", "reference_data"):
            shutil.copytree(
                REPO / name,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        inputs = [
            OBSERVATIONS,
            FOLDS,
            BASE / "run_manifest.json",
            Path(__file__).with_name("frozen_resources.py"),
            Path(__file__).with_name("run_qe_standard_mode_experiment.py"),
            Path(__file__).with_name("run_qe_atlas_experiment.py"),
        ]
        inputs.extend(REFS / f"exclude_{f}.json" for f in range(5))
        inputs.extend(BASE / "runs" / d / "full_annotation.pkl" for d in projects)
        protocol = dict(
            purpose="Numerical RT diagnostic ONLY; no annotation changes, classifier fits, candidates or acceptance. NOT annotation performance,90%coverage or deployment validation.",
            settings="Exact-class/CDB residuals from OTHER-project mapped endogenous observations,one value per project/class/CDB. Parent-atlas bounded class-balanced affine normalization. >=3projects,MAD<=5s,absmedian<=30s,shrink n/(n+3). SAMEother-class automaticwarp/leave-CDB-out localalignment. Unsupported compositions keep parent polynomial. No own query curation in fitting/alignment; curator compositions are diagnostic evaluation locations ONLY.",
            datasets=33,
            project_groups=30,
            numerical_reference_fits=5,
            prediction_scope="Use mapped endogenous reference observations only as numerical RT evaluation targets. X4excluded by user;unresolved rows cannot have an assessed originalRT,so absent here but NOT removed from full annotation benchmark denominators. Known prior development exposure;not untouched validation.",
            source_snapshot_sha256={
                p.relative_to(snapshot).as_posix(): sha(p)
                for p in snapshot.rglob("*")
                if p.is_file()
            },
            input_sha256={str(p): sha(p) for p in inputs},
        )
        dump(OUT / "protocol.json", protocol)
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    for path, digest in protocol["source_snapshot_sha256"].items():
        assert sha(OUT / "source_snapshot" / path) == digest
    for path, digest in protocol["input_sha256"].items():
        assert sha(Path(path)) == digest
    configure_frozen_resources(OUT / "source_snapshot")
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.empirical_retention import (
        fit_empirical_retention,
        predict_aligned_empirical_retention,
    )
    from matlab_annotation.retention_atlas import predict_aligned_retention_atlas

    observations = pd.read_pickle(OBSERVATIONS)
    assert observations.cohort.eq(observations.dataset.str[:7]).all()
    assert set(observations.dataset) == set(projects)
    bundles, models = {}, {}
    # Freeze ALL numerical reference fits before evaluating a held-out RT.
    for outer in range(5):
        bundle = json.loads((REFS / f"exclude_{outer}.json").read_text())
        allowed = {d for d in projects if folds[d] != outer}
        assert set(bundle["training_datasets"]) == allowed
        train = observations[
            observations.dataset.isin(allowed) & observations.mapped.astype(bool)
        ].copy()
        assert not set(train.cohort) & set(bundle["excluded_groups"])
        model = fit_empirical_retention(
            train,
            bundle["atlas"],
            training_groups=bundle["training_groups"],
            excluded_groups=bundle["excluded_groups"],
        )
        dump(OUT / f"fold_{outer}_numerical_reference.json", model)
        models[outer], bundles[outer] = model, bundle
    records = []
    for dataset in projects:
        outer = folds[dataset]
        model, atlas = models[outer], bundles[outer]["atlas"]
        assert (
            dataset[:7] in model["excluded_groups"]
            and dataset[:7] not in model["training_groups"]
        )
        baseline = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        query = observations[
            observations.dataset.eq(dataset) & observations.mapped.astype(bool)
        ].copy()
        query["class"] = query["class"].map(canonical_lipid_class)
        query = query[~query["class"].eq("X4")]
        for stream, part in query.groupby("stream", sort=False):
            keys = ["stream", "class", "carbon", "db"]
            targets = part[keys].drop_duplicates().reset_index(drop=True)
            automatic = baseline["streams"][stream]["annotations"]
            before = predict_aligned_retention_atlas(
                targets, automatic, atlas, stream=stream
            )
            after = predict_aligned_empirical_retention(
                targets, automatic, atlas, model, stream=stream, query_group=dataset[:7]
            )
            lookup = targets.copy()
            lookup["expected_before"] = before.atlas_rt_seconds
            lookup["expected_after"] = after.atlas_rt_seconds
            lookup["empirical_reference_available"] = (
                after.empirical_offset_seconds.notna()
            )
            detail = part.merge(lookup, on=keys, validate="many_to_one")
            for side in ("before", "after"):
                expected = detail["expected_" + side]
                detail["available_" + side] = np.isfinite(expected) & expected.gt(0)
                detail["absolute_error_" + side] = (expected - detail.rt_seconds).abs()
                detail["within_10_" + side] = detail["available_" + side] & detail[
                    "absolute_error_" + side
                ].le(10)
            detail["gained_within_10"] = (
                ~detail.within_10_before & detail.within_10_after
            )
            detail["lost_within_10"] = detail.within_10_before & ~detail.within_10_after
            records.append(detail)
        print("NUMERICAL RT DIAGNOSTIC", dataset, flush=True)
    all_rows = pd.concat(records, ignore_index=True)
    all_rows.to_pickle(OUT / "private_numerical_predictions.pkl")
    for name, keys in (
        ("by_dataset_mode", ["dataset", "stream"]),
        ("by_mode_class", ["stream", "class"]),
        ("by_dataset_mode_class", ["dataset", "stream", "class"]),
        ("by_mode", ["stream"]),
    ):
        result = all_rows.groupby(keys, as_index=False).agg(
            assessed_original_RTs=("mapped", "size"),
            empirical_reference_available=("empirical_reference_available", "sum"),
            predictions_available_before=("available_before", "sum"),
            predictions_available_after=("available_after", "sum"),
            within_10_seconds_before=("within_10_before", "sum"),
            within_10_seconds_after=("within_10_after", "sum"),
            newly_within_10_seconds=("gained_within_10", "sum"),
            no_longer_within_10_seconds=("lost_within_10", "sum"),
            median_absolute_RT_error_before=("absolute_error_before", "median"),
            median_absolute_RT_error_after=("absolute_error_after", "median"),
        )
        result.to_csv(OUT / (name + ".csv"), index=False)
    summary = dict(
        annotation_changes=False,
        classifier_fits=0,
        numerical_reference_fits=5,
        assessed_original_RTs=len(all_rows),
        datasets=all_rows.dataset.nunique(),
        dataset_modes=all_rows.groupby(["dataset", "stream"]).ngroups,
        empirical_reference_available=int(all_rows.empirical_reference_available.sum()),
        within_10_seconds_before=int(all_rows.within_10_before.sum()),
        within_10_seconds_after=int(all_rows.within_10_after.sum()),
        newly_within_10_seconds=int(all_rows.gained_within_10.sum()),
        no_longer_within_10_seconds=int(all_rows.lost_within_10.sum()),
        source_hash_checks=len(protocol["source_snapshot_sha256"]),
        input_hash_checks=len(protocol["input_sha256"]),
        note=protocol["purpose"] + " " + protocol["prediction_scope"],
    )
    assert summary["datasets"] == 33 and summary["dataset_modes"] == 132
    dump(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
