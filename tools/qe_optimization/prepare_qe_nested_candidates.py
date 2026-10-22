"""Prepare label-FREE nested predictors only; no classifier/annotation changes."""

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from context import WORKSPACE as ROOT, REPO
from frozen_resources import configure_frozen_resources

QE = ROOT / "outputs/qe_optimization_20261022"
MODELS = QE / "nested_reference_development_v2"
PARITY = QE / "nested_preparation_parity_v1"
OUT = QE / "nested_candidate_preparation_v2"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def initialize():
    proof = json.loads((PARITY / "summary.json").read_text())
    assert proof["all_passed"] and proof["dataset_fold_checks"] == 165
    models = json.loads((MODELS / "summary.json").read_text())
    assert models["all_complete"] and models["reference_bundles"] == 25
    for path, digest in json.loads(
        (PARITY / "source_manifest.json").read_text()
    ).items():
        assert sha(Path(path)) == digest, (
            path,
            "Preparation differs from verified parity source",
        )
    projects = json.loads((QE / "baseline/run_manifest.json").read_text())["datasets"]
    if not (OUT / "protocol.json").exists():
        OUT.mkdir(exist_ok=False)
        snapshot = OUT / "source_snapshot"
        snapshot.mkdir()
        for name in ("app", "reference_data", "methods"):
            shutil.copytree(
                REPO / name,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        sources = {
            p.relative_to(snapshot).as_posix(): sha(p)
            for p in snapshot.rglob("*")
            if p.is_file()
        }
        paths = [
            Path(__file__).with_name("frozen_resources.py"),
            PARITY / "summary.json",
            MODELS / "summary.json",
            QE / "rt_atlas_development/fold_assignment.json",
        ]
        paths.extend(MODELS / name for name in models["model_sha256"])
        for dataset in projects:
            paths.append(QE / "baseline/runs" / dataset / "full_annotation.pkl")
            paths.extend((QE / "experiment_004/raw_evidence" / dataset).glob("*.pkl"))
            # Only the one correctly held-out cache is read for schema/parity.
            fold = json.loads(
                (QE / "rt_atlas_development/fold_assignment.json").read_text()
            )[dataset]
            paths.append(
                QE / "experiment_007/prepared" / f"fold_{fold}" / (dataset + ".pkl")
            )
        protocol = dict(
            purpose="LABEL-FREE PREDICTOR PREPARATION ONLY, not annotation performance. Each query project excluded from its RT/prior/envelope. No manual comparison or acceptance threshold is used for generation. Legacy007 cached supervised flags are read ONLY in parity caches, ignored and never attached to new predictors. Inherited baseline prior exposure remains.",
            datasets=33,
            expected_dataset_mask_tables=363,
            numerical_workers=3,
            source_sha256=sources,
            resource_receipt=configure_frozen_resources(snapshot),
            replaces="v1 preserved after failed candidate parity: missing bundled assets caused installed PE30-48 fallback instead of frozen PE28-50. No annotation scoring/acceptance occurred; discard v1 tables,including its two completed datasets.",
            input_sha256={str(p): sha(p) for p in paths},
        )
        (OUT / "protocol.json").write_text(
            json.dumps(protocol, indent=2), encoding="utf-8"
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
    protocol = json.loads((OUT / "protocol.json").read_text())
    for path, digest in protocol["source_sha256"].items():
        assert sha(OUT / "source_snapshot" / path) == digest, path
    for path, digest in protocol["input_sha256"].items():
        assert sha(Path(path)) == digest, path
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    assert (
        configure_frozen_resources(OUT / "source_snapshot")
        == protocol["resource_receipt"]
    )
    return protocol, projects


def prepare(dataset):
    receipt = OUT / "checkpoints" / (dataset + ".json")
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        for path, digest in saved["prepared_sha256"].items():
            assert sha(Path(path)) == digest, path
        return saved
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.candidate_preparation import (
        prepare_candidate_pool,
        enrich_candidate_pools,
    )
    from matlab_annotation.nested_feature_folds import plan_nested_feature_folds

    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    plan = plan_nested_feature_folds(folds, {d: d[:7] for d in folds})
    masks = [m for m in plan["models"] if folds[dataset] in m]
    assert len(masks) == 11
    result = pd.read_pickle(QE / "baseline/runs" / dataset / "full_annotation.pkl")
    bundles, physical, priors = {}, {}, {}
    for mask in masks:
        name = "exclude_" + "_".join(map(str, mask))
        bundle = json.loads((MODELS / (name + ".json")).read_text())
        assert (
            dataset in bundle["excluded_datasets"]
            and dataset[:7] in bundle["excluded_groups"]
        )
        assert dataset[:7] not in bundle["training_groups"]
        assert set(bundle["training_groups"]) == set(
            plan["models"][mask]["training_groups"]
        )
        bundles[mask] = bundle
        priors[mask] = bundle["composition_priors"]
        physical[mask] = prepare_candidate_pool(
            result, bundle["atlas"], bundle["composition_envelope"]
        )
    quality = {}
    for path in (QE / "experiment_004/raw_evidence" / dataset).glob("*.pkl"):
        stream, charge = path.stem.rsplit("_z", 1)
        frame = pd.read_pickle(path).set_index("feature_id")
        quality[(stream, int(charge))] = frame[[c for c in frame if c.startswith("q_")]]
    tables, _ = enrich_candidate_pools(
        physical, result, priors, raw_quality_cache=quality
    )
    single = (folds[dataset],)
    original = pd.read_pickle(
        QE / "experiment_007/prepared" / f"fold_{folds[dataset]}" / (dataset + ".pkl")
    )
    keys = ["stream", "name", "adduct", "feat_feature_id"]
    # One-block model equals the existing held-out model: exact query ORDER and
    # predictor parity, not just membership, protects stable competition ties.
    pd.testing.assert_frame_equal(
        tables[single]["pool"][keys].reset_index(drop=True),
        original["pool"][keys].reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        tables[single]["evidence"].reindex(columns=original["evidence"].columns),
        original["evidence"],
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    records, files = [], {}
    for mask, data in tables.items():
        pool, evidence = data["pool"], data["evidence"]
        assert not {
            "training_positive",
            "training_label_known",
            "same_manual_peak",
            "manual_feature_id",
        } & set(pool)
        assert set(evidence) == set(original["evidence"]), (
            dataset,
            mask,
            "Predictor schema changed",
        )
        data["evidence"] = evidence.reindex(columns=original["evidence"].columns)
        data["query_dataset"] = dataset
        data["query_group"] = dataset[:7]
        data["excluded_folds"] = list(mask)
        data["reference_training_groups"] = bundles[mask]["training_groups"]
        data["reference_excluded_groups"] = bundles[mask]["excluded_groups"]
        name = "exclude_" + "_".join(map(str, mask))
        data["reference_sha256"] = sha(MODELS / (name + ".json"))
        folder = OUT / "prepared" / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (dataset + ".pkl")
        pd.to_pickle(data, path)
        files[str(path)] = sha(path)
        records.append(
            dict(
                dataset=dataset,
                excluded_folds="_".join(map(str, mask)),
                excluded_blocks=len(mask),
                reference_training_groups=len(bundles[mask]["training_groups"]),
                candidate_rows=len(pool),
                query_project_excluded=True,
                supervised_labels_used_for_predictors=False,
                legacy_label_flags_ignored_in_parity_cache=True,
                original_single_fold_parity=(len(mask) == 1),
            )
        )
    saved = dict(
        dataset=dataset,
        tables=records,
        prepared_sha256=files,
        note="Private original-query tables. No supervised labels attached to predictors or annotation selections. Legacy parity-cache flags ignored. Do not upload/commit.",
    )
    receipt.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return saved


def main():
    protocol, projects = initialize()
    (OUT / "checkpoints").mkdir(exist_ok=True)
    records = []
    # Keep only one job per worker in flight. On a parity failure, do not
    # silently drain a queue containing the entire remaining collection.
    executor = ProcessPoolExecutor(max_workers=protocol["numerical_workers"])
    remaining = iter(projects)
    pending = {
        executor.submit(prepare, d)
        for d in list(projects)[: protocol["numerical_workers"]]
    }
    for _ in range(len(pending)):
        next(remaining)
    try:
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            # Resolve all completed jobs before submitting anything else.
            completed = [future.result() for future in done]
            for result in completed:
                records.extend(result["tables"])
                print(
                    "NESTED PREDICTORS PREPARED",
                    result["dataset"],
                    len(result["tables"]),
                    "tables",
                    flush=True,
                )
                dataset = next(remaining, None)
                if dataset is not None:
                    pending.add(executor.submit(prepare, dataset))
    except BaseException:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    assert len(records) == 363
    table = pd.DataFrame(records).sort_values(["dataset", "excluded_folds"])
    table.to_csv(OUT / "by_dataset_mask.csv", index=False)
    summary = dict(
        datasets=33,
        dataset_mask_tables=363,
        all_complete=True,
        supervised_labels_used_for_predictors=False,
        legacy_label_flags_ignored_in_parity_cache=True,
        query_exclusion_checks=363,
        single_fold_candidate_and_evidence_parity_checks=33,
        annotation_changes=False,
        note=protocol["purpose"],
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
