"""Add nested independent declared-standard context; no annotation decisions."""

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from context import WORKSPACE, REPO
from frozen_resources import configure_frozen_resources

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OLD = QE / "nested_candidate_preparation_v2"
MODELS = QE / "nested_standard_development_v1"
OUT = QE / "standard_context_preparation_v1"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def initialize():
    old = json.loads((OLD / "summary.json").read_text())
    models = json.loads((MODELS / "summary.json").read_text())
    assert old["all_complete"] and old["dataset_mask_tables"] == 363
    assert models["all_complete"] and models["reference_bundles"] == 25
    assert models["single_fold_005_parity_checks"] == 5
    assert models["chemical_catalogue_parity_checks"] == 33
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
        paths = [
            Path(__file__).with_name("frozen_resources.py"),
            OLD / "summary.json",
            MODELS / "summary.json",
            MODELS / "protocol.json",
            QE / "rt_atlas_development/fold_assignment.json",
        ]
        model_protocol = json.loads((MODELS / "protocol.json").read_text())
        for path, digest in model_protocol["source_sha256"].items():
            assert sha(MODELS / "source_snapshot" / path) == digest
        for name, digest in models["model_sha256"].items():
            path = MODELS / name
            assert sha(path) == digest
            paths.append(path)
        for dataset in projects:
            receipt_path = OLD / "checkpoints" / (dataset + ".json")
            paths.extend(
                [
                    receipt_path,
                    QE / "baseline/runs" / dataset / "full_annotation.pkl",
                    MODELS / "automatic_inputs" / (dataset + ".pkl"),
                ]
            )
            # Full preparation data are label-free; no manual comparison opens.
            for path, digest in json.loads(receipt_path.read_text())[
                "prepared_sha256"
            ].items():
                assert sha(Path(path)) == digest
                paths.append(Path(path))
            staged = MODELS / "automatic_inputs" / (dataset + ".pkl")
            assert sha(staged) == model_protocol["input_sha256"][str(staged)]
        protocol = dict(
            purpose="LABEL-FREE standard-context preparation ONLY. Exact022 candidate pools and all old predictors immutable; add15numeric predictors from independently detected declared spikes. No proposed endo calls, query manual comparison or annotation selection used.",
            datasets=33,
            expected_dataset_mask_tables=363,
            numerical_workers=3,
            new_features=15,
            resource_receipt=configure_frozen_resources(snapshot),
            source_sha256={
                p.relative_to(snapshot).as_posix(): sha(p)
                for p in snapshot.rglob("*")
                if p.is_file()
            },
            input_sha256={str(p): sha(p) for p in paths},
            scope="Match standard-reference exclusion mask to each original query/outer/inner mask. Exact reference training/excluded groups required. No self/companion standard anchors. Raw uniqueness before occupancy. Full scientific resources frozen; inherited blind baseline exposure remains.",
        )
        dump(OUT / "protocol.json", protocol)
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
    protocol = json.loads((OUT / "protocol.json").read_text())
    for path, digest in protocol["source_sha256"].items():
        assert sha(OUT / "source_snapshot" / path) == digest
    for path, digest in protocol["input_sha256"].items():
        assert sha(Path(path)) == digest
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
            assert sha(Path(path)) == digest
        return saved
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.declared_standard_context import (
        independent_standard_measurements,
        declared_standard_context_evidence,
        COLUMNS,
        PREFIX,
    )
    from matlab_annotation.evidence_training import validate_feature_scope

    baseline = pd.read_pickle(QE / "baseline/runs" / dataset / "full_annotation.pkl")
    staged = pd.read_pickle(MODELS / "automatic_inputs" / (dataset + ".pkl"))
    catalogue = staged["declared"]
    old_receipt = json.loads((OLD / "checkpoints" / (dataset + ".json")).read_text())
    records, files = [], {}
    for previous in sorted(old_receipt["prepared_sha256"]):
        previous = Path(previous)
        name = previous.parent.name
        model_path = MODELS / (name + ".json")
        model = json.loads(model_path.read_text())
        old = pd.read_pickle(previous)
        assert old["query_dataset"] == dataset and old["query_group"] == dataset[:7]
        assert set(model["excluded_folds"]) == set(old["excluded_folds"])
        assert (
            dataset in model["excluded_datasets"]
            and dataset[:7] in model["excluded_groups"]
        )
        assert set(model["training_groups"]) == set(old["reference_training_groups"])
        assert set(model["excluded_groups"]) == set(old["reference_excluded_groups"])
        assert not {
            "training_positive",
            "training_label_known",
            "same_manual_peak",
            "manual_feature_id",
        } & set(old["pool"])
        assert not any(str(c).startswith(PREFIX) for c in old["evidence"])
        # Use precisely the staged blind AUTOMATIC alignment columns for which
        # model fitting was audited. Raw features come from the same baseline.
        blind = dict(baseline)
        blind["streams"] = {
            s: dict(payload, annotations=staged["automatic"][s])
            for s, payload in baseline["streams"].items()
        }
        measured, audit = independent_standard_measurements(blind, catalogue, model)
        context = declared_standard_context_evidence(old["pool"], measured)
        assert list(context) == [PREFIX + c for c in COLUMNS]
        assert all(pd.api.types.is_numeric_dtype(context[c]) for c in context)
        data = dict(old)
        data["evidence"] = pd.concat([old["evidence"], context], axis=1)
        data["standard_reference_training_groups"] = model["training_groups"]
        data["standard_reference_excluded_groups"] = model["excluded_groups"]
        data["standard_reference_sha256"] = sha(model_path)
        pd.testing.assert_frame_equal(data["pool"], old["pool"])
        pd.testing.assert_frame_equal(
            data["evidence"][old["evidence"].columns], old["evidence"]
        )
        assert len(data["evidence"].columns) == len(old["evidence"].columns) + 15
        validate_feature_scope(
            data,
            set(model["training_groups"]),
            set(model["excluded_groups"]),
            query_is_training=False,
        )
        folder = OUT / "prepared" / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (dataset + ".pkl")
        pd.to_pickle(data, path)
        files[str(path)] = sha(path)
        private = OUT / "measurements" / name
        private.mkdir(parents=True, exist_ok=True)
        measured.to_pickle(private / (dataset + ".pkl"))
        audit.to_pickle(private / (dataset + "_audit.pkl"))
        records.append(
            dict(
                dataset=dataset,
                excluded_folds="_".join(map(str, old["excluded_folds"])),
                reference_training_groups=len(model["training_groups"]),
                candidate_rows=len(old["pool"]),
                independently_detected_standard_ions=len(measured),
                query_rows_with_same_class_context=int(
                    context[PREFIX + "same_class_compounds"].gt(0).sum()
                ),
                query_rows_with_same_adduct_context=int(
                    context[PREFIX + "same_adduct_compounds"].gt(0).sum()
                ),
                known_spike_query_rows=int(context[PREFIX + "spike_feature"].sum()),
                new_numeric_features=15,
                original_pool_exact_parity=True,
                original_predictors_exact_parity=True,
                standard_query_exclusion_checked=True,
                supervised_labels_used_for_predictors=False,
            )
        )
    assert len(records) == 11
    saved = dict(
        dataset=dataset,
        tables=records,
        prepared_sha256=files,
        note="Private original features and independently measured standard context; no manual comparison labels or endogenous selection. Never commit/upload feature or model tables.",
    )
    dump(receipt, saved)
    return saved


def main():
    protocol, projects = initialize()
    (OUT / "checkpoints").mkdir(exist_ok=True)
    records = []
    executor = ProcessPoolExecutor(max_workers=protocol["numerical_workers"])
    remaining = iter(projects)
    pending = {
        executor.submit(prepare, next(remaining))
        for _ in range(protocol["numerical_workers"])
    }
    try:
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            complete = [future.result() for future in done]
            for saved in complete:
                records.extend(saved["tables"])
                print(
                    "STANDARD CONTEXT PREPARED",
                    saved["dataset"],
                    len(saved["tables"]),
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
    pd.DataFrame(records).sort_values(["dataset", "excluded_folds"]).to_csv(
        OUT / "by_dataset_mask.csv", index=False
    )
    summary = dict(
        datasets=33,
        dataset_mask_tables=363,
        all_complete=True,
        new_numeric_features=15,
        supervised_labels_used_for_predictors=False,
        original_pool_exact_parity_checks=363,
        original_predictor_exact_parity_checks=363,
        standard_query_exclusion_checks=363,
        annotation_changes=False,
        note=protocol["purpose"],
    )
    dump(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
