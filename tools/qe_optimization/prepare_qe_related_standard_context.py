"""Label-free related-standard descriptors only; makes NO annotation decisions."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from context import WORKSPACE, REPO
from frozen_resources import configure_frozen_resources

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OLD = QE / "standard_context_preparation_v1"
OUT = QE / "related_standard_preparation_v1"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def initialize():
    ready = json.loads((OLD / "summary.json").read_text())
    assert ready["all_complete"] and ready["dataset_mask_tables"] == 363
    projects = json.loads((QE / "baseline/run_manifest.json").read_text())["datasets"]
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
        paths = [
            OLD / "summary.json",
            OLD / "protocol.json",
            Path(__file__).with_name("frozen_resources.py"),
        ]
        for dataset in projects:
            paths.append(QE / "baseline/runs" / dataset / "full_annotation.pkl")
            receipt = OLD / "checkpoints" / (dataset + ".json")
            paths.append(receipt)
            for previous, digest in json.loads(receipt.read_text())[
                "prepared_sha256"
            ].items():
                previous = Path(previous)
                assert sha(previous) == digest
                paths.extend(
                    [
                        previous,
                        OLD
                        / "measurements"
                        / previous.parent.name
                        / (dataset + ".pkl"),
                    ]
                )
        dump(
            OUT / "protocol.json",
            dict(
                purpose="Label-free preparation ONLY; no annotation decisions or manual references. "
                "Same physical candidates and existing 15 exact-standard predictors, adding16 "
                "separate RELATED-standard descriptors. No structural identity/RT order assumed.",
                datasets=33,
                expected_dataset_mask_tables=363,
                new_features=16,
                supervised_labels_used_for_predictors=False,
                resource_receipt=configure_frozen_resources(snapshot),
                source_sha256={
                    p.relative_to(snapshot).as_posix(): sha(p)
                    for p in snapshot.rglob("*")
                    if p.is_file()
                },
                input_sha256={str(p): sha(p) for p in paths},
            ),
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    for p, digest in protocol["source_sha256"].items():
        assert sha(OUT / "source_snapshot" / p) == digest
    for p, digest in protocol["input_sha256"].items():
        assert sha(Path(p)) == digest
    assert (
        configure_frozen_resources(OUT / "source_snapshot")
        == protocol["resource_receipt"]
    )
    return projects


def prepare(dataset):
    receipt = OUT / "checkpoints" / (dataset + ".json")
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        for p, digest in saved["prepared_sha256"].items():
            assert sha(Path(p)) == digest
        return saved
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.declared_standard_context import (
        declared_standard_context_evidence,
    )
    from matlab_annotation.related_standard_context import (
        PREFIX,
        related_standard_context_evidence,
    )
    from matlab_annotation.evidence_training import validate_feature_scope

    baseline = pd.read_pickle(QE / "baseline/runs" / dataset / "full_annotation.pkl")
    old_receipt = json.loads((OLD / "checkpoints" / (dataset + ".json")).read_text())
    records, files = [], {}
    for previous, digest in sorted(old_receipt["prepared_sha256"].items()):
        previous = Path(previous)
        assert sha(previous) == digest
        old = pd.read_pickle(previous)
        measured = pd.read_pickle(
            OLD / "measurements" / previous.parent.name / (dataset + ".pkl")
        )
        assert not {
            "training_positive",
            "training_label_known",
            "same_manual_peak",
        } & set(old["pool"])
        assert old["query_group"] == dataset[:7]
        assert dataset[:7] not in old["standard_reference_training_groups"]
        assert dataset[:7] in old["standard_reference_excluded_groups"]
        # Independently saved measurements were not hashed in old receipts.
        # Reproduce ALL 15 old descriptor values against their hashed tables,
        # verify measured raw coordinates, and freeze measured inputs now.
        exact = declared_standard_context_evidence(old["pool"], measured)
        pd.testing.assert_frame_equal(old["evidence"][exact.columns], exact)
        if len(measured):
            for stream, part in measured.groupby("stream"):
                raw = baseline["streams"][stream]["features"].copy()
                raw.index = raw.feature_id.astype(str)
                assert raw.index.is_unique
                matched = raw.loc[part.feat_feature_id.astype(str)]
                for source, target in (
                    ("mz", "feat_mz"),
                    ("rt", "feat_rt"),
                    ("intensity", "feat_intensity"),
                ):
                    np.testing.assert_allclose(
                        matched[source].to_numpy(float),
                        part[target].to_numpy(float),
                        rtol=1e-12,
                        atol=1e-12,
                    )
        context = related_standard_context_evidence(old["pool"], measured)
        assert len(context.columns) == 16
        assert all(pd.api.types.is_numeric_dtype(context[c]) for c in context)
        assert all(c.startswith(PREFIX) for c in context)
        assert not any(str(c).startswith(PREFIX) for c in old["evidence"])
        data = dict(old, evidence=pd.concat([old["evidence"], context], axis=1))
        pd.testing.assert_frame_equal(data["pool"], old["pool"])
        pd.testing.assert_frame_equal(
            data["evidence"][old["evidence"].columns], old["evidence"]
        )
        validate_feature_scope(
            data,
            old["standard_reference_training_groups"],
            old["standard_reference_excluded_groups"],
            query_is_training=False,
        )
        folder = OUT / "prepared" / previous.parent.name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (dataset + ".pkl")
        data["related_standard_measurements_sha256"] = sha(
            OLD / "measurements" / previous.parent.name / (dataset + ".pkl")
        )
        pd.to_pickle(data, path)
        files[str(path)] = sha(path)
        records.append(
            dict(
                dataset=dataset,
                excluded_folds=previous.parent.name,
                candidate_rows=len(old["pool"]),
                rows_with_related_context=int(
                    context[PREFIX + "related_compounds"].gt(0).sum()
                ),
                pool_exact_parity=True,
                old_predictor_exact_parity=True,
                exact_standard_context_reproduced=True,
                raw_standard_coordinates_checked=True,
                new_numeric_features=16,
                supervised_labels_used_for_predictors=False,
            )
        )
    assert len(records) == 11
    saved = dict(dataset=dataset, tables=records, prepared_sha256=files)
    dump(receipt, saved)
    return saved


def main():
    projects = initialize()
    (OUT / "checkpoints").mkdir(exist_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for saved in executor.map(prepare, projects):
            records.extend(saved["tables"])
            print("LABEL-FREE RELATED CONTEXT", saved["dataset"], flush=True)
    assert len(records) == 363
    pd.DataFrame(records).to_csv(OUT / "by_dataset_mask.csv", index=False)
    dump(
        OUT / "summary.json",
        dict(
            all_complete=True,
            datasets=33,
            dataset_mask_tables=363,
            new_numeric_features=16,
            existing_exact_standard_features=15,
            original_pool_exact_parity_checks=363,
            original_predictor_exact_parity_checks=363,
            standard_query_exclusion_checks=363,
            supervised_labels_used_for_predictors=False,
            annotation_changes=False,
        ),
    )
    print("363 TABLES PREPARED; NO ANNOTATIONS PRODUCED", flush=True)


if __name__ == "__main__":
    main()
