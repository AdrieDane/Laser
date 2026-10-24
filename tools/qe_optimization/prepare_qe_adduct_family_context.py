"""Label-free native-adduct companion measurements; no annotation decisions."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from context import WORKSPACE, REPO
from frozen_resources import configure_frozen_resources

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OLD = QE / "standard_context_preparation_v1"
OUT = QE / "adduct_family_preparation_v1"
KEYS = ["stream", "class", "adduct", "ref_mz", "feat_feature_id"]


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
                assert sha(Path(previous)) == digest
                paths.append(Path(previous))
        dump(
            OUT / "protocol.json",
            dict(
                purpose="Label-free same-acquisition adduct-aware companion measurements ONLY. "
                "No manual references, new candidates, wider windows or annotation decisions. "
                "Keep all029 physical pools/old predictors and independent STD scopes exact. "
                "Native review-only ions are evidence only, never new accepted identities.",
                datasets=33,
                expected_dataset_mask_tables=363,
                new_features=12,
                supervised_labels_used_for_predictors=False,
                partner_mass_ppm=3,
                partner_RT_seconds=2,
                minimum_profile_pairs=4,
                profile_support_correlation=0.9,
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
        assert sha(Path(saved["measurement_path"])) == saved["measurement_sha256"]
        return saved
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.adduct_family_evidence import PREFIX, adduct_family_evidence
    from matlab_annotation.evidence_training import validate_feature_scope

    baseline = pd.read_pickle(QE / "baseline/runs" / dataset / "full_annotation.pkl")
    old_receipt = json.loads((OLD / "checkpoints" / (dataset + ".json")).read_text())
    tables = {}
    for previous, digest in sorted(old_receipt["prepared_sha256"].items()):
        path = Path(previous)
        assert sha(path) == digest
        old = pd.read_pickle(path)
        assert not {
            "training_positive",
            "training_label_known",
            "same_manual_peak",
        } & set(old["pool"])
        assert old["query_group"] == dataset[:7]
        assert dataset[:7] not in old["standard_reference_training_groups"]
        assert dataset[:7] in old["standard_reference_excluded_groups"]
        tables[path] = old
    assert len(tables) == 11
    # Measurements depend ONLY on this query's raw acquisition and native
    # chemistry. Reusing identical physical queries across nested masks is safe;
    # no masked RT/prior/score quantities enter these measurements.
    union = pd.concat([t["pool"][KEYS] for t in tables.values()], ignore_index=True)
    union = union.drop_duplicates(KEYS).reset_index(drop=True)
    contexts = []
    for stream, part in union.groupby("stream", sort=False):
        contexts.append(
            adduct_family_evidence(
                part,
                baseline["streams"][stream]["features"],
                OUT / "source_snapshot/reference_data/matlab_pipeline",
                stream=stream,
            )
        )
    context = pd.concat(contexts).reindex(union.index)
    assert len(context.columns) == 12 and context.index.equals(union.index)
    assert all(
        c.startswith(PREFIX) and pd.api.types.is_numeric_dtype(context[c])
        for c in context
    )
    lookup = context.copy()
    lookup.index = pd.MultiIndex.from_frame(union[KEYS])
    assert lookup.index.is_unique
    folder = OUT / "measurements"
    folder.mkdir(exist_ok=True)
    measurement = folder / (dataset + ".pkl")
    pd.to_pickle(dict(queries=union, evidence=context), measurement)
    records, files = [], {}
    for previous, old in tables.items():
        extra = lookup.reindex(pd.MultiIndex.from_frame(old["pool"][KEYS]))
        extra.index = old["pool"].index
        assert extra[PREFIX + "source_configured"].notna().all()
        assert not any(str(c).startswith(PREFIX) for c in old["evidence"])
        data = dict(old, evidence=pd.concat([old["evidence"], extra], axis=1))
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
        data["adduct_family_measurements_sha256"] = sha(measurement)
        pd.to_pickle(data, path)
        files[str(path)] = sha(path)
        records.append(
            dict(
                dataset=dataset,
                excluded_folds=previous.parent.name,
                candidate_rows=len(old["pool"]),
                rows_with_profile_support=int(
                    extra[PREFIX + "profile_supported_ions"].gt(0).sum()
                ),
                pool_exact_parity=True,
                old_predictor_exact_parity=True,
                new_numeric_features=12,
                supervised_labels_used_for_predictors=False,
            )
        )
    saved = dict(
        dataset=dataset,
        tables=records,
        prepared_sha256=files,
        physical_queries_measured=len(union),
        measurement_path=str(measurement),
        measurement_sha256=sha(measurement),
    )
    dump(receipt, saved)
    return saved


def main():
    projects = initialize()
    (OUT / "checkpoints").mkdir(exist_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for saved in executor.map(prepare, projects):
            records.extend(saved["tables"])
            print(
                "LABEL-FREE ADDUCT CONTEXT",
                saved["dataset"],
                saved["physical_queries_measured"],
                flush=True,
            )
    assert len(records) == 363
    pd.DataFrame(records).to_csv(OUT / "by_dataset_mask.csv", index=False)
    dump(
        OUT / "summary.json",
        dict(
            all_complete=True,
            datasets=33,
            dataset_mask_tables=363,
            new_numeric_features=12,
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
