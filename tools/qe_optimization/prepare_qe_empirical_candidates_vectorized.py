"""036 preparation v2: equivalent vectorized RTlookup; scientific rules unchanged."""

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
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
REFERENCES = QE / "nested_reference_development_v2"
EMPIRICAL = QE / "nested_empirical_RT_references_v1"
OUT = QE / "empirical_RT_candidate_preparation_v2"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def initialize():
    ready = json.loads((OLD / "summary.json").read_text())
    assert ready["all_complete"] and ready["dataset_mask_tables"] == 363
    empirical_ready = json.loads((EMPIRICAL / "summary.json").read_text())
    assert (
        empirical_ready["all_complete"] and empirical_ready["reference_bundles"] == 25
    )
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
        inputs = [
            OLD / "summary.json",
            OLD / "protocol.json",
            REFERENCES / "summary.json",
            Path(__file__).with_name("frozen_resources.py"),
        ]
        inputs.extend(REFERENCES.glob("exclude_*.json"))
        inputs.extend(EMPIRICAL.glob("exclude_*.json"))
        inputs.extend([EMPIRICAL / "summary.json", EMPIRICAL / "protocol.json"])
        inputs.extend((QE / "nested_standard_development_v1").glob("exclude_*.json"))
        for d in projects:
            receipt = OLD / "checkpoints" / (d + ".json")
            inputs.extend(
                [
                    receipt,
                    QE / "baseline/runs" / d / "full_annotation.pkl",
                    QE
                    / "nested_standard_development_v1/automatic_inputs"
                    / (d + ".pkl"),
                ]
            )
            for p, digest in json.loads(receipt.read_text())["prepared_sha256"].items():
                p = Path(p)
                assert sha(p) == digest
                inputs.extend([p, OLD / "measurements" / p.parent.name / (d + ".pkl")])
            inputs.extend((QE / "experiment_004/raw_evidence" / d).glob("*.pkl"))
        dump(
            OUT / "protocol.json",
            dict(
                engineering_note="Version2 vectorizes deterministic empirical RTlookup ONLY;no crossquery/model prediction cache. Exact full-table equivalence against ALLcompleted scalarv1datasets required BEFOREannotationfits. Incompletev1preserved,no v1scorefits. AllRT/physics/references unchanged.",
                purpose="Label-free empirical composition RTunion. Preserve all029physical candidates/observed coordinates; appendONLYrawabsppm<3/RT<=10s original-feature pairs. Same native+old-envelope/ranges/adducts. No query manual labels, rarecatalogue or annotation decisions.",
                dataset_mask_tables=363,
                datasets=33,
                numerical_workers=3,
                predictor_policy="Recompute029competition/context on expanded pools. Preserve old identities/coordinates;reproduce15exactSTDfields. Add8numeric empiricalRTdescriptors. NO033standard-RT/034classcut/035familymodels.",
                query_scope="Same363query/outer/innerexclusion masks. References and measuredSTD "
                "context match each mask,ownproject excluded. ExistingblindAUTOanchors only.",
                annotation_changes=False,
                supervised_labels_used_for_predictors=False,
                resource_receipt=configure_frozen_resources(snapshot),
                source_sha256={
                    p.relative_to(snapshot).as_posix(): sha(p)
                    for p in snapshot.rglob("*")
                    if p.is_file()
                },
                input_sha256={str(p): sha(p) for p in inputs},
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
    return protocol, projects


def prepare(dataset):
    receipt = OUT / "checkpoints" / (dataset + ".json")
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        for p, digest in saved["prepared_sha256"].items():
            assert sha(Path(p)) == digest
        return saved
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.chemical_targets import compile_additional_targets
    from matlab_annotation.empirical_candidates import (
        prepare_empirical_candidates,
        empirical_retention_evidence,
    )
    from matlab_annotation.candidate_preparation import enrich_candidate_pools, _quality
    from matlab_annotation.declared_standard_context import (
        declared_standard_context_evidence,
        PREFIX,
        COLUMNS,
    )
    from matlab_annotation.evidence_training import validate_feature_scope

    result = pd.read_pickle(QE / "baseline/runs" / dataset / "full_annotation.pkl")
    staged = pd.read_pickle(
        QE / "nested_standard_development_v1/automatic_inputs" / (dataset + ".pkl")
    )
    declared = staged["declared"]
    aligned_context = {}
    old_receipt = json.loads((OLD / "checkpoints" / (dataset + ".json")).read_text())
    physical, priors, old_tables, bundles, measurements, audits, new_counts = (
        {},
        {},
        {},
        {},
        {},
        [],
        {},
    )
    for previous in sorted(old_receipt["prepared_sha256"]):
        previous = Path(previous)
        name = previous.parent.name
        old = pd.read_pickle(previous)
        assert not {
            "training_positive",
            "training_label_known",
            "same_manual_peak",
            "manual_feature_id",
        } & set(old["pool"])
        bundle = json.loads((REFERENCES / (name + ".json")).read_text())
        assert (
            dataset[:7] not in bundle["training_groups"]
            and dataset[:7] in bundle["excluded_groups"]
        )
        assert set(old["reference_training_groups"]) == set(bundle["training_groups"])
        assert set(old["reference_excluded_groups"]) == set(bundle["excluded_groups"])
        assert set(old["standard_reference_training_groups"]) == set(
            bundle["training_groups"]
        )
        assert set(old["standard_reference_excluded_groups"]) == set(
            bundle["excluded_groups"]
        )
        old_tables[name], bundles[name] = old, bundle
        measurements[name] = pd.read_pickle(
            OLD / "measurements" / name / (dataset + ".pkl")
        )
        exact = declared_standard_context_evidence(old["pool"], measurements[name])
        pd.testing.assert_frame_equal(old["evidence"][exact.columns], exact)
        standard_path = QE / "nested_standard_development_v1" / (name + ".json")
        standard_model = json.loads(standard_path.read_text())
        assert sha(standard_path) == old["standard_reference_sha256"]
        assert set(standard_model["training_groups"]) == set(bundle["training_groups"])
        assert set(standard_model["excluded_groups"]) == set(bundle["excluded_groups"])
        assert standard_model["atlas"] == bundle["atlas"]
        empirical_path = EMPIRICAL / (name + ".json")
        empirical_bundle = json.loads(empirical_path.read_text())
        assert empirical_bundle["parent_reference_sha256"] == sha(
            REFERENCES / (name + ".json")
        )
        assert empirical_bundle["training_groups"] == bundle["training_groups"]
        assert empirical_bundle["excluded_groups"] == bundle["excluded_groups"]
        assert empirical_bundle["excluded_folds"] == bundle["excluded_folds"]
        empirical = empirical_bundle["empirical_reference"]
        assert empirical["training_groups"] == bundle["training_groups"]
        assert empirical["excluded_groups"] == bundle["excluded_groups"]
        extras, aligned_context[name] = [], {}

        for stream, payload in result["streams"].items():
            local_declared = declared[declared.stream.eq(stream)].copy()
            generated, ranges, audit = compile_additional_targets(
                payload["reference"],
                bundle["composition_envelope"],
                stream=stream,
                declared_standards=local_declared,
            )
            targets = pd.concat([payload["reference"], generated], ignore_index=True)
            targets["chemical_target_generated"] = (
                targets.get(
                    "chemical_target_generated", pd.Series(False, index=targets.index)
                )
                .fillna(False)
                .astype(bool)
            )
            extra, chemical_targets, predictions, parent = prepare_empirical_candidates(
                targets,
                payload["features"],
                payload["annotations"],
                local_declared,
                measurements[name][measurements[name].stream.eq(stream)].copy(),
                bundle["atlas"],
                empirical,
                stream=stream,
                query_group=dataset[:7],
                ranges=ranges,
            )
            aligned_context[name][stream] = (chemical_targets, predictions, parent)
            audits.append(
                dict(
                    dataset=dataset,
                    excluded_folds=name,
                    stream=stream,
                    targets=len(chemical_targets),
                    supported_aligned_targets=int(
                        predictions.atlas_rt_seconds.notna().sum()
                    ),
                    physical_candidates=len(extra),
                )
            )
            if len(extra):
                extra = _quality(extra, payload["features"], spreads=True)
                assert extra.ppm_error.abs().lt(3).all()
                assert (extra.rt_error.abs() * 60).le(10 + 1e-9).all()
                extras.append(extra)
        old_pool = old["pool"]
        expanded = pd.concat([old_pool, *extras], ignore_index=True)
        keys = ["stream", "name", "adduct", "feat_feature_id"]
        expanded = expanded.drop_duplicates(keys, keep="first").reset_index(drop=True)
        new_counts[name] = len(expanded) - len(old_pool)
        # Enrichment must recompute crossmode descriptors, not reuse stale
        # values or count new hypotheses as independent evidence.
        expanded = expanded.drop(
            columns=[c for c in expanded if c.startswith("q_cross_mode_")]
        )
        physical[name] = expanded
        priors[name] = bundle["composition_priors"]
    quality = {}
    for p in (QE / "experiment_004/raw_evidence" / dataset).glob("*.pkl"):
        stream, charge = p.stem.rsplit("_z", 1)
        frame = pd.read_pickle(p).set_index("feature_id")
        quality[(stream, int(charge))] = frame[[c for c in frame if c.startswith("q_")]]
    prepared, _ = enrich_candidate_pools(
        physical, result, priors, raw_quality_cache=quality
    )
    records, files = [], {}
    coords = [
        "stream",
        "name",
        "adduct",
        "feat_feature_id",
        "feat_mz",
        "feat_rt",
        "feat_intensity",
    ]
    for name, data in prepared.items():
        old, bundle = old_tables[name], bundles[name]
        keys = ["stream", "name", "adduct", "feat_feature_id"]
        present = pd.MultiIndex.from_frame(data["pool"][keys])
        desired = pd.MultiIndex.from_frame(physical[name][keys])
        assert present.is_unique and desired.is_unique
        positions = present.get_indexer(desired)
        assert np.all(positions >= 0) and len(positions) == len(present)
        # Per-stream enrichment groups new rows with their own mode. Restore
        # original old-rows-first ordering jointly for pool AND predictors.
        data["pool"] = data["pool"].iloc[positions].reset_index(drop=True)
        data["evidence"] = data["evidence"].iloc[positions].reset_index(drop=True)
        old_rows = len(old["pool"])
        pd.testing.assert_frame_equal(
            data["pool"].iloc[:old_rows][coords].reset_index(drop=True),
            old["pool"][coords].reset_index(drop=True),
            check_dtype=False,
        )
        context = declared_standard_context_evidence(data["pool"], measurements[name])
        assert list(context) == [PREFIX + c for c in COLUMNS]
        data["evidence"] = pd.concat([data["evidence"], context], axis=1)
        assert set(data["evidence"]) == set(old["evidence"]), (
            dataset,
            name,
            "Unexpected predictor-schema change",
        )
        data["evidence"] = data["evidence"].reindex(columns=old["evidence"].columns)
        additions = []
        for stream, part in data["pool"].groupby("stream", sort=False):
            targets, predictions, parent = aligned_context[name][stream]
            additions.append(
                empirical_retention_evidence(part, targets, predictions, parent)
            )
        additions = pd.concat(additions).reindex(data["pool"].index)
        assert len(additions.columns) == 8
        assert all(pd.api.types.is_numeric_dtype(additions[c]) for c in additions)
        data["evidence"] = pd.concat([data["evidence"], additions], axis=1)
        for key in (
            "query_dataset",
            "query_group",
            "excluded_folds",
            "reference_training_groups",
            "reference_excluded_groups",
            "reference_sha256",
            "standard_reference_training_groups",
            "standard_reference_excluded_groups",
            "standard_reference_sha256",
        ):
            data[key] = old[key]
        empirical_path = EMPIRICAL / (name + ".json")
        data["empirical_reference_training_groups"] = bundle["training_groups"]
        data["empirical_reference_excluded_groups"] = bundle["excluded_groups"]
        data["empirical_reference_sha256"] = sha(empirical_path)
        validate_feature_scope(
            data,
            bundle["training_groups"],
            bundle["excluded_groups"],
            query_is_training=False,
        )
        folder = OUT / "prepared" / name
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / (dataset + ".pkl")
        pd.to_pickle(data, p)
        files[str(p)] = sha(p)
        records.append(
            dict(
                dataset=dataset,
                excluded_folds=name,
                old_candidate_rows=old_rows,
                added_candidate_rows=new_counts[name],
                total_candidate_rows=len(data["pool"]),
                original_candidates_and_coordinates_preserved=True,
                old_predictor_schema_preserved=True,
                new_numeric_features=8,
                query_exclusion_checked=True,
                supervised_labels_used_for_predictors=False,
            )
        )
    assert len(records) == 11
    pd.DataFrame(audits).to_csv(OUT / "audits" / (dataset + ".csv"), index=False)
    saved = dict(dataset=dataset, tables=records, prepared_sha256=files)
    dump(receipt, saved)
    return saved


def main():
    protocol, projects = initialize()
    (OUT / "checkpoints").mkdir(exist_ok=True)
    (OUT / "audits").mkdir(exist_ok=True)
    records = []
    remaining = iter(projects)
    executor = ProcessPoolExecutor(max_workers=protocol["numerical_workers"])
    pending = {
        executor.submit(prepare, next(remaining))
        for _ in range(protocol["numerical_workers"])
    }
    try:
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            results = [future.result() for future in done]
            for saved in results:
                records.extend(saved["tables"])
                print(
                    "EMPIRICAL-RT CANDIDATES PREPARED",
                    saved["dataset"],
                    sum(t["added_candidate_rows"] for t in saved["tables"]),
                    "candidate rows across11masks;NOTannotations",
                    flush=True,
                )
                d = next(remaining, None)
                if d is not None:
                    pending.add(executor.submit(prepare, d))
    except BaseException:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    assert len(records) == 363
    pd.DataFrame(records).to_csv(OUT / "by_dataset_mask.csv", index=False)
    dump(
        OUT / "summary.json",
        dict(
            all_complete=True,
            datasets=33,
            dataset_mask_tables=363,
            original_candidate_coordinate_parity_checks=363,
            predictor_schema_checks=363,
            query_exclusion_checks=363,
            supervised_labels_used_for_predictors=False,
            annotation_changes=False,
            new_numeric_features=8,
        ),
    )
    print("363LABEL-FREE TABLES COMPLETE;NOANNOTATIONS", flush=True)


if __name__ == "__main__":
    main()
