"""All25nested references/363numeric descriptor layers; no annotation fits."""

from concurrent.futures import ProcessPoolExecutor
import json
import shutil

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import matrix_nested_data as data
import matrix_peer_composition as peer


def fit(mask):
    configure_frozen_resources(data.OUT / "source_snapshot")
    rows, allowed, excluded, reads = data.training_rows(mask)
    signatures = json.loads((data.pilot.OUT / "native_signatures.json").read_text())
    model = peer.fit_reference(
        rows,
        {d: s for d, s in signatures.items() if d[:7] in allowed},
        training_groups=allowed,
        excluded_groups=excluded,
    )
    folder = data.OUT / "models" / data.mask_name(mask)
    folder.mkdir(parents=True, exist_ok=False)
    rows.to_pickle(folder / "authorized_compositions.pkl")
    pd.DataFrame(reads).to_csv(folder / "training_role_audit.csv", index=False)
    data.pilot.dump(folder / "model.json", model)
    if len(mask) == 1:
        old = data.pilot.OUT / "models" / f"fold_{mask[0]}" / "model.json"
        data.pilot.same_model(model, json.loads(old.read_text()))
    print(
        "MATCHING NESTED MATRIX REFERENCE",
        mask,
        len(reads),
        len(model["classes"]),
        flush=True,
    )
    return dict(
        excluded_folds=list(mask),
        training_dataset_uses=len(reads),
        class_mode_models=len(model["classes"]),
    )


def prepare(dataset):
    configure_frozen_resources(data.OUT / "source_snapshot")
    _, folds = data.pilot.inventory()
    signatures = json.loads((data.pilot.OUT / "native_signatures.json").read_text())
    rows = []
    for mask in data.original.masks_for(folds[dataset]):
        reference_path = data.OUT / "models" / data.mask_name(mask) / "model.json"
        model = json.loads(reference_path.read_text())
        original = data.original.load_predictors(dataset, mask)
        before_pool = original["pool"].copy(deep=True)
        before_evidence = original["evidence"].copy(deep=True)
        all_datasets, all_folds = data.pilot.inventory()
        allowed = {d[:7] for d in all_datasets if all_folds[d] not in mask}
        excluded = {d[:7] for d in all_datasets if all_folds[d] in mask}
        data.validate_scope(original, model, allowed, excluded, query_is_training=False)
        evidence = peer.predict(
            original["pool"], signatures[dataset], model, query_group=dataset[:7]
        )
        pd.testing.assert_frame_equal(original["pool"], before_pool, check_exact=True)
        pd.testing.assert_frame_equal(
            original["evidence"], before_evidence, check_exact=True
        )
        if len(mask) == 1:
            pd.testing.assert_frame_equal(
                evidence,
                pd.read_pickle(data.pilot.OUT / "private" / (dataset + ".pkl")),
                check_exact=True,
            )
        folder = data.layer_path(dataset, mask).parent
        folder.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(
            dict(
                dataset=dataset,
                excluded_folds=sorted(mask),
                source_pool_sha256=data.pilot.sha(
                    data.original.prepared_path(dataset, mask)
                ),
                reference_sha256=data.pilot.sha(reference_path),
                evidence=evidence,
            ),
            data.layer_path(dataset, mask),
        )
        loaded = data.load_predictors(dataset, mask)
        pd.testing.assert_frame_equal(
            loaded["evidence"][before_evidence.columns],
            before_evidence,
            check_exact=True,
        )
        pd.testing.assert_frame_equal(loaded["pool"], before_pool, check_exact=True)
        rows.append(
            dict(
                dataset=dataset,
                excluded_folds=sorted(mask),
                candidate_rows=len(evidence),
                supported_rows=int(evidence.iloc[:, 0].notna().sum()),
                old_pool_predictors_exact=True,
                layer_sha256=data.pilot.sha(data.layer_path(dataset, mask)),
            )
        )
    assert len(rows) == 11
    print("ALL11 MATRIX NESTED QUERY LAYERS", dataset, flush=True)
    return rows


def main():
    checked = json.loads((data.pilot.CHECK / "receipt.json").read_text())
    assert (
        checked["all_passed"]
        and checked["datasets"] == 33
        and checked["query_gold_validation_tables_opened"] == 0
    )
    assert checked["source_input_sha256"] == data.pilot.inputs()
    hashes = data.inputs()
    data.OUT.mkdir(exist_ok=False)
    snapshot = data.OUT / "source_snapshot"
    snapshot.mkdir()
    for folder in ("app", "methods", "reference_data"):
        shutil.copytree(
            REPO / folder,
            snapshot / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
    resources = configure_frozen_resources(snapshot)
    data.pilot.dump(
        data.OUT / "protocol.json",
        dict(
            source_input_sha256=hashes,
            numeric_runtime=data.pilot.environment(),
            resources=resources,
            numerical_workers=2,
            reference_masks=25,
            query_tables=363,
            new_numeric_columns=8,
            old_columns=108,
            new_columns=116,
            classifier_fits=0,
            annotation_changes=False,
            policy="EXACT frozen matrix pilot descriptors, matching each own/outer/inner reference mask. Delta layers only; no outer-only descriptor reuse. Other-project composition TRAINING roles only, no query validation labels.",
        ),
    )
    with ProcessPoolExecutor(max_workers=2) as executor:
        models = list(executor.map(fit, data.models()))
    assert len(models) == 25 and sum(m["training_dataset_uses"] for m in models) == 462
    datasets, _ = data.pilot.inventory()
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(prepare, datasets))
    rows = [r for dataset in results for r in dataset]
    assert len(rows) == 363 and hashes == data.inputs()
    data.pilot.dump(data.OUT / "by_reference_mask.json", models)
    data.pilot.dump(data.OUT / "by_query_mask.json", rows)
    artifacts = {
        str(p): data.pilot.sha(p)
        for p in data.OUT.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    data.pilot.dump(
        data.OUT / "blind_receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            dataset_modes=132,
            reference_masks=25,
            query_tables=363,
            authorized_training_table_uses=462,
            singlefold_numeric_parities=33,
            all_old_pool_predictors_exact=True,
            source_input_sha256=hashes,
            artifact_sha256=artifacts,
            numeric_runtime=data.pilot.environment(),
            query_gold_validation_tables_opened=0,
            annotation_changes=False,
            classifier_fits=0,
            thresholds_selected=False,
        ),
    )
    print(
        "ALL363 NESTED MATRIX NUMERIC LAYERS COMPLETE; INDEPENDENT CHECK REQUIRED",
        flush=True,
    )


if __name__ == "__main__":
    main()
