"""All25reference/363layer scalar verification; no heldout query labels."""

from concurrent.futures import ProcessPoolExecutor
import json

import pandas as pd

from frozen_resources import configure_frozen_resources
import matrix_nested_data as data
import matrix_peer_independent as oracle


def independent_training(mask):
    datasets, folds = data.pilot.inventory()
    allowed = {d[:7] for d in datasets if folds[d] not in mask}
    excluded = {d[:7] for d in datasets if folds[d] in mask}
    rows, count = [], 0
    for dataset in datasets:
        if folds[dataset] in mask:
            continue
        assert dataset[:7] in allowed and dataset[:7] not in excluded
        m = pd.read_pickle(
            data.BASE / "evaluation_v2" / dataset / "manual_comparison.pkl"
        )
        count += 1
        m = m.loc[
            ~m.internal_standard & ~m.lipid_class.eq("X4"),
            ["stream", "species", "lipid_class"],
        ]
        parts = m.species.astype(str).str.extract(
            r"^[^()]+\((?:[dtm]|(?:O|P)-|O\+P-)?([0-9]+):([0-9]+)\)$"
        )
        for index, r in m.iterrows():
            if pd.isna(parts.loc[index, 0]):
                continue
            rows.append(
                dict(
                    dataset=dataset,
                    cohort=dataset[:7],
                    stream=r.stream,
                    lipid_class=r.lipid_class,
                    carbon=int(parts.loc[index, 0]),
                    db=int(parts.loc[index, 1]),
                )
            )
    return pd.DataFrame(rows), allowed, excluded, count


def reference(mask):
    configure_frozen_resources(data.OUT / "source_snapshot")
    rows, allowed, excluded, count = independent_training(mask)
    folder = data.OUT / "models" / data.mask_name(mask)
    pd.testing.assert_frame_equal(
        rows, pd.read_pickle(folder / "authorized_compositions.pkl"), check_exact=True
    )
    signatures = json.loads((data.pilot.OUT / "native_signatures.json").read_text())
    expected = oracle.fit_scalar(
        rows,
        {d: s for d, s in signatures.items() if d[:7] in allowed},
        training_groups=allowed,
        excluded_groups=excluded,
    )
    data.pilot.same_model(expected, json.loads((folder / "model.json").read_text()))
    audit = pd.read_csv(folder / "training_role_audit.csv")
    assert (
        len(audit) == count
        and audit.authorized_training.all()
        and set(audit.cohort) == allowed
    )
    if len(mask) == 1:
        data.pilot.same_model(
            expected,
            json.loads(
                (
                    data.pilot.OUT / "models" / f"fold_{mask[0]}" / "model.json"
                ).read_text()
            ),
        )
    print("INDEPENDENT NESTED MATRIX REFERENCE", mask, count, flush=True)
    return count


def query(dataset):
    configure_frozen_resources(data.OUT / "source_snapshot")
    _, folds = data.pilot.inventory()
    signature = json.loads((data.pilot.OUT / "native_signatures.json").read_text())[
        dataset
    ]
    result = []
    for mask in data.original.masks_for(folds[dataset]):
        original = data.original.load_predictors(dataset, mask)
        loaded = data.load_predictors(dataset, mask)
        model = json.loads(
            (data.OUT / "models" / data.mask_name(mask) / "model.json").read_text()
        )
        pool = original["pool"]
        columns = ["stream", "class", "carbon", "db", "is_is"]
        chemical = pool[columns].drop_duplicates()
        scalar = oracle.predict_scalar(
            chemical, signature, model, query_group=dataset[:7]
        )

        def key(values):
            return tuple(None if pd.isna(v) else v for v in values)

        lookup = {
            key(row): scalar.loc[i].to_numpy(float) for i, row in chemical.iterrows()
        }
        expected = pd.DataFrame(
            [
                lookup[key(row)]
                for row in pool[columns].itertuples(index=False, name=None)
            ],
            index=pool.index,
            columns=scalar.columns,
        )
        actual = pd.read_pickle(data.layer_path(dataset, mask))["evidence"]
        pd.testing.assert_frame_equal(
            actual, expected, check_exact=False, rtol=0, atol=1e-12
        )
        pd.testing.assert_frame_equal(loaded["pool"], pool, check_exact=True)
        pd.testing.assert_frame_equal(
            loaded["evidence"][original["evidence"].columns],
            original["evidence"],
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            loaded["evidence"][scalar.columns], actual, check_exact=True
        )
        assert actual.loc[pool.is_is].isna().all().all()
        if len(mask) == 1:
            pd.testing.assert_frame_equal(
                actual,
                pd.read_pickle(data.pilot.OUT / "private" / (dataset + ".pkl")),
                check_exact=True,
            )
        result.append(
            dict(
                dataset=dataset,
                excluded_folds=sorted(mask),
                candidate_rows=len(actual),
                all_passed=True,
            )
        )
    assert len(result) == 11
    print("ALL11 INDEPENDENT MATRIX QUERY LAYERS PASS", dataset, flush=True)
    return result


def main():
    configure_frozen_resources(data.OUT / "source_snapshot")
    receipt_path = data.OUT / "blind_receipt.json"
    blind = json.loads(receipt_path.read_text())
    assert (
        blind["all_passed"]
        and blind["query_tables"] == 363
        and blind["query_gold_validation_tables_opened"] == 0
    )
    assert (
        blind["source_input_sha256"] == data.inputs()
        and blind["numeric_runtime"] == data.pilot.environment()
    )
    for path, h in blind["artifact_sha256"].items():
        assert data.pilot.sha(path) == h
    data.CHECK.mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=2) as executor:
        uses = list(executor.map(reference, data.models()))
    assert len(uses) == 25 and sum(uses) == 462
    datasets, _ = data.pilot.inventory()
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(query, datasets))
    rows = [r for dataset in results for r in dataset]
    assert len(rows) == 363 and blind["source_input_sha256"] == data.inputs()
    data.pilot.dump(data.CHECK / "by_query_mask.json", rows)
    data.pilot.dump(
        data.CHECK / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            query_tables=363,
            reference_masks=25,
            authorized_training_table_uses=462,
            all_original_pool_predictors_exact=True,
            independent_numeric_tolerance=1e-12,
            singlefold_numeric_parities=33,
            blind_receipt_sha256=data.pilot.sha(receipt_path),
            source_input_sha256=blind["source_input_sha256"],
            query_gold_validation_tables_opened=0,
            annotation_changes=False,
            classifier_fits=0,
            thresholds_selected=False,
        ),
    )
    print(
        "ALL25/363 NESTED MATRIX CHECKS PASS; ACTUAL METHOD STILL NEEDS OWN PLAN",
        flush=True,
    )


if __name__ == "__main__":
    main()
