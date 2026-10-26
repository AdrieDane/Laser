"""All33 separate scalar composition-reference/signature/descriptor checks."""

import json

import pandas as pd

from frozen_resources import configure_frozen_resources
import matrix_peer_pilot_data as data
import matrix_peer_independent as oracle
import matrix_peer_composition as peer


def training_scalar(outer):
    datasets, folds = data.inventory()
    allowed = {d[:7] for d in datasets if folds[d] != outer}
    excluded = {d[:7] for d in datasets if folds[d] == outer}
    rows, reads = [], 0
    for dataset in datasets:
        if folds[dataset] == outer:
            continue
        assert dataset[:7] in allowed and dataset[:7] not in excluded
        m = pd.read_pickle(
            data.BASE / "evaluation_v2" / dataset / "manual_comparison.pkl"
        )
        reads += 1
        m = m.loc[
            ~m.internal_standard & ~m.lipid_class.eq("X4"),
            ["stream", "species", "lipid_class"],
        ].copy()
        composition = m.species.astype(str).str.extract(
            r"^[^()]+\((?:[dtm]|(?:O|P)-|O\+P-)?([0-9]+):([0-9]+)\)$"
        )
        for index, r in m.iterrows():
            if pd.isna(composition.loc[index, 0]):
                continue
            c, d = int(composition.loc[index, 0]), int(composition.loc[index, 1])
            rows.append(
                dict(
                    dataset=dataset,
                    cohort=dataset[:7],
                    stream=r.stream,
                    lipid_class=r.lipid_class,
                    carbon=c,
                    db=d,
                )
            )
    return pd.DataFrame(rows), allowed, excluded, reads


def main():
    configure_frozen_resources(data.OUT / "source_snapshot")
    receipt_path = data.OUT / "blind_receipt.json"
    blind = json.loads(receipt_path.read_text())
    assert (
        blind["all_passed"]
        and blind["datasets"] == 33
        and blind["query_gold_validation_tables_opened"] == 0
    )
    assert blind["source_input_sha256"] == data.inputs()
    assert blind["numeric_runtime"] == data.environment()
    for path, h in blind["artifact_sha256"].items():
        assert data.sha(path) == h
    data.CHECK.mkdir(exist_ok=False)
    datasets, folds = data.inventory()
    signatures = {d: oracle.signature_scalar(data.native_inputs(d)) for d in datasets}
    data.same_model(
        signatures, json.loads((data.OUT / "native_signatures.json").read_text())
    )
    reads = 0
    models = {}
    for outer in range(5):
        rows, allowed, excluded, count = training_scalar(outer)
        reads += count
        folder = data.OUT / "models" / f"fold_{outer}"
        pd.testing.assert_frame_equal(
            rows,
            pd.read_pickle(folder / "authorized_composition_rows.pkl"),
            check_exact=True,
        )
        models[outer] = oracle.fit_scalar(
            rows,
            {d: s for d, s in signatures.items() if folds[d] != outer},
            training_groups=allowed,
            excluded_groups=excluded,
        )
        data.same_model(models[outer], json.loads((folder / "model.json").read_text()))
        audit = pd.read_csv(folder / "training_role_audit.csv")
        assert (
            len(audit) == count
            and audit.authorized_training.all()
            and not set(audit.cohort) & excluded
        )
        assert set(audit.cohort) == allowed
        print(
            "INDEPENDENT MATRIX PEER REFERENCE",
            outer,
            count,
            len(models[outer]["classes"]),
            flush=True,
        )
    assert reads == 132
    checks = []
    for dataset in datasets:
        original = data.load_pool(dataset, folds[dataset])
        pool = original["pool"]
        # Pure chemical equivalence reduction, NOT peak selection: every raw
        # option is restored below and full index/multiplicity is compared.
        columns = ["stream", "class", "carbon", "db", "is_is"]
        chemical = pool[columns].drop_duplicates().copy()
        scalar = oracle.predict_scalar(
            chemical,
            signatures[dataset],
            models[folds[dataset]],
            query_group=dataset[:7],
        )

        def chemical_key(values):
            return tuple(None if pd.isna(v) else v for v in values)

        lookup = {
            chemical_key(r): scalar.loc[i].to_numpy(float)
            for i, r in chemical.iterrows()
        }
        expected = pd.DataFrame(
            [
                lookup[chemical_key(r)]
                for r in pool[columns].itertuples(index=False, name=None)
            ],
            index=pool.index,
            columns=scalar.columns,
        )
        actual = pd.read_pickle(data.OUT / "private" / (dataset + ".pkl"))
        pd.testing.assert_frame_equal(
            actual, expected, check_exact=False, rtol=0, atol=1e-12
        )
        assert list(actual) == [peer.PREFIX + f for f in peer.FIELDS]
        # Quantities cannot depend on original peak or adduct. Unsupported
        # entries remain NaN, not a fallback numerical zero.
        assert actual.loc[pool.is_is].isna().all().all()
        for indices in pool.groupby(
            ["stream", "class", "carbon", "db", "is_is"], sort=False
        ).groups.values():
            assert actual.loc[indices].nunique(dropna=False).le(1).all()
        checks.append(
            dict(
                dataset=dataset,
                candidate_rows=len(pool),
                supported_rows=int(actual.iloc[:, 0].notna().sum()),
                all_passed=True,
            )
        )
        print("INDEPENDENT MATRIX PEER ALL OPTIONS", dataset, len(pool), flush=True)
    assert blind["source_input_sha256"] == data.inputs()
    pd.DataFrame(checks).to_csv(data.CHECK / "by_dataset_check.csv", index=False)
    data.dump(
        data.CHECK / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            dataset_modes=132,
            authorized_training_table_uses=reads,
            blind_receipt_sha256=data.sha(receipt_path),
            source_input_sha256=blind["source_input_sha256"],
            query_gold_validation_tables_opened=0,
            classifier_fits=0,
            thresholds_selected=False,
            annotation_changes=False,
            independent_numeric_tolerance=1e-12,
            all_original_feature_coordinates_checked=True,
        ),
    )
    print(
        "ALL33 MATRIX PEER INDEPENDENT CHECKS PASS; QUERY DIAGNOSTIC NOW ALLOWED",
        flush=True,
    )


if __name__ == "__main__":
    main()
