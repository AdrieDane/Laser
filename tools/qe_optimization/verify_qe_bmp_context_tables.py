"""Independent ALL363 BMP graph/physical/numeric and original131 integrity."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import prepare_qe_bmp_context_tables as prep
import verify_qe_bmp_context_pilot as independent

OUT = prep.QE / "BMP_candidate_context_construction_check_v1"


def one(item):
    d, own, folds = item
    configure_frozen_resources(REPO)
    from matlab_annotation.adduct_family_evidence import native_adduct_families

    _, features, maps = prep.pilot.load(d, own)
    families = native_adduct_families(str(REPO / "reference_data/matlab_pipeline"))
    counts = []
    for mask in prep.masks_for(own):
        raw = prep.original.load_predictors(d, mask)
        allowed = {p[:7] for p in folds if folds[p] not in mask}
        excluded = {p[:7] for p in folds if folds[p] in mask}
        assert set(raw["reference_training_groups"]) == allowed
        assert set(raw["reference_excluded_groups"]) == excluded
        protections = {
            s: prep.pilot.common.common.physical_protection(d, mask, s, raw)
            for s in ("NP_Pos", "NP_Neg")
        }
        physical = independent.eligibility(raw["pool"], protections)
        expected = independent.independent_measure(
            raw["pool"], features, maps, families, physical
        )
        sidecar = pd.read_pickle(prep.sidecar_path(d, mask))
        assert sidecar["query_dataset"] == d and set(sidecar["excluded_folds"]) == set(
            mask
        )
        assert set(sidecar["reference_training_groups"]) == allowed
        assert set(sidecar["reference_excluded_groups"]) == excluded
        assert sidecar["source131_sha256"] == prep.original.sha(
            prep.original.preparation.prepared_path(d, mask)
        )
        pd.testing.assert_frame_equal(
            sidecar["evidence"], expected, check_exact=False, atol=1e-6, rtol=1e-8
        )
        augmented = prep.load_augmented(d, mask)
        prep.validate_scope(augmented, allowed, excluded, query_is_training=False)
        pd.testing.assert_frame_equal(augmented["pool"], raw["pool"], check_exact=True)
        pd.testing.assert_frame_equal(
            augmented["evidence"].loc[:, raw["evidence"].columns],
            raw["evidence"],
            check_exact=True,
        )
        reconstructed = pd.concat(
            [raw["evidence"], expected.reindex(raw["pool"].index)], axis=1
        )
        pd.testing.assert_frame_equal(
            augmented["evidence"],
            reconstructed,
            check_exact=False,
            atol=1e-6,
            rtol=1e-8,
        )
        assert augmented["evidence"].shape == (len(raw["pool"]), 163)
        if set(mask) == {own}:
            pilot = pd.read_pickle(prep.pilot.OUT / "private" / (d + ".pkl"))
            pd.testing.assert_frame_equal(
                sidecar["evidence"], pilot["evidence"], check_exact=True
            )
            pd.testing.assert_series_equal(
                physical, pilot["physical"], check_names=False, check_exact=True
            )
        counts.append(
            dict(
                dataset=d,
                mask=prep.mask_name(mask),
                new_numeric_cells=expected.size,
                original_pool_rows=len(raw["pool"]),
                old_predictors_checked=131,
                total_predictors_checked=163,
            )
        )
        print(
            "INDEPENDENT MASKED BMP GRAPH CHECKED",
            d,
            prep.mask_name(mask),
            expected.size,
            flush=True,
        )
    return counts


def main():
    datasets, folds = prep.original.inventory()
    source_receipt = prep.OUT / "receipt.json"
    receipt = json.loads(source_receipt.read_text())
    assert receipt["all_passed"] and receipt["unique_nested_tables"] == 363
    assert receipt["manual_files_opened"] == receipt["classifier_fits"] == 0
    assert receipt["graph_membership_unioned"] is False
    hashes = {
        **receipt["source_input_sha256"],
        **receipt["artifact_sha256"],
        str(source_receipt): prep.original.sha(source_receipt),
        str(Path(__file__)): prep.original.sha(Path(__file__)),
        str(Path(independent.__file__)): prep.original.sha(Path(independent.__file__)),
        str(Path(independent.numeric.__file__)): prep.original.sha(
            Path(independent.numeric.__file__)
        ),
    }
    assert all(prep.original.sha(Path(p)) == h for p, h in hashes.items())
    OUT.mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(one, [(d, folds[d], folds) for d in datasets]))
    rows = [r for table in results for r in table]
    assert len(rows) == 363 and len(results) == 33
    assert all(prep.original.sha(Path(p)) == h for p, h in hashes.items())
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "numeric_check_by_mask.csv", index=False)
    prep.original.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            unique_nested_tables=363,
            new_numeric_cells_checked=int(table.new_numeric_cells.sum()),
            old_predictors=131,
            total_predictors=163,
            all33_outer_pilot_exact_parity=True,
            graph_membership_unioned=False,
            manual_files_opened=0,
            classifier_fits=0,
            annotation_changes=False,
            preparation_receipt_sha256=prep.original.sha(source_receipt),
            source_input_sha256=hashes,
        ),
    )


if __name__ == "__main__":
    main()
