"""Freeze all33 four-mode comparative-ion descriptors before gold analysis."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import prepare_qe_formula_isotope_pilot as previous
import prepare_qe_raw_mass_profile_tables as original
import competing_ion_family_measurements as measurement

common = previous.common
OUT = common.QE / "competing_ion_family_pilot_v1"
PLAN = REPO / "docs/validation/next_competing_ion_family_pilot.txt"


def inputs(datasets, folds):
    paths = set(previous.inputs(datasets, folds))
    paths.update(
        str(p)
        for p in (
            Path(__file__),
            Path(measurement.__file__),
            Path(original.__file__),
            PLAN,
            Path(__file__).with_name("competing_ion_family_independent.py"),
            Path(__file__).with_name("verify_qe_competing_ion_family_pilot.py"),
            Path(__file__).with_name("diagnose_qe_competing_ion_family_pilot.py"),
            Path(__file__).with_name("test_competing_ion_family_measurements.py"),
            original.OUT / "receipt.json",
            common.QE / "raw_mass_profile_independent_check_v1/receipt.json",
        )
    )
    built = json.loads((original.OUT / "receipt.json").read_text())
    checked = json.loads(
        (common.QE / "raw_mass_profile_independent_check_v1/receipt.json").read_text()
    )
    assert (
        built["all_passed"]
        and checked["all_passed"]
        and checked["unique_nested_tables"] == 363
    )
    assert checked["preparation_receipt_sha256"] == common.sha(
        original.OUT / "receipt.json"
    )
    for d in datasets:
        path = original.prepared_path(d, {folds[d]})
        assert common.sha(path) == built["artifact_sha256"][str(path)]
        paths.add(str(path))
    return {p: common.sha(Path(p)) for p in sorted(paths)}


def load(d, fold):
    raw = pd.read_pickle(original.prepared_path(d, {fold}))
    assert raw["query_dataset"] == d and set(raw["excluded_folds"]) == {fold}
    assert raw["evidence"].shape == (len(raw["pool"]), 131)
    return raw


def one(item):
    d, fold = item
    configure_frozen_resources(REPO)
    from matlab_annotation.evidence_training import validate_feature_scope
    from matlab_annotation.adduct_family_evidence import native_adduct_families
    from verify_qe_bmp_context_pilot import eligibility

    datasets, folds = common.inventory()
    raw = load(d, fold)
    validate_feature_scope(
        raw,
        {x[:7] for x in datasets if folds[x] != fold},
        {x[:7] for x in datasets if folds[x] == fold},
        query_is_training=False,
    )
    pool, evidence = raw["pool"], raw["evidence"]
    old_pool, old_evidence = pool.copy(deep=True), evidence.copy(deep=True)
    protections = {
        s: common.common.physical_protection(d, {fold}, s, raw) for s in common.MODES
    }
    physical = eligibility(pool, protections)
    families = native_adduct_families(str(REPO / "reference_data/matlab_pipeline"))
    extra, links = measurement.measure(pool, evidence, families, physical)
    assert extra.index.equals(pool.index) and list(extra) == [
        measurement.PREFIX + f for f in measurement.FIELDS
    ]
    assert np.isfinite(extra.to_numpy(float)[~extra.isna().to_numpy()]).all()
    pd.testing.assert_frame_equal(pool, old_pool, check_exact=True)
    pd.testing.assert_frame_equal(evidence, old_evidence, check_exact=True)
    path = OUT / "private" / (d + ".pkl")
    pd.to_pickle(
        dict(
            query_dataset=d,
            excluded_folds=[fold],
            evidence=extra,
            links=links,
            physical=physical,
            source_prepared_sha256=common.sha(original.prepared_path(d, {fold})),
        ),
        path,
    )
    rows = []
    for (s, c), part in pool.groupby(["stream", "class"]):
        values = extra.loc[part.index]
        rows.append(
            dict(
                dataset=d,
                stream=s,
                lipid_class=c,
                candidate_options=len(part),
                with_alternative_family=int(
                    values[measurement.PREFIX + "alternative_families"].gt(0).sum()
                ),
                with_measured_alternative=int(
                    values[measurement.PREFIX + "measured_alternative_families"]
                    .gt(0)
                    .sum()
                ),
                with_raw_contrast=int(
                    values[
                        measurement.PREFIX
                        + "own_minus_alternative_best_raw_correlation"
                    ]
                    .notna()
                    .sum()
                ),
            )
        )
    print("BLIND COMPETING-ION TABLE", d, len(extra), len(links), flush=True)
    return rows, {str(path): common.sha(path)}


def main():
    configure_frozen_resources(REPO)
    datasets, folds = common.inventory()
    hashes = inputs(datasets, folds)
    OUT.mkdir(exist_ok=False)
    (OUT / "private").mkdir()
    common.dump(OUT / "planned_inputs.json", hashes)
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(one, [(d, folds[d]) for d in datasets]))
    rows = [r for part, _ in results for r in part]
    artifacts = {p: h for _, part in results for p, h in part.items()}
    assert len(artifacts) == 33 and hashes == inputs(datasets, folds)
    pd.DataFrame(rows).to_csv(
        OUT / "blind_coverage_by_dataset_mode_class.csv", index=False
    )
    common.dump(
        OUT / "blind_receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            all33_tables_frozen=True,
            fields=len(measurement.FIELDS),
            source_input_sha256=hashes,
            physical_sha256=artifacts,
            manual_files_opened=0,
            annotation_changes=False,
            classifier_fits=0,
            thresholds_selected=False,
            note="Single-outer-excluded measurement pilot, NOT new363nested construction or annotation experiment. Existing independently checked raw43 fields reused, not its rejected models/calls.",
        ),
    )


if __name__ == "__main__":
    main()
