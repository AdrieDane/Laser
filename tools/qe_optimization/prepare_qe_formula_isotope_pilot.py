"""ALL33 four-mode formula descriptors frozen before any gold analysis."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import prepare_qe_cross_phase_candidate_pilot as previous
import formula_isotope_measurements as measurement

common = previous.common
OUT = common.QE / "formula_isotope_measurement_pilot_v1"
PLAN = REPO / "docs/validation/next_formula_isotope_pilot.txt"


def inputs(datasets, folds):
    paths = set(previous.inputs(datasets, folds))
    paths.update(
        str(p)
        for p in (
            Path(__file__),
            Path(measurement.__file__),
            PLAN,
            Path(__file__).with_name("formula_isotope_independent.py"),
            Path(__file__).with_name("verify_qe_formula_isotope_pilot.py"),
            Path(__file__).with_name("diagnose_qe_formula_isotope_pilot.py"),
            Path(__file__).with_name("test_formula_isotope_measurements.py"),
        )
    )
    return {p: common.sha(Path(p)) for p in sorted(paths)}


def one(item):
    d, fold = item
    configure_frozen_resources(REPO)
    from matlab_annotation.evidence_training import validate_feature_scope

    datasets, folds = common.inventory()
    raw = common.common.load_predictors(d, {fold})
    validate_feature_scope(
        raw,
        {x[:7] for x in datasets if folds[x] != fold},
        {x[:7] for x in datasets if folds[x] == fold},
        query_is_training=False,
    )
    pool = raw["pool"]
    before = pool.copy(deep=True)
    root = REPO / "reference_data/matlab_pipeline"
    evidence, audit = measurement.measure(
        pool, measurement.definitions(root), measurement.isotope_table(root)
    )
    assert list(evidence) == [measurement.PREFIX + f for f in measurement.FIELDS]
    assert evidence.index.equals(pool.index) and audit.index.equals(pool.index)
    assert np.isfinite(evidence.to_numpy(float)[~evidence.isna().to_numpy()]).all()
    pd.testing.assert_frame_equal(pool, before, check_exact=True)
    path = OUT / "private" / (d + ".pkl")
    pd.to_pickle(
        dict(
            query_dataset=d,
            excluded_folds=[fold],
            evidence=evidence,
            audit=audit,
            source_prepared_sha256=common.sha(common.common.prepared_path(d, {fold})),
        ),
        path,
    )
    table = (
        pool[["stream", "class"]]
        .join(audit[["status"]])
        .join(
            evidence[
                [
                    measurement.PREFIX + "upper_log_ratio",
                    measurement.PREFIX + "lower_inverse_log_ratio",
                    measurement.PREFIX + "lower2_inverse_log_ratio",
                ]
            ]
        )
    )
    rows = []
    for (s, cls, status), g in table.groupby(["stream", "class", "status"]):
        rows.append(
            dict(
                dataset=d,
                stream=s,
                lipid_class=cls,
                status=status,
                candidate_options=len(g),
                upper_measurable=int(
                    g[measurement.PREFIX + "upper_log_ratio"].notna().sum()
                ),
                lower_measurable=int(
                    g[measurement.PREFIX + "lower_inverse_log_ratio"].notna().sum()
                ),
                lower2_measurable=int(
                    g[measurement.PREFIX + "lower2_inverse_log_ratio"].notna().sum()
                ),
            )
        )
    print(
        "BLIND FORMULA ISOTOPE TABLE",
        d,
        len(evidence),
        audit.status.value_counts().to_dict(),
        flush=True,
    )
    return rows, {str(path): common.sha(path)}


def main():
    configure_frozen_resources(REPO)
    datasets, folds = common.inventory()
    hashes = inputs(datasets, folds)
    OUT.mkdir(exist_ok=False)
    (OUT / "private").mkdir()
    common.dump(OUT / "planned_inputs.json", hashes)
    with ProcessPoolExecutor(max_workers=2) as executor:
        result = list(executor.map(one, [(d, folds[d]) for d in datasets]))
    rows = [r for part, _ in result for r in part]
    artifacts = {p: h for _, part in result for p, h in part.items()}
    assert len(artifacts) == 33 and hashes == inputs(datasets, folds)
    pd.DataFrame(rows).to_csv(
        OUT / "blind_coverage_by_dataset_mode_class.csv", index=False
    )
    common.dump(
        OUT / "blind_receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            measurement_streams=list(common.MODES),
            fields=len(measurement.FIELDS),
            all33_tables_frozen=True,
            manual_files_opened=0,
            classifier_fits=0,
            thresholds_selected=False,
            annotation_changes=False,
            source_input_sha256=hashes,
            physical_sha256=artifacts,
            note="Single outer-excluded read-only measurement pilot; NOT363 nested construction or annotation benchmark.",
        ),
    )


if __name__ == "__main__":
    main()
