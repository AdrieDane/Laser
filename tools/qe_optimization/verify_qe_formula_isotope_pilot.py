"""All33 independent raw-TSV formula/native mass/polynomial ratio audit."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import prepare_qe_formula_isotope_pilot as prep
import formula_isotope_independent as independent

OUT = prep.common.QE / "formula_isotope_independent_check_v1"


def one(item):
    d, fold = item
    configure_frozen_resources(REPO)
    raw = prep.common.common.load_predictors(d, {fold})
    saved = pd.read_pickle(prep.OUT / "private" / (d + ".pkl"))
    assert saved["query_dataset"] == d and saved["excluded_folds"] == [fold]
    assert saved["source_prepared_sha256"] == prep.common.sha(
        prep.common.common.prepared_path(d, {fold})
    )
    definitions, isotopes = independent.resources(
        REPO / "reference_data/matlab_pipeline"
    )
    evidence, audit = independent.independent_measure(
        raw["pool"],
        definitions,
        isotopes,
        prep.measurement.FIELDS,
        prep.measurement.PREFIX,
    )
    pd.testing.assert_frame_equal(
        saved["evidence"], evidence, check_exact=False, atol=1e-8, rtol=1e-8
    )
    pd.testing.assert_frame_equal(saved["audit"], audit, check_exact=True)
    print("INDEPENDENT FORMULA ISOTOPE CHECK", d, len(evidence), flush=True)
    return dict(
        dataset=d,
        candidate_options=len(evidence),
        numeric_cells=int(evidence.size),
        formula_available=int(evidence[prep.measurement.PREFIX + "available"].sum()),
    )


def main():
    configure_frozen_resources(REPO)
    path = prep.OUT / "blind_receipt.json"
    receipt = json.loads(path.read_text())
    datasets, folds = prep.common.inventory()
    hashes = prep.inputs(datasets, folds)
    assert (
        receipt["all_passed"]
        and receipt["datasets"] == 33
        and receipt["manual_files_opened"] == 0
    )
    assert hashes == receipt["source_input_sha256"]
    assert all(
        prep.common.sha(Path(p)) == h for p, h in receipt["physical_sha256"].items()
    )
    OUT.mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=2) as executor:
        checks = list(executor.map(one, [(d, folds[d]) for d in datasets]))
    assert len(checks) == 33 and hashes == prep.inputs(datasets, folds)
    prep.common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            checks=checks,
            blind_receipt_sha256=prep.common.sha(path),
            source_input_sha256=hashes,
            physical_sha256=receipt["physical_sha256"],
            manual_files_opened=0,
            annotation_changes=False,
            classifier_fits=0,
            thresholds_selected=False,
        ),
    )


if __name__ == "__main__":
    main()
