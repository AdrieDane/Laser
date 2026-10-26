"""All33 separate scalar comparative-ion checks before gold analysis."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import prepare_qe_competing_ion_family_pilot as prep
from competing_ion_family_independent import independent_measure

OUT = prep.common.QE / "competing_ion_family_independent_check_v1"


def one(item):
    d, fold = item
    configure_frozen_resources(REPO)
    from matlab_annotation.adduct_family_evidence import native_adduct_families
    from diagnose_qe_combined_peak_ranks import physical_masks

    raw = prep.load(d, fold)
    saved = pd.read_pickle(prep.OUT / "private" / (d + ".pkl"))
    assert saved["query_dataset"] == d and saved["excluded_folds"] == [fold]
    assert saved["source_prepared_sha256"] == prep.common.sha(
        prep.original.prepared_path(d, {fold})
    )
    physical = pd.Series(False, index=raw["pool"].index)
    for s in prep.common.MODES:
        part = raw["pool"].loc[raw["pool"].stream.eq(s)]
        protection = prep.common.common.physical_protection(d, {fold}, s, raw)
        physical.loc[part.index] = physical_masks(part, s, protection)[0]
    pd.testing.assert_series_equal(saved["physical"], physical, check_exact=True)
    families = native_adduct_families(str(REPO / "reference_data/matlab_pipeline"))
    m = prep.measurement
    evidence, links = independent_measure(
        raw["pool"],
        raw["evidence"],
        families,
        physical,
        m.FIELDS,
        m.MEASURES,
        m.CONTRASTS,
        m.PREFIX,
        m.RAW_PREFIX,
    )
    pd.testing.assert_frame_equal(
        saved["evidence"], evidence, check_exact=False, atol=1e-10, rtol=1e-10
    )
    pd.testing.assert_frame_equal(saved["links"], links, check_exact=True)
    print("INDEPENDENT COMPETING-ION CHECK", d, len(evidence), len(links), flush=True)
    return dict(
        dataset=d,
        candidate_options=len(evidence),
        numeric_cells=int(evidence.size),
        links=len(links),
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
