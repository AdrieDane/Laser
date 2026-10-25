"""ALL33 independent scalar physics/graph/native-ion/NumPy-profile check."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import prepare_qe_cross_phase_candidate_pilot as prep
from cross_phase_candidate_independent import independent_measure
from verify_qe_bmp_context_pilot import eligibility

OUT = prep.common.QE / "cross_phase_class_candidate_independent_check_v1"


def one(item):
    d, fold = item
    configure_frozen_resources(REPO)
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.adduct_family_evidence import native_adduct_families

    raw, features, maps = prep.previous.load(d, fold)
    saved = pd.read_pickle(prep.OUT / "private" / (d + ".pkl"))
    assert saved["query_dataset"] == d and saved["excluded_folds"] == [fold]
    assert saved["source_prepared_sha256"] == prep.common.sha(
        prep.common.common.prepared_path(d, {fold})
    )
    protections = {
        s: prep.common.common.physical_protection(d, {fold}, s, raw)
        for s in prep.measurement.MODES
    }
    assert set(protections) == set(saved["protections"])
    for s in protections:
        assert set(protections[s]["protected_features"]) == set(
            saved["protections"][s]["protected_features"]
        )
        assert list(protections[s]["declared_masses"]) == list(
            saved["protections"][s]["declared_masses"]
        )
    p = raw["pool"]
    classes = {canonical_lipid_class(c) for c in ("FA", "FA-OH")}
    q = p.loc[
        p.stream.isin(prep.measurement.MODES)
        & p["class"].map(canonical_lipid_class).isin(classes)
    ]
    pd.testing.assert_series_equal(
        eligibility(q, protections), saved["physical"].loc[q.index], check_exact=True
    )
    families = native_adduct_families(str(REPO / "reference_data/matlab_pipeline"))
    evidence, links = independent_measure(
        p, features, maps, families, saved["physical"]
    )
    pd.testing.assert_frame_equal(
        saved["evidence"], evidence, check_exact=False, atol=1e-6, rtol=1e-8
    )
    pd.testing.assert_frame_equal(
        saved["links"], links, check_exact=False, atol=1e-6, rtol=1e-8
    )
    print(
        "INDEPENDENT CROSS-PHASE CLASS CHECK", d, len(evidence), len(links), flush=True
    )
    return dict(dataset=d, numeric_cells=int(evidence.size), physical_links=len(links))


def main():
    configure_frozen_resources(REPO)
    receipt_path = prep.OUT / "blind_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert (
        receipt["all_passed"]
        and receipt["datasets"] == 33
        and receipt["manual_files_opened"] == 0
    )
    datasets, folds = prep.common.inventory()
    hashes = prep.inputs(datasets, folds)
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
            blind_receipt_sha256=prep.common.sha(receipt_path),
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
