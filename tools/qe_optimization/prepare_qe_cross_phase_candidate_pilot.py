"""ALL33 label-free outer-only FA/FA-OH cross-phase graph/profile pilot."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from context import REPO, SOURCE
from frozen_resources import configure_frozen_resources
import prepare_qe_bmp_context_pilot as previous
import cross_phase_candidate_measurements as measurement

common = previous.common
OUT = common.QE / "cross_phase_class_candidate_pilot_v1"
PLAN = REPO / "docs/validation/next_cross_phase_candidate_pilot.txt"


def inputs(datasets, folds):
    paths = {
        Path(__file__),
        Path(measurement.__file__),
        Path(previous.__file__),
        Path(previous.guards.__file__),
        Path(previous.old.__file__),
        Path(common.__file__),
        Path(common.common.__file__),
        PLAN,
        Path(__file__).with_name("context.py"),
        Path(__file__).with_name("frozen_resources.py"),
        Path(__file__).with_name("cross_phase_candidate_independent.py"),
        Path(__file__).with_name("verify_qe_cross_phase_candidate_pilot.py"),
        Path(__file__).with_name("diagnose_qe_cross_phase_candidate_pilot.py"),
        Path(__file__).with_name("verify_qe_raw_counterpart_pilot.py"),
        Path(__file__).with_name("verify_qe_bmp_context_pilot.py"),
        common.QE / "experiment_041/integrity_receipt.json",
        common.QE / "rt_atlas_development/fold_assignment.json",
        previous.old.PAIR / "summary.json",
    }
    paths.update(
        p
        for root in ("app", "methods", "reference_data")
        for p in (REPO / root).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
    )
    for d in datasets:
        paths.update(
            {
                SOURCE / d / "MS_metadata.mat",
                common.QE / "baseline/runs" / d / "full_annotation.pkl",
                common.common.prepared_path(d, {folds[d]}),
                common.common.PREPARED
                / "measurements"
                / f"exclude_{folds[d]}"
                / (d + ".pkl"),
                common.QE
                / "nested_standard_development_v1"
                / f"exclude_{folds[d]}.json",
            }
        )
        mapping = previous.old.PAIR / "private_maps" / (d + ".json")
        if mapping.exists():
            paths.add(mapping)
    return {str(p): common.sha(p) for p in sorted(paths)}


def one(item):
    d, fold = item
    configure_frozen_resources(REPO)
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.adduct_family_evidence import native_adduct_families
    from matlab_annotation.evidence_training import validate_feature_scope

    datasets, folds = common.inventory()
    raw, features, maps = previous.load(d, fold)
    allowed = {x[:7] for x in datasets if folds[x] != fold}
    excluded = {x[:7] for x in datasets if folds[x] == fold}
    validate_feature_scope(raw, allowed, excluded, query_is_training=False)
    pool = raw["pool"]
    before = pool.copy(deep=True)
    physical = pd.Series(False, index=pool.index)
    protections = {}
    for s in measurement.MODES:
        protections[s] = common.common.physical_protection(d, {fold}, s, raw)
        part = pool.loc[pool.stream.eq(s)]
        physical.loc[part.index] = previous.guards.physical_masks(
            part, s, protections[s]
        )[0]
    families = native_adduct_families(str(REPO / "reference_data/matlab_pipeline"))
    evidence, links = measurement.measure(pool, features, maps, families, physical)
    assert list(evidence) == [measurement.PREFIX + c for c in measurement.FIELDS]
    assert np.isfinite(evidence.to_numpy(float)[~evidence.isna().to_numpy()]).all()
    pd.testing.assert_frame_equal(pool, before, check_exact=True)
    path = OUT / "private" / (d + ".pkl")
    pd.to_pickle(
        dict(
            query_dataset=d,
            excluded_folds=[fold],
            evidence=evidence,
            links=links,
            physical=physical,
            protections=protections,
            pairing_available=bool(maps),
            source_prepared_sha256=common.sha(common.common.prepared_path(d, {fold})),
        ),
        path,
    )
    rows = []
    for s in measurement.MODES:
        for cls in ("FA", "FA-OH"):
            idx = pool.loc[evidence.index].index[
                pool.loc[evidence.index, "stream"].eq(s)
                & pool.loc[evidence.index, "class"]
                .map(canonical_lipid_class)
                .eq(canonical_lipid_class(cls))
            ]
            part = evidence.loc[idx]
            rows.append(
                dict(
                    dataset=d,
                    stream=s,
                    lipid_class=cls,
                    candidate_options=len(part),
                    physical_options=int(
                        part[measurement.PREFIX + "source_physical"].sum()
                    ),
                    options_with_partner=int(
                        part[measurement.PREFIX + "candidate_counterparts"].gt(0).sum()
                    ),
                    options_with_profile=int(
                        part[measurement.PREFIX + "counterpart_profiles"].gt(0).sum()
                    ),
                    metadata_pairing_available=bool(maps),
                )
            )
    print(
        "BLIND CROSS-PHASE CLASS MEASUREMENTS FROZEN",
        d,
        len(evidence),
        len(links),
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
        results = list(executor.map(one, [(d, folds[d]) for d in datasets]))
    rows = [r for part, _ in results for r in part]
    artifacts = {p: h for _, part in results for p, h in part.items()}
    assert (
        len(rows) == 132 and len(artifacts) == 33 and hashes == inputs(datasets, folds)
    )
    pd.DataFrame(rows).to_csv(
        OUT / "availability_by_dataset_mode_class.csv", index=False
    )
    common.dump(
        OUT / "blind_receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            measurement_streams=list(measurement.MODES),
            lipid_classes=["FA", "FA-OH"],
            availability_rows=132,
            all33_tables_frozen=True,
            manual_files_opened=0,
            classifier_fits=0,
            thresholds_selected=False,
            annotation_changes=False,
            source_input_sha256=hashes,
            physical_sha256=artifacts,
            note="Single outer-excluded measurement pilot,NOT363trainingconstruction or annotation experiment. 132availability rows are33datasets*2negative modes*2classes,NOT132four-mode performance cases.",
        ),
    )


if __name__ == "__main__":
    main()
