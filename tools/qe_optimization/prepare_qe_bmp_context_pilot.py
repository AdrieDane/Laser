"""ALL33 label-free, single outer-query-excluded BMP measurement pilot."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from context import REPO, SOURCE
from frozen_resources import configure_frozen_resources
import prepare_qe_raw_counterpart_pilot as old
import diagnose_qe_combined_peak_ranks as guards
import bmp_context_measurements as measurement

common = old.common
OUT = common.QE / "BMP_candidate_context_pilot_v1"
PLAN = REPO / "docs/validation/next_BMP_candidate_context_pilot.txt"
SLIDES = SOURCE.parent / "archive/annotatie_gateways.pptx"


def inputs(datasets, folds):
    paths = {
        Path(__file__),
        Path(measurement.__file__),
        Path(guards.__file__),
        Path(old.__file__),
        Path(common.__file__),
        Path(common.common.__file__),
        PLAN,
        SLIDES,
        Path(__file__).with_name("context.py"),
        Path(__file__).with_name("frozen_resources.py"),
        common.QE / "rt_atlas_development/fold_assignment.json",
        old.PAIR / "summary.json",
    }
    paths.update(
        p
        for root in ("app", "reference_data", "methods")
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
        saved = old.PAIR / "private_maps" / (d + ".json")
        if saved.exists():
            paths.add(saved)
    return {str(p): common.sha(p) for p in sorted(paths)}


def load(d, fold):
    from matlab_annotation.specimen_pairs import explicit_specimen_pairs

    raw = common.common.load_predictors(d, {fold})
    assert raw["query_dataset"] == d and set(raw["excluded_folds"]) == {fold}
    baseline = pd.read_pickle(common.QE / "baseline/runs" / d / "full_annotation.pkl")
    features = {s: v["features"] for s, v in baseline["streams"].items()}
    metadata = loadmat(
        SOURCE / d / "MS_metadata.mat",
        variable_names=["MS_metadata"],
        simplify_cells=True,
    )["MS_metadata"]
    try:
        maps, _ = explicit_specimen_pairs(metadata, features)
        assert maps == json.loads(
            (old.PAIR / "private_maps" / (d + ".json")).read_text()
        )
    except ValueError:
        maps = {}
        assert not (old.PAIR / "private_maps" / (d + ".json")).exists()
    return raw, features, maps


def one(item):
    d, fold = item
    configure_frozen_resources(REPO)
    from matlab_annotation.adduct_family_evidence import native_adduct_families

    raw, features, maps = load(d, fold)
    pool = raw["pool"]
    physical = pd.Series(False, index=pool.index)
    protections = {}
    for stream in ("NP_Pos", "NP_Neg"):
        protections[stream] = common.common.physical_protection(d, {fold}, stream, raw)
        part = pool.loc[pool.stream.eq(stream)]
        physical.loc[part.index] = guards.physical_masks(
            part, stream, protections[stream]
        )[0]
    families = native_adduct_families(str(REPO / "reference_data/matlab_pipeline"))
    result = measurement.measure(pool, features, maps, families, physical)
    assert list(result) == [measurement.PREFIX + f for f in measurement.FIELDS]
    assert np.isfinite(result.to_numpy(float)[~result.isna().to_numpy()]).all()
    pd.testing.assert_frame_equal(
        pool, common.common.load_predictors(d, {fold})["pool"], check_exact=True
    )
    path = OUT / "private" / (d + ".pkl")
    pd.to_pickle(
        dict(
            evidence=result,
            physical=physical,
            query_dataset=d,
            excluded_folds=[fold],
            protections=protections,
            original_prepared_sha256=common.sha(common.common.prepared_path(d, {fold})),
            pairing_available=bool(maps),
        ),
        path,
    )
    rows = []
    for stream in ("NP_Pos", "NP_Neg"):
        part = result.loc[pool.loc[result.index, "stream"].eq(stream)]
        rows.append(
            dict(
                dataset=d,
                stream=stream,
                candidate_options=len(part),
                physical_options=int(
                    part[measurement.PREFIX + "source_physical"].sum()
                ),
                options_with_class_RT_counterpart=int(
                    part[measurement.PREFIX + "candidate_counterparts"].gt(0).sum()
                ),
                options_with_counterpart_profile=int(
                    part[measurement.PREFIX + "counterpart_profiles"].gt(0).sum()
                ),
                options_with_PA_primary_candidate=int(
                    part[measurement.PREFIX + "PA_candidates"].gt(0).sum()
                ),
                options_with_PA_profile=int(
                    part[measurement.PREFIX + "PA_profiles"].gt(0).sum()
                ),
                metadata_pairing_available=bool(maps),
            )
        )
    print("BLIND BMP CONTEXT MEASUREMENTS FROZEN", d, len(result), flush=True)
    return rows, {str(path): common.sha(path)}


def main():
    configure_frozen_resources(REPO)
    datasets, folds = common.inventory()
    hashes = inputs(datasets, folds)
    OUT.mkdir(exist_ok=False)
    (OUT / "private").mkdir()
    common.dump(OUT / "planned_inputs.json", hashes)
    with ProcessPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(one, [(d, folds[d]) for d in datasets]))
    rows = [r for table, _ in outputs for r in table]
    artifacts = {p: h for _, paths in outputs for p, h in paths.items()}
    assert len(artifacts) == 33 and len(rows) == 66
    assert hashes == inputs(datasets, folds)
    pd.DataFrame(rows).to_csv(OUT / "availability_by_dataset_mode.csv", index=False)
    common.dump(
        OUT / "blind_receipt.json",
        dict(
            all_passed=True,
            all33_physical_tables_frozen=True,
            manual_files_opened=0,
            classifier_fits=0,
            annotation_changes=False,
            single_outer_exclusion_pilot_not363_training=True,
            source_input_sha256=hashes,
            physical_sha256=artifacts,
        ),
    )


if __name__ == "__main__":
    main()
