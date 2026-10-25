"""Label-free ALL363 original pools+108 predictors+23 raw physical fields.

Raw-only union memoization, not fitted/reference/mask-dependent measurements.
No models, labels, annotations, RT links or added candidate identities.
"""

from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from context import REPO, SOURCE
from frozen_resources import configure_frozen_resources
import prepare_qe_raw_counterpart_pilot as pilot
import run_qe_add_only_class_experiment as common

QE = common.QE
OUT = QE / "raw_mass_profile_preparation_v2"
DIAGNOSTIC = QE / "raw_counterpart_diagnostic_041_v2"
KEYS = ["stream", "class", "adduct", "ref_mz", "feat_feature_id", "feat_rt", "feat_mz"]


def masks_for(own):
    others = sorted(set(range(5)) - {own})
    return [
        frozenset({own, *extra})
        for size in range(3)
        for extra in combinations(others, size)
    ]


def mask_name(mask):
    return "exclude_" + "_".join(map(str, sorted(mask)))


def prepared_path(dataset, mask):
    return OUT / "prepared" / mask_name(mask) / (dataset + ".pkl")


def physical_queries(pool):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.standard_rt_library import ion_key
    from matlab_annotation.raw_counterpart_rt import SUPERVISION

    if SUPERVISION & set(pool):
        raise ValueError("Raw physical memoization refuses curation")
    p = pool[KEYS].copy()
    p["class"] = p["class"].map(canonical_lipid_class)
    p["adduct"] = p.adduct.map(ion_key)
    p["feat_feature_id"] = p.feat_feature_id.astype(str)
    return p


def exact_lookup(queries, measurements, pool):
    """Exact tuples only: memoizing evidence cannot expand a query's pool."""
    if queries.duplicated(KEYS).any() or not queries.index.equals(measurements.index):
        raise ValueError("Unique aligned physical measurement tuples required")
    index = {
        key: i for i, key in enumerate(queries[KEYS].itertuples(index=False, name=None))
    }
    wanted = physical_queries(pool)
    tuples = list(wanted.itertuples(index=False, name=None))
    if any(key not in index for key in tuples):
        raise ValueError("Missing raw physical measurement; no fallback allowed")
    result = measurements.iloc[[index[key] for key in tuples]].copy()
    result.index = pool.index
    return result


def bindings(datasets, folds):
    files = set(pilot.input_paths(datasets, folds))
    files.update(
        [
            Path(__file__),
            DIAGNOSTIC / "receipt.json",
            pilot.OUT / "blind_physical_receipt.json",
            QE / "raw_counterpart_feature_independent_check_v1/receipt.json",
            REPO / "docs/validation/next_raw_counterpart_mass_profile_completion.txt",
        ]
    )
    for d in datasets:
        files.add(pilot.OUT / "private" / (d + ".pkl"))
        for mask in masks_for(folds[d]):
            files.update(
                [
                    common.common.prepared_path(d, mask),
                    QE / "nested_standard_development_v1" / (mask_name(mask) + ".json"),
                ]
            )
    return {str(p): common.sha(p) for p in sorted(files)}


def prepare_one(item):
    d, own, folds = item
    configure_frozen_resources(REPO)
    from matlab_annotation.raw_counterpart_evidence import (
        raw_counterpart_evidence,
        FIELDS,
        PREFIX,
    )
    from matlab_annotation.specimen_pairs import explicit_specimen_pairs
    from matlab_annotation.evidence_training import validate_feature_scope

    masks = masks_for(own)
    original = {mask: common.common.load_predictors(d, mask) for mask in masks}
    for mask, raw in original.items():
        assert len(raw["evidence"].columns) == 108
        validate_feature_scope(
            raw,
            {p[:7] for p in folds if folds[p] not in mask},
            {p[:7] for p in folds if folds[p] in mask},
            query_is_training=False,
        )
    union = (
        pd.concat(
            [physical_queries(raw["pool"]) for raw in original.values()],
            ignore_index=True,
        )
        .drop_duplicates(KEYS)
        .reset_index(drop=True)
    )
    # Legacy tables can differ by one floating-point RT bit across masks.
    # Keep those exact tuples DISTINCT, not normalized/merged. The constructor
    # checks EVERY coordinate against the original raw source at8eps, the
    # already-tested v1 pilot tolerance. Original pools/108values stay exact.
    baseline_path = QE / "baseline/runs" / d / "full_annotation.pkl"
    baseline = pd.read_pickle(baseline_path)
    features = {s: v["features"] for s, v in baseline["streams"].items()}
    metadata_path = SOURCE / d / "MS_metadata.mat"
    metadata = loadmat(
        metadata_path, variable_names=["MS_metadata"], simplify_cells=True
    )["MS_metadata"]
    try:
        maps, _ = explicit_specimen_pairs(metadata, features)
        assert maps == json.loads(
            (pilot.PAIR / "private_maps" / (d + ".json")).read_text()
        )
        pairing_status = "explicit_metadata_links_checked"
    except ValueError:
        maps, pairing_status = {}, "metadata_conflict_or_unsupported"
        assert not (pilot.PAIR / "private_maps" / (d + ".json")).exists()
    parts = []
    fields = [PREFIX + f for f in FIELDS[:23]]
    assert FIELDS[22] == "profile_margin" and len(fields) == 23
    for stream in common.MODES:
        part = union.loc[union.stream.eq(stream)]
        measured = raw_counterpart_evidence(
            part,
            features,
            REPO / "reference_data/matlab_pipeline",
            maps,
            stream=stream,
            rt_links=None,
        )
        parts.append(measured.loc[:, fields])
    measured = pd.concat(parts).reindex(union.index)
    assert (
        measured.shape == (len(union), 23)
        and not np.isinf(measured.to_numpy(float)).any()
    )
    query_path = OUT / "union" / (d + "_queries.pkl")
    measure_path = OUT / "union" / (d + "_measurements.pkl")
    union.to_pickle(query_path)
    measured.to_pickle(measure_path)
    artifacts = {str(p): common.sha(p) for p in (query_path, measure_path)}
    rows = []
    for mask, raw in original.items():
        extra = exact_lookup(union, measured, raw["pool"])
        evidence = pd.concat([raw["evidence"], extra], axis=1)
        assert evidence.shape == (len(raw["pool"]), 131)
        pd.testing.assert_frame_equal(
            evidence.loc[:, raw["evidence"].columns], raw["evidence"], check_exact=True
        )
        if mask == {own}:
            previous = pd.read_pickle(pilot.OUT / "private" / (d + ".pkl"))[
                "evidence"
            ].loc[:, fields]
            pd.testing.assert_frame_equal(extra, previous, check_exact=True)
        payload = dict(
            raw,
            evidence=evidence,
            raw_mass_profile_provenance=dict(
                mask=sorted(mask),
                original_prepared_sha256=common.sha(
                    common.common.prepared_path(d, mask)
                ),
                baseline_sha256=common.sha(baseline_path),
                metadata_sha256=common.sha(metadata_path),
                union_queries_sha256=common.sha(query_path),
                union_measurements_sha256=common.sha(measure_path),
                pairing_status=pairing_status,
                new_numeric_fields=23,
                fitted_RT_links_used=False,
                curation_runtime_input=False,
                cache_kind="exact raw-only physical tuple memoization",
            ),
        )
        path = prepared_path(d, mask)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(payload, path)
        artifacts[str(path)] = common.sha(path)
        old = common.common.load_predictors(d, mask)
        pd.testing.assert_frame_equal(raw["pool"], old["pool"], check_exact=True)
        pd.testing.assert_frame_equal(
            raw["evidence"], old["evidence"], check_exact=True
        )
        for s, g in raw["pool"].groupby("stream"):
            rows.append(
                dict(
                    dataset=d,
                    mask=mask_name(mask),
                    stream=s,
                    rows=len(g),
                    profile_rows=int(
                        extra.loc[
                            g.index, PREFIX + "profile_measurement_available"
                        ].sum()
                    ),
                )
            )
    old = pd.read_pickle(baseline_path)
    for s in common.MODES:
        pd.testing.assert_frame_equal(
            features[s], old["streams"][s]["features"], check_exact=True
        )
        assert features[s].attrs == old["streams"][s]["features"].attrs
    print(
        "RAW-ONLY UNION AND ALL11 ORIGINAL MASK TABLES PREPARED",
        d,
        len(union),
        flush=True,
    )
    return rows, artifacts, dict(dataset=d, physical_queries=len(union))


def main():
    datasets, folds = common.inventory()
    diagnostic = json.loads((DIAGNOSTIC / "receipt.json").read_text())
    assert (
        diagnostic["all_passed"]
        and diagnostic["historical_all_manual_entries_checked"] == 92100
    )
    pilot_receipt = json.loads((pilot.OUT / "blind_physical_receipt.json").read_text())
    assert all(
        common.sha(Path(p)) == h
        for p, h in pilot_receipt["source_input_sha256"].items()
    )
    hashes = bindings(datasets, folds)
    OUT.mkdir(exist_ok=False)
    (OUT / "union").mkdir()
    common.dump(OUT / "planned_inputs.json", hashes)
    rows, artifacts, counts = [], {}, []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for table, paths, summary in executor.map(
            prepare_one, [(d, folds[d], folds) for d in datasets]
        ):
            rows.extend(table)
            artifacts.update(paths)
            counts.append(summary)
    table = pd.DataFrame(rows)
    assert (
        len(table) == 1452 and table.groupby("dataset")["mask"].nunique().eq(11).all()
    )
    table.to_csv(OUT / "by_dataset_mask_mode.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "by_dataset_union.csv", index=False)
    assert hashes == bindings(datasets, folds)
    common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            unique_nested_tables=363,
            dataset_mask_modes=1452,
            training_tables=330,
            outer_tables=33,
            original_predictors=108,
            new_numeric_fields=23,
            union_physical_queries=sum(r["physical_queries"] for r in counts),
            classifier_fits=0,
            annotation_changes=False,
            manual_files_opened_for_measurements=0,
            source_input_sha256=hashes,
            artifact_sha256=artifacts,
            note="Raw-only physical tuple memoization,no fittedRTlinks/curation/otherdatasets/modelscore/reference membership. Originalpools/108fields/rawquant unchanged,ALL33outerfirst23bitexactpilot. V1failedexactRTuniquenessassertionbeforemeasurements;source/log/planpreserved. V2keepslegacy1-bitRTaliasesasDISTINCTexacttuples;everycoordinatecheckedagainstoriginalrawatunchanged8eps,notRTwindowwidening. All33/363rerun. Notfitpermission:independentALLunionnumeric/ALL363lookup/scopeaudit and25parent/model/exportpreflight/fullrunner/verifier required before100freshclassifierfits. No annotation gains/FDR/deploymentclaim.",
        ),
    )
    print("ALL363 RAW MASS/PROFILE TABLES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
