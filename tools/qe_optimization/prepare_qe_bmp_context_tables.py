"""ALL363 independently masked BMP graphs; sparse32-field numeric sidecars.

No graph unions, model scores, curation reads, fits or annotations. Original
131 fields and original pools are not rewritten. Sidecars bind exact source
tables, masks and reference provenance; loading reconstructs163 predictors.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import prepare_qe_bmp_context_pilot as pilot
import run_qe_raw_mass_profile_experiment as original
import bmp_context_measurements as measurement

QE = original.QE
OUT = QE / "BMP_candidate_context_preparation_v1"
PLAN = REPO / "docs/validation/next_BMP_candidate_context_experiment.txt"
masks_for = original.preparation.masks_for
mask_name = original.preparation.mask_name


def sidecar_path(d, mask):
    return OUT / "prepared" / mask_name(mask) / (d + ".pkl")


def load_augmented(d, mask):
    raw = original.load_predictors(d, mask)
    sidecar = pd.read_pickle(sidecar_path(d, mask))
    if sidecar["query_dataset"] != d or set(sidecar["excluded_folds"]) != set(mask):
        raise ValueError("BMP context query/mask mismatch")
    if sidecar["source131_sha256"] != original.sha(
        original.preparation.prepared_path(d, mask)
    ):
        raise ValueError("BMP context source131 changed")
    for name in ("reference_training_groups", "reference_excluded_groups"):
        if set(sidecar[name]) != set(raw[name]):
            raise ValueError("BMP context reference scope mismatch")
    if raw["query_group"] in set(sidecar["reference_training_groups"]) or raw[
        "query_group"
    ] not in set(sidecar["reference_excluded_groups"]):
        raise ValueError("Query project entered BMP context references")
    extra = sidecar["evidence"]
    if (
        list(extra) != [measurement.PREFIX + f for f in measurement.FIELDS]
        or not extra.index.is_unique
        or not extra.index.isin(raw["pool"].index).all()
    ):
        raise ValueError("BMP context sidecar schema/index mismatch")
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.raw_counterpart_rt import SUPERVISION

    expected = (
        raw["pool"]
        .loc[
            raw["pool"].stream.isin(["NP_Pos", "NP_Neg"])
            & raw["pool"]["class"].map(canonical_lipid_class).eq("BMP")
            & ~raw["pool"].is_is.astype(bool)
        ]
        .index
    )
    if not extra.index.equals(expected) or SUPERVISION & set(extra):
        raise ValueError("Exact original BMP row membership required")
    evidence = pd.concat([raw["evidence"], extra.reindex(raw["pool"].index)], axis=1)
    assert evidence.shape == (len(raw["pool"]), 163) and evidence.columns.is_unique
    pd.testing.assert_frame_equal(
        evidence.loc[:, raw["evidence"].columns], raw["evidence"], check_exact=True
    )
    return dict(
        raw,
        evidence=evidence,
        bmp_context_provenance=dict(
            query_dataset=d,
            mask=sorted(mask),
            source131_sha256=sidecar["source131_sha256"],
            sidecar_sha256=original.sha(sidecar_path(d, mask)),
            reference_training_groups=sidecar["reference_training_groups"],
            reference_excluded_groups=sidecar["reference_excluded_groups"],
            new_numeric_fields=32,
            curation_runtime_input=False,
            graph_membership_unioned=False,
        ),
    )


def validate_scope(raw, training, excluded, *, query_is_training):
    """Scope of NEW reference-dependent measurements checked explicitly too."""
    from matlab_annotation.evidence_training import validate_feature_scope

    validate_feature_scope(raw, training, excluded, query_is_training=query_is_training)
    p = raw.get("bmp_context_provenance", {})
    required = {
        "query_dataset",
        "mask",
        "source131_sha256",
        "sidecar_sha256",
        "reference_training_groups",
        "reference_excluded_groups",
        "new_numeric_fields",
        "curation_runtime_input",
        "graph_membership_unioned",
    }
    if (
        not required.issubset(p)
        or p["curation_runtime_input"] is not False
        or p["graph_membership_unioned"] is not False
        or p["new_numeric_fields"] != 32
    ):
        raise ValueError("Explicit mask-specific BMP context provenance required")
    if p["query_dataset"] != raw["query_dataset"] or set(p["mask"]) != set(
        raw["excluded_folds"]
    ):
        raise ValueError("BMP context query/mask scope mismatch")
    for key in ("reference_training_groups", "reference_excluded_groups"):
        if set(p[key]) != set(raw[key]):
            raise ValueError("BMP context reference membership mismatch")
    refs = set(p["reference_training_groups"])
    if (
        not refs
        or not refs.issubset(set(training))
        or refs & set(excluded)
        or raw["query_group"] in refs
    ):
        raise ValueError("Excluded group entered BMP context candidate graph")


def bindings(datasets, folds):
    paths = set(map(Path, pilot.inputs(datasets, folds)))
    paths.update(
        {
            Path(__file__),
            PLAN,
            Path(__file__).with_name("verify_qe_bmp_context_tables.py"),
            QE / "BMP_candidate_context_independent_check_v1/receipt.json",
            QE / "BMP_candidate_context_diagnostic_041_v1/receipt.json",
        }
    )
    for d in datasets:
        paths.add(pilot.OUT / "private" / (d + ".pkl"))
        for mask in masks_for(folds[d]):
            paths.add(original.preparation.prepared_path(d, mask))
            paths.add(pilot.common.common.prepared_path(d, mask))
            paths.add(
                QE / "nested_standard_development_v1" / (mask_name(mask) + ".json")
            )
            paths.add(
                pilot.common.common.PREPARED
                / "measurements"
                / mask_name(mask)
                / (d + ".pkl")
            )
    return {str(p): original.sha(p) for p in sorted(paths)}


def one(item):
    d, own, folds = item
    configure_frozen_resources(REPO)
    from matlab_annotation.adduct_family_evidence import native_adduct_families
    from matlab_annotation.evidence_training import validate_feature_scope

    _, features, maps = pilot.load(d, own)
    families = native_adduct_families(str(REPO / "reference_data/matlab_pipeline"))
    artifacts, rows = {}, []
    for mask in masks_for(own):
        raw = original.load_predictors(d, mask)
        allowed = {p[:7] for p in folds if folds[p] not in mask}
        excluded = {p[:7] for p in folds if folds[p] in mask}
        validate_feature_scope(raw, allowed, excluded, query_is_training=False)
        pool = raw["pool"]
        physical = pd.Series(False, index=pool.index)
        for stream in ("NP_Pos", "NP_Neg"):
            protection = pilot.common.common.physical_protection(d, mask, stream, raw)
            part = pool.loc[pool.stream.eq(stream)]
            physical.loc[part.index] = pilot.guards.physical_masks(
                part, stream, protection
            )[0]
        measured = measurement.measure(pool, features, maps, families, physical)
        assert not np.isinf(measured.to_numpy(float)).any()
        if set(mask) == {own}:
            old = pd.read_pickle(pilot.OUT / "private" / (d + ".pkl"))
            pd.testing.assert_frame_equal(measured, old["evidence"], check_exact=True)
            pd.testing.assert_series_equal(physical, old["physical"], check_exact=True)
        path = sidecar_path(d, mask)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            query_dataset=d,
            excluded_folds=sorted(mask),
            source131_sha256=original.sha(original.preparation.prepared_path(d, mask)),
            evidence=measured,
            reference_training_groups=raw["reference_training_groups"],
            reference_excluded_groups=raw["reference_excluded_groups"],
        )
        pd.to_pickle(payload, path)
        augmented = load_augmented(d, mask)
        validate_scope(augmented, allowed, excluded, query_is_training=False)
        pd.testing.assert_frame_equal(augmented["pool"], raw["pool"], check_exact=True)
        artifacts[str(path)] = original.sha(path)
        rows.append(
            dict(
                dataset=d,
                mask=mask_name(mask),
                total_original_rows=len(pool),
                measured_BMP_rows=len(measured),
                new_numeric_cells=measured.size,
                old_predictors=131,
                total_predictors=163,
            )
        )
        print("SCOPED BMP GRAPH FROZEN", d, mask_name(mask), len(measured), flush=True)
    return rows, artifacts


def main():
    configure_frozen_resources(REPO)
    datasets, folds = original.inventory()
    receipt = json.loads(
        (QE / "BMP_candidate_context_independent_check_v1/receipt.json").read_text()
    )
    assert receipt["all_passed"] and receipt["datasets"] == 33
    assert all(
        original.sha(Path(p)) == h for p, h in receipt["source_input_sha256"].items()
    )
    hashes = bindings(datasets, folds)
    OUT.mkdir(exist_ok=False)
    original.dump(OUT / "planned_inputs.json", hashes)
    with ProcessPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(one, [(d, folds[d], folds) for d in datasets]))
    rows = [r for table, _ in outputs for r in table]
    artifacts = {p: h for _, paths in outputs for p, h in paths.items()}
    assert len(rows) == len(artifacts) == 363
    assert hashes == bindings(datasets, folds)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    original.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            unique_nested_tables=363,
            datasets=33,
            total_predictors=163,
            old_predictors=131,
            new_numeric_fields=32,
            graph_membership_unioned=False,
            manual_files_opened=0,
            classifier_fits=0,
            annotation_changes=False,
            source_input_sha256=hashes,
            artifact_sha256=artifacts,
        ),
    )


if __name__ == "__main__":
    main()
