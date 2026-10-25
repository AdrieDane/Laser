"""Label-free ALL33 raw counterpart measurements; no models/annotations/curation.

The separate retrospective diagnostic may open labels ONLY after this complete
blind receipt exists. This pilot uses outer-query-excluded standards, not the
future full nested model feature preparation (which still needs all363 tables).
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from context import REPO, SOURCE
from frozen_resources import configure_frozen_resources
import run_qe_add_only_class_experiment as common

QE = common.QE
OUT = QE / "raw_counterpart_feature_pilot_v1"
RT = QE / "raw_counterpart_RT_availability_v1"
PAIR = QE / "explicit_specimen_pair_audit_v1"


def input_paths(datasets, folds):
    files = {
        Path(__file__),
        Path(common.__file__),
        Path(common.common.__file__),
        Path(__file__).with_name("context.py"),
        Path(__file__).with_name("frozen_resources.py"),
        REPO / "docs/validation/next_raw_counterpart_evidence.txt",
        QE / "rt_atlas_development/fold_assignment.json",
        common.OLD / "run_manifest.json",
        RT / "receipt.json",
        PAIR / "summary.json",
        PAIR / "by_dataset_mode.csv",
    }
    files.update(
        p
        for root in ("app", "reference_data", "methods")
        for p in (REPO / root).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
    )
    for d in datasets:
        mask = f"exclude_{folds[d]}"
        files.update(
            [
                SOURCE / d / "MS_metadata.mat",
                QE / "baseline/runs" / d / "full_annotation.pkl",
                common.common.prepared_path(d, {folds[d]}),
                common.common.PREPARED / "measurements" / mask / (d + ".pkl"),
                QE / "nested_standard_development_v1" / (mask + ".json"),
                QE / "nested_standard_development_v1/automatic_inputs" / (d + ".pkl"),
            ]
        )
        for phase in ("RP", "NP"):
            files.add(RT / "private" / (d + "_" + phase + "_links.json"))
        saved_map = PAIR / "private_maps" / (d + ".json")
        if saved_map.exists():
            files.add(saved_map)
    return sorted(files)


def prepare_one(item):
    d, fold = item
    configure_frozen_resources(REPO)
    from matlab_annotation.raw_counterpart_evidence import (
        raw_counterpart_evidence,
        PREFIX,
        FIELDS,
    )
    from matlab_annotation.specimen_pairs import explicit_specimen_pairs

    raw = common.common.load_predictors(d, {fold})
    pool = raw["pool"]
    assert not {"training_positive", "training_label_known", "same_manual_peak"} & set(
        pool
    )
    baseline = pd.read_pickle(QE / "baseline/runs" / d / "full_annotation.pkl")
    features = {s: v["features"] for s, v in baseline["streams"].items()}
    metadata = loadmat(
        SOURCE / d / "MS_metadata.mat",
        variable_names=["MS_metadata"],
        simplify_cells=True,
    )["MS_metadata"]
    try:
        maps, _ = explicit_specimen_pairs(metadata, features)
        status, error = "explicit_metadata_links_checked", ""
        # Reproduce the earlier audit; never silently switch specimen pairing.
        assert maps == json.loads((PAIR / "private_maps" / (d + ".json")).read_text())
    except ValueError as exc:
        maps, status, error = {}, "metadata_conflict_or_unsupported", str(exc)
        assert not (PAIR / "private_maps" / (d + ".json")).exists()
    old_status = pd.read_csv(PAIR / "by_dataset_mode.csv")
    assert set(old_status.loc[old_status.dataset.eq(d), "status"]) == {status}
    parts, rows = [], []
    for stream in common.MODES:
        common.common.physical_protection(d, {fold}, stream, raw)
        links = json.loads(
            (RT / "private" / (d + "_" + stream[:2] + "_links.json")).read_text()
        )
        links = links["positive" if stream.endswith("Pos") else "negative"]
        part = pool.loc[pool.stream.eq(stream)]
        result = raw_counterpart_evidence(
            part,
            features,
            REPO / "reference_data/matlab_pipeline",
            maps,
            stream=stream,
            rt_links=links,
        )
        assert result.index.equals(part.index) and list(result) == [
            PREFIX + x for x in FIELDS
        ]
        assert np.isfinite(result.to_numpy(float)[~result.isna().to_numpy()]).all()
        assert result[PREFIX + "source_configured"].isin([0.0, 1.0]).all()
        assert result[PREFIX + "pairing_available"].eq(bool(maps)).all()
        assert not {
            "automatic_accepted",
            "automatic_candidate",
            "laser_annotation",
        } & set(result)
        parts.append(result)
        rows.append(
            dict(
                dataset=d,
                stream=stream,
                candidate_rows=len(result),
                pairing_status=status,
                pairing_error=error,
                configured_rows=int(result[PREFIX + "source_configured"].sum()),
                raw_mass_partner_rows=int(
                    result[PREFIX + "distinct_mass_hits"].gt(0).sum()
                ),
                measured_profile_rows=int(
                    result[PREFIX + "profile_measurement_available"].sum()
                ),
                raw_correlation_ge_08_rows=int(
                    result[PREFIX + "best_raw_correlation"].ge(0.8).sum()
                ),
                within_group_correlation_ge_08_rows=int(
                    result[PREFIX + "best_within_group_correlation"].ge(0.8).sum()
                ),
                independent_RT_available_rows=int(
                    result[PREFIX + "RT_map_available"].sum()
                ),
                RT_supported_profile_rows=int(
                    result[PREFIX + "RT_supported_profile_hits"].gt(0).sum()
                ),
            )
        )
    result = pd.concat(parts).reindex(pool.index)
    # No old inputs, candidate pool or source quantification may change.
    original = common.common.load_predictors(d, {fold})
    pd.testing.assert_frame_equal(raw["pool"], original["pool"], check_exact=True)
    pd.testing.assert_frame_equal(
        raw["evidence"], original["evidence"], check_exact=True
    )
    reloaded = pd.read_pickle(QE / "baseline/runs" / d / "full_annotation.pkl")
    for stream in common.MODES:
        pd.testing.assert_frame_equal(
            features[stream], reloaded["streams"][stream]["features"], check_exact=True
        )
        assert features[stream].attrs == reloaded["streams"][stream]["features"].attrs
    path = OUT / "private" / (d + ".pkl")
    pd.to_pickle(
        dict(
            evidence=result,
            query_dataset=d,
            excluded_folds=[fold],
            original_prepared_sha256=common.sha(common.common.prepared_path(d, {fold})),
            standard_reference_sha256=raw["standard_reference_sha256"],
            pairing_status=status,
        ),
        path,
    )
    print("BLIND RAW COUNTERPART MEASUREMENTS FROZEN", d, len(result), flush=True)
    return rows, (str(path), common.sha(path))


def main():
    datasets, folds = common.inventory()
    rt_receipt = json.loads((RT / "receipt.json").read_text())
    assert rt_receipt["all_passed"] and rt_receipt["manual_files_opened"] == 0
    # RT links cannot be imported from a changed native/RT implementation.
    for p, h in rt_receipt["source_input_sha256"].items():
        assert common.sha(Path(p)) == h
    hashes = {str(p): common.sha(p) for p in input_paths(datasets, folds)}
    OUT.mkdir(exist_ok=False)
    (OUT / "private").mkdir()
    common.dump(OUT / "planned_inputs.json", hashes)
    rows, artifacts = [], {}
    with ProcessPoolExecutor(max_workers=2) as executor:
        for table, artifact in executor.map(
            prepare_one, [(d, folds[d]) for d in datasets]
        ):
            rows.extend(table)
            artifacts[artifact[0]] = artifact[1]
    table = pd.DataFrame(rows)
    assert len(table) == 132 and table.dataset.nunique() == 33
    table.to_csv(OUT / "by_dataset_mode.csv", index=False)
    artifacts[str(OUT / "by_dataset_mode.csv")] = common.sha(
        OUT / "by_dataset_mode.csv"
    )
    assert all(common.sha(Path(p)) == h for p, h in hashes.items())
    common.dump(
        OUT / "blind_physical_receipt.json",
        dict(
            all_passed=True,
            all33_physical_tables_frozen=True,
            datasets=33,
            dataset_modes=132,
            manual_files_opened=0,
            classifier_fits=0,
            annotation_changes=False,
            new_numeric_fields=32,
            source_input_sha256=hashes,
            physical_sha256=artifacts,
            note="Label-free measurement availability ONLY; not new annotation accuracy/precision/FDR. Same-phase opposite-polarity original raw peaks may be unassigned. Original108predictors/pools/quantifications unchanged. Missing RT/pairing stays unknown. Query-excluded outer standard references ONLY; full363nested model preparation still required before fits.",
        ),
    )
    print("ALL33/132 BLIND PHYSICAL TABLES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
