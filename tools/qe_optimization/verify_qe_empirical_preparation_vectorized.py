"""Reconstruct numerical references, independent spikes and all masked tables.

Isolated scratch copies ONLY; original research artifacts remain untouched.
No annotations, score fits or query curation used for physical reconstruction.
"""

from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from context import WORKSPACE
from frozen_resources import configure_frozen_resources
from run_qe_standard_mode_experiment import sha, dump

QE = WORKSPACE / "outputs/qe_optimization_20261022"
PREPARED = QE / "empirical_RT_candidate_preparation_v2"
EMPIRICAL = QE / "nested_empirical_RT_references_v1"
OLD = QE / "standard_context_preparation_v1"


def same(actual, expected):
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    elif isinstance(actual, pd.Series):
        pd.testing.assert_series_equal(actual, expected, check_exact=True)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            same(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            same(a, b)
    elif isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected or (pd.isna(actual) and pd.isna(expected))


def verify_dataset(dataset):
    configure_frozen_resources(PREPARED / "source_snapshot")
    from matlab_annotation.declared_standard_context import (
        independent_standard_measurements,
    )

    receipt = json.loads((PREPARED / "checkpoints" / (dataset + ".json")).read_text())
    baseline = pd.read_pickle(QE / "baseline/runs" / dataset / "full_annotation.pkl")
    staged = pd.read_pickle(
        QE / "nested_standard_development_v1/automatic_inputs" / (dataset + ".pkl")
    )
    blind = dict(
        baseline,
        streams={
            s: dict(p, annotations=staged["automatic"][s])
            for s, p in baseline["streams"].items()
        },
    )
    for path in receipt["prepared_sha256"]:
        name = Path(path).parent.name
        model = json.loads(
            (QE / "nested_standard_development_v1" / (name + ".json")).read_text()
        )
        assert dataset[:7] not in model["training_groups"]
        assert dataset[:7] in model["excluded_groups"]
        measured, _ = independent_standard_measurements(
            blind, staged["declared"], model
        )
        same(measured, pd.read_pickle(OLD / "measurements" / name / (dataset + ".pkl")))
    spec = importlib.util.spec_from_file_location(
        "isolated_empirical_preparer", PREPARED / "runner_source.py"
    )
    preparer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparer)
    # prepare() has no cache hit in a NEW, validated own temporary directory.
    # The symlink resolves to the SAME frozen scientific resources, not a new
    # installed/default resource context or modified source implementation.
    with tempfile.TemporaryDirectory(prefix="qe036-preparation-replay-") as scratch:
        root = Path(scratch)
        (root / "source_snapshot").symlink_to(
            PREPARED / "source_snapshot", target_is_directory=True
        )
        (root / "checkpoints").mkdir()
        (root / "audits").mkdir()
        preparer.OUT = root
        replay = preparer.prepare(dataset)
        assert len(replay["tables"]) == 11
        same(replay["tables"], receipt["tables"])
        for path, digest in receipt["prepared_sha256"].items():
            path = Path(path)
            assert sha(path) == digest
            reproduced = root / "prepared" / path.parent.name / path.name
            same(pd.read_pickle(reproduced), pd.read_pickle(path))
    print(
        "036 ALL11MASKS/ALLPHYSICAL/PREDICTORS/INDEPENDENTSTDS REPRODUCED",
        dataset,
        flush=True,
    )
    return 11


def verify_references():
    configure_frozen_resources(PREPARED / "source_snapshot")
    from matlab_annotation.empirical_retention import fit_empirical_retention

    observations = pd.read_pickle(QE / "rt_atlas_development/observations.pkl")
    checks = 0
    for path in sorted(EMPIRICAL.glob("exclude_*.json")):
        record = json.loads(path.read_text())
        parent_path = QE / "nested_reference_development_v2" / path.name
        assert sha(parent_path) == record["parent_reference_sha256"]
        parent = json.loads(parent_path.read_text())
        train = observations[
            observations.dataset.isin(parent["training_datasets"])
            & observations.mapped.astype(bool)
        ]
        rebuilt = fit_empirical_retention(
            train,
            parent["atlas"],
            training_groups=parent["training_groups"],
            excluded_groups=parent["excluded_groups"],
        )
        assert rebuilt == record["empirical_reference"]
        checks += 1
    assert checks == 25
    return checks


def main():
    assert (PREPARED / "runner_source.py").read_bytes() == Path(__file__).with_name(
        "prepare_qe_empirical_candidates_vectorized.py"
    ).read_bytes()
    assert (EMPIRICAL / "runner_source.py").read_bytes() == Path(__file__).with_name(
        "build_qe_empirical_rt_references.py"
    ).read_bytes()
    summary = json.loads((PREPARED / "summary.json").read_text())
    assert summary["all_complete"] and summary["dataset_mask_tables"] == 363
    source_count, input_count = 0, 0
    for root in (EMPIRICAL, PREPARED):
        protocol = json.loads((root / "protocol.json").read_text())
        for path, digest in protocol["source_sha256"].items():
            assert sha(root / "source_snapshot" / path) == digest
            source_count += 1
        for path, digest in protocol["input_sha256"].items():
            assert sha(Path(path)) == digest
            input_count += 1
    references = verify_references()
    projects = json.loads((QE / "baseline/run_manifest.json").read_text())["datasets"]
    with ProcessPoolExecutor(max_workers=2) as executor:
        counts = list(executor.map(verify_dataset, projects))
    assert sum(counts) == 363
    result = dict(
        all_passed=True,
        annotation_changes=False,
        classifier_fits=0,
        source_sha256=sha(Path(__file__)),
        numerical_reference_exact_reproductions=references,
        full_masked_table_reproductions=363,
        independent_standard_mask_measurement_reproductions=363,
        datasets=33,
        source_hash_checks=source_count,
        input_hash_checks=input_count,
        note="Engineering ONLY,not annotation accuracy. ALLold+new physical rows,ALLpredictors and metadata reproduced,not samplechecks. Matchingquery/outer/inner references; rawspikes remeasured. Temporary generated verification copies cleaned up; original data/research outputs preserved.",
    )
    dump(PREPARED / "integrity_receipt.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
