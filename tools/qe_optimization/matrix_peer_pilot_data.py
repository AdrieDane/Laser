"""Explicit authorized reference roles; no query labels in prediction inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import REPO
from paired_gap_pilot_data import QE, BASE, REFERENCE, inventory, sha
from paired_gap_pilot_data import dump as dump, environment as environment

OUT = QE / "matrix_peer_composition_pilot_v1"
CHECK = QE / "matrix_peer_composition_independent_check_v1"
DIAG = QE / "matrix_peer_composition_diagnostic_041_v1"
PLAN = REPO / "docs/validation/next_matrix_peer_composition_pilot.txt"
SOURCES = (
    "matrix_peer_composition.py",
    "matrix_peer_independent.py",
    "matrix_peer_pilot_data.py",
    "prepare_qe_matrix_peer_pilot.py",
    "verify_qe_matrix_peer_pilot.py",
    "diagnose_qe_matrix_peer_pilot.py",
    "test_matrix_peer_composition.py",
    "paired_gap_pilot_data.py",
    "diagnose_qe_raw_counterpart_pilot.py",
    "frozen_resources.py",
    "context.py",
)


def inputs():
    datasets, folds = inventory()
    paths = {Path(__file__).with_name(n) for n in SOURCES}
    paths.update(
        (
            PLAN,
            BASE / "run_manifest.json",
            QE / "rt_atlas_development/fold_assignment.json",
            REFERENCE / "integrity_receipt.json",
            QE / "paired_isomer_gap_diagnostic_041_v1/receipt.json",
        )
    )
    for dataset in datasets:
        paths.update(
            (
                BASE / "runs" / dataset / "full_annotation.pkl",
                BASE / "evaluation_v2" / dataset / "manual_comparison.pkl",
                REFERENCE / "evaluation_v2" / dataset / "manual_comparison.pkl",
                REFERENCE / "runs" / dataset / "automatic_annotations.csv",
                pool_path(dataset, folds[dataset]),
            )
        )
    for folder in ("app", "methods", "reference_data"):
        paths.update(
            p
            for p in (REPO / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
        )
    return {str(p): sha(p) for p in sorted(paths)}


def pool_path(dataset, fold):
    return (
        QE
        / "standard_context_preparation_v1/prepared"
        / f"exclude_{fold}"
        / (dataset + ".pkl")
    )


def load_pool(dataset, fold):
    from matlab_annotation.evidence_training import validate_feature_scope

    datasets, folds = inventory()
    original = pd.read_pickle(pool_path(dataset, fold))
    allowed = {d[:7] for d in datasets if folds[d] != fold}
    excluded = {d[:7] for d in datasets if folds[d] == fold}
    assert original["query_dataset"] == dataset and set(original["excluded_folds"]) == {
        fold
    }
    validate_feature_scope(original, allowed, excluded, query_is_training=False)
    return original


def native_inputs(dataset):
    """Select blind native fields and verify anchors against ORIGINAL peaks."""
    from standard_lipidomics.validation import canonical_feature_id

    baseline = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
    columns = [
        "class",
        "carbon",
        "db",
        "is_is",
        "ref_mz",
        "feat_mz",
        "feat_intensity",
        "post_regression_confidence",
    ]
    tables = []
    for stream, payload in baseline["streams"].items():
        a = payload["annotations"]
        f = payload["features"]
        ids = f.feature_id.map(canonical_feature_id)
        assert ids.is_unique
        original = f.set_index(ids).loc[a.feat_feature_id.map(canonical_feature_id)]
        for candidate, raw in (
            ("feat_mz", "mz"),
            ("feat_rt", "rt"),
            ("feat_intensity", "intensity"),
        ):
            np.testing.assert_allclose(
                a[candidate].to_numpy(float),
                original[raw].to_numpy(float),
                rtol=8 * np.finfo(float).eps,
                atol=0,
                equal_nan=True,
            )
        tables.append(a[columns].assign(stream=stream))
    return pd.concat(tables, ignore_index=True)


def training_rows(outer):
    """Own-query project excluded BEFORE each curated training table open."""
    datasets, folds = inventory()
    allowed = {d[:7] for d in datasets if folds[d] != outer}
    excluded = {d[:7] for d in datasets if folds[d] == outer}
    rows, reads = [], []
    for dataset in datasets:
        if folds[dataset] == outer:
            continue
        assert dataset[:7] in allowed and dataset[:7] not in excluded
        path = BASE / "evaluation_v2" / dataset / "manual_comparison.pkl"
        manual = pd.read_pickle(path)[
            ["stream", "species", "lipid_class", "internal_standard"]
        ]
        for r in manual.itertuples():
            if r.internal_standard or r.lipid_class == "X4":
                continue
            composition = parse_composition(r.species)
            if composition is None:
                continue
            c, d = composition
            rows.append(
                dict(
                    dataset=dataset,
                    cohort=dataset[:7],
                    stream=r.stream,
                    lipid_class=r.lipid_class,
                    carbon=c,
                    db=d,
                )
            )
        reads.append(
            dict(
                dataset=dataset,
                cohort=dataset[:7],
                outer=outer,
                authorized_training=True,
                manual_sha256=sha(path),
            )
        )
    return pd.DataFrame(rows), allowed, excluded, reads


def parse_composition(species):
    """Same physical C/DB naming as native catalogue, including d/t/O+P."""
    from matlab_annotation.chemical_targets import COMPOSITION

    match = COMPOSITION.fullmatch(str(species))
    return None if match is None else (int(match.group(2)), int(match.group(3)))


def same_model(a, b):
    """JSON-normalized metadata exact; only numerical arithmetic tolerance."""
    a, b = (json.loads(json.dumps(v)) for v in (a, b))
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            same_model(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same_model(x, y)
    elif isinstance(a, float):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    else:
        assert a == b
