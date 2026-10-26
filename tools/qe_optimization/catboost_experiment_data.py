"""Explicit048 source paths, old108 inputs and authorized training labels.

No app import before caller configures its FINAL frozen scientific root.
No model fitting, threshold selection or annotation writes in this module.
"""

from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import REPO, WORKSPACE

QE = WORKSPACE / "outputs/qe_optimization_20261022"
BASE = QE / "baseline"
CORE = QE / "experiment_029"
PREVIOUS = QE / "experiment_041"
PREPARED = QE / "standard_context_preparation_v1"
STD = QE / "experiment_005"
MASS = QE / "experiment_027"
OUT = QE / "experiment_048"
PREFLIGHT = QE / "catboost_model_preflight_v1"
PLAN = REPO / "docs/validation/next_catboost_model_family_experiment.txt"
MODES = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def inventory():
    datasets = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    assert len(datasets) == 33 and len({d[:7] for d in datasets}) == 30
    assert set(folds.values()) == set(range(5))
    assert all(
        len({folds[d] for d in datasets if d[:7] == group}) == 1
        for group in {d[:7] for d in datasets}
    )
    return datasets, folds


def masks_for(own):
    other = sorted(set(range(5)) - {own})
    return [{own, *extra} for size in range(3) for extra in combinations(other, size)]


def prepared_path(dataset, mask):
    return (
        PREPARED
        / "prepared"
        / ("exclude_" + "_".join(map(str, sorted(mask))))
        / (dataset + ".pkl")
    )


def parent_path(outer, inner, root=OUT):
    return root / (
        f"fold_{outer}_nested_model.pkl"
        if inner is None
        else f"calibration/fold_{outer}/inner_block_{inner}_model.pkl"
    )


def parents():
    return [(o, i) for o in range(5) for i in [None, *[j for j in range(5) if j != o]]]


def load_predictors(dataset, mask):
    from matlab_annotation.evidence_training import validate_feature_scope

    datasets, folds = inventory()
    assert dataset in datasets and folds[dataset] in mask and len(mask) in {1, 2, 3}
    data = pd.read_pickle(prepared_path(dataset, mask))
    assert data["query_dataset"] == dataset and data["query_group"] == dataset[:7]
    assert set(data["excluded_folds"]) == set(mask)
    assert data["pool"].index.is_unique and data["pool"].index.equals(
        data["evidence"].index
    )
    assert data["evidence"].shape == (len(data["pool"]), 108)
    assert not {
        "training_positive",
        "training_label_known",
        "calibration_identity_in_manual",
        "same_manual_peak",
        "manual_feature_id",
    } & set(data["pool"])
    assert set(data["standard_reference_training_groups"]) == set(
        data["reference_training_groups"]
    )
    assert set(data["standard_reference_excluded_groups"]) == set(
        data["reference_excluded_groups"]
    )
    training = {d[:7] for d in datasets if folds[d] not in mask}
    excluded = {d[:7] for d in datasets if folds[d] in mask}
    assert set(data["reference_training_groups"]) == training
    assert set(data["reference_excluded_groups"]) == excluded
    validate_feature_scope(data, training, excluded, query_is_training=False)
    return data


def authorize_labels(raw, training, excluded):
    """Same029labels, but permission checked BEFORE opening a manual reference."""
    from matlab_annotation.evidence_training import validate_feature_scope
    from standard_lipidomics.validation import _species
    from matlab_annotation.standard_rt_library import ion_key

    validate_feature_scope(raw, training, excluded, query_is_training=True)
    assert raw["query_group"] in training and raw["query_group"] not in excluded
    manual = pd.read_pickle(
        BASE / "evaluation_v2" / raw["query_dataset"] / "manual_comparison.pkl"
    )
    manual = manual.loc[~manual.internal_standard].copy()
    keys = ["stream", "species", "adduct", "feature_id"]
    exact = set(manual.loc[manual.mapped, keys].itertuples(index=False, name=None))
    unresolved = set(
        manual.loc[~manual.mapped, keys[:3]].itertuples(index=False, name=None)
    )
    identities = set(manual[keys[:3]].itertuples(index=False, name=None))
    pool = raw["pool"].copy()
    query = [
        (str(s), _species(n), ion_key(a), str(f))
        for s, n, a, f in pool[
            ["stream", "name", "adduct", "feat_feature_id"]
        ].itertuples(index=False, name=None)
    ]
    pool["training_positive"] = [k in exact for k in query]
    pool["training_label_known"] = [
        k[:3] not in unresolved or k in exact for k in query
    ]
    pool["calibration_identity_in_manual"] = [k[:3] in identities for k in query]
    return dict(raw, pool=pool), manual


def training_records(outer, inner):
    datasets, folds = inventory()
    blocked = {outer} if inner is None else {outer, inner}
    training = {d[:7] for d in datasets if folds[d] not in blocked}
    excluded = {d[:7] for d in datasets if folds[d] in blocked}
    records = {}
    for d in datasets:
        if folds[d] in blocked:
            continue
        raw = load_predictors(d, blocked | {folds[d]})
        records[d], _ = authorize_labels(raw, training, excluded)
    return records, training, excluded


def independent_fitting_matrix(records, training, excluded, mode):
    """Separate positional construction of original known rows and weights."""
    from matlab_annotation.evidence_training import validate_feature_scope

    matrices, labels, groups = [], [], []
    for record in records.values():
        validate_feature_scope(record, training, excluded, query_is_training=True)
        pool, evidence = record["pool"], record["evidence"]
        indices = np.flatnonzero(pool.stream.to_numpy() == mode)
        known = pool.training_label_known.to_numpy(dtype=bool)[indices]
        selected = indices[known]
        matrices.append(evidence.iloc[selected].copy())
        labels.extend(pool.training_positive.to_numpy(dtype=int)[selected].tolist())
        groups.extend([record["query_group"]] * len(selected))
    x = pd.concat(matrices, ignore_index=True)
    categorical = []
    for c in x:
        if pd.api.types.is_numeric_dtype(x[c]):
            values = np.array(pd.to_numeric(x[c], errors="coerce"), dtype=float)
            values[~np.isfinite(values)] = np.nan
            x[c] = values
        else:
            categorical.append(c)
            x[c] = x[c].map(str)
    frequency = Counter(groups)
    assert set(frequency) == set(training)
    weights = np.empty(len(groups), dtype=float)
    # Preserve the original fitter's two successive divisions, including IEEE
    # rounding. Equivalent single-division arithmetic can differ by one ULP.
    project_total = len(groups) / len(training)
    for i, group in enumerate(groups):
        weights[i] = project_total / frequency[group]
    return x, np.asarray(labels, int), np.asarray(groups), weights, categorical
