"""Full nested numeric delta layers; no training labels/annotations/score fits."""

import json
from pathlib import Path

import pandas as pd

from context import REPO
import catboost_experiment_data as original
import matrix_peer_pilot_data as pilot
import matrix_peer_composition as peer

QE, BASE = pilot.QE, pilot.BASE
OUT = QE / "matrix_peer_nested_preparation_v1"
CHECK = QE / "matrix_peer_nested_independent_check_v1"
PLAN = REPO / "docs/validation/next_matrix_nested_preparation.txt"


def mask_name(mask):
    return "exclude_" + "_".join(map(str, sorted(mask)))


def models():
    datasets, folds = pilot.inventory()
    masks = {tuple(sorted(m)) for d in datasets for m in original.masks_for(folds[d])}
    assert len(masks) == 25
    return sorted(masks, key=lambda m: (len(m), m))


def inputs():
    datasets, folds = pilot.inventory()
    files = {
        PLAN,
        pilot.OUT / "blind_receipt.json",
        pilot.CHECK / "receipt.json",
        pilot.OUT / "native_signatures.json",
    }
    for name in (
        "matrix_nested_data.py",
        "prepare_qe_matrix_nested.py",
        "verify_qe_matrix_nested.py",
        "test_matrix_nested_data.py",
        "catboost_experiment_data.py",
        "matrix_peer_composition.py",
        "matrix_peer_independent.py",
        "matrix_peer_pilot_data.py",
        "paired_gap_pilot_data.py",
        "context.py",
        "frozen_resources.py",
    ):
        files.add(Path(__file__).with_name(name))
    for d in datasets:
        files.add(BASE / "evaluation_v2" / d / "manual_comparison.pkl")
        for mask in original.masks_for(folds[d]):
            files.add(original.prepared_path(d, mask))
    for folder in ("app", "methods", "reference_data"):
        files.update(
            p
            for p in (REPO / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
        )
    # Include all dependencies already bound in the original pilot, not its
    # descriptor outputs as predictors; signatures are separately authorized.
    files.update(Path(p) for p in pilot.inputs())
    return {str(p): pilot.sha(p) for p in sorted(files)}


def training_rows(mask):
    datasets, folds = pilot.inventory()
    assert 1 <= len(set(mask)) <= 3 and set(mask) <= set(range(5))
    allowed = {d[:7] for d in datasets if folds[d] not in mask}
    excluded = {d[:7] for d in datasets if folds[d] in mask}
    rows, reads = [], []
    for dataset in datasets:
        if folds[dataset] in mask:
            continue
        assert dataset[:7] in allowed and dataset[:7] not in excluded
        path = BASE / "evaluation_v2" / dataset / "manual_comparison.pkl"
        table = pd.read_pickle(path)[
            ["stream", "species", "lipid_class", "internal_standard"]
        ]
        for r in table.itertuples():
            if r.internal_standard or r.lipid_class == "X4":
                continue
            comp = pilot.parse_composition(r.species)
            if comp is not None:
                rows.append(
                    dict(
                        dataset=dataset,
                        cohort=dataset[:7],
                        stream=r.stream,
                        lipid_class=r.lipid_class,
                        carbon=comp[0],
                        db=comp[1],
                    )
                )
        reads.append(
            dict(
                dataset=dataset,
                cohort=dataset[:7],
                authorized_training=True,
                manual_sha256=pilot.sha(path),
            )
        )
    return pd.DataFrame(rows), allowed, excluded, reads


def validate_scope(raw, model, allowed, excluded, *, query_is_training):
    from matlab_annotation.evidence_training import validate_feature_scope

    validate_feature_scope(raw, allowed, excluded, query_is_training=query_is_training)
    refs, blocked = set(model["training_groups"]), set(model["excluded_groups"])
    if refs != set(raw["reference_training_groups"]) or blocked != set(
        raw["reference_excluded_groups"]
    ):
        raise ValueError("Matrix references must match original exact nested mask")
    if (
        refs - set(allowed)
        or refs & set(excluded)
        or raw["query_group"] in refs
        or raw["query_group"] not in blocked
    ):
        raise ValueError("Own/outer/inner query entered matrix reference")


def layer_path(dataset, mask):
    return OUT / "layers" / mask_name(mask) / (dataset + ".pkl")


def load_predictors(dataset, mask):
    raw = original.load_predictors(dataset, mask)
    saved = pd.read_pickle(layer_path(dataset, mask))
    reference_path = OUT / "models" / mask_name(mask) / "model.json"
    model = json.loads(reference_path.read_text())
    assert saved["dataset"] == dataset and set(saved["excluded_folds"]) == set(mask)
    assert saved["source_pool_sha256"] == pilot.sha(
        original.prepared_path(dataset, mask)
    )
    assert saved["reference_sha256"] == pilot.sha(reference_path)
    datasets, folds = pilot.inventory()
    allowed = {d[:7] for d in datasets if folds[d] not in mask}
    excluded = {d[:7] for d in datasets if folds[d] in mask}
    validate_scope(raw, model, allowed, excluded, query_is_training=False)
    added = saved["evidence"]
    assert added.index.equals(raw["pool"].index) and list(added) == [
        peer.PREFIX + f for f in peer.FIELDS
    ]
    assert not set(added) & set(raw["evidence"])
    output = dict(
        raw,
        evidence=pd.concat([raw["evidence"], added], axis=1),
        matrix_reference_training_groups=model["training_groups"],
        matrix_reference_excluded_groups=model["excluded_groups"],
        matrix_reference_sha256=pilot.sha(reference_path),
    )
    assert output["evidence"].shape == (len(raw["pool"]), 116)
    pd.testing.assert_frame_equal(output["pool"], raw["pool"], check_exact=True)
    pd.testing.assert_frame_equal(
        output["evidence"][raw["evidence"].columns], raw["evidence"], check_exact=True
    )
    return output
