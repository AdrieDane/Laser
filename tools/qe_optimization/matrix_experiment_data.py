"""Prospective049 exact inputs/labels, no actual fits or annotation writes.

The runner MUST wait for complete independent363/preflight receipts. This
module verifies numeric-layer artifact hashes on every load, not only metadata
references to the old physical pool/model. Old048 code is imported READONLY.
"""

import json

from context import REPO
import catboost_experiment_data as old
import matrix_nested_data as nested
import matrix_peer_composition as peer

QE, BASE, CORE, PREVIOUS = old.QE, old.BASE, old.CORE, old.PREVIOUS
OUT = QE / "experiment_049"
PREFLIGHT = QE / "matrix_context_model_preflight_v1"
PLAN = REPO / "docs/validation/next_matrix_context_annotation_experiment.txt"
MODES = old.MODES
inventory, parents = old.inventory, old.parents


def parameters():
    classifier = json.loads((CORE / "protocol.json").read_text())["classifier"]
    result = {k: v for k, v in classifier.items() if k != "kind"}
    assert result["max_iter"] == 180 and result["early_stopping"] is False
    return result


def model_path(outer, inner):
    return old.parent_path(outer, inner, root=OUT)


def validate_record(raw, allowed, excluded, *, query_is_training):
    from matlab_annotation.evidence_training import validate_feature_scope

    validate_feature_scope(raw, allowed, excluded, query_is_training=query_is_training)
    required = {
        "matrix_reference_training_groups",
        "matrix_reference_excluded_groups",
        "matrix_reference_sha256",
        "matrix_layer_sha256",
    }
    if not required <= set(raw):
        raise ValueError("Explicit scoped matrix reference metadata required")
    refs, blocked = (
        set(raw["matrix_reference_training_groups"]),
        set(raw["matrix_reference_excluded_groups"]),
    )
    if (
        not refs
        or refs != set(raw["reference_training_groups"])
        or blocked != set(raw["reference_excluded_groups"])
    ):
        raise ValueError("Matrix references must match exact original nested scope")
    if (
        refs - set(allowed)
        or refs & set(excluded)
        or raw["query_group"] in refs
        or raw["query_group"] not in blocked
    ):
        raise ValueError("Own/outer/inner project entered matrix reference")
    added = [peer.PREFIX + f for f in peer.FIELDS]
    if len(raw["evidence"].columns) != 116 or not set(added) <= set(raw["evidence"]):
        raise ValueError("Exactly108old plus8matrix descriptor schema required")
    if not all(
        str(raw["evidence"][c].dtype).startswith(("float", "int")) for c in added
    ):
        raise ValueError("Matrix descriptors must be numeric with missing unknown")
    if not raw["pool"].stream.astype(str).equals(raw["evidence"].stream.astype(str)):
        raise ValueError("Physical and predictor acquisition routes differ")


def load_predictors(dataset, mask):
    receipt = json.loads((nested.OUT / "blind_receipt.json").read_text())
    assert receipt["all_passed"] and receipt["query_tables"] == 363
    layer = nested.layer_path(dataset, mask)
    reference = nested.OUT / "models" / nested.mask_name(mask) / "model.json"
    for path in (layer, reference):
        assert nested.pilot.sha(path) == receipt["artifact_sha256"][str(path)]
    raw = nested.load_predictors(dataset, mask)
    datasets, folds = inventory()
    allowed = {d[:7] for d in datasets if folds[d] not in mask}
    excluded = {d[:7] for d in datasets if folds[d] in mask}
    raw = dict(raw, matrix_layer_sha256=nested.pilot.sha(layer))
    validate_record(raw, allowed, excluded, query_is_training=False)
    assert not peer.FORBIDDEN & set(raw["pool"])
    assert raw["matrix_layer_sha256"] == receipt["artifact_sha256"][str(layer)]
    return raw


def authorize_labels(raw, allowed, excluded):
    """Reference and scoring permission checked BEFORE manual table opening."""
    validate_record(raw, allowed, excluded, query_is_training=True)
    return old.authorize_labels(raw, allowed, excluded)


def training_records(outer, inner):
    datasets, folds = inventory()
    blocked = {outer} if inner is None else {outer, inner}
    allowed = {d[:7] for d in datasets if folds[d] not in blocked}
    excluded = {d[:7] for d in datasets if folds[d] in blocked}
    records = {}
    for dataset in datasets:
        if folds[dataset] in blocked:
            continue
        raw = load_predictors(dataset, blocked | {folds[dataset]})
        records[dataset], _ = authorize_labels(raw, allowed, excluded)
    return records, allowed, excluded
