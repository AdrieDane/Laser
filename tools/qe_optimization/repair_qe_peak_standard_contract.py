"""Preserve011 storage-fixed snapshot; allow ONLY protected duplicate standard labels."""

from collections import Counter
import hashlib
import json
import shutil

import pandas as pd

from context import WORKSPACE as ROOT, REPO

OUT = ROOT / "outputs/qe_optimization_20261022/experiment_011"
PREVIOUS = ROOT / "outputs/qe_optimization_20261022/experiment_009"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


protocol_path = OUT / "protocol.json"
protocol = json.loads(protocol_path.read_text())
assert not (OUT / "source_snapshot_before_protected_standard_fix").exists(), (
    "Repair already applied"
)
source = OUT / "source_snapshot"
for rel, expected in protocol["source_snapshot_sha256"].items():
    assert sha(source / rel) == expected, rel
allowed = {
    "app/matlab_annotation/conditional_peak_choice.py",
    "app/tests/test_conditional_peak_choice.py",
}
changed = {
    rel
    for rel in protocol["source_snapshot_sha256"]
    if (REPO / rel).is_file() and sha(source / rel) != sha(REPO / rel)
}
assert changed == allowed, changed
completed = {}
keys = ["stream", "name", "adduct", "is_is"]
for marker in (OUT / "runs").glob("*/complete.json"):
    d = marker.parent.name
    before = pd.read_csv(
        PREVIOUS / "runs" / d / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        low_memory=False,
    )
    after = pd.read_csv(
        marker.parent / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        low_memory=False,
    )
    assert not before.duplicated(keys, keep=False).any(), (
        "Already completed input contains duplicated identities; audit first"
    )
    assert Counter(before[keys].itertuples(index=False, name=None)) == Counter(
        after[keys].itertuples(index=False, name=None)
    )
    std = before.is_is.fillna(False).astype(bool)
    pd.testing.assert_frame_equal(
        before.loc[std].reset_index(drop=True),
        after.loc[std, before.columns].reset_index(drop=True),
        check_dtype=False,
    )
    completed[d] = sha(marker.parent / "automatic_annotations.csv")
models = {p.name: sha(p) for p in OUT.glob("fold_*_conditional_peak_model.pkl")}
shutil.copy2(protocol_path, OUT / "protocol_before_protected_standard_fix.json")
shutil.move(source, OUT / "source_snapshot_before_protected_standard_fix")
shutil.copytree(OUT / "source_snapshot_before_protected_standard_fix", source)
for rel in allowed:
    shutil.copy2(REPO / rel, source / rel)
protocol["source_snapshot_sha256"] = {
    rel: sha(source / rel) for rel in protocol["source_snapshot_sha256"]
}
protocol["protected_standard_contract_repair"] = dict(
    kind="Permit inherited duplicated STANDARD/manual identities; keep every protected row/peak unchanged. Nonprotected duplicates still rejected. Identity MULTISET invariant enforced.",
    preserved_snapshot="source_snapshot_before_protected_standard_fix",
    changed_paths=sorted(allowed),
    unchanged_models_sha256=models,
    completed_outputs_sha256=completed,
    scientific_rules="No score/threshold/replacement/training changes; earlier completed inputs have no duplicate identities so this guard change cannot affect their decisions.",
)
protocol_path.write_text(
    json.dumps(protocol, indent=2, allow_nan=False), encoding="utf-8"
)
for name, expected in models.items():
    assert sha(OUT / name) == expected
for d, expected in completed.items():
    assert sha(OUT / "runs" / d / "automatic_annotations.csv") == expected
print(
    "PROTECTED-STANDARD CONTRACT FIX ONLY; prior snapshots/models/outputs preserved:",
    len(completed),
)
