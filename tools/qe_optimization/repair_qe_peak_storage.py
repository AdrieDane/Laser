"""Preserve initial011 snapshot; repair ONLY CSV column compatibility in same run."""

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
assert not (OUT / "source_snapshot_initial").exists(), (
    "Repair already applied; inspect before repeating"
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
completed = []
for marker in (OUT / "runs").glob("*/complete.json"):
    state = json.loads(marker.read_text())
    assert not state["actions"], "An initial run changed peaks; audit before repair"
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
    pd.testing.assert_frame_equal(before, after, check_dtype=False)
    completed.append(d)
models = {p.name: sha(p) for p in OUT.glob("fold_*_conditional_peak_model.pkl")}
shutil.copy2(protocol_path, OUT / "protocol_initial.json")
shutil.move(source, OUT / "source_snapshot_initial")
shutil.copytree(OUT / "source_snapshot_initial", source)
for rel in allowed:
    shutil.copy2(REPO / rel, source / rel)
protocol["source_snapshot_sha256"] = {
    rel: sha(source / rel) for rel in protocol["source_snapshot_sha256"]
}
protocol["engineering_repair"] = dict(
    kind="CSV empty-text column dtype compatibility only; model/replacement choices unchanged",
    preserved_snapshot="source_snapshot_initial",
    preserved_protocol="protocol_initial.json",
    changed_paths=sorted(allowed),
    unchanged_models_sha256=models,
    completed_before_repair=completed,
)
protocol_path.write_text(
    json.dumps(protocol, indent=2, allow_nan=False), encoding="utf-8"
)
for name, expected in models.items():
    assert sha(OUT / name) == expected
print(
    "REPAIRED STORAGE ONLY; original snapshot preserved, models unchanged, completed zero-change outputs identical:",
    completed,
)
