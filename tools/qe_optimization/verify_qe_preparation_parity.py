"""Verify a computational acceleration against the frozen original candidates."""

import hashlib
import json
import pickle
import sys
import time

import pandas as pd

from context import WORKSPACE as ROOT, REPO

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
EXP = (
    ROOT
    / "outputs/qe_optimization_20261022/experiment_001_preparation_before_acceleration"
)
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
sys.path.insert(0, str(REPO / "app"))
from matlab_annotation.class_candidates import (
    prepare_class_candidates,
    candidate_evidence,
    fit_class_composition_priors,
    attach_composition_prior_evidence,
)

dataset = "21P0116_Zoemba__fff844b15f6b"
with (BASE / "runs" / dataset / "full_annotation.pkl").open("rb") as handle:
    result = pickle.load(handle)
observations = pd.read_pickle(ATLAS / "observations.pkl")
folds = json.loads((ATLAS / "fold_assignment.json").read_text())
records = []
start = time.monotonic()
for fold in range(5):
    old = pd.read_pickle(EXP / "prepared" / f"fold_{fold}" / (dataset + ".pkl"))
    atlas = json.loads((ATLAS / f"fold_{fold}_atlas.json").read_text())["atlas"]
    priors = fit_class_composition_priors(
        observations[observations.dataset.map(folds).ne(fold) & observations.mapped]
    )
    pools = []
    for stream, payload in result["streams"].items():
        pool, _ = prepare_class_candidates(
            payload["annotations"],
            payload["reference"],
            payload["features"],
            stream=stream,
            atlas=atlas,
        )
        pools.append(pool)
    pool = pd.concat(pools, ignore_index=True)
    old_pool = old["pool"]
    assert len(pool) == len(old_pool)
    pd.testing.assert_frame_equal(
        pool, old_pool[pool.columns], check_exact=False, rtol=1e-12, atol=1e-12
    )
    for col in [c for c in old_pool if c.startswith("q_")]:
        pool[col] = old_pool[col].to_numpy()
    evidence = pd.concat(
        [
            attach_composition_prior_evidence(candidate_evidence(group), priors)
            for _, group in pool.groupby("stream", sort=False)
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(
        evidence, old["evidence"], check_exact=False, rtol=1e-12, atol=1e-12
    )
    records.append(
        dict(
            fold=fold,
            pairs=len(pool),
            all_candidate_fields_match=True,
            all_model_evidence_matches=True,
        )
    )
    print(
        "PARITY PASSED",
        fold,
        len(pool),
        "pairs",
        round(time.monotonic() - start, 2),
        "seconds",
        flush=True,
    )
output = dict(
    dataset=dataset,
    folds=records,
    elapsed_seconds=time.monotonic() - start,
    note="Only loop vectorization and memoization of immutable composition decisions. No changed selection settings or scientific outputs.",
    new_sources={
        name: hashlib.sha256(
            (REPO / "app/matlab_annotation" / name).read_bytes()
        ).hexdigest()
        for name in ["retention_atlas.py", "class_candidates.py"]
    },
)
(
    ROOT / "outputs/qe_optimization_20261022/preparation_acceleration_parity.json"
).write_text(json.dumps(output, indent=2), encoding="utf-8")
