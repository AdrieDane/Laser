"""Engineering parity ONLY: no refits, selections, defaults or source mutations."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

from context import WORKSPACE as ROOT, REPO

QE = ROOT / "outputs/qe_optimization_20261022"
OUT = QE / "nested_preparation_parity_v1"
EXP = QE / "experiment_007"
sys.path.insert(0, str(REPO / "app"))
from matlab_annotation.candidate_preparation import (
    prepare_candidate_pool,
    enrich_candidate_pools,
)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check(dataset):
    dest = OUT / "checkpoints" / (dataset + ".json")
    if dest.exists():
        return json.loads(dest.read_text())
    paths = [QE / "baseline/runs" / dataset / "full_annotation.pkl"]
    paths.extend(EXP / "prepared" / f"fold_{f}" / (dataset + ".pkl") for f in range(5))
    receipt = {str(p): sha(p) for p in paths}
    result = pd.read_pickle(paths[0])
    pools = {}
    priors = {}
    old = {}
    for f in range(5):
        atlas_path = QE / "rt_atlas_development" / f"fold_{f}_atlas.json"
        envelope_path = QE / "experiment_003" / f"fold_{f}_composition_envelope.json"
        prior_path = QE / "experiment_001" / f"fold_{f}_selection_model.json"
        for p in (atlas_path, envelope_path, prior_path):
            receipt[str(p)] = sha(p)
        atlas = json.loads(atlas_path.read_text())["atlas"]
        envelope = json.loads(envelope_path.read_text())["envelope"]
        priors[f] = json.loads(prior_path.read_text())["composition_priors"]
        pools[f] = prepare_candidate_pool(result, atlas, envelope)
        old[f] = pd.read_pickle(paths[f + 1])
    cache = {}
    for p in (QE / "experiment_004/raw_evidence" / dataset).glob("*.pkl"):
        receipt[str(p)] = sha(p)
        stream, charge = p.stem.rsplit("_z", 1)
        frame = pd.read_pickle(p).set_index("feature_id")
        cache[(stream, int(charge))] = frame[[c for c in frame if c.startswith("q_")]]
    new, _ = enrich_candidate_pools(pools, result, priors, raw_quality_cache=cache)
    records = []
    keys = ["stream", "name", "adduct", "feat_feature_id"]
    coordinates = [
        "class",
        "carbon",
        "db",
        "ref_mz",
        "ref_rt",
        "feat_mz",
        "feat_rt",
        "feat_intensity",
        "ppm_error",
        "rt_error",
    ]
    for f in range(5):
        a = new[f]["pool"].copy()
        b = old[f]["pool"].copy()
        a.feat_feature_id = a.feat_feature_id.astype(str)
        b.feat_feature_id = b.feat_feature_id.astype(str)
        assert not a.duplicated(keys).any() and not b.duplicated(keys).any()
        assert set(a[keys].itertuples(index=False, name=None)) == set(
            b[keys].itertuples(index=False, name=None)
        ), (dataset, f, "Physical keys changed")
        ai = pd.MultiIndex.from_frame(a[keys])
        bi = pd.MultiIndex.from_frame(b[keys])
        ac = a[coordinates].set_axis(ai).reindex(bi)
        bc = b[coordinates].set_axis(bi)
        pd.testing.assert_frame_equal(
            ac, bc, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12
        )
        ae = new[f]["evidence"].copy()
        be = old[f]["evidence"]
        assert set(ae) == set(be), (
            dataset,
            f,
            "Evidence schema changed",
            set(ae) ^ set(be),
        )
        ae = ae.set_axis(ai).reindex(bi).set_axis(be.index).reindex(columns=be.columns)
        pd.testing.assert_frame_equal(
            ae, be, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12
        )
        records.append(
            dict(
                fold=f,
                candidate_rows=len(a),
                physical_keys_match=True,
                coordinates_match=True,
                all_model_evidence_match=True,
            )
        )
    for p, digest in receipt.items():
        assert sha(Path(p)) == digest, p
    proof = dict(
        dataset=dataset, folds=records, input_sha256=receipt, annotation_changes=False
    )
    dest.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    return proof


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "checkpoints").mkdir(exist_ok=True)
    files = [REPO / "app/matlab_annotation/candidate_preparation.py"]
    helpers = [
        "matlab_annotation/chemical_targets.py",
        "matlab_annotation/class_candidates.py",
        "matlab_annotation/dual_candidates.py",
        "matlab_annotation/retention_atlas.py",
        "matlab_annotation/run_context_evidence.py",
        "matlab_annotation/cross_mode_evidence.py",
        "annotator/peak_evidence.py",
        "annotator/composition_ranges.py",
    ]
    for name in helpers:
        assert (REPO / "app" / name).read_bytes() == (
            EXP / "source_snapshot/app" / name
        ).read_bytes(), name
        files.append(REPO / "app" / name)
    source = {str(p): sha(p) for p in files}
    manifest = OUT / "source_manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == source
    else:
        manifest.write_text(json.dumps(source, indent=2), encoding="utf-8")
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        shutil.copy2(files[0], OUT / "candidate_preparation_source.py")
    projects = json.loads((QE / "baseline/run_manifest.json").read_text())["datasets"]
    results = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        for future in as_completed([executor.submit(check, d) for d in projects]):
            result = future.result()
            results.append(result)
            print(
                "UNIFIED PARITY PASSED",
                result["dataset"],
                sum(r["candidate_rows"] for r in result["folds"]),
                flush=True,
            )
    rows = [dict(dataset=r["dataset"], **fold) for r in results for fold in r["folds"]]
    pd.DataFrame(rows).sort_values(["dataset", "fold"]).to_csv(
        OUT / "by_dataset_fold.csv", index=False
    )
    summary = dict(
        datasets=len(results),
        dataset_fold_checks=len(rows),
        all_passed=True,
        source_sha256=source,
        purpose="Engineering candidate/evidence parity,NOT new annotation accuracy. No label/model fitting or acceptance/default change; original007queries and every predictor match within1e-12.",
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
