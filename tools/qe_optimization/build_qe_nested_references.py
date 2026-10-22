"""Reference-model preparation only; no annotation selections/performance."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from context import WORKSPACE as ROOT, REPO
from frozen_resources import configure_frozen_resources

QE = ROOT / "outputs/qe_optimization_20261022"
OUT = QE / "nested_reference_development_v2"
PARITY = QE / "nested_preparation_parity_v1/summary.json"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def same(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert np.isclose(a, b, rtol=1e-12, atol=1e-12), (a, b)
    else:
        assert a == b, (a, b)


def initialize():
    proof = json.loads(PARITY.read_text())
    assert (
        proof["all_passed"]
        and proof["datasets"] == 33
        and proof["dataset_fold_checks"] == 165
    )
    if not (OUT / "protocol.json").exists():
        OUT.mkdir(exist_ok=False)
        snapshot = OUT / "source_snapshot"
        snapshot.mkdir()
        for name in ("app", "reference_data", "methods"):
            shutil.copytree(
                REPO / name,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        hashes = {
            p.relative_to(snapshot).as_posix(): sha(p)
            for p in snapshot.rglob("*")
            if p.is_file()
        }
        inputs = [
            Path(__file__).with_name("frozen_resources.py"),
            PARITY,
            QE / "rt_atlas_development/observations.pkl",
            QE / "rt_atlas_development/fold_assignment.json",
        ]
        for fold in range(5):
            inputs.extend(
                [
                    QE / "rt_atlas_development" / f"fold_{fold}_atlas.json",
                    QE / "experiment_001" / f"fold_{fold}_selection_model.json",
                    QE / "experiment_003" / f"fold_{fold}_composition_envelope.json",
                ]
            )
        protocol = dict(
            purpose="REFERENCE MODEL PREPARATION ONLY. No classifier fitting, annotation acceptance or production default changes. Five old single-block references must match; twenty new pair/triple exclusions prepare nested training.",
            datasets=33,
            project_groups=30,
            expected_masks=25,
            numerical_workers=3,
            source_sha256=hashes,
            resource_receipt=configure_frozen_resources(snapshot),
            replaces="v1 preserved: app-only snapshot fell back to installed older PE ranges. No annotation outputs were produced. Rebuild all25 bundles with full explicit frozen resources.",
            input_sha256={str(p): sha(p) for p in inputs},
        )
        (OUT / "protocol.json").write_text(
            json.dumps(protocol, indent=2), encoding="utf-8"
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
    protocol = json.loads((OUT / "protocol.json").read_text())
    for name, digest in protocol["source_sha256"].items():
        assert sha(OUT / "source_snapshot" / name) == digest, name
    for path, digest in protocol["input_sha256"].items():
        assert sha(Path(path)) == digest, path
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    assert (
        configure_frozen_resources(OUT / "source_snapshot")
        == protocol["resource_receipt"]
    )
    from matlab_annotation.nested_feature_folds import plan_nested_feature_folds

    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    plan = plan_nested_feature_folds(folds, {d: d[:7] for d in folds})
    return protocol, list(plan["models"])


def fit(mask):
    dest = OUT / ("exclude_" + "_".join(map(str, mask)) + ".json")
    if dest.exists():
        return json.loads(dest.read_text())
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.nested_reference_models import fit_nested_reference_bundle

    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    data = pd.read_pickle(QE / "rt_atlas_development/observations.pkl")
    model = fit_nested_reference_bundle(data, folds, {d: d[:7] for d in folds}, mask)
    if len(mask) == 1:
        f = mask[0]
        same(
            model["atlas"],
            json.loads(
                (QE / "rt_atlas_development" / f"fold_{f}_atlas.json").read_text()
            )["atlas"],
        )
        same(
            model["composition_priors"],
            json.loads(
                (QE / "experiment_001" / f"fold_{f}_selection_model.json").read_text()
            )["composition_priors"],
        )
        same(
            model["composition_envelope"],
            json.loads(
                (
                    QE / "experiment_003" / f"fold_{f}_composition_envelope.json"
                ).read_text()
            )["envelope"],
        )
        model["original_single_fold_parity"] = True
    dest.write_text(json.dumps(model, indent=2, allow_nan=False), encoding="utf-8")
    return model


def main():
    protocol, masks = initialize()
    rows = []
    with ProcessPoolExecutor(max_workers=protocol["numerical_workers"]) as executor:
        for future in as_completed([executor.submit(fit, m) for m in masks]):
            model = future.result()
            rows.append(
                dict(
                    excluded_folds="_".join(map(str, model["excluded_folds"])),
                    excluded_blocks=len(model["excluded_folds"]),
                    training_groups=len(model["training_groups"]),
                    training_rows=model["training_rows"],
                    rt_class_models=len(model["atlas"]["models"]),
                    composition_class_models=len(model["composition_priors"]),
                    range_class_models=len(model["composition_envelope"]["models"]),
                    old_single_fold_parity=model.get(
                        "original_single_fold_parity", False
                    ),
                )
            )
            print("NESTED REFERENCES FIT", rows[-1], flush=True)
    assert len(rows) == 25
    pd.DataFrame(rows).sort_values("excluded_folds").to_csv(
        OUT / "by_exclusion_mask.csv", index=False
    )
    summary = dict(
        reference_bundles=25,
        original_single_fold_parity_checks=5,
        all_complete=True,
        annotation_changes=False,
        note=protocol["purpose"],
        model_sha256={p.name: sha(p) for p in OUT.glob("exclude_*.json")},
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
