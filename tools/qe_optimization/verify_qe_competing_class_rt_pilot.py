"""Reproduce ALL33 blind numerical predictions and independent spike inputs.

No annotations/classifiers/truth or sample subset shortcuts. Fresh scratch
outputs only; master PASS requires all source/input/artifact hashes unchanged.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import tempfile

import pandas as pd

from context import WORKSPACE
from frozen_resources import configure_frozen_resources
from run_qe_standard_mode_experiment import sha, dump
import diagnose_qe_competing_class_rt as pilot

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = pilot.OUT


def verify_case(dataset):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.declared_standard_context import (
        independent_standard_measurements,
    )

    folds = json.loads(pilot.FOLDS.read_text())
    baseline = pd.read_pickle(pilot.BASE / "runs" / dataset / "full_annotation.pkl")
    staged = pd.read_pickle(
        QE / "nested_standard_development_v1/automatic_inputs" / (dataset + ".pkl")
    )
    model = json.loads(
        (
            QE / "nested_standard_development_v1" / f"exclude_{folds[dataset]}.json"
        ).read_text()
    )
    assert (
        dataset[:7] in model["excluded_groups"]
        and dataset[:7] not in model["training_groups"]
    )
    blind = dict(
        baseline,
        streams={
            s: dict(p, annotations=staged["automatic"][s])
            for s, p in baseline["streams"].items()
        },
    )
    measured, _ = independent_standard_measurements(blind, staged["declared"], model)
    original = pd.read_pickle(
        pilot.STD / "measurements" / f"exclude_{folds[dataset]}" / (dataset + ".pkl")
    )
    pd.testing.assert_frame_equal(measured, original, check_exact=True)
    spec = importlib.util.spec_from_file_location(
        "isolated_coherent_pilot", Path(pilot.__file__)
    )
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    with tempfile.TemporaryDirectory(
        prefix="laser-coherent-numeric-replay-"
    ) as scratch:
        root = Path(scratch)
        (root / "source_snapshot").symlink_to(
            OUT / "source_snapshot", target_is_directory=True
        )
        for name in ("checkpoints", "predictions", "audits"):
            (root / name).mkdir()
        replay.OUT = root
        replay.predict(dataset)
        for name in ("predictions", "audits"):
            pd.testing.assert_frame_equal(
                pd.read_pickle(root / name / (dataset + ".pkl")),
                pd.read_pickle(OUT / name / (dataset + ".pkl")),
                check_exact=True,
            )
    print("COHERENT RT ALL4PREDICTIONS/FRESH_SPIKES EXACT", dataset, flush=True)
    return dataset


def main():
    protocol, projects, folds = pilot.initialize()
    ready = json.loads((OUT / "blind_predictions_complete.json").read_text())
    assert ready["all_33_complete"] and ready["dataset_modes"] == 132
    assert ready["manual_evaluation_loaded"] is False
    artifacts = ready["prediction_sha256"]
    additional = [
        QE / "nested_standard_development_v1" / f"exclude_{f}.json"
        for f in set(folds.values())
    ]
    binding = {str(p): sha(p) for p in additional}
    own_hash = sha(Path(__file__))
    for p, h in artifacts.items():
        assert sha(Path(p)) == h
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(verify_case, d) for d in projects]
        completed = [f.result() for f in as_completed(futures)]
    assert set(completed) == set(projects) and len(completed) == 33
    pilot.check_binding(protocol)
    assert sha(Path(__file__)) == own_hash
    for p, h in {**artifacts, **binding}.items():
        assert sha(Path(p)) == h
    dump(
        OUT / "integrity_receipt.json",
        dict(
            all_passed=True,
            annotation_changes=False,
            classifier_fits=0,
            manual_evaluation_loaded=False,
            datasets=33,
            full_blind_stream_prediction_reproductions=132,
            full_blind_dataset_audit_reproductions=33,
            freshly_reproduced_independent_standard_inputs=33,
            source_sha256=own_hash,
            protocol_sha256=sha(OUT / "protocol.json"),
            blind_all33_receipt_sha256=sha(OUT / "blind_predictions_complete.json"),
            source_hash_checks=len(protocol["source_sha256"]),
            input_hash_checks=len(protocol["input_sha256"]) + len(binding),
            bound_prediction_artifacts=len(artifacts),
            note="Engineering replay,not identification performance or analytical FDR. ALL33/132 entire numerical tables reproduced exactly in new scratch;originals untouched. Fresh spike measurements matched frozen scope-matched inputs exactly. No manual truth loaded by this verifier.",
        ),
    )


if __name__ == "__main__":
    main()
