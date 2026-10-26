"""Fresh native HGB mode experts plus explicit matrix-reference scope controls.

Numerical family/parameters/weights/labels unchanged from029. Actual049 runner
must enforce independent363/preflight permission before using real QE records.
No file writes, operating points, annotation selection or query gold here.
"""

from collections import Counter
import hashlib

import numpy as np
import pandas as pd

import matrix_experiment_data as data


def digest(values):
    array = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(array.shape).encode())
    h.update(array.dtype.str.encode())
    h.update(array.tobytes())
    return h.hexdigest()


def fitting_ledger(records, model):
    """Independent ordered mode rows/labels/category codes/equalproject weights."""
    ledgers = {}
    for mode in data.MODES:
        frames, labels, groups = [], [], []
        for raw in records.values():
            pool, evidence = raw["pool"], raw["evidence"]
            selected = np.flatnonzero(
                pool.stream.eq(mode).to_numpy()
                & pool.training_label_known.to_numpy(bool)
            )
            frames.append(evidence.iloc[selected])
            labels.extend(pool.training_positive.to_numpy(int)[selected])
            groups.extend([raw["query_group"]] * len(selected))
        frame = pd.concat(frames, ignore_index=True)
        expert = model["experts"][mode]
        arrays = []
        for column in expert["columns"]:
            if column in expert["categories"]:
                arrays.append(
                    np.asarray(
                        [expert["categories"][column][str(v)] for v in frame[column]],
                        float,
                    )
                )
            else:
                arrays.append(
                    pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
                )
        matrix = np.column_stack(arrays)
        counts = Counter(groups)
        target = len(groups) / len(counts)
        weights = np.asarray([target / counts[g] for g in groups])
        ledgers[mode] = dict(
            rows=len(frame),
            positives=int(sum(labels)),
            contributors=sorted(counts),
            matrix_sha256=digest(matrix),
            labels_sha256=digest(np.asarray(labels, int)),
            weights_sha256=digest(weights),
            equal_project_weight=target,
        )
    return ledgers


def fit(records, *, training_groups, excluded_groups, parameters):
    from matlab_annotation.mode_evidence_experts import fit_mode_evidence_experts

    if parameters != data.parameters():
        raise ValueError("Exactly original029 native HGB parameters required")
    if {r["query_group"] for r in records.values()} != set(training_groups):
        raise ValueError("Every permitted project must contribute")
    for raw in records.values():
        data.validate_record(
            raw, training_groups, excluded_groups, query_is_training=True
        )
    model = fit_mode_evidence_experts(
        records,
        training_groups=training_groups,
        excluded_groups=excluded_groups,
        parameters=parameters,
    )
    model["matrix_feature_scopes"] = {
        d: dict(
            training=r["matrix_reference_training_groups"],
            excluded=r["matrix_reference_excluded_groups"],
            reference_sha256=r["matrix_reference_sha256"],
            layer_sha256=r["matrix_layer_sha256"],
            excluded_folds=r.get("excluded_folds", []),
        )
        for d, r in records.items()
    }
    model["independent_fitting_ledger"] = fitting_ledger(records, model)
    for mode, expert in model["experts"].items():
        assert expert["estimator"].n_iter_ == 180
        assert (
            expert["training_pairs"]
            == model["independent_fitting_ledger"][mode]["rows"]
        )
        assert (
            expert["positives"]
            == model["independent_fitting_ledger"][mode]["positives"]
        )
        assert "sample_weight_policy" not in expert
    return model


def predict(raw, model):
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts

    data.validate_record(
        raw,
        model["training_cohorts"],
        model["excluded_groups"],
        query_is_training=False,
    )
    if data.peer.FORBIDDEN & set(raw["pool"]):
        raise ValueError("Prediction refuses query supervision")
    scores = predict_mode_evidence_experts(raw, model)
    assert scores.index.equals(raw["pool"].index)
    assert np.isfinite(scores).all() and scores.between(0, 1).all()
    return scores
