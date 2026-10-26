"""Research-only alternate learner on existing validated evidence, no annotation.

No files, curation loading, feature generation or automatic acceptance here.
Production has no CatBoost dependency. Install only in an isolated research env.
"""

from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd

FORMAT = "laser-fixed-four-mode-catboost-evidence-v1"
MODES = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")
PARAMETERS = dict(
    iterations=800,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    loss_function="Logloss",
    eval_metric="Logloss",
    boosting_type="Ordered",
    bootstrap_type="Bayesian",
    bagging_temperature=1.0,
    random_strength=1.0,
    max_ctr_complexity=1,
    one_hot_max_size=2,
    nan_mode="Min",
    random_seed=20261026,
    thread_count=1,
    task_type="CPU",
    allow_writing_files=False,
    use_best_model=False,
    verbose=False,
)


def environment():
    import catboost
    import catboost._catboost
    import scipy
    import sklearn

    if catboost.__version__ != "1.2.10":
        raise RuntimeError("Pinned CatBoost1.2.10 research runtime required")
    binary = Path(catboost._catboost.__file__)
    with binary.open("rb") as f:
        binary_sha = hashlib.file_digest(f, "sha256").hexdigest()
    return dict(
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        machine=platform.machine(),
        catboost=catboost.__version__,
        catboost_native_sha256=binary_sha,
        numpy=np.__version__,
        pandas=pd.__version__,
        scipy=scipy.__version__,
        sklearn=sklearn.__version__,
    )


def matrix(evidence, categorical_columns):
    from matlab_annotation.evidence_training import FORBIDDEN

    if FORBIDDEN & set(evidence) or not set(categorical_columns).issubset(evidence):
        raise ValueError("Only validated evidence and an aligned schema are allowed")
    result = evidence.copy()
    for c in result:
        if c in categorical_columns:
            result[c] = result[c].astype(str)
        else:
            result[c] = pd.to_numeric(result[c], errors="coerce").to_numpy(float)
            result[c] = result[c].where(np.isfinite(result[c]), np.nan)
    return result


def fitting_matrix(records, training_groups, excluded_groups):
    """EXACT original known rows/labels and equal-project weighting, not tuning."""
    from matlab_annotation.evidence_training import validate_feature_scope

    xs, ys, groups = [], [], []
    columns = None
    for data in records.values():
        validate_feature_scope(
            data, training_groups, excluded_groups, query_is_training=True
        )
        pool, evidence = data["pool"], data["evidence"]
        if not {"training_positive", "training_label_known"}.issubset(pool):
            raise ValueError("Separate authorized boolean training labels required")
        if (
            not pool[["training_positive", "training_label_known"]]
            .isin([False, True])
            .all()
            .all()
        ):
            raise ValueError("Explicit boolean training flags required")
        if (pool.training_positive & ~pool.training_label_known).any():
            raise ValueError("Known positive training rows required")
        if columns is None:
            columns = list(evidence)
        if list(evidence) != columns:
            raise ValueError("Identical ordered predictor schemas required")
        known = pool.training_label_known.astype(bool)
        xs.append(evidence.loc[known, columns])
        ys.extend(pool.loc[known, "training_positive"].astype(int))
        groups.extend([data["query_group"]] * int(known.sum()))
    if set(groups) != set(training_groups):
        raise ValueError("Every permitted project must contribute known rows")
    x = pd.concat(xs, ignore_index=True)
    categorical = [c for c in x if not pd.api.types.is_numeric_dtype(x[c])]
    x = matrix(x, categorical)
    counts = Counter(groups)
    weights = np.array([len(groups) / len(counts) / counts[g] for g in groups], float)
    return x, np.asarray(ys, dtype=int), np.asarray(groups), weights, categorical


def fingerprint(x, y, groups, weights):
    digest = hashlib.sha256(json.dumps(list(x)).encode())
    digest.update(pd.util.hash_pandas_object(x, index=False).to_numpy().tobytes())
    digest.update(np.asarray(y, dtype=np.int64).tobytes())
    digest.update(json.dumps(list(map(str, groups))).encode())
    digest.update(np.asarray(weights, dtype=np.float64).tobytes())
    return digest.hexdigest()


def fit_four_modes(records, *, training_groups, excluded_groups, parameters=None):
    from catboost import CatBoostClassifier
    from matlab_annotation.mode_evidence_experts import _subset

    parameters = dict(PARAMETERS if parameters is None else parameters)
    if (
        parameters.get("allow_writing_files") is not False
        or parameters.get("use_best_model") is not False
        or parameters.get("thread_count") != 1
        or parameters.get("task_type") != "CPU"
    ):
        raise ValueError(
            "No external files, hidden validation selection or variable CPU threading"
        )
    if any(
        c in parameters
        for c in ("early_stopping_rounds", "od_type", "od_wait", "od_pval", "eval_set")
    ):
        raise ValueError("No row-wise early stopping or evaluation data permitted")
    runtime = environment()
    experts = {}
    for mode in MODES:
        subset = {d: _subset(data, mode) for d, data in records.items()}
        subset = {d: data for d, data in subset.items() if len(data["pool"])}
        x, y, groups, weights, categorical = fitting_matrix(
            subset, training_groups, excluded_groups
        )
        model = CatBoostClassifier(**parameters, cat_features=categorical)
        model.fit(x, y, sample_weight=weights)
        assert (
            model.classes_.tolist() == [0, 1]
            and model.tree_count_ == parameters["iterations"]
        )
        experts[mode] = dict(
            estimator=model,
            columns=list(x),
            categorical_columns=categorical,
            training_pairs=len(x),
            positives=int(y.sum()),
            training_cohorts=sorted(training_groups),
            excluded_groups=sorted(excluded_groups),
            fitting_matrix_sha256=fingerprint(x, y, groups, weights),
            project_rows=dict(Counter(groups)),
            native_parameters=model.get_all_params(),
        )
    columns = next(iter(experts.values()))["columns"]
    assert all(m["columns"] == columns for m in experts.values())
    return dict(
        format=FORMAT,
        experts=experts,
        columns=columns,
        training_cohorts=sorted(training_groups),
        excluded_groups=sorted(excluded_groups),
        parameters=parameters,
        runtime=runtime,
        training_pairs=sum(m["training_pairs"] for m in experts.values()),
        positives=sum(m["positives"] for m in experts.values()),
        note="Fixed CPU CatBoost curator-agreement model on original validated inputs. Not a calibrated chemical identification probability/FDR. No annotation decisions or implicit validation set.",
    )


def predict_four_modes(data, model):
    from matlab_annotation.evidence_training import validate_feature_scope
    from matlab_annotation.mode_evidence_experts import _subset

    if model.get("format") != FORMAT or set(model["experts"]) != set(MODES):
        raise ValueError("Complete fixed four-mode CatBoost model required")
    validate_feature_scope(
        data,
        model["training_cohorts"],
        model["excluded_groups"],
        query_is_training=False,
    )
    if list(data["evidence"]) != model["columns"]:
        raise ValueError("Query evidence schema differs from the model")
    parts = []
    for mode in MODES:
        subset = _subset(data, mode)
        if not len(subset["pool"]):
            continue
        expert = model["experts"][mode]
        if (
            expert["training_cohorts"] != model["training_cohorts"]
            or expert["excluded_groups"] != model["excluded_groups"]
        ):
            raise ValueError("Expert and enclosing model scopes disagree")
        x = matrix(subset["evidence"], expert["categorical_columns"])
        predicted = expert["estimator"].predict_proba(x, thread_count=1)[:, 1]
        parts.append(pd.Series(predicted, index=subset["pool"].index))
    scores = (
        pd.concat(parts).reindex(data["pool"].index)
        if parts
        else pd.Series(index=data["pool"].index, dtype=float)
    )
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError("Exactly one finite native score per query required")
    return scores
