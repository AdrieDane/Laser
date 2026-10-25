"""Unused research: separate feature detection and conditional label ranking.

No files, old scorer outputs, annotation decisions, thresholds or defaults.
All ranks describe curator agreement, not chemical identity probabilities/FDR.
"""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .add_only_class_completion import MODES
from .evidence_training import evidence_matrix, validate_feature_scope
from .feature_label_examples import (
    GROUP,
    feature_evidence,
    feature_training_targets,
    physical_label_keys,
)

FORMAT = "laser-feature-label-factorization-v1"
SUPERVISION = {
    "training_positive",
    "training_label_known",
    "calibration_identity_in_manual",
    "same_manual_peak",
    "manual_feature_id",
    "manual_confirmed",
    "manually_confirmed",
    "manual_annotation",
}


def _fit(x, y, projects, parameters):
    """Actual supervised contributors, not the full reference permission set."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    y = np.asarray(y, bool)
    projects = np.asarray(projects, str)
    if len(x) != len(y) or len(y) != len(projects):
        raise ValueError("Aligned supervised observations/projects required")
    support = dict(
        observations=len(y),
        positives=int(y.sum()),
        negatives=int((~y).sum()),
        actual_projects=sorted(set(projects)),
        positive_projects=sorted(set(projects[y])),
        negative_projects=sorted(set(projects[~y])),
    )
    if (
        y.sum() < 20
        or (~y).sum() < 20
        or len(set(projects[y])) < 3
        or len(set(projects[~y])) < 3
    ):
        return None, support
    categories = {
        c: {v: i for i, v in enumerate(sorted(x[c].astype(str).unique()))}
        for c in x
        if not pd.api.types.is_numeric_dtype(x[c])
    }
    if any(len(v) > 255 for v in categories.values()):
        raise ValueError("Fixed HGB categorical support exceeded")
    model = dict(columns=list(x), categories=categories, **support)
    counts = Counter(projects)
    weights = np.asarray([len(y) / len(counts) / counts[g] for g in projects])
    estimator = HistGradientBoostingClassifier(
        **parameters, categorical_features=[c in categories for c in x]
    )
    estimator.fit(evidence_matrix(x, model), y.astype(int), sample_weight=weights)
    model["estimator"] = estimator
    return model, support


def _supported_classes(counts, projects):
    return sorted(c for c, n in counts.items() if n >= 20 and len(projects[c]) >= 3)


def prepare_feature_label_training(
    data_by_dataset, *, training_groups, excluded_groups
):
    """Expose the exact no-fit preparation for independent preflight/audits."""
    allowed, excluded = set(training_groups), set(excluded_groups)
    if {d["query_group"] for d in data_by_dataset.values()} != allowed:
        raise ValueError("All permitted training projects must be represented")
    records = {
        s: dict(
            detector_x=[],
            detector_y=[],
            detector_groups=[],
            label_x=[],
            label_y=[],
            label_groups=[],
            class_counts=Counter(),
            class_projects=defaultdict(set),
            conditional_counts=Counter(),
            conditional_projects=defaultdict(set),
        )
        for s in MODES
    }
    query_audit, original_columns, detector_columns = {}, None, None
    for dataset, data in data_by_dataset.items():
        validate_feature_scope(data, allowed, excluded, query_is_training=True)
        pool, evidence = data["pool"], data["evidence"]
        x = feature_evidence(pool, evidence)
        targets, conditional = feature_training_targets(pool)
        keys = physical_label_keys(pool)
        if original_columns is None:
            original_columns = list(evidence)
            detector_columns = list(x)
        if set(original_columns) != set(evidence) or set(detector_columns) != set(x):
            raise ValueError("Unchanged candidate and feature schemas required")
        assert x.index.equals(targets.index)
        project = data["query_group"]
        mode_counts = {}
        for stream in MODES:
            r = records[stream]
            feature_rows = (
                x.index.get_level_values("stream") == stream
            ) & targets.known.to_numpy()
            rows = pool.stream.eq(stream)
            positive = (
                rows
                & pool.training_label_known
                & pool.training_positive
                & ~keys.standard
            )
            cond = rows & conditional
            r["detector_x"].append(x.loc[feature_rows])
            r["detector_y"].extend(targets.loc[feature_rows, "positive"].tolist())
            r["detector_groups"].extend([project] * int(feature_rows.sum()))
            r["label_x"].append(evidence.loc[cond])
            r["label_y"].extend(pool.loc[cond, "training_positive"].tolist())
            r["label_groups"].extend([project] * int(cond.sum()))
            # Physical candidate versions of one positive label count ONCE.
            unique = keys.loc[positive].drop_duplicates([*GROUP, "species", "ion"])
            for cls, n in unique.lipid_class.value_counts().items():
                r["class_counts"][cls] += int(n)
                r["class_projects"][cls].add(project)
            unique = keys.loc[cond & pool.training_positive].drop_duplicates(
                [*GROUP, "species", "ion"]
            )
            for cls, n in unique.lipid_class.value_counts().items():
                r["conditional_counts"][cls] += int(n)
                r["conditional_projects"][cls].add(project)
            mode_counts[stream] = dict(
                feature_known=int(feature_rows.sum()),
                feature_positive=int(targets.loc[feature_rows, "positive"].sum()),
                conditional_rows=int(cond.sum()),
                conditional_positive=int(pool.loc[cond, "training_positive"].sum()),
            )
        query_audit[dataset] = dict(
            project=project,
            reference_training_groups=list(data["reference_training_groups"]),
            reference_excluded_groups=list(data["reference_excluded_groups"]),
            modes=mode_counts,
        )
    return records, query_audit, original_columns, detector_columns


def fit_feature_label_model(
    data_by_dataset, *, training_groups, excluded_groups, parameters
):
    records, query_audit, original_columns, detector_columns = (
        prepare_feature_label_training(
            data_by_dataset,
            training_groups=training_groups,
            excluded_groups=excluded_groups,
        )
    )
    experts = {}
    for stream, r in records.items():
        detector, detector_support = _fit(
            pd.concat(r["detector_x"], ignore_index=True),
            r["detector_y"],
            r["detector_groups"],
            parameters,
        )
        labels, label_support = _fit(
            pd.concat(r["label_x"], ignore_index=True),
            r["label_y"],
            r["label_groups"],
            parameters,
        )
        experts[stream] = dict(
            detector=detector,
            conditional=labels,
            detector_support=detector_support,
            conditional_support=label_support,
            singleton_supported_classes=_supported_classes(
                r["class_counts"], r["class_projects"]
            ),
            ambiguous_supported_classes=_supported_classes(
                r["conditional_counts"], r["conditional_projects"]
            ),
            class_positive_observations=dict(r["class_counts"]),
            class_positive_projects={
                k: sorted(v) for k, v in r["class_projects"].items()
            },
            conditional_class_positive_observations=dict(r["conditional_counts"]),
            conditional_class_positive_projects={
                k: sorted(v) for k, v in r["conditional_projects"].items()
            },
        )
    return dict(
        format=FORMAT,
        training_reference_permissions=sorted(training_groups),
        excluded_groups=sorted(excluded_groups),
        columns=original_columns,
        feature_columns=detector_columns,
        experts=experts,
        training_query_audit=query_audit,
        fresh_classifier_fits=sum(
            e[k] is not None
            for e in experts.values()
            for k in ("detector", "conditional")
        ),
        note="Two distinct curator-agreement targets,not chemical posterior probabilities/FDR. No oldmodelscore/annotation/cutoff inputs. Actualsupervisedcontributors separatefromreferencepermissions;unsupportedwholeclusterswithheld.",
    )


def _predict(frame, model):
    return model["estimator"].predict_proba(evidence_matrix(frame, model))[:, 1]


def predict_feature_label_model(data, model):
    if model.get("format") != FORMAT or set(model["experts"]) != set(MODES):
        raise ValueError("Complete explicit feature/label model required")
    if SUPERVISION & set(data["pool"]):
        raise ValueError("Final query prediction refuses supervision")
    validate_feature_scope(
        data,
        model["training_reference_permissions"],
        model["excluded_groups"],
        query_is_training=False,
    )
    pool, evidence = data["pool"], data["evidence"]
    if set(evidence) != set(model["columns"]):
        raise ValueError("Original candidate predictor schema changed")
    features = feature_evidence(pool, evidence)
    if set(features) != set(model["feature_columns"]):
        raise ValueError("Feature predictor schema changed")
    keys = physical_label_keys(pool)
    out = pd.DataFrame(
        0.0,
        index=pool.index,
        columns=[
            "feature_detection_rank",
            "conditional_label_rank",
            "joint_agreement_rank",
        ],
    )
    out["supported_cluster"] = False
    for stream, expert in model["experts"].items():
        for stage in ("detector", "conditional"):
            fitted = expert[stage]
            if fitted is not None and (
                not set(fitted["actual_projects"]).issubset(
                    model["training_reference_permissions"]
                )
                or set(fitted["actual_projects"]) & set(model["excluded_groups"])
            ):
                raise ValueError(
                    "Stage supervised contributors escape permitted reference scope"
                )
        if expert["detector"] is None:
            continue
        rows = pool.stream.eq(stream)
        if not rows.any():
            continue
        x = features.loc[features.index.get_level_values("stream") == stream]
        detector = pd.Series(_predict(x, expert["detector"]), index=x.index)
        row_keys = keys.loc[rows]
        local = row_keys.copy()
        local["raw_conditional_rank"] = (
            _predict(evidence.loc[rows], expert["conditional"])
            if expert["conditional"] is not None
            else 0.0
        )
        labels = (
            local.groupby([*GROUP, "species", "ion"], sort=False)
            .agg(
                raw_conditional_rank=("raw_conditional_rank", "max"),
                lipid_class=("lipid_class", "first"),
                standard=("standard", "any"),
            )
            .reset_index()
        )
        label_ranks = {}
        for key, group in labels.groupby(GROUP, sort=False):
            detection = float(detector.loc[key])
            indices = row_keys.index[
                row_keys.stream.eq(key[0]) & row_keys.feat_feature_id.eq(key[1])
            ]
            out.loc[indices, "feature_detection_rank"] = detection
            if group.standard.any():
                continue
            if len(group) == 1:
                supported = set(group.lipid_class).issubset(
                    expert["singleton_supported_classes"]
                )
                conditional = np.ones(1)
            else:
                supported = expert["conditional"] is not None and set(
                    group.lipid_class
                ).issubset(expert["ambiguous_supported_classes"])
                total = group.raw_conditional_rank.sum()
                if not np.isfinite(total) or total <= 0:
                    continue
                conditional = group.raw_conditional_rank.to_numpy(float) / total
            if not supported:
                continue
            for row, rank in zip(group.itertuples(), conditional):
                label_ranks[(row.stream, row.feat_feature_id, row.species, row.ion)] = (
                    float(rank)
                )
            out.loc[indices, "supported_cluster"] = True
        for i, row in row_keys.iterrows():
            rank = label_ranks.get(
                (row.stream, row.feat_feature_id, row.species, row.ion), 0.0
            )
            out.loc[i, "conditional_label_rank"] = rank
            out.loc[i, "joint_agreement_rank"] = (
                out.loc[i, "feature_detection_rank"] * rank
            )
    return out
