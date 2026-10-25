"""Unused research: predeclared chemical-family/mode scoring, not class winners.

Fixed routing does not use curated class presence, matrix, project or identity
as a predictor. ALL query classes still compete together downstream. Sparse
training families fall back to the explicit unchanged four-mode scorer.
"""

from collections import Counter
import re

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .evidence_training import validate_feature_scope, evidence_matrix
from .mode_evidence_experts import MODES, predict_mode_evidence_experts

FORMAT = "laser-fixed-family-mode-experts-v1"
FAMILIES = ("glycerophospholipid", "sphingolipid", "glycerolipid", "fatty_acid_ester")


def chemical_family(value):
    """Deterministic catalogue class routing, not a learned whitelist."""
    cls = re.sub(r"^[12]-acyl\s*", "", canonical_lipid_class(value)).split("[")[0]
    if cls in {"BMP", "SLBPA", "CL", "MLCL", "DLCL"} or cls.split("-")[0] in {
        "PC",
        "PE",
        "PI",
        "PS",
        "PG",
        "PA",
        "LPC",
        "LPE",
        "LPI",
        "LPS",
        "LPG",
        "LPA",
    }:
        return "glycerophospholipid"
    if (
        "Cer" in cls
        or cls.split("-")[0] in {"SM", "SM4", "SPH", "C1P", "S1P"}
        or re.match(r"G[AMDTQ][1-4](?:-|$)", cls)
    ):
        return "sphingolipid"
    if cls.split("-")[0] in {"MG", "DG", "TG"}:
        return "glycerolipid"
    if cls.split("-")[0] in {"FA", "AC", "CE", "WE"}:
        return "fatty_acid_ester"
    return "unsupported"


def _keys(evidence):
    if (
        not {"stream", "class"}.issubset(evidence)
        or not evidence.stream.isin(MODES).all()
    ):
        raise ValueError(
            "Explicit known acquisition stream and chemical class required"
        )
    return evidence.stream.astype(str) + "|" + evidence["class"].map(chemical_family)


def _check_fallback(fallback, training_groups, excluded_groups, columns):
    if set(fallback["training_cohorts"]) != set(training_groups) or set(
        fallback["excluded_groups"]
    ) != set(excluded_groups):
        raise ValueError("Fallback model scope differs from family model")
    if set(fallback["columns"]) != set(columns):
        raise ValueError("Fallback predictor schema differs from family model")
    if set(fallback["experts"]) != set(MODES):
        raise ValueError("All four fallback mode experts required")


def fit_family_evidence_experts(
    data_by_dataset,
    *,
    training_groups,
    excluded_groups,
    parameters,
    fallback_model,
    minimum_rows=200,
    minimum_positive=20,
    minimum_negative=20,
    minimum_positive_projects=3,
):
    """Same180-round family, equal project weight WITHIN each chemical family.

    Reference permissions remain ALL allowed training projects. Actual supervised
    contributors are separately recorded; zero rows are NOT invented to satisfy
    an equality check. Eligibility uses known TRAINING labels only. No model is
    chosen by outer performance. No base labels/predictors are changed.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    if parameters.get("early_stopping") is not False:
        raise ValueError("Implicit row-wise validation must be disabled")
    if (
        min(minimum_rows, minimum_positive, minimum_negative) < 1
        or minimum_positive_projects < 2
    ):
        raise ValueError("Explicit positive training support required")
    xs, ys, projects, contributors = [], [], [], set()
    columns = None
    for data in data_by_dataset.values():
        validate_feature_scope(
            data, training_groups, excluded_groups, query_is_training=True
        )
        pool, evidence = data["pool"], data["evidence"]
        if not {"training_positive", "training_label_known"}.issubset(pool):
            raise ValueError("Separate authorized supervision required")
        if (
            not pool[["training_positive", "training_label_known"]]
            .isin([True, False])
            .all()
            .all()
        ):
            raise ValueError("Explicit boolean labels required")
        if (pool.training_positive & ~pool.training_label_known).any():
            raise ValueError("Positive training rows must be known")
        if columns is None:
            columns = list(evidence)
        if set(columns) != set(evidence):
            raise ValueError("All predictor schemas must agree")
        _keys(evidence)
        known = pool.training_label_known.astype(bool)
        xs.append(evidence.loc[known, columns])
        ys.extend(pool.loc[known, "training_positive"].astype(int))
        projects.extend([data["query_group"]] * int(known.sum()))
        if known.any():
            contributors.add(data["query_group"])
    if contributors != set(training_groups):
        raise ValueError("Every allowed project must contribute known global rows")
    _check_fallback(fallback_model, training_groups, excluded_groups, columns)
    x, y, groups = (
        pd.concat(xs, ignore_index=True),
        np.asarray(ys),
        np.asarray(projects),
    )
    if (
        len(x) != fallback_model["training_pairs"]
        or int(y.sum()) != fallback_model["positives"]
    ):
        raise ValueError("Fallback and new fitting supervision must be identical")
    routing = _keys(x)
    experts, audits = {}, []
    for mode in MODES:
        for family in FAMILIES:
            key = mode + "|" + family
            use = routing.eq(key).to_numpy()
            n, positive, negative = (
                int(use.sum()),
                int(y[use].sum()),
                int((1 - y[use]).sum()),
            )
            actual = sorted(set(groups[use]))
            positive_projects = sorted(set(groups[use & y.astype(bool)]))
            fit = (
                n >= minimum_rows
                and positive >= minimum_positive
                and negative >= minimum_negative
                and len(positive_projects) >= minimum_positive_projects
            )
            audits.append(
                dict(
                    key=key,
                    fitted=fit,
                    known_rows=n,
                    positives=positive,
                    negatives=negative,
                    actual_supervised_projects=actual,
                    positive_projects=positive_projects,
                )
            )
            if not fit:
                continue
            part = x.loc[use].copy()
            categories = {
                c: {v: i for i, v in enumerate(sorted(part[c].astype(str).unique()))}
                for c in part
                if not pd.api.types.is_numeric_dtype(part[c])
            }
            if any(len(v) > 255 for v in categories.values()):
                raise ValueError("Categorical cardinality exceeds HGB support")
            expert = dict(
                columns=columns,
                categories=categories,
                training_cohorts=sorted(training_groups),
                excluded_groups=sorted(excluded_groups),
                supervised_contributing_groups=actual,
                training_pairs=n,
                positives=positive,
            )
            counts = Counter(groups[use])
            weights = np.asarray([n / len(counts) / counts[g] for g in groups[use]])
            estimator = HistGradientBoostingClassifier(
                **parameters, categorical_features=[c in categories for c in part]
            )
            estimator.fit(evidence_matrix(part, expert), y[use], sample_weight=weights)
            expert["estimator"] = estimator
            experts[key] = expert
    return dict(
        format=FORMAT,
        experts=experts,
        columns=columns,
        training_cohorts=sorted(training_groups),
        excluded_groups=sorted(excluded_groups),
        training_pairs=len(x),
        positives=int(y.sum()),
        family_support=audits,
        note="Predeclared family/mode routing; real supervised contributors distinct from allowed reference projects. Sparse training-family fallback is unchanged explicit029model. No held-out class choice or analytical probability/FDR.",
    )


def predict_family_evidence_experts(data, model, fallback_model):
    if model.get("format") != FORMAT:
        raise ValueError("Unsupported family expert model")
    validate_feature_scope(
        data,
        model["training_cohorts"],
        model["excluded_groups"],
        query_is_training=False,
    )
    _check_fallback(
        fallback_model,
        model["training_cohorts"],
        model["excluded_groups"],
        model["columns"],
    )
    if set(model["columns"]) != set(data["evidence"]):
        raise ValueError("Family predictor schema differs from fitting")
    route = _keys(data["evidence"])
    output = predict_mode_evidence_experts(data, fallback_model).copy()
    for key, expert in model["experts"].items():
        if key not in {m + "|" + f for m in MODES for f in FAMILIES}:
            raise ValueError("Unknown family expert routing key")
        if set(expert["training_cohorts"]) != set(model["training_cohorts"]) or set(
            expert["excluded_groups"]
        ) != set(model["excluded_groups"]):
            raise ValueError("Family expert scopes disagree with enclosing model")
        use = route.eq(key)
        if use.any():
            output.loc[use] = expert["estimator"].predict_proba(
                evidence_matrix(data["evidence"].loc[use], expert)
            )[:, 1]
    if not np.isfinite(output.to_numpy(float)).all():
        raise ValueError("All routed predictions must be finite")
    return output
