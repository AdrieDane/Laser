"""Unused research-only BMP completion: audited training, physics and grid.

No file reads/writes, curation lookup, candidate generation or app defaults.
Callers must authorize labels first and enforce complete source/model audits.
"""

from collections import Counter
from itertools import product
import hashlib

import numpy as np
import pandas as pd

from bmp_context_measurements import PREFIX
from prepare_qe_bmp_context_tables import validate_scope

MODES = ("NP_Pos", "NP_Neg")
LEVELS = tuple([round(0.50 + 0.05 * i, 2) for i in range(10)] + [0.975, 0.99, 1.0])
FORMAT = "laser-BMP-candidate-context-completion-v1"


def matrix_receipt(part):
    from matlab_annotation.evidence_training import evidence_matrix

    arrays = dict(
        matrix=evidence_matrix(part["x"], part["metadata"]),
        targets=part["y"],
        sample_weights=part["weights"],
    )
    return {
        name + "_sha256": hashlib.sha256(
            np.ascontiguousarray(value).tobytes()
        ).hexdigest()
        for name, value in arrays.items()
    }


def prepare_training(data_by_dataset, *, training_groups, excluded_groups):
    """Expose exact matrices/targets/cohort weights BEFORE any classifier fit."""
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.mode_evidence_experts import _subset

    allowed, excluded = set(training_groups), set(excluded_groups)
    if {d["query_group"] for d in data_by_dataset.values()} != allowed:
        raise ValueError("Every permitted project must be represented")
    schema = None
    for data in data_by_dataset.values():
        validate_scope(data, allowed, excluded, query_is_training=True)
        audit = data.get("bmp_partial_supervision_audit", {})
        pool = data["pool"]
        if (
            not {"training_positive", "training_label_known"}.issubset(pool)
            or not pool.training_positive.isin([False, True]).all()
            or not pool.training_label_known.isin([False, True]).all()
        ):
            raise ValueError("Explicit supervised flags required")
        if (
            audit.get("evaluation_labels_changed") is not False
            or audit.get("positive_rows_before") != audit.get("positive_rows_after")
            or audit.get("positive_rows_after") != int(pool.training_positive.sum())
            or audit.get("known_rows_after") != int(pool.training_label_known.sum())
        ):
            raise ValueError("Explicit immutable-positive partial-label audit required")
        if schema is None:
            schema = list(data["evidence"])
        elif list(data["evidence"]) != schema:
            raise ValueError("Identical163-column predictor order required")
    if len(schema or []) != 163:
        raise ValueError("Original131 plus32 predictors required")
    result = {}
    for mode in MODES:
        xs, ys, projects = [], [], []
        positives, positive_projects, other_known = 0, set(), 0
        for data in data_by_dataset.values():
            part = _subset(data, mode)
            p, e = part["pool"], part["evidence"]
            known = p.training_label_known.astype(bool)
            positive = p.training_positive.astype(bool)
            bmp = p["class"].map(canonical_lipid_class).eq("BMP") & ~p.is_is.astype(
                bool
            )
            npos = int((known & positive & bmp).sum())
            positives += npos
            if npos:
                positive_projects.add(data["query_group"])
            other_known += int((known & ~bmp).sum())
            xs.append(e.loc[known, schema])
            ys.extend(p.loc[known, "training_positive"].astype(int))
            projects.extend([data["query_group"]] * int(known.sum()))
        if set(projects) != allowed or not other_known:
            raise ValueError("Actual all-class project contributors required")
        x = pd.concat(xs, ignore_index=True)
        categories = {
            c: {v: i for i, v in enumerate(sorted(x[c].astype(str).unique()))}
            for c in x
            if not pd.api.types.is_numeric_dtype(x[c])
        }
        if any(len(v) > 255 for v in categories.values()):
            raise ValueError("Categorical cardinality exceeds HGB support")
        counts = Counter(projects)
        weights = np.array([len(projects) / len(counts) / counts[g] for g in projects])
        metadata = dict(
            columns=list(x),
            categories=categories,
            training_cohorts=sorted(set(projects)),
            excluded_groups=sorted(excluded),
            training_pairs=len(x),
            positives=int(sum(ys)),
            note="Research curator agreement, not chemical confidence/FDR.",
        )
        result[mode] = dict(
            x=x,
            y=np.asarray(ys),
            weights=weights,
            projects=projects,
            metadata=metadata,
            supported=positives >= 20 and len(positive_projects) >= 3,
            support=dict(
                known_BMP_positive_pairs=positives,
                BMP_positive_projects=sorted(positive_projects),
                other_class_known_context_rows=other_known,
                known_rows=len(x),
                known_positive_rows=int(sum(ys)),
                actual_projects=sorted(counts),
                known_rows_by_project=dict(counts),
            ),
        )
    return result


def fit_context(data_by_dataset, *, training_groups, excluded_groups, parameters):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from matlab_annotation.evidence_training import evidence_matrix

    if parameters.get("early_stopping") is not False or "kind" in parameters:
        raise ValueError("Explicit HGB kwargs/no implicit rowwise validation required")
    prepared = prepare_training(
        data_by_dataset,
        training_groups=training_groups,
        excluded_groups=excluded_groups,
    )
    experts = {}
    for mode, part in prepared.items():
        if not part["supported"]:
            experts[mode] = None
            continue
        meta = part["metadata"]
        estimator = HistGradientBoostingClassifier(
            **parameters,
            categorical_features=[c in meta["categories"] for c in meta["columns"]],
        )
        estimator.fit(
            evidence_matrix(part["x"], meta), part["y"], sample_weight=part["weights"]
        )
        experts[mode] = dict(meta, estimator=estimator)
    return dict(
        format=FORMAT,
        experts=experts,
        support={s: p["support"] for s, p in prepared.items()},
        training_cohorts=sorted(training_groups),
        excluded_groups=sorted(excluded_groups),
        columns=next(iter(prepared.values()))["metadata"]["columns"],
        fresh_mode_classifier_fits=sum(v is not None for v in experts.values()),
        training_matrix_receipts={s: matrix_receipt(p) for s, p in prepared.items()},
    )


def predict_context(raw, model):
    from matlab_annotation.raw_counterpart_rt import SUPERVISION
    from matlab_annotation.mode_evidence_experts import _subset
    from matlab_annotation.evidence_training import predict_grouped_evidence

    if SUPERVISION & set(raw["pool"]):
        raise ValueError("Final prediction refuses supervision")
    if model.get("format") != FORMAT or set(model.get("experts", {})) != set(MODES):
        raise ValueError("Complete explicit BMP context model required")
    validate_scope(
        raw,
        model["training_cohorts"],
        model["excluded_groups"],
        query_is_training=False,
    )
    output = pd.Series(
        np.nan,
        index=raw["pool"].loc[raw["pool"].stream.isin(MODES)].index,
        name="BMP_context_agreement_rank",
    )
    for mode, expert in model["experts"].items():
        if expert is not None:
            assert (
                expert["training_cohorts"] == model["training_cohorts"]
                and expert["excluded_groups"] == model["excluded_groups"]
            )
            part = _subset(raw, mode)
            output.loc[part["pool"].index] = predict_grouped_evidence(part, expert)
    return output


def profile_gate(evidence):
    """Fixed domain safeguards; no unknown evidence becomes proof of support."""
    required = [
        PREFIX + f
        for f in (
            "source_physical",
            "candidate_counterparts",
            "counterpart_profiles",
            "counterpart_n",
            "counterpart_raw",
            "counterpart_normalized",
            "counterpart_leave_one_out",
        )
    ]
    if not set(required).issubset(evidence):
        raise ValueError("Complete scoped physical/profile measurements required")
    values = evidence.loc[:, required].apply(pd.to_numeric, errors="coerce")
    return (
        pd.Series(np.isfinite(values).all(axis=1), index=evidence.index)
        & values[required[0]].eq(1)
        & values[required[1]].ge(1)
        & values[required[2]].ge(1)
        & values[required[3]].ge(8)
        & values[required[4]].ge(0.8)
        & values[required[5]].ge(0.8)
        & values[required[6]].ge(0.6)
    )


def select_additions(
    pool,
    scores,
    evidence,
    previous,
    *,
    stream,
    threshold,
    protected_features,
    declared_masses,
):
    from matlab_annotation.np_context_bmp_completion import select_context_bmp_additions

    if not pool.index.equals(evidence.index):
        raise ValueError("Aligned original candidates and scoped measurements required")
    selected = select_context_bmp_additions(
        pool,
        scores,
        previous,
        stream=stream,
        threshold=threshold,
        protected_features=protected_features,
        declared_masses=declared_masses,
    )
    # Fixed global identity/feature winners are selected BEFORE this gate.
    # There is no fallback to a lower-ranked but better-correlating candidate.
    return selected.loc[profile_gate(evidence).reindex(selected.index)].copy()


def choose_joint_grid(table):
    """Complete NP Cartesian grid under PROSPECTIVE budgets, explicit disable.

    Table rows summarize each mode/threshold over authorized INNER queries.
    No truth-reading or manual-presence filtering is performed by this helper.
    """
    required = {
        "stream",
        "threshold",
        "manual",
        "endogenous_manual",
        "previous_correct",
        "added_correct",
        "extras",
        "extras_without_opposite_composition",
        "different_peak",
        "positive_projects",
    }
    if not required.issubset(table) or set(table.stream) != set(MODES):
        raise ValueError("Complete two-mode training grid required")
    numeric = table.loc[:, sorted(required - {"stream"})].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric).all().all() or numeric.lt(0).any().any():
        raise ValueError("Finite nonnegative complete training counts required")
    if (
        table.endogenous_manual.gt(table.manual).any()
        or (table.previous_correct + table.added_correct).gt(table.manual).any()
        or table.extras_without_opposite_composition.gt(table.extras).any()
    ):
        raise ValueError("Inconsistent manual/extra counts")
    rows = {}
    for s in MODES:
        part = table.loc[table.stream.eq(s)]
        if sorted(part.threshold.tolist()) != sorted(LEVELS):
            raise ValueError("Every declared threshold including disable required")
        if (
            not part[["manual", "endogenous_manual", "previous_correct"]]
            .nunique()
            .eq(1)
            .all()
        ):
            raise ValueError("Fixed manual denominator and prior checkpoint required")
        rows[s] = list(part.itertuples(index=False))
    grid = []
    for pos, neg in product(rows[MODES[0]], rows[MODES[1]]):
        pair = [pos, neg]
        correct, extras, outside, wrong = (
            sum(getattr(r, c) for r in pair)
            for c in (
                "added_correct",
                "extras",
                "extras_without_opposite_composition",
                "different_peak",
            )
        )
        manual, endo = (
            sum(r.manual for r in pair),
            sum(r.endogenous_manual for r in pair),
        )
        supported = all(
            (r.threshold == 1 and r.added_correct == r.extras == r.different_peak == 0)
            or (r.threshold < 1 and r.added_correct >= 20 and r.positive_projects >= 3)
            for r in pair
        )
        eligible = bool(
            supported
            and correct > 0
            and extras <= correct
            and extras <= 0.5 * manual
            and outside <= 0.1 * endo
            and wrong <= 0.005 * endo
            and all(r.different_peak <= 0.005 * r.endogenous_manual for r in pair)
        )
        coverage = sum(
            min((r.previous_correct + r.added_correct) / r.manual, 0.9)
            if r.manual
            else 0
            for r in pair
        )
        grid.append(
            dict(
                NP_Pos=pos.threshold,
                NP_Neg=neg.threshold,
                added_correct=correct,
                extras=extras,
                extras_without_opposite_composition=outside,
                different_peak=wrong,
                capped90_sum=coverage,
                eligible=eligible,
            )
        )
    full = pd.DataFrame(grid)
    eligible = full.loc[full.eligible]
    if eligible.empty:
        return {s: 1.0 for s in MODES}, full
    selected = eligible.sort_values(
        [
            "capped90_sum",
            "added_correct",
            "extras",
            "different_peak",
            "NP_Pos",
            "NP_Neg",
        ],
        ascending=[False, False, True, True, False, False],
        kind="stable",
    ).iloc[0]
    return {s: float(selected[s]) for s in MODES}, full
