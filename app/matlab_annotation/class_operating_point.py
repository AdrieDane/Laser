"""Unused research: TRAINING-only class/mode operating-point allocation.

No candidate admissions, scoring, runtime annotation or curation reads here.
Caller must construct tables from grouped OOF TRAINING predictions, excluding
the evaluated projects. This is curator agreement, not analytical FDR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_class_operating_candidates(
    pool,
    scores,
    thresholds,
    *,
    fallback_threshold,
    ambiguity_margin=0.05,
):
    """One stream only; class cutoffs cannot promote a losing feature label.

    Cutoffs/fallback must be trained without this stream's evaluated project.
    Same global feature winners and same identity winners as ordinary operating
    selection. Only their acceptance cutoffs differ. No candidate is added.
    """
    from annotator.composition_ranges import canonical_lipid_class
    from .operating_point import select_operating_candidates

    values = [fallback_threshold, *thresholds.values()]
    if any(not np.isfinite(v) or not 0.05 <= v <= 1 for v in values):
        raise ValueError("Class/fallback cutoffs must be between .05 and1")
    if pool.empty:
        return select_operating_candidates(
            pool,
            scores,
            threshold=fallback_threshold,
            ambiguity_margin=ambiguity_margin,
        )
    if pool.stream.nunique() != 1:
        raise ValueError("Class operating selection requires ONE measurement stream")
    canonical = pool["class"].map(canonical_lipid_class)
    if canonical.groupby([pool.name, pool.adduct]).nunique().gt(1).any():
        raise ValueError("One identity cannot belong to conflicting canonical classes")
    chosen = select_operating_candidates(
        pool, scores, threshold=0.05, ambiguity_margin=ambiguity_margin
    )
    keys = chosen.stream.astype(str) + "|" + chosen["class"].map(canonical_lipid_class)
    chosen["operating_threshold"] = [
        thresholds.get(k, fallback_threshold) for k in keys
    ]
    chosen = chosen[chosen.class_selection_score.ge(chosen.operating_threshold)].copy()
    chosen["auto_annotation_status"] = "class_training_operating_point"
    return chosen


def choose_class_training_points(
    table,
    *,
    training_groups,
    excluded_groups,
    extras_per_manual=0.095,
    target=0.9,
    minimum_cohorts=3,
    minimum_manual=30,
    time_limit=60,
):
    """Exact multiple-choice integer optimization of class coverage deficits.

    Choose one cutoff per class/mode. Keep under-supported classes at their
    reference cutoff. Protect each class and each mode at min(reference correct,
    ceil(target*manual)); total correct cannot fall below reference. Respect a
    pooled TRAINING extras budget. First maximize sum of per-class coverage
    capped at target, then total correct, then fewer wrong peaks/extras. Equal
    count points prefer higher cutoffs. A missing/nonoptimal solver solution
    returns None explicitly; never silently relax a safeguard.

    Reference and proposed counts MUST use identical global feature/identity
    winners. Applying class thresholds must not promote a lower-ranked label
    after withholding the top feature winner. Otherwise counts aren't additive.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    if (
        not training_groups
        or not excluded_groups
        or set(training_groups) & set(excluded_groups)
    ):
        raise ValueError(
            "Distinct explicit training and excluded project groups required"
        )
    if (
        not 0 <= extras_per_manual <= 1
        or not 0 < target <= 1
        or minimum_cohorts < 2
        or minimum_manual < 1
        or time_limit <= 0
    ):
        raise ValueError("Invalid class operating-point settings")
    columns = {
        "stream",
        "lipid_class",
        "threshold",
        "manual",
        "correct",
        "extras",
        "wrong_peak",
        "training_cohorts",
        "reference",
    }
    if not columns.issubset(table) or table.empty:
        raise ValueError("Complete TRAINING class operating-point counts required")
    data = table[list(sorted(columns))].copy()
    counts = ["manual", "correct", "extras", "wrong_peak", "training_cohorts"]
    numeric = data[["threshold", *counts]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all() or (numeric[counts] < 0).any().any():
        raise ValueError("Finite nonnegative operating-point counts required")
    if (
        not numeric[counts].eq(np.floor(numeric[counts])).all().all()
        or numeric.correct.gt(numeric.manual).any()
    ):
        raise ValueError("Invalid integer match/extra/manual counts")
    if (
        not numeric.threshold.between(0.05, 1).all()
        or not data.reference.isin([True, False]).all()
    ):
        raise ValueError("Invalid threshold/reference flags")
    data[numeric.columns] = numeric
    keys = ["stream", "lipid_class"]
    if data.duplicated([*keys, "threshold"]).any():
        raise ValueError("Duplicate class/mode thresholds")
    if not data.stream.isin(["RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"]).all():
        raise ValueError("Unknown measurement stream")
    parts, references, floors = [], [], {}
    for key, part in data.groupby(keys, sort=True):
        reference = part[part.reference.astype(bool)]
        if (
            len(reference) != 1
            or part.manual.nunique() != 1
            or part.training_cohorts.nunique() != 1
        ):
            raise ValueError(
                "One reference point and fixed denominators/cohorts per class required"
            )
        ref = reference.iloc[0]
        references.append(ref)
        floor = min(int(ref.correct), int(np.ceil(target * ref.manual - 1e-9)))
        floors[key] = floor
        if ref.training_cohorts < minimum_cohorts or ref.manual < minimum_manual:
            part = reference
        else:
            part = part[part.correct.ge(floor)]
        # Safely remove identical-count alternatives; no scientific tie-breaking
        # by a held-out score or arbitrary input row order.
        part = part.sort_values("threshold", ascending=False).drop_duplicates(
            ["manual", "correct", "extras", "wrong_peak"]
        )
        parts.append(part)
    reference = pd.DataFrame(references)
    options = pd.concat(parts, ignore_index=True)
    option_keys = list(zip(options.stream, options.lipid_class))
    n = len(options)
    rows, lo, hi = [], [], []
    for key in floors:
        rows.append(np.asarray([k == key for k in option_keys], float))
        lo.append(1.0)
        hi.append(1.0)
    rows.append(options.extras.to_numpy(float))
    lo.append(-np.inf)
    budget = extras_per_manual * reference.manual.sum()
    hi.append(budget)
    rows.append(options.correct.to_numpy(float))
    lo.append(float(reference.correct.sum()))
    hi.append(np.inf)
    for stream, refs in reference.groupby("stream", sort=True):
        rows.append(
            options.correct.to_numpy(float) * options.stream.eq(stream).to_numpy()
        )
        lo.append(
            min(
                float(refs.correct.sum()),
                float(np.ceil(target * refs.manual.sum() - 1e-9)),
            )
        )
        hi.append(np.inf)
    denominator = options.manual.to_numpy(float)
    coverage = np.minimum(
        np.divide(
            options.correct.to_numpy(float),
            denominator,
            out=np.zeros(n),
            where=denominator > 0,
        ),
        target,
    )
    messages = []

    def solve(objective):
        result = milp(
            np.asarray(objective, float),
            integrality=np.ones(n),
            bounds=Bounds(np.zeros(n), np.ones(n)),
            constraints=LinearConstraint(
                np.asarray(rows), np.asarray(lo), np.asarray(hi)
            ),
            options=dict(time_limit=float(time_limit), mip_rel_gap=0.0),
        )
        messages.append(str(result.message))
        if not result.success or result.status != 0 or result.x is None:
            return None
        x = np.rint(result.x)
        if np.max(np.abs(result.x - x)) > 1e-5:
            return None
        return x

    first = solve(-coverage)
    if first is None:
        return None
    macro = float(coverage @ first)
    rows.append(coverage)
    lo.append(macro - 1e-8)
    hi.append(np.inf)
    second = solve(-options.correct.to_numpy(float))
    if second is None:
        return None
    correct = int(np.rint(options.correct.to_numpy(float) @ second))
    rows.append(options.correct.to_numpy(float))
    lo.append(correct)
    hi.append(np.inf)
    max_extras = sum(float(g.extras.max()) for _, g in options.groupby(keys))
    third = solve(
        options.wrong_peak.to_numpy(float)
        + options.extras.to_numpy(float) / (max_extras + 1)
    )
    if third is None:
        return None
    chosen = options.loc[third.astype(bool)].copy()
    assert len(chosen) == len(floors) and not chosen.duplicated(keys).any()
    assert chosen.extras.sum() <= budget + 1e-6
    assert chosen.correct.sum() >= reference.correct.sum()
    assert all(
        r.correct >= floors[(r.stream, r.lipid_class)] for r in chosen.itertuples()
    )
    thresholds = {
        r.stream + "|" + r.lipid_class: float(r.threshold) for r in chosen.itertuples()
    }
    return dict(
        format="laser-training-class-operating-points-v1",
        thresholds=thresholds,
        target=float(target),
        training_groups=sorted(set(training_groups)),
        excluded_groups=sorted(set(excluded_groups)),
        selected=chosen.to_dict("records"),
        reference_correct=int(reference.correct.sum()),
        correct=int(chosen.correct.sum()),
        extras=int(chosen.extras.sum()),
        wrong_peak=int(chosen.wrong_peak.sum()),
        manual=int(reference.manual.sum()),
        pooled_training_extra_budget=float(budget),
        capped_class_coverage_sum=float(coverage @ third),
        solver_messages=messages,
        production_enabled=False,
    )
