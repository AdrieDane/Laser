"""Unused research: training-only class cutoffs with fixed peak competition.

No files, query labels, candidate generation or scoring model. Raw scores are
curator-agreement rankings, NOT calibrated probabilities of chemical identity.
Low absolute scores can be class-calibrated only with other-project evidence.
"""

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class

KEYS = ["stream", "lipid_class"]
COUNTS = ["manual", "correct", "extras", "different_peak"]
LEVELS = tuple(sorted({*(i / 20 for i in range(1, 21)), 0.88, 0.89, 0.9, 0.91, 0.92}))


def clear_identity_winners(pool, scores, *, ambiguity_margin=0.05):
    """Original rank/ambiguity/identity rules BEFORE any acceptance cutoff.

    One stream per call. Identity classes must be unambiguous so subsequent
    class-dependent filtering cannot change which feature wins an identity.
    """
    required = {
        "stream",
        "class",
        "name",
        "adduct",
        "feat_feature_id",
        "feat_intensity",
    }
    if (
        not required.issubset(pool)
        or not pool.index.is_unique
        or not pool.index.equals(scores.index)
    ):
        raise ValueError("Aligned unique physical candidate/score rows required")
    if pool.stream.nunique() > 1:
        raise ValueError("Peak competition must remain within one stream")
    values = scores.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Finite agreement scores between zero and one required")
    if not np.isfinite(ambiguity_margin) or not 0 <= ambiguity_margin <= 1:
        raise ValueError("Invalid ambiguity margin")
    classes = pool["class"].map(canonical_lipid_class)
    if (
        pool.assign(_chemical_class=classes)
        .groupby(["name", "adduct"])
        ._chemical_class.nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("Each chemical identity must have one consistent class")
    ranked = pool.assign(class_selection_score=scores)
    ranked = ranked.sort_values(
        ["class_selection_score", "feat_intensity"], ascending=False, kind="stable"
    )
    ranked = ranked.drop_duplicates(["feat_feature_id", "name", "adduct"])
    first = ranked.groupby("feat_feature_id").cumcount().eq(0)
    second = ranked.groupby("feat_feature_id").class_selection_score.shift(-1)
    clear = second.isna() | (ranked.class_selection_score - second).ge(ambiguity_margin)
    return ranked[first & clear].drop_duplicates(["name", "adduct"]).copy()


def select_class_operating_candidates(
    pool, scores, *, baseline_threshold, class_thresholds, ambiguity_margin=0.05
):
    """Filter FIXED winners, never promote a losing label or alternative peak."""
    cuts = {}
    for key, value in class_thresholds.items():
        cls, value = canonical_lipid_class(key), float(value)
        if cls in cuts and cuts[cls] != value:
            raise ValueError("Conflicting cutoffs for the same canonical class")
        cuts[cls] = value
    values = [float(baseline_threshold), *cuts.values()]
    if not np.isfinite(values).all() or not all(0 <= v <= 1 for v in values):
        raise ValueError("Class and baseline cutoffs must be between zero and one")
    winners = clear_identity_winners(pool, scores, ambiguity_margin=ambiguity_margin)
    classes = winners["class"].map(canonical_lipid_class)
    thresholds = classes.map(cuts).fillna(float(baseline_threshold))
    selected = winners[winners.class_selection_score.ge(thresholds)].copy()
    selected["class_operating_threshold"] = thresholds.loc[selected.index]
    selected["automatic_accepted"] = True
    selected["auto_annotation_status"] = "inner_class_calibrated_selected"
    return selected


def class_training_curves(
    winners,
    manual,
    baseline_thresholds,
    *,
    training_groups,
    excluded_groups,
    levels=LEVELS,
    minimum_manual=100,
    minimum_projects=3,
    minimum_positives=20,
):
    """Caller must authorize/group-exclude data BEFORE attaching these labels.

    Manual denominators include unresolved rows; internal standards must already
    be excluded. Sparse/unsupported classes have ONLY their baseline cutoff.
    Eligible cutoffs are positive-score coverage quantiles, not a global score
    floor. Include the exact baseline recall boundary for conservative savings.
    """
    training, excluded = set(training_groups), set(excluded_groups)
    if not training or not excluded or training & excluded:
        raise ValueError("Distinct explicit training and excluded projects required")
    required = {
        "stream",
        "lipid_class",
        "project",
        "class_selection_score",
        "training_positive",
        "training_label_known",
        "calibration_identity_in_manual",
    }
    if not required.issubset(winners) or not {
        "stream",
        "lipid_class",
        "project",
    }.issubset(manual):
        raise ValueError(
            "Explicit training-only winners, labels and manual support required"
        )
    if not all(np.isfinite(v) and 0 < v <= 1 for v in levels):
        raise ValueError("Coverage levels must be positive fractions")
    if (set(winners.project) | set(manual.project)) - training:
        raise ValueError("Excluded or unknown projects entered calibration labels")
    if minimum_manual < 1 or minimum_projects < 2 or minimum_positives < 1:
        raise ValueError("Positive counts and independent project support required")
    if "internal_standard" in manual and manual.internal_standard.fillna(False).any():
        raise ValueError("Standards are excluded from endogenous calibration")
    for flag in (
        "training_positive",
        "training_label_known",
        "calibration_identity_in_manual",
    ):
        if not winners[flag].isin([True, False]).all():
            raise ValueError("Explicit boolean calibration labels required")
    if (winners.training_positive & ~winners.training_label_known).any():
        raise ValueError("A positive training match must be known")
    values = winners.class_selection_score.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Finite agreement scores between zero and one required")
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in baseline_thresholds.values()):
        raise ValueError("Finite baseline cutoffs between zero and one required")
    w, m = winners.copy(), manual.copy()
    w["lipid_class"] = w.lipid_class.map(canonical_lipid_class)
    m["lipid_class"] = m.lipid_class.map(canonical_lipid_class)
    wg = {k: g for k, g in w.groupby(KEYS, sort=True)}
    mg = {k: g for k, g in m.groupby(KEYS, sort=True)}
    rows = []
    for key in sorted(set(wg) | set(mg)):
        stream, cls = key
        candidates = wg.get(key, w.iloc[:0])
        reference = mg.get(key, m.iloc[:0])
        n, groups = len(reference), reference.project.nunique()
        positives = candidates[candidates.training_positive]
        if len(positives) > n:
            raise ValueError("Candidate and manual class definitions disagree")
        eligible = (
            n >= minimum_manual
            and groups >= minimum_projects
            and len(positives) >= minimum_positives
            and positives.project.nunique() >= minimum_projects
        )
        base = float(baseline_thresholds[stream])
        thresholds = {base}
        if eligible:
            positive_scores = np.sort(positives.class_selection_score.to_numpy(float))[
                ::-1
            ]
            counts = {int(np.ceil(level * n)) for level in levels}
            counts.add(int(positive_scores.size))
            counts.add(int((positive_scores >= base).sum()))
            thresholds.add(1.0)
            thresholds.update(
                float(positive_scores[k - 1])
                for k in counts
                if 1 <= k <= len(positive_scores)
            )
        for threshold in sorted(thresholds, reverse=True):
            selected = candidates[candidates.class_selection_score.ge(threshold)]
            correct = int(selected.training_positive.sum())
            if correct > n:
                raise ValueError("Candidate and manual class definitions disagree")
            rows.append(
                dict(
                    stream=stream,
                    lipid_class=cls,
                    threshold=threshold,
                    manual=n,
                    projects=int(groups),
                    positive_projects=int(positives.project.nunique()),
                    eligible=bool(eligible),
                    baseline=threshold == base,
                    correct=correct,
                    extras=int((~selected.calibration_identity_in_manual).sum()),
                    different_peak=int(
                        (
                            selected.calibration_identity_in_manual
                            & selected.training_label_known
                            & ~selected.training_positive
                        ).sum()
                    ),
                )
            )
    return pd.DataFrame(rows)


def choose_class_training_points(
    table,
    *,
    training_groups,
    excluded_groups,
    extras_per_manual=0.095,
    target=0.9,
    time_limit=120,
):
    """Lexicographic class coverage / correct / wrong / extras / stability.

    Require NO lower correct count in any eligible training class; fixed sparse
    classes keep the baseline. Wrong-peak counts cannot rise in ANY stream.
    Pooled endogenous extras budget is unchanged. No test-data guarantee/FDR.
    If an optimum is not certified, conservatively return the whole baseline.
    """
    from scipy.optimize import milp, Bounds, LinearConstraint

    training, excluded = set(training_groups), set(excluded_groups)
    if not training or not excluded or training & excluded:
        raise ValueError("Distinct explicit training and excluded projects required")

    required = {*KEYS, *COUNTS, "threshold", "eligible", "baseline"}
    if not required.issubset(table) or table.empty:
        raise ValueError("Complete training operating-point table required")
    if not 0 <= extras_per_manual <= 1 or not 0 < target <= 1 or time_limit <= 0:
        raise ValueError("Invalid training budget, target or solver limit")
    t = table.reset_index(drop=True).copy()
    if t.duplicated([*KEYS, "threshold"]).any():
        raise ValueError("Duplicate class operating points")
    numbers = t[[*COUNTS, "threshold"]].to_numpy(float)
    if not np.isfinite(numbers).all() or (numbers < 0).any() or t.threshold.gt(1).any():
        raise ValueError("Nonnegative finite counts/cutoffs required")
    if (
        not np.equal(t[COUNTS], np.floor(t[COUNTS])).all().all()
        or t.correct.gt(t.manual).any()
    ):
        raise ValueError("Integral correct counts cannot exceed manual denominators")
    if (
        not t.eligible.isin([True, False]).all()
        or not t.baseline.isin([True, False]).all()
    ):
        raise ValueError("Explicit boolean eligibility/baseline flags required")
    if t.loc[t.eligible.astype(bool), "manual"].le(0).any():
        raise ValueError("Eligible classes require positive manual support")
    base = t[t.baseline.astype(bool)]
    groups = t.groupby(KEYS, sort=False)
    if len(base) != groups.ngroups or base.duplicated(KEYS).any():
        raise ValueError("Exactly one baseline per class/stream required")
    if groups.manual.nunique().gt(1).any() or groups.eligible.nunique().gt(1).any():
        raise ValueError(
            "Class denominators/eligibility must be identical across points"
        )
    if t[~t.eligible.astype(bool)].groupby(KEYS).size().gt(1).any():
        raise ValueError("Sparse/unsupported classes must retain only the baseline")
    budget = int(np.floor(extras_per_manual * base.manual.sum() + 1e-9))
    if base.extras.sum() > budget:
        raise ValueError("Original baseline is not feasible; do not relax the budget")
    fixed = base[~base.eligible.astype(bool)]
    eligible = t[t.eligible.astype(bool)].copy()
    base_correct = base.set_index(KEYS).correct
    min_correct = pd.MultiIndex.from_frame(eligible[KEYS]).map(base_correct)
    eligible = eligible[eligible.correct.to_numpy() >= np.asarray(min_correct)]
    eligible["point_index"] = eligible.index
    p = eligible.reset_index(drop=True)
    audits = []

    def result(indices, fallback):
        selected = t.loc[indices]
        assert len(selected) == len(base) and not selected.duplicated(KEYS).any()
        original_counts = base.set_index(KEYS).correct
        current_counts = selected.set_index(KEYS).correct.reindex(original_counts.index)
        assert current_counts.ge(original_counts).all()
        assert int(selected.extras.sum()) <= budget
        for stream, group in base.groupby("stream"):
            current = selected[selected.stream.eq(stream)]
            assert int(current.correct.sum()) >= int(group.correct.sum())
            assert int(current.different_peak.sum()) <= int(group.different_peak.sum())
        cuts = {s: {} for s in base.stream.unique()}
        for row in selected[selected.eligible.astype(bool)].itertuples():
            cuts[row.stream][row.lipid_class] = float(row.threshold)
        return dict(
            training_groups=sorted(training),
            excluded_groups=sorted(excluded),
            thresholds=cuts,
            selected_point_indices=list(map(int, indices)),
            solver_fallback=bool(fallback),
            solver_audit=audits,
            training_extras_budget=budget,
            eligible_class_modes=int(base.eligible.sum()),
        )

    if p.empty:
        return result(base.index.tolist(), False)
    rows, lower, upper = [], [], []
    for indices in p.groupby(KEYS, sort=False).groups.values():
        row = np.zeros(len(p))
        row[list(indices)] = 1
        rows.append(row)
        lower.append(1)
        upper.append(1)
    for stream, group in base.groupby("stream"):
        use = p.stream.eq(stream).to_numpy(float)
        old_fixed = fixed[fixed.stream.eq(stream)]
        rows.append(p.correct.to_numpy(float) * use)
        lower.append(float(group.correct.sum() - old_fixed.correct.sum()))
        upper.append(np.inf)
        rows.append(p.different_peak.to_numpy(float) * use)
        lower.append(-np.inf)
        upper.append(float(group.different_peak.sum() - old_fixed.different_peak.sum()))
    rows.append(p.extras.to_numpy(float))
    lower.append(-np.inf)
    upper.append(float(budget - fixed.extras.sum()))
    cap = np.minimum(
        100 * p.correct.to_numpy(float) / p.manual.to_numpy(float), 100 * target
    )
    correct = p.correct.to_numpy(float)
    wrong, extra = p.different_peak.to_numpy(float), p.extras.to_numpy(float)
    baselines = base.set_index(KEYS).threshold
    base_threshold = np.asarray(pd.MultiIndex.from_frame(p[KEYS]).map(baselines), float)
    distance = np.abs(p.threshold.to_numpy(float) - base_threshold)
    stages = [
        ("class_coverage", -cap),
        ("correct", -correct),
        ("wrong", wrong),
        ("extras", extra),
        ("stability", distance),
    ]
    chosen = None
    for name, cost in stages:
        solved = milp(
            cost,
            integrality=np.ones(len(p)),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(
                np.vstack(rows), np.asarray(lower), np.asarray(upper)
            ),
            options=dict(time_limit=float(time_limit), mip_rel_gap=0.0),
        )
        audits.append(
            dict(stage=name, status=int(solved.status), optimal=bool(solved.success))
        )
        if solved.status == 2:
            raise ValueError(
                "Class calibration unexpectedly infeasible despite baseline"
            )
        if not solved.success or solved.x is None:
            return result(base.index.tolist(), True)
        rounded = np.rint(solved.x).astype(int)
        assert np.max(np.abs(solved.x - rounded)) < 1e-5
        chosen = np.flatnonzero(rounded)
        value = float(cost @ rounded)
        rows.append(cost.copy())
        lower.append(-np.inf)
        upper.append(value + (1e-7 if name in {"class_coverage", "stability"} else 0.1))
    indices = [*fixed.index.tolist(), *p.iloc[chosen].point_index.tolist()]
    return result(indices, False)
