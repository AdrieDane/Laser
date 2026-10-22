"""Research-only operating points fitted on separate TRAINING-group predictions.

Scores estimate curator agreement, not analytical identity probabilities. The
caller must supply strict physical candidates and exclude all evaluation groups
from both score fitting and operating-point selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_operating_candidates(pool, scores, *, threshold, ambiguity_margin=0.05):
    """Same one-feature/one-identity competition, allowing a calibrated cutoff.

    No composition, mass or RT candidate is added here. Threshold=.5 is exactly
    the existing class selector's behavior, including stable intensity tie-breaks.
    """
    if not np.isfinite(threshold) or not 0.05 <= threshold <= 1:
        raise ValueError("Research operating threshold must be between .05 and 1")
    if not np.isfinite(ambiguity_margin) or not 0 <= ambiguity_margin <= 1:
        raise ValueError("Invalid research ambiguity margin")
    if not scores.index.equals(pool.index):
        raise ValueError("Candidate score indices must match candidate rows")
    ranked = pool.assign(class_selection_score=scores)
    ranked = ranked.sort_values(
        ["class_selection_score", "feat_intensity"], ascending=False, kind="stable"
    )
    ranked = ranked.drop_duplicates(["feat_feature_id", "name", "adduct"])
    rank = ranked.groupby("feat_feature_id").cumcount()
    next_score = ranked.groupby("feat_feature_id").class_selection_score.shift(-1)
    clear = next_score.isna() | (ranked.class_selection_score - next_score).ge(
        ambiguity_margin
    )
    selected = ranked[
        rank.eq(0) & clear & ranked.class_selection_score.ge(threshold)
    ].copy()
    selected = selected.drop_duplicates(["name", "adduct"])
    selected["automatic_accepted"] = True
    selected["auto_annotation_status"] = "class_rt_atlas_selected"
    return selected


def choose_training_operating_point(table, *, extras_per_manual=0.09):
    """Highest TRAINING correct count within a fixed unconfirmed-extras budget.

    Ties prefer fewer extras and then a higher cutoff. No feasible point returns
    ``None`` so callers can explicitly fall back/withhold, never hide a budget
    failure. This budget is NOT a false-discovery-rate estimate.
    """
    required = {"threshold", "manual", "correct", "extras"}
    if not required.issubset(table) or table.empty:
        raise ValueError("Operating point table needs threshold/manual/correct/extras")
    if not 0 <= extras_per_manual <= 1:
        raise ValueError("Invalid extras per manual budget")
    values = table[list(sorted(required))].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all() or values.manual.le(0).any():
        raise ValueError(
            "Operating point counts must be finite with positive denominators"
        )
    if (values[["correct", "extras"]] < 0).any().any() or values.correct.gt(
        values.manual
    ).any():
        raise ValueError("Invalid operating point match/extra counts")
    if values.manual.nunique() != 1:
        raise ValueError("Operating points must use identical manual denominators")
    feasible = values[values.extras.le(extras_per_manual * values.manual)]
    if feasible.empty:
        return None
    return float(
        feasible.sort_values(
            ["correct", "extras", "threshold"], ascending=[False, True, False]
        )
        .iloc[0]
        .threshold
    )


def choose_balanced_training_points(table, *, extras_per_manual=0.095, target=0.9):
    """Allocate a POOLED training extras budget to the weakest measurement mode.

    First maximize minimum per-mode correct coverage up to the requested target,
    then total correct matches, then fewer wrong-peak choices/extras, then higher
    cutoffs. Evaluation labels are not inputs. This is not an FDR guarantee.
    """
    required = {"stream", "threshold", "manual", "correct", "extras", "different_peak"}
    if not required.issubset(table) or table.empty:
        raise ValueError("Balanced operating points require per-mode training counts")
    if not 0 <= extras_per_manual <= 1 or not 0 < target <= 1:
        raise ValueError("Invalid training budget or coverage target")
    parts = []
    streams = sorted(table.stream.unique())
    if not 2 <= len(streams) <= 4:
        raise ValueError("Balanced selection requires two to four measurement streams")
    for stream in streams:
        p = (
            table[table.stream.eq(stream)]
            .copy()
            .sort_values("threshold", ascending=False)
        )
        choose_training_operating_point(p, extras_per_manual=1.0)
        if (
            p.threshold.duplicated().any()
            or p.different_peak.lt(0).any()
            or not np.isfinite(p.different_peak).all()
        ):
            raise ValueError("Invalid/duplicate training operating points")
        parts.append(p.reset_index(drop=True))
    grids = np.meshgrid(*[np.arange(len(p)) for p in parts], indexing="ij")
    indices = [g.ravel() for g in grids]
    total_manual = sum(float(p.manual.iloc[0]) for p in parts)
    correct = np.stack([p.correct.to_numpy(float)[i] for p, i in zip(parts, indices)])
    extras = sum(p.extras.to_numpy(float)[i] for p, i in zip(parts, indices))
    wrong = sum(p.different_peak.to_numpy(float)[i] for p, i in zip(parts, indices))
    feasible = np.flatnonzero(extras <= extras_per_manual * total_manual)
    if not len(feasible):
        return None
    denominator = np.asarray([float(p.manual.iloc[0]) for p in parts])[:, None]
    minimum = np.minimum((correct / denominator).min(axis=0), target)
    total_correct = correct.sum(axis=0)
    # meshgrid follows descending threshold order: stable final tie-break.
    best = max(
        feasible,
        key=lambda j: (minimum[j], total_correct[j], -wrong[j], -extras[j], -j),
    )
    return {
        s: float(p.iloc[i[best]].threshold) for s, p, i in zip(streams, parts, indices)
    }
