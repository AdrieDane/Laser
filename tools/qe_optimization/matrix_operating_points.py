"""Exact029mode grid/objective and041completion policy on new INNER scores.

No files, score fitting, outer evaluation, threshold sweep changes or new
physical candidates. Class completion uses the original conservative helper.
"""

from collections import Counter

import numpy as np
import pandas as pd

GRID = tuple(round(0.2 + 0.05 * i, 2) for i in range(17))
MODES = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")


def mode_curves(cases, manual, *, training_groups, excluded_groups):
    from matlab_annotation.operating_point import select_operating_candidates

    allowed, excluded = set(training_groups), set(excluded_groups)
    if not allowed or not excluded or allowed & excluded:
        raise ValueError("Explicit disjoint calibration scopes required")
    if set(manual.project) - allowed or manual.internal_standard.any():
        raise ValueError("Only authorized endogenous INNER validation manual rows")
    counts = manual.stream.value_counts().to_dict()
    if set(counts) != set(MODES):
        raise ValueError("Complete four-mode calibration required")
    for case in cases.values():
        if case["project"] not in allowed or case["project"] in excluded:
            raise ValueError("Excluded query in calibration")
        pool, scores = case["pool"], case["scores"]
        if (
            not pool.index.equals(scores.index)
            or not np.isfinite(scores).all()
            or not scores.between(0, 1).all()
        ):
            raise ValueError("Aligned finite unmodified INNER scores required")
        for key in (
            "training_positive",
            "training_label_known",
            "calibration_identity_in_manual",
        ):
            if not pool[key].isin([True, False]).all():
                raise ValueError("Explicit training labels required")
        if (pool.training_positive & ~pool.training_label_known).any():
            raise ValueError("Positive must be known")
    rows = []
    for threshold in GRID:
        total = {s: Counter(correct=0, extras=0, different_peak=0) for s in MODES}
        for case in cases.values():
            for stream, pool in case["pool"].groupby("stream", sort=False):
                chosen = select_operating_candidates(
                    pool,
                    case["scores"].loc[pool.index],
                    threshold=threshold,
                    ambiguity_margin=0.05,
                )
                total[stream]["correct"] += int(chosen.training_positive.sum())
                total[stream]["extras"] += int(
                    (~chosen.calibration_identity_in_manual).sum()
                )
                total[stream]["different_peak"] += int(
                    (
                        chosen.calibration_identity_in_manual
                        & chosen.training_label_known
                        & ~chosen.training_positive
                    ).sum()
                )
        rows.extend(
            dict(stream=s, threshold=threshold, manual=counts[s], **dict(total[s]))
            for s in MODES
        )
    return pd.DataFrame(rows)


def choose_modes(table):
    from matlab_annotation.operating_point import choose_balanced_training_points

    chosen = choose_balanced_training_points(table, extras_per_manual=0.095, target=0.9)
    if chosen is None:
        raise RuntimeError("No feasible original029 mode point; do not relax budget")
    return chosen


def completion_curves(
    proposals, manual, base_counts, *, training_groups, excluded_groups
):
    from matlab_annotation.add_only_class_completion import (
        class_completion_curves,
        choose_completion_thresholds,
    )

    table = class_completion_curves(
        proposals,
        manual,
        base_counts,
        training_groups=training_groups,
        excluded_groups=excluded_groups,
    )
    thresholds, chosen = choose_completion_thresholds(table)
    return table, thresholds, chosen
