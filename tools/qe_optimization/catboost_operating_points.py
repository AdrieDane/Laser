"""Native-domain/disabled semantics, otherwise existing fixed competition.

Class-aware calibration is the existing034 TRAINING optimizer, not new outer
label choices. No source-data edits, annotation exports or file reads here.
"""

import numpy as np

THRESHOLDS = tuple(i / 40 for i in range(41))


def select_candidates(
    pool, scores, *, baseline_threshold, class_thresholds=None, ambiguity_margin=0.05
):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.guarded_class_operating_point import clear_identity_winners

    base = float(baseline_threshold)
    thresholds = {}
    for cls, value in (class_thresholds or {}).items():
        key = canonical_lipid_class(cls)
        if key in thresholds and thresholds[key] != float(value):
            raise ValueError("Conflicting canonical class cutoffs")
        thresholds[key] = float(value)
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in [base, *thresholds.values()]):
        raise ValueError("Native valid score-domain cutoffs required")
    winners = clear_identity_winners(pool, scores, ambiguity_margin=ambiguity_margin)
    cut = winners["class"].map(canonical_lipid_class).map(thresholds).fillna(base)
    result = winners.loc[cut.lt(1) & winners.class_selection_score.ge(cut)].copy()
    result["operating_threshold"] = base
    result["class_operating_threshold"] = cut.loc[result.index]
    result["automatic_accepted"] = True
    result["auto_annotation_status"] = "catboost_class_calibrated_selected"
    return result


def class_curves(*args, **kwargs):
    from matlab_annotation.guarded_class_operating_point import class_training_curves

    result = class_training_curves(*args, **kwargs)
    # Score1 is possible for a generic software fixture. Disable is explicit,
    # not an assumption about whether this particular learner saturates.
    result.loc[result.threshold.eq(1), ["correct", "extras", "different_peak"]] = 0
    return result


def mode_counts(winners, threshold):
    picked = (
        winners.loc[winners.class_selection_score.ge(threshold)]
        if threshold < 1
        else winners.iloc[:0]
    )
    return dict(
        correct=int(picked.training_positive.sum()),
        extras=int((~picked.calibration_identity_in_manual).sum()),
        different_peak=int(
            (
                picked.calibration_identity_in_manual
                & picked.training_label_known
                & ~picked.training_positive
            ).sum()
        ),
    )


def validate_curve_rows(table, winners):
    """Direct fixed-winner replay, including explicit disabled threshold1."""
    for row in table.itertuples():
        part = winners.loc[winners.stream.eq(row.stream)]
        actual = mode_counts(part, float(row.threshold))
        assert all(actual[k] == getattr(row, k) for k in actual)
    return True
