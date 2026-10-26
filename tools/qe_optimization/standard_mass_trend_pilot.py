"""Descriptive standard-only mass trends, NOT an annotation/calibration rule.

Already-qualified spikes are censored to abs(rawppm)<3. Their errors cannot
establish how an unrestricted mass population behaves. No files or gold read.
"""

from statistics import median

import numpy as np
import pandas as pd

MINIMUM_NEUTRALS = 6
MINIMUM_PAIR_SPACING_DA = 50.0


def grouped_anchors(anchors):
    required = {"stream", "neutral_key", "feature_id", "reference_mz", "ppm"}
    if not required.issubset(anchors):
        raise ValueError("Explicit original independently measured standards required")
    if {"training_positive", "training_label_known", "same_manual_peak"} & set(anchors):
        raise ValueError("No curated annotation flags permitted")
    if anchors.duplicated().any():
        raise ValueError("Duplicate physical anchor rows are not independent")
    for c in ("reference_mz", "ppm"):
        if not np.isfinite(anchors[c].to_numpy(float)).all():
            raise ValueError("Finite standard masses/errors required")
    if not anchors.reference_mz.gt(0).all() or not anchors.ppm.abs().lt(3).all():
        raise ValueError("Pilot uses only existing qualified<3ppm anchors")
    if anchors.neutral_key.isna().any() or anchors.feature_id.isna().any():
        raise ValueError("Explicit neutral and original feature IDs required")
    if anchors.groupby(["stream", "feature_id"]).neutral_key.nunique().gt(1).any():
        raise ValueError("A feature cannot represent two independent neutrals")
    return anchors.groupby(["stream", "neutral_key"], as_index=False, sort=True).agg(
        reference_mz=("reference_mz", "median"),
        ppm=("ppm", "median"),
        physical_ions=("feature_id", "size"),
    )


def robust_line(x, y):
    """Theil-Sen descriptive ppm/mass trend; well-separated pairs only."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < MINIMUM_NEUTRALS - 1:
        return None
    a, b = np.triu_indices(len(x), 1)
    dx = x[b] - x[a]
    valid = np.abs(dx) >= MINIMUM_PAIR_SPACING_DA
    if not valid.any():
        return None
    slope = float(np.median((y[b[valid]] - y[a[valid]]) / dx[valid]))
    center = float(np.median(x))
    intercept = float(np.median(y - slope * (x - center)))
    return center, intercept, slope


def independent_line(x, y):
    """Separate scalar median/pair enumeration, including identical spacings."""
    pairs = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            delta = float(x[j]) - float(x[i])
            if abs(delta) >= MINIMUM_PAIR_SPACING_DA:
                pairs.append((float(y[j]) - float(y[i])) / delta)
    if len(x) < MINIMUM_NEUTRALS - 1 or not pairs:
        return None
    slope = median(pairs)
    center = median(map(float, x))
    offset = median(
        [float(b) - slope * (float(a) - center) for a, b in zip(x, y, strict=True)]
    )
    return center, offset, slope


def leave_neutral_out(grouped):
    rows = []
    for stream, part in grouped.groupby("stream", sort=True):
        if len(part) < MINIMUM_NEUTRALS:
            continue
        x, y = part.reference_mz.to_numpy(float), part.ppm.to_numpy(float)
        for i, neutral in enumerate(part.neutral_key):
            other = np.arange(len(part)) != i
            a, b = x[other], y[other]
            constant = float(np.median(b))
            assert constant == median(map(float, b))
            fit, independent = robust_line(a, b), independent_line(a, b)
            assert (fit is None) == (independent is None)
            prediction = slope = np.nan
            if fit is not None:
                np.testing.assert_allclose(fit, independent, rtol=0, atol=1e-12)
                center, intercept, slope = fit
                prediction = intercept + slope * (x[i] - center)
                independent_prediction = independent[1] + independent[2] * (
                    float(x[i]) - independent[0]
                )
                np.testing.assert_allclose(
                    prediction, independent_prediction, rtol=0, atol=1e-12
                )
            rows.append(
                dict(
                    stream=stream,
                    heldout_neutral=neutral,
                    reference_mz=x[i],
                    observed_ppm=y[i],
                    training_neutrals=len(a),
                    inside_training_mass_span=bool(a.min() <= x[i] <= a.max()),
                    diagnostic_constant_ppm_prediction=constant,
                    diagnostic_trend_ppm_prediction=prediction,
                    trend_slope_ppm_per_1000Da=slope * 1000,
                    constant_LOO_absolute_error_ppm=abs(y[i] - constant),
                    trend_LOO_absolute_error_ppm=abs(y[i] - prediction),
                )
            )
    return pd.DataFrame(rows)
