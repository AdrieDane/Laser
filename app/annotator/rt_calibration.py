"""RT models copied from LASER-4D a188f72 (2026-11-04)."""
from __future__ import annotations
import numpy as np
import pandas as pd


_RT_CALIBRATION_COLUMNS = [
    "standard", "feat_feature_id", "name", "adduct", "ref_rt", "feat_rt",
    "raw_rt_error", "adjusted_ref_rt", "adjusted_rt_error", "model_weight",
]


RT_CALIBRATION_MODELS = {
    "Constant shift": "constant_shift",
    "Linear": "linear",
    "Robust linear": "robust_linear",
    "Monotonic piecewise linear": "monotonic_piecewise",
}


SUPPORTED_RT_CALIBRATION_KINDS = frozenset(RT_CALIBRATION_MODELS.values())


def _fit_rt_model(
    reference_rt: np.ndarray,
    observed_rt: np.ndarray,
    model: str,
) -> tuple[float, float, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return summary coefficients, weights, and optional monotonic knots."""
    weights = np.ones(len(reference_rt), dtype=float)
    reference_knots = observed_knots = None
    if model == "constant_shift":
        slope = 1.0
        intercept = float(np.median(observed_rt - reference_rt))
    elif model == "robust_linear":
        coefficients = np.polyfit(reference_rt, observed_rt, 1)
        for _ in range(30):
            residual = observed_rt - np.polyval(coefficients, reference_rt)
            centre = float(np.median(residual))
            scale = float(1.4826 * np.median(np.abs(residual - centre)))
            if not np.isfinite(scale) or scale <= np.finfo(float).eps:
                break
            cutoff = 1.345 * scale
            centred_residual = residual - centre
            updated_weights = np.minimum(
                1.0, cutoff / np.maximum(np.abs(centred_residual), 1e-15)
            )
            updated = np.polyfit(
                reference_rt, observed_rt, 1, w=np.sqrt(updated_weights)
            )
            weights = updated_weights
            if np.allclose(coefficients, updated, rtol=1e-12, atol=1e-14):
                coefficients = updated
                break
            coefficients = updated
        slope, intercept = map(float, coefficients)
    elif model == "monotonic_piecewise":
        order = np.argsort(reference_rt)
        reference_knots = reference_rt[order]
        observed_knots = observed_rt[order]
        if np.any(np.diff(reference_knots) <= 0):
            raise ValueError(
                "Monotonic piecewise RT calibration requires a distinct "
                "reference RT for every confirmed standard"
            )
        if np.any(np.diff(observed_knots) <= 0):
            raise ValueError(
                "Monotonic piecewise RT calibration requires the confirmed "
                "standards to preserve their reference elution order"
            )
        span = float(reference_knots[-1] - reference_knots[0])
        if span <= 0 or float(np.max(np.diff(reference_knots))) > 0.5 * span:
            raise ValueError(
                "Monotonic piecewise RT calibration requires standards spread "
                "across the RT range without a gap larger than half that range"
            )
        slope = float(
            (observed_knots[-1] - observed_knots[0])
            / (reference_knots[-1] - reference_knots[0])
        )
        intercept = float(observed_knots[0] - slope * reference_knots[0])
    else:
        slope, intercept = map(float, np.polyfit(reference_rt, observed_rt, 1))
    return slope, intercept, weights, reference_knots, observed_knots


def _apply_rt_arrays(
    values: np.ndarray,
    *,
    kind: str,
    slope: float,
    intercept: float,
    reference_knots: np.ndarray | None = None,
    observed_knots: np.ndarray | None = None,
) -> np.ndarray:
    if kind != "monotonic_piecewise":
        return slope * values + intercept
    assert reference_knots is not None and observed_knots is not None
    shape = values.shape
    flat = values.reshape(-1)
    adjusted = np.interp(flat, reference_knots, observed_knots)
    below = flat < reference_knots[0]
    above = flat > reference_knots[-1]
    adjusted[below] = flat[below] + observed_knots[0] - reference_knots[0]
    adjusted[above] = flat[above] + observed_knots[-1] - reference_knots[-1]
    return adjusted.reshape(shape)


def _rt_leave_one_out_errors(
    reference_rt: np.ndarray,
    observed_rt: np.ndarray,
    model: str,
) -> np.ndarray:
    if model in {"linear", "robust_linear"} and len(reference_rt) <= 2:
        return np.asarray([], dtype=float)
    errors: list[float] = []
    for index in range(len(reference_rt)):
        keep = np.arange(len(reference_rt)) != index
        try:
            slope, intercept, _, reference_knots, observed_knots = _fit_rt_model(
                reference_rt[keep], observed_rt[keep], model
            )
            predicted = _apply_rt_arrays(
                np.asarray([reference_rt[index]], dtype=float),
                kind=model,
                slope=slope,
                intercept=intercept,
                reference_knots=reference_knots,
                observed_knots=observed_knots,
            )[0]
        except (ValueError, np.linalg.LinAlgError):
            return np.asarray([], dtype=float)
        errors.append(float(observed_rt[index] - predicted))
    return np.asarray(errors, dtype=float)


def _empty_rt_calibration(model: str, minimum_standards: int) -> dict:
    title = model.replace("_", " ").title()
    return {
        "kind": "none", "model": model, "n_used": 0,
        "slope": 1.0, "intercept": 0.0,
        "rmse_before_min": np.nan, "rmse_after_min": np.nan,
        "loo_rmse_min": np.nan, "loo_median_absolute_error_min": np.nan,
        "loo_max_absolute_error_min": np.nan,
        "r_squared": np.nan, "quality_warnings": [],
        "label": (
            f"{title} RT calibration requires at least "
            f"{minimum_standards} confirmed standards"
        ),
        "anchors": pd.DataFrame(columns=_RT_CALIBRATION_COLUMNS),
    }


def _rt_validation_summary(
    x: np.ndarray,
    y: np.ndarray,
    model: str,
    weights: np.ndarray,
    rmse_before: float,
) -> tuple[float, float, float, list[str]]:
    loo_errors = _rt_leave_one_out_errors(x, y, model)
    loo_rmse = (
        float(np.sqrt(np.mean(loo_errors ** 2))) if len(loo_errors) else np.nan
    )
    loo_median = (
        float(np.median(np.abs(loo_errors))) if len(loo_errors) else np.nan
    )
    loo_max = float(np.max(np.abs(loo_errors))) if len(loo_errors) else np.nan
    warnings: list[str] = []
    if len(x) == 2:
        warnings.append(
            "This two-point RT line has no independent in-sample fit validation."
        )
    if model == "monotonic_piecewise":
        warnings.append(
            "Piecewise interpolation passes through every confirmed standard; "
            "use the leave-one-standard-out error to judge generalization. "
            "Outside the confirmed RT range LASER applies the nearest boundary "
            "shift instead of extrapolating a segment slope."
        )
    if np.isfinite(loo_rmse) and loo_rmse >= rmse_before:
        warnings.append(
            "Leave-one-standard-out error does not improve on unadjusted RT; "
            "prefer a simpler model or keep RT calibration disabled."
        )
    if model == "robust_linear" and np.any(weights < 0.999):
        warnings.append(
            "Robust fitting downweighted one or more standards; inspect their "
            "model weights and identities before applying the model."
        )
    return loo_rmse, loo_median, loo_max, warnings


def _rt_calibration_result(
    d: pd.DataFrame,
    model: str,
    slope: float,
    intercept: float,
    fitted: np.ndarray,
    weights: np.ndarray,
    reference_knots: np.ndarray | None,
    observed_knots: np.ndarray | None,
) -> dict:
    x = d["ref_rt"].to_numpy(float)
    y = d["feat_rt"].to_numpy(float)
    residual = y - fitted
    rmse_before = float(np.sqrt(np.mean((y - x) ** 2)))
    rmse_after = float(np.sqrt(np.mean(residual ** 2)))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - np.sum(residual ** 2) / total) if total > 0 else 1.0
    d["raw_rt_error"], d["adjusted_ref_rt"] = y - x, fitted
    d["adjusted_rt_error"], d["model_weight"] = residual, weights
    loo_rmse, loo_median, loo_max, warnings = _rt_validation_summary(
        x, y, model, weights, rmse_before
    )
    equation = (
        f"database RT {intercept:+.3f} min" if model == "constant_shift"
        else f"{len(reference_knots)} monotonic knots"
        if model == "monotonic_piecewise" and reference_knots is not None
        else f"{slope:.5f} × database RT {intercept:+.3f} min"
    )
    knots = (
        {
            "reference_knots": [float(value) for value in reference_knots],
            "adjusted_knots": [float(value) for value in observed_knots],
        }
        if reference_knots is not None and observed_knots is not None else {}
    )
    return {
        "kind": model, "model": model, "n_used": int(len(d)),
        "slope": float(slope), "intercept": float(intercept),
        "rmse_before_min": rmse_before, "rmse_after_min": rmse_after,
        "loo_rmse_min": loo_rmse,
        "loo_median_absolute_error_min": loo_median,
        "loo_max_absolute_error_min": loo_max,
        "r_squared": r_squared, "quality_warnings": warnings,
        "calibration_rt_min": float(np.min(x)),
        "calibration_rt_max": float(np.max(x)), **knots,
        "label": (
            f"{model.replace('_', ' ').title()} RT model ({len(d)}): adjusted "
            f"reference = {equation}; RMSE {rmse_before:.3f} → "
            f"{rmse_after:.3f} min"
        ),
        "anchors": d.reindex(columns=_RT_CALIBRATION_COLUMNS).copy(),
    }


def fit_confirmed_internal_standard_rt_calibration(
    confirmed_candidates: pd.DataFrame,
    model: str = "linear",
) -> dict:
    """Fit a guarded RT-reference adjustment from confirmed lipid standards."""
    model = str(model).strip().casefold().replace(" ", "_")
    if model not in SUPPORTED_RT_CALIBRATION_KINDS:
        raise ValueError(f"Unsupported RT-calibration model: {model}")
    minimum_standards = {
        "constant_shift": 3,
        "linear": 2,
        "robust_linear": 4,
        "monotonic_piecewise": 6,
    }[model]
    model_title = model.replace("_", " ").title()
    empty = _empty_rt_calibration(model, minimum_standards)
    required = {"standard", "ref_rt", "feat_rt"}
    if (confirmed_candidates.empty
            or not required.issubset(confirmed_candidates.columns)):
        return empty
    d = confirmed_candidates.dropna(subset=["ref_rt", "feat_rt"]).copy()
    if len(d) < minimum_standards or d["ref_rt"].nunique() < minimum_standards:
        return {
            **empty,
            "label": (
                f"{model_title} RT calibration requires at least "
                f"{minimum_standards} distinct standards"
            ),
        }
    if d["standard"].duplicated().any():
        return {
            **empty,
            "label": "Select no more than one candidate per internal standard",
        }

    x = d["ref_rt"].to_numpy(float)
    y = d["feat_rt"].to_numpy(float)
    try:
        slope, intercept, weights, reference_knots, observed_knots = _fit_rt_model(
            x, y, model
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {**empty, "label": str(exc)}
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0:
        return {**empty, "label": "RT calibration model is invalid"}
    fitted = _apply_rt_arrays(
        x,
        kind=model,
        slope=slope,
        intercept=intercept,
        reference_knots=reference_knots,
        observed_knots=observed_knots,
    )
    return _rt_calibration_result(
        d, model, slope, intercept, fitted, weights,
        reference_knots, observed_knots,
    )


def apply_internal_standard_rt_calibration(ref_rt, calibration: dict):
    """Adjust reference RT values with a supported frozen RT model."""
    values = np.asarray(ref_rt, float)
    reference_knots = calibration.get("reference_knots")
    adjusted_knots = calibration.get("adjusted_knots")
    return _apply_rt_arrays(
        values,
        kind=str(calibration.get("kind", "linear")),
        slope=float(calibration.get("slope", 1.0)),
        intercept=float(calibration.get("intercept", 0.0)),
        reference_knots=(
            np.asarray(reference_knots, dtype=float)
            if reference_knots is not None else None
        ),
        observed_knots=(
            np.asarray(adjusted_knots, dtype=float)
            if adjusted_knots is not None else None
        ),
    )


def is_rt_calibration_model(calibration: dict | None) -> bool:
    """Return whether a persisted RT calibration is supported and safe."""
    if not calibration or calibration.get("kind") not in SUPPORTED_RT_CALIBRATION_KINDS:
        return False
    slope = float(calibration.get("slope", np.nan))
    intercept = float(calibration.get("intercept", np.nan))
    if not np.isfinite(slope) or slope <= 0 or not np.isfinite(intercept):
        return False
    if calibration.get("kind") == "constant_shift" and not np.isclose(
        slope, 1.0, rtol=0.0, atol=1e-12
    ):
        return False
    if calibration.get("kind") != "monotonic_piecewise":
        return True
    reference = np.asarray(calibration.get("reference_knots", ()), dtype=float)
    adjusted = np.asarray(calibration.get("adjusted_knots", ()), dtype=float)
    return bool(
        len(reference) >= 6
        and len(reference) == len(adjusted)
        and np.isfinite(reference).all()
        and np.isfinite(adjusted).all()
        and np.all(np.diff(reference) > 0)
        and np.all(np.diff(adjusted) > 0)
    )
