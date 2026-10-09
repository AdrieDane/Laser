"""Match features to reference lipids and score how well they fit the grid.

Two ideas:

1. **m/z matching** - a feature can be annotated as a reference lipid only if its
   measured m/z is within a ppm tolerance of the theoretical m/z.

2. **RT grid fit** - within a lipid class the species sit on a regular grid in the
   RT-vs-m/z plane (RT rises with carbon number, falls with double bonds).  The
   reference database already encodes the expected RT of every species, so a good
   annotation is one whose feature RT matches the reference RT.  Because a new
   batch/column can shift RT globally, we optionally fit a robust linear
   alignment (reference RT -> observed RT) from the confident 1:1 matches before
   judging the RT residual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ppm_window(mz: float, tol_ppm: float) -> tuple[float, float]:
    delta = mz * tol_ppm * 1e-6
    return mz - delta, mz + delta


def _match_by_mz(features: pd.DataFrame, ref: pd.DataFrame, tol_ppm: float) -> pd.DataFrame:
    """All (feature, reference) pairs within the m/z tolerance."""
    ref_sorted = ref.sort_values("ref_mz").reset_index(drop=True)
    ref_mz = ref_sorted["ref_mz"].to_numpy()
    fmz = features["mz"].to_numpy()
    delta = fmz * tol_ppm * 1e-6
    starts = np.searchsorted(ref_mz, fmz - delta, side="left")
    stops = np.searchsorted(ref_mz, fmz + delta, side="right")
    counts = stops - starts
    matched = counts > 0
    if not matched.any():
        return pd.DataFrame()

    feat_idx = np.repeat(np.flatnonzero(matched), counts[matched])
    ref_idx = np.concatenate([
        np.arange(start, stop)
        for start, stop in zip(starts[matched], stops[matched], strict=True)
    ])

    f = features.iloc[feat_idx].reset_index(drop=True)
    r = ref_sorted.iloc[ref_idx].reset_index(drop=True)
    ref_cols = [
        "name", "class", "adduct", "ref_mz", "ref_rt", "n_RT",
        "carbon", "db", "base", "is_is", "polarity",
    ]
    ref_cols += [
        c for c in ("formula", "charge", "ref_ccs", "ref_ccs_min",
                    "ref_ccs_max", "ref_ccs_n", "ref_ccs_names",
                    "ref_ccs_mapping")
        if c in r.columns
    ]
    pairs = pd.concat(
        [f.add_prefix("feat_"), r[ref_cols]],
        axis=1,
    )
    pairs["ppm_error"] = (pairs["feat_mz"] - pairs["ref_mz"]) / pairs["ref_mz"] * 1e6
    return pairs


def _robust_rt_alignment(pairs: pd.DataFrame, rt_tol: float,
                         iterations: int = 3) -> tuple[float, float, int]:
    """Fit observed_rt ~= slope * ref_rt + intercept using confident anchors.

    Anchors are features that match exactly one reference within the ppm window
    (unambiguous).  A few rounds of residual trimming make the fit robust to the
    remaining mis-matches.  Returns (slope, intercept, n_anchors).
    """
    counts = pairs.groupby("feat_feature_id")["name"].transform("size")
    anchors = pairs[(counts == 1)]
    if len(anchors) < 8:
        anchors = pairs  # too few unique hits; use everything
    x = anchors["ref_rt"].to_numpy(dtype=float)
    y = anchors["feat_rt"].to_numpy(dtype=float)

    slope, intercept = 1.0, 0.0
    mask: np.ndarray = np.ones(len(x), dtype=bool)
    for _ in range(iterations):
        if mask.sum() < 5:
            break
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        resid = y - (slope * x + intercept)
        keep = np.abs(resid) <= max(rt_tol, 2.5 * np.std(resid[mask]))
        if keep.sum() == mask.sum():
            mask = keep
            break
        mask = keep
    return float(slope), float(intercept), int(mask.sum())


def one_over_k0_to_ccs(one_over_k0, mz, charge):
    """Convert Bruker inverse reduced mobility to nitrogen CCS in Å².

    This is the inverse of Metabo-Wizard's Mason–Schamp implementation, using
    its exact nitrogen mass, TIMS temperature, and standard gas density.
    """
    k_b = 1.380649e-23
    e_c = 1.602176634e-19
    atomic_mass = 1.66053906660e-27
    n_0 = 2.6867774e25
    temperature = 305.15
    charge = np.abs(pd.to_numeric(charge, errors="coerce"))
    ion_mass = pd.to_numeric(mz, errors="coerce") * charge * atomic_mass
    nitrogen_mass = 28.014 * atomic_mass
    reduced_mass = ion_mass * nitrogen_mass / (ion_mass + nitrogen_mass)
    inverse_k0 = pd.to_numeric(one_over_k0, errors="coerce") * 1e4
    ccs_m2 = ((3 * charge * e_c) / (16 * n_0)
              * np.sqrt(2 * np.pi / (reduced_mass * k_b * temperature))
              * inverse_k0)
    return ccs_m2 / 1e-20


def ccs_to_one_over_k0(ccs, mz, charge):
    """Convert nitrogen CCS in Å² to Bruker inverse reduced mobility."""
    k_b = 1.380649e-23
    e_c = 1.602176634e-19
    atomic_mass = 1.66053906660e-27
    n_0 = 2.6867774e25
    temperature = 305.15
    charge = np.abs(pd.to_numeric(charge, errors="coerce"))
    ion_mass = pd.to_numeric(mz, errors="coerce") * charge * atomic_mass
    nitrogen_mass = 28.014 * atomic_mass
    reduced_mass = ion_mass * nitrogen_mass / (ion_mass + nitrogen_mass)
    ccs_m2 = pd.to_numeric(ccs, errors="coerce") * 1e-20
    k0 = ((3 * charge * e_c) / (16 * n_0)
          * np.sqrt(2 * np.pi / (reduced_mass * k_b * temperature))
          / ccs_m2)
    return 1.0 / (k0 * 1e4)


POSITIVE_CCS_INTERNAL_STANDARDS = {
    "LPC(14:0)": "1-acyl LPC(14:0) (IS)",
    "PC(28:0)": "PC(28:0) (IS)",
    "LPG(14:0)": "LPG(14:0) (IS)",
    "PG(28:0)": "PG(28:0) (IS)",
    "DG(28:0)": "DG(28:0) (IS)",
    "TG(42:0)": "TG(42:0) (IS)",
    "LPA(14:0)": "LPA(14:0) (IS)",
    "PA(28:0)": "PA(28:0) (IS)",
    "SM(d30:1)": "SM(d30:1) (IS)",
    "CL(56:0)": "CL(56:0) (IS)",
    "LPE(14:0)": "1-acyl LPE(14:0) (IS)",
    "PE(28:0)": "PE(28:0) (IS)",
    "PI(16:0)": "PI(16:0) (IS)",
    "PS(28:0)": "PS(28:0) (IS)",
    "SPH(d17:1)": "SPH(d17:1) (IS)",
    "SPH(d17:0)": "SPH(d17:0) (IS)",
    "Cer(d30:1)": "Cer(d30:1) (IS)",
}

NEGATIVE_CCS_INTERNAL_STANDARDS = {
    "LPC(14:0)": "1-acyl LPC(14:0) (IS)",
    "PC(28:0)": "PC(28:0) (IS)",
    "LPG(14:0)": "LPG(14:0) (IS)",
    "PG(28:0)": "PG(28:0) (IS)",
    "LPE(14:0)": "1-acyl LPE(14:0) (IS)",
    "PE(28:0)": "PE(28:0) (IS)",
    "PS(28:0)": "PS(28:0) (IS)",
    "SM(d30:1)": "SM(d30:1) (IS)",
}

# Agilent ESI-L tune-mix reference inverse reduced mobilities from the
# polarity-specific ES-TOF CCS compendium supplied with the instrument
# workflow. Reference CCS is derived from these values with the same
# Mason–Schamp implementation used for measured MZmine mobility values.
AGILENT_TUNE_MIX_REFERENCE = {
    "pos": (
        ("TuneMix 118", 118.086255, 0.5446),
        ("TuneMix 322", 322.048121, 0.7363),
        ("TuneMix 622", 622.028960, 0.9915),
        ("TuneMix 922", 922.009798, 1.1986),
        ("TuneMix 1221", 1221.990637, 1.3934),
        ("TuneMix 1521", 1521.971475, 1.5685),
        ("TuneMix 1821", 1821.952313, 1.7407),
        ("TuneMix 2121", 2121.933152, 1.9003),
        ("TuneMix 2421", 2421.913990, 2.0504),
        ("TuneMix 2721", 2721.894829, 2.1921),
    ),
    "neg": (
        ("TuneMix 301", 301.998139, 0.6690),
        ("TuneMix 601", 601.978977, 0.8824),
        ("TuneMix 1033", 1033.988109, 1.2582),
        ("TuneMix 1333", 1333.968947, 1.4073),
        ("TuneMix 1633", 1633.949786, 1.5797),
        ("TuneMix 1933", 1933.930624, 1.7479),
        ("TuneMix 2233", 2233.911463, 1.8995),
        ("TuneMix 2533", 2533.892301, 2.0511),
        ("TuneMix 2833", 2833.873139, 2.1498),
    ),
}

_CCS_CALIBRATION_COLUMNS = [
    "standard", "feat_feature_id", "name", "adduct", "feat_mz_raw", "feat_mz",
    "ref_mz", "feat_rt", "ref_rt", "ref_mobility",
    "ppm_error", "rt_expected", "rt_error", "feat_intensity",
    "feat_ccs_raw", "ref_ccs", "raw_ccs_error_percent", "calibrated_ccs",
    "calibrated_ccs_error_percent",
]

_RT_CALIBRATION_COLUMNS = [
    "standard", "feat_feature_id", "name", "adduct", "ref_rt", "feat_rt",
    "raw_rt_error", "adjusted_ref_rt", "adjusted_rt_error",
]

_MASS_CALIBRATION_COLUMNS = [
    "standard", "feat_feature_id", "name", "adduct", "source",
    "feat_mz", "ref_mz",
    "raw_ppm_error", "calibrated_mz", "calibrated_ppm_error",
    "feat_rt", "ref_rt", "rt_error", "feat_intensity", "model_weight",
]

MASS_CALIBRATION_MODELS = {
    "Constant ppm offset": "constant_ppm",
    "Linear": "linear",
    "Robust linear": "robust_linear",
    "Quadratic": "quadratic",
}
SUPPORTED_MASS_CALIBRATION_KINDS = frozenset(MASS_CALIBRATION_MODELS.values())


def highest_abundance_internal_standard_indices(
    candidates: pd.DataFrame,
) -> pd.Index:
    """Return one highest-intensity candidate index per detected standard."""
    if candidates.empty:
        return candidates.index
    return (
        candidates.sort_values(
            ["standard", "feat_intensity"],
            ascending=[True, False],
            na_position="last",
        )
        .drop_duplicates("standard", keep="first")
        .index
    )


# A straight line through two points passes through both by construction, so
# its RMSE is exactly zero and its R² exactly one regardless of where those two
# points sit. Fit-quality numbers only start carrying information once the line
# is over-determined, which is why the standard count is checked separately and
# the quality numbers are suppressed below this threshold.
CCS_CALIBRATION_MIN_RELIABLE_STANDARDS = 4
CCS_CALIBRATION_MIN_R_SQUARED = 0.95
CCS_CALIBRATION_MAX_RMSE_A2 = 5.0


def assess_ccs_calibration_quality(
    n_used: int,
    rmse_a2: float,
    r_squared: float,
) -> dict:
    """Judge a fitted CCS line by standard count first, fit quality second.

    Returns ``quality`` (``ok``/``caution``/``unverifiable``), whether the fit
    metrics carry information at this standard count, and human-readable
    warnings. ``unverifiable`` means the line cannot be checked at all, not
    that it is known to be wrong.
    """
    metrics_meaningful = n_used >= CCS_CALIBRATION_MIN_RELIABLE_STANDARDS
    warnings: list[str] = []
    if n_used < 2:
        return {
            "quality": "unverifiable",
            "quality_metrics_meaningful": False,
            "quality_warnings": ["No CCS calibration line was fitted."],
        }
    if n_used == 2:
        return {
            "quality": "unverifiable",
            "quality_metrics_meaningful": False,
            "quality_warnings": [
                "Only 2 internal standards: the line passes through both by "
                "construction, so its RMSE and R² cannot reveal an error. "
                "Treat this calibration as unverified and check the standards "
                "reported as missing before relying on it.",
            ],
        }
    if not metrics_meaningful:
        warnings.append(
            f"Only {n_used} internal standards: the line is barely "
            "over-determined, so its RMSE and R² are weak evidence. Check the "
            "standards reported as missing."
        )
    else:
        if np.isfinite(r_squared) and r_squared < CCS_CALIBRATION_MIN_R_SQUARED:
            warnings.append(
                f"Fit R² {r_squared:.3f} is below "
                f"{CCS_CALIBRATION_MIN_R_SQUARED:.2f}: the standards do not "
                "fall on a common line."
            )
        if np.isfinite(rmse_a2) and rmse_a2 > CCS_CALIBRATION_MAX_RMSE_A2:
            warnings.append(
                f"Fit RMSE {rmse_a2:.2f} Å² exceeds "
                f"{CCS_CALIBRATION_MAX_RMSE_A2:.1f} Å²: calibrated CCS will "
                "carry that error into every candidate."
            )
    return {
        "quality": "caution" if warnings else "ok",
        "quality_metrics_meaningful": metrics_meaningful,
        "quality_warnings": warnings,
    }


def _empty_ccs_calibration(kind: str, reason: str) -> dict:
    return {
        "kind": kind,
        "n_used": 0,
        "slope": 1.0,
        "intercept": 0.0,
        "rmse_a2": np.nan,
        "r_squared": np.nan,
        "label": reason,
        "anchors": pd.DataFrame(columns=_CCS_CALIBRATION_COLUMNS),
        **assess_ccs_calibration_quality(0, np.nan, np.nan),
    }


def detect_internal_standard_candidates(
    features: pd.DataFrame,
    ref: pd.DataFrame,
    ppm_tol: float,
    rt_tol: float,
    align_rt: bool = True,
    use_ccs_criterion: bool = False,
    ccs_tol_percent: float = 5.0,
    standards: dict[str, str] | None = None,
) -> dict:
    """Find lipid-IS candidates for CCS, RT, and mass calibration.

    Candidate detection uses precursor m/z and aligned RT. Raw, uncalibrated CCS
    may optionally be used as an additional detection criterion, but defaults to
    off because the purpose of this step is to correct a systematic CCS shift.
    ``rt_candidates`` retains detected standards without reference CCS so the
    optional RT and mass fits can use the full available lipid-IS panel.
    """
    standards = standards or POSITIVE_CCS_INTERNAL_STANDARDS
    target_names = set(standards.values())
    target_ref = ref[ref["name"].isin(target_names)].copy()
    pairs = _match_by_mz(features, ref, ppm_tol)
    if pairs.empty or target_ref.empty:
        return {
            "candidates": pd.DataFrame(), "alignment": (1.0, 0.0, 0),
            "rt_candidates": pd.DataFrame(),
            "missing": list(standards),
            "missing_no_mz_match": list(standards),
            "missing_outside_rt_window": [],
            "ccs_unavailable": [],
        }

    if align_rt:
        slope, intercept, n_anchors = _robust_rt_alignment(pairs, rt_tol)
    else:
        slope, intercept, n_anchors = 1.0, 0.0, 0
    pairs = pairs[pairs["name"].isin(target_names)].copy()
    pairs["rt_expected"] = slope * pairs["ref_rt"] + intercept
    pairs["rt_error"] = pairs["feat_rt"] - pairs["rt_expected"]
    pairs["score_2d"] = np.sqrt(
        (pairs["ppm_error"] / ppm_tol) ** 2
        + (pairs["rt_error"] / rt_tol) ** 2
    )
    if {"feat_mobility", "ref_mz", "charge"}.issubset(pairs.columns):
        pairs["feat_ccs_raw"] = one_over_k0_to_ccs(
            pairs["feat_mobility"], pairs["ref_mz"], pairs["charge"]
        )
    else:
        pairs["feat_ccs_raw"] = np.nan
    if "ref_ccs" not in pairs:
        pairs["ref_ccs"] = np.nan
    pairs["raw_ccs_error_percent"] = 100.0 * (
        pairs["feat_ccs_raw"] / pairs["ref_ccs"] - 1.0
    )
    reverse_names = {
        database_name: short_name
        for short_name, database_name in standards.items()
    }
    pairs["standard"] = pairs["name"].map(reverse_names)
    rt_keep = (
        (pairs["ppm_error"].abs() <= ppm_tol)
        & (pairs["rt_error"].abs() <= rt_tol)
    )
    rt_candidates = pairs[rt_keep].sort_values(
        ["standard", "score_2d", "feat_intensity"],
        ascending=[True, True, False], na_position="last",
    ).reset_index(drop=True)
    if not rt_candidates.empty:
        rt_candidates["candidate_key"] = (
            rt_candidates["standard"].astype(str) + "|"
            + rt_candidates["feat_feature_id"].astype(str)
        )
    keep = (
        rt_keep
        & pairs["feat_ccs_raw"].notna()
        & pairs["ref_ccs"].notna()
    )
    if use_ccs_criterion:
        keep &= pairs["raw_ccs_error_percent"].abs() <= ccs_tol_percent
    candidates = pairs[keep].sort_values(
        ["standard", "score_2d", "feat_intensity"],
        ascending=[True, True, False], na_position="last",
    ).reset_index(drop=True)
    candidates["candidate_key"] = (
        candidates["standard"].astype(str) + "|"
        + candidates["feat_feature_id"].astype(str)
    )
    found = set(rt_candidates["standard"])
    ccs_found = set(candidates["standard"])
    missing = [
        standard for standard in standards
        if standard not in found
    ]
    # A standard whose m/z matched but whose RT fell outside the window is
    # present in the data and was rejected by the search window. Reporting that
    # separately from "no m/z match" keeps a misplaced RT window from being
    # misread as a missing standard in the sample.
    mz_matched = set(pairs["standard"].dropna())
    missing_no_mz_match = [
        standard for standard in missing if standard not in mz_matched
    ]
    missing_outside_rt_window = [
        standard for standard in missing if standard in mz_matched
    ]
    return {
        "candidates": candidates,
        "rt_candidates": rt_candidates,
        "alignment": (float(slope), float(intercept), int(n_anchors)),
        "missing": missing,
        "missing_no_mz_match": missing_no_mz_match,
        "missing_outside_rt_window": missing_outside_rt_window,
        "ccs_unavailable": sorted(found - ccs_found),
    }


def detect_agilent_tune_mix_candidates(
    features: pd.DataFrame,
    polarity: str,
    ppm_tol: float = 5.0,
    rt_max: float = 3.0,
) -> dict:
    """Detect early-run Agilent ESI-L tune ions for CCS and mass calibration."""
    references = AGILENT_TUNE_MIX_REFERENCE.get(str(polarity), ())
    rows = []
    observed_standards = set()
    if references and {"feature_id", "mz", "rt"}.issubset(features.columns):
        feature_mz = pd.to_numeric(features["mz"], errors="coerce")
        feature_rt = pd.to_numeric(features["rt"], errors="coerce")
        for standard, ref_mz, ref_mobility in references:
            ref_ccs = float(one_over_k0_to_ccs(ref_mobility, ref_mz, 1))
            ppm_error = (feature_mz - ref_mz) / ref_mz * 1e6
            matches = features[
                ppm_error.abs().le(ppm_tol)
                & feature_rt.le(float(rt_max))
            ].copy()
            if matches.empty:
                continue
            observed_standards.add(standard)
            matches["standard"] = standard
            matches["name"] = standard
            matches["adduct"] = "[M]+" if polarity == "pos" else "[M]-"
            matches["ref_mz"] = ref_mz
            matches["ref_rt"] = np.nan
            matches["ref_mobility"] = ref_mobility
            matches["ref_ccs"] = ref_ccs
            matches["charge"] = 1 if polarity == "pos" else -1
            matches["ppm_error"] = (
                (pd.to_numeric(matches["mz"], errors="coerce") - ref_mz)
                / ref_mz * 1e6
            )
            matches["rt_expected"] = np.nan
            matches["rt_error"] = np.nan
            matches["score_2d"] = matches["ppm_error"].abs() / ppm_tol
            mobility = pd.to_numeric(
                matches.get("mobility", pd.Series(np.nan, index=matches.index)),
                errors="coerce",
            )
            matches["feat_ccs_raw"] = one_over_k0_to_ccs(
                mobility, ref_mz, 1
            )
            matches["raw_ccs_error_percent"] = 100.0 * (
                matches["feat_ccs_raw"] / ref_ccs - 1.0
            )
            renamed = matches.rename(columns={
                column: f"feat_{column}"
                for column in matches.columns
                if column not in {
                    "standard", "name", "adduct", "ref_mz", "ref_rt",
                    "ref_mobility", "ref_ccs", "charge", "ppm_error", "rt_expected",
                    "rt_error", "score_2d", "feat_ccs_raw",
                    "raw_ccs_error_percent",
                }
            })
            rows.append(renamed)
    candidates = (
        pd.concat(rows, ignore_index=True, sort=False)
        if rows else pd.DataFrame()
    )
    if not candidates.empty:
        candidates["candidate_key"] = (
            candidates["standard"].astype(str) + "|"
            + candidates["feat_feature_id"].astype(str)
        )
        candidates = candidates.sort_values(
            ["standard", "score_2d", "feat_intensity"],
            ascending=[True, True, False], na_position="last",
        ).reset_index(drop=True)
    return {
        "candidates": candidates,
        "missing": [
            standard for standard, _, _ in references
            if standard not in observed_standards
        ],
        "rt_max": float(rt_max),
    }


def fit_confirmed_internal_standard_ccs_calibration(
    confirmed_candidates: pd.DataFrame,
) -> dict:
    """Fit ``reference CCS = slope × raw CCS + intercept`` from confirmed IS."""
    required = {"standard", "feat_ccs_raw", "ref_ccs"}
    if (confirmed_candidates.empty
            or not required.issubset(confirmed_candidates.columns)):
        return _empty_ccs_calibration(
            "none", "CCS calibration requires at least two confirmed standards"
        )
    d = confirmed_candidates.dropna(subset=["feat_ccs_raw", "ref_ccs"]).copy()
    d = d[(d["feat_ccs_raw"] > 0) & (d["ref_ccs"] > 0)]
    if len(d) < 2 or d["feat_ccs_raw"].nunique() < 2:
        return _empty_ccs_calibration(
            "none", "CCS calibration requires at least two distinct standards"
        )
    if d["standard"].duplicated().any():
        return _empty_ccs_calibration(
            "none", "Select no more than one candidate per internal standard"
        )

    x = d["feat_ccs_raw"].to_numpy(float)
    y = d["ref_ccs"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0:
        return _empty_ccs_calibration("none", "CCS calibration line is invalid")
    fitted = slope * x + intercept
    residual = y - fitted
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - np.sum(residual ** 2) / total) if total > 0 else 1.0
    d["raw_ccs_error_percent"] = 100.0 * (d["feat_ccs_raw"] / d["ref_ccs"] - 1.0)
    d["calibrated_ccs"] = fitted
    d["calibrated_ccs_error_percent"] = 100.0 * (fitted / d["ref_ccs"] - 1.0)
    anchors = d.reindex(columns=_CCS_CALIBRATION_COLUMNS).copy()
    quality = assess_ccs_calibration_quality(len(d), rmse, r_squared)
    # Below the reliable-standard count the fit metrics are not evidence, so
    # they are deliberately kept out of the headline label rather than being
    # displayed as reassurance.
    label = (
        f"IS CCS line ({len(d)}): calibrated = {slope:.5f} × raw "
        f"{intercept:+.3f} Å²; RMSE {rmse:.2f} Å²"
        if quality["quality_metrics_meaningful"] else
        f"IS CCS line ({len(d)}): calibrated = {slope:.5f} × raw "
        f"{intercept:+.3f} Å²; fit quality not assessable at "
        f"{len(d)} standards"
    )
    return {
        "kind": "linear",
        "n_used": int(len(d)),
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse_a2": rmse,
        "r_squared": r_squared,
        "label": label,
        "anchors": anchors,
        **quality,
    }


def apply_internal_standard_ccs_calibration(ccs, calibration: dict):
    """Apply the confirmed linear CCS calibration while preserving NaNs."""
    values = np.asarray(ccs, float)
    slope = float(calibration.get("slope", 1.0))
    intercept = float(calibration.get("intercept", 0.0))
    corrected = slope * values + intercept
    return np.where(corrected > 0, corrected, np.nan)


def fit_confirmed_internal_standard_rt_calibration(
    confirmed_candidates: pd.DataFrame,
) -> dict:
    """Fit observed RT from database RT using the confirmed lipid standards."""
    required = {"standard", "ref_rt", "feat_rt"}
    if (confirmed_candidates.empty
            or not required.issubset(confirmed_candidates.columns)):
        return {
            "kind": "none", "n_used": 0, "slope": 1.0, "intercept": 0.0,
            "rmse_before_min": np.nan, "rmse_after_min": np.nan,
            "r_squared": np.nan,
            "label": "RT calibration requires at least two confirmed standards",
            "anchors": pd.DataFrame(columns=_RT_CALIBRATION_COLUMNS),
        }
    d = confirmed_candidates.dropna(subset=["ref_rt", "feat_rt"]).copy()
    if len(d) < 2 or d["ref_rt"].nunique() < 2:
        return {
            "kind": "none", "n_used": 0, "slope": 1.0, "intercept": 0.0,
            "rmse_before_min": np.nan, "rmse_after_min": np.nan,
            "r_squared": np.nan,
            "label": "RT calibration requires at least two distinct standards",
            "anchors": pd.DataFrame(columns=_RT_CALIBRATION_COLUMNS),
        }
    if d["standard"].duplicated().any():
        return {
            "kind": "none", "n_used": 0, "slope": 1.0, "intercept": 0.0,
            "rmse_before_min": np.nan, "rmse_after_min": np.nan,
            "r_squared": np.nan,
            "label": "Select no more than one candidate per internal standard",
            "anchors": pd.DataFrame(columns=_RT_CALIBRATION_COLUMNS),
        }

    x = d["ref_rt"].to_numpy(float)
    y = d["feat_rt"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0:
        return {
            "kind": "none", "n_used": 0, "slope": 1.0, "intercept": 0.0,
            "rmse_before_min": np.nan, "rmse_after_min": np.nan,
            "r_squared": np.nan, "label": "RT calibration line is invalid",
            "anchors": pd.DataFrame(columns=_RT_CALIBRATION_COLUMNS),
        }
    fitted = slope * x + intercept
    residual = y - fitted
    rmse_before = float(np.sqrt(np.mean((y - x) ** 2)))
    rmse_after = float(np.sqrt(np.mean(residual ** 2)))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - np.sum(residual ** 2) / total) if total > 0 else 1.0
    d["raw_rt_error"] = y - x
    d["adjusted_ref_rt"] = fitted
    d["adjusted_rt_error"] = residual
    anchors = d.reindex(columns=_RT_CALIBRATION_COLUMNS).copy()
    return {
        "kind": "linear",
        "n_used": int(len(d)),
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse_before_min": rmse_before,
        "rmse_after_min": rmse_after,
        "r_squared": r_squared,
        "label": (
            f"IS RT line ({len(d)}): adjusted reference = {slope:.5f} × "
            f"database {intercept:+.3f} min; RMSE {rmse_before:.3f} → "
            f"{rmse_after:.3f} min"
        ),
        "anchors": anchors,
    }


def apply_internal_standard_rt_calibration(ref_rt, calibration: dict):
    """Adjust reference RT values with a frozen internal-standard line."""
    values = np.asarray(ref_rt, float)
    return (float(calibration.get("slope", 1.0)) * values
            + float(calibration.get("intercept", 0.0)))


def fit_confirmed_mass_calibration(
    confirmed_candidates: pd.DataFrame,
    model: str = "linear",
) -> dict:
    """Fit a guarded precursor-mass correction from confirmed standards.

    ``constant_ppm`` is a one-parameter multiplicative correction. ``linear``
    preserves the historical two-parameter model. ``robust_linear`` uses Huber
    iteratively reweighted least squares, and ``quadratic`` is only enabled for
    at least five standards spanning 100 m/z so a three-parameter curve cannot
    be fitted to a narrow or exactly determined panel.
    """
    model = str(model).strip().casefold().replace(" ", "_")
    if model not in SUPPORTED_MASS_CALIBRATION_KINDS:
        raise ValueError(f"Unsupported mass-calibration model: {model}")
    minimum_standards = {
        "constant_ppm": 2,
        "linear": 2,
        "robust_linear": 3,
        "quadratic": 5,
    }[model]
    empty = {
        "kind": "none",
        "model": model,
        "n_used": 0,
        "slope": 1.0,
        "intercept": 0.0,
        "coefficients": [1.0, 0.0],
        "rmse_before_ppm": np.nan,
        "rmse_after_ppm": np.nan,
        "r_squared": np.nan,
        "quality_warnings": [],
        "label": (
            f"{model.replace('_', ' ').title()} mass calibration requires at "
            f"least {minimum_standards} confirmed standards"
        ),
        "anchors": pd.DataFrame(columns=_MASS_CALIBRATION_COLUMNS),
    }
    required = {"standard", "feat_mz", "ref_mz"}
    if (confirmed_candidates.empty
            or not required.issubset(confirmed_candidates.columns)):
        return empty
    d = confirmed_candidates.dropna(
        subset=["standard", "feat_mz", "ref_mz"]
    ).copy()
    d = d[(d["feat_mz"] > 0) & (d["ref_mz"] > 0)]
    if len(d) < minimum_standards or d["feat_mz"].nunique() < minimum_standards:
        return {
            **empty,
            "label": (
                f"{model.replace('_', ' ').title()} mass calibration requires "
                f"at least {minimum_standards} distinct standards"
            ),
        }
    if d["standard"].duplicated().any():
        return {
            **empty,
            "label": "Select no more than one candidate per calibration standard",
        }
    observed = d["feat_mz"].to_numpy(float)
    reference = d["ref_mz"].to_numpy(float)
    weights: np.ndarray = np.ones(len(d), dtype=float)
    if model == "constant_ppm":
        slope = float(np.median(reference / observed))
        intercept = 0.0
        coefficients = np.asarray([slope, intercept], dtype=float)
    elif model == "robust_linear":
        coefficients = np.polyfit(observed, reference, 1)
        for _ in range(30):
            residual = reference - np.polyval(coefficients, observed)
            centre = float(np.median(residual))
            scale = float(1.4826 * np.median(np.abs(residual - centre)))
            if not np.isfinite(scale) or scale <= np.finfo(float).eps:
                break
            cutoff = 1.345 * scale
            new_weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-15))
            updated = np.polyfit(observed, reference, 1, w=np.sqrt(new_weights))
            weights = new_weights
            if np.allclose(coefficients, updated, rtol=1e-12, atol=1e-14):
                coefficients = updated
                break
            coefficients = updated
        slope, intercept = map(float, coefficients)
    elif model == "quadratic":
        if float(np.ptp(observed)) < 100.0:
            return {
                **empty,
                "label": (
                    "Quadratic mass calibration requires standards spanning at "
                    "least 100 m/z"
                ),
            }
        coefficients = np.polyfit(observed, reference, 2)
        slope = float(coefficients[1])
        intercept = float(coefficients[2])
        derivative = 2.0 * float(coefficients[0]) * np.asarray(
            [observed.min(), observed.max()]
        ) + slope
        if not np.isfinite(derivative).all() or np.any(derivative <= 0):
            return {
                **empty,
                "label": "Quadratic mass calibration is non-monotonic and unsafe",
            }
    else:
        coefficients = np.polyfit(observed, reference, 1)
        slope, intercept = map(float, coefficients)
    if not np.isfinite(coefficients).all() or slope <= 0:
        return {**empty, "label": "Mass calibration model is invalid"}
    calibrated = np.polyval(coefficients, observed)
    raw_ppm = (observed - reference) / reference * 1e6
    calibrated_ppm = (calibrated - reference) / reference * 1e6
    total = float(np.sum((reference - np.mean(reference)) ** 2))
    residual = reference - calibrated
    r_squared = (
        float(1.0 - np.sum(residual ** 2) / total)
        if total > 0 else 1.0
    )
    d["raw_ppm_error"] = raw_ppm
    d["calibrated_mz"] = calibrated
    d["calibrated_ppm_error"] = calibrated_ppm
    d["model_weight"] = weights
    anchors = d.reindex(columns=_MASS_CALIBRATION_COLUMNS).copy()
    rmse_before = float(np.sqrt(np.mean(raw_ppm ** 2)))
    rmse_after = float(np.sqrt(np.mean(calibrated_ppm ** 2)))
    warnings = []
    if len(d) == minimum_standards and model in {"linear", "quadratic"}:
        warnings.append(
            "The model is minimally determined; add standards before relying on "
            "its fit statistics or extrapolating beyond the calibration range."
        )
    if rmse_after >= rmse_before:
        warnings.append(
            "The selected model does not improve RMS mass error; keep calibration "
            "disabled or review the selected standards."
        )
    model_title = model.replace("_", " ").title()
    equation = (
        f"{coefficients[0]:.3e} × observed² {coefficients[1]:+.9f} × observed "
        f"{coefficients[2]:+.6f}"
        if model == "quadratic"
        else f"{slope:.9f} × observed {intercept:+.6f}"
    )
    return {
        "kind": model,
        "model": model,
        "n_used": int(len(d)),
        "slope": float(slope),
        "intercept": float(intercept),
        "coefficients": [float(value) for value in coefficients],
        "rmse_before_ppm": rmse_before,
        "rmse_after_ppm": rmse_after,
        "r_squared": r_squared,
        "quality_warnings": warnings,
        "calibration_mz_min": float(observed.min()),
        "calibration_mz_max": float(observed.max()),
        "label": (
            f"{model_title} mass model ({len(d)}): calibrated = {equation}; "
            f"RMS ppm {rmse_before:.3f} → "
            f"{rmse_after:.3f}"
        ),
        "anchors": anchors,
    }


def apply_mass_calibration(mz, calibration: dict):
    """Apply any supported frozen mass calibration while preserving NaNs."""
    values = np.asarray(mz, float)
    coefficients = calibration.get("coefficients")
    if coefficients is None:
        coefficients = [
            float(calibration.get("slope", 1.0)),
            float(calibration.get("intercept", 0.0)),
        ]
    corrected = np.polyval(np.asarray(coefficients, dtype=float), values)
    return np.where(corrected > 0, corrected, np.nan)


def is_mass_calibration_model(calibration: dict | None) -> bool:
    """Return whether a persisted calibration uses a supported model."""
    return bool(
        calibration
        and str(calibration.get("kind", "")) in SUPPORTED_MASS_CALIBRATION_KINDS
    )


def annotate(
    features: pd.DataFrame,
    ref: pd.DataFrame,
    ppm_tol: float = 5.0,
    rt_tol: float = 0.5,
    align_rt: bool = True,
    ccs_tol_percent: float = 10.0,
    use_mobility: bool = True,
    ccs_calibration: dict | None = None,
    rt_calibration: dict | None = None,
    use_rt: bool = True,
) -> dict:
    """Annotate features against the reference.

    Returns a dict with:
      * ``annotations`` - best annotation per (feature, lipid) that passes both
        tolerances, with ppm/RT deviations and a grid-fit score.
      * ``alignment``   - (slope, intercept, n_anchors) used for RT alignment.
      * ``all_pairs``   - every m/z candidate (also the RT-failing ones) so the UI
        can show near-misses.
    """
    pairs = _match_by_mz(features, ref, ppm_tol)
    calibration = (
        dict(ccs_calibration)
        if ccs_calibration is not None
        else _empty_ccs_calibration("none", "CCS calibration not applied")
    )
    calibration.setdefault("slope", 1.0)
    calibration.setdefault("intercept", 0.0)
    calibration.setdefault("kind", "none")
    frozen_rt_calibration = (
        dict(rt_calibration)
        if rt_calibration is not None and rt_calibration.get("kind") == "linear"
        else None
    )
    if pairs.empty:
        return {"annotations": pairs, "alignment": (1.0, 0.0, 0),
                "ccs_evidence_n": 0,
                "ccs_calibration": calibration,
                "rt_calibration": frozen_rt_calibration,
                "all_pairs": pairs}

    if not use_rt:
        slope, intercept, n_anchors = 1.0, 0.0, 0
    elif frozen_rt_calibration:
        slope = float(frozen_rt_calibration.get("slope", 1.0))
        intercept = float(frozen_rt_calibration.get("intercept", 0.0))
        n_anchors = int(frozen_rt_calibration.get("n_used", 0))
    elif align_rt:
        slope, intercept, n_anchors = _robust_rt_alignment(pairs, rt_tol)
    else:
        slope, intercept, n_anchors = 1.0, 0.0, 0

    pairs["rt_expected"] = (
        slope * pairs["ref_rt"] + intercept if use_rt else np.nan
    )
    pairs["rt_error"] = (
        pairs["feat_rt"] - pairs["rt_expected"] if use_rt else np.nan
    )
    pairs["has_rt_evidence"] = bool(use_rt)

    # First calculate the 2D evidence, then convert Bruker 1/K0 to raw CCS for
    # each candidate (charge is candidate-specific). A calibration line fitted
    # from user-confirmed internal standards is applied before CCS filtering and
    # scoring.
    rt_component = (
        (pairs["rt_error"] / rt_tol) ** 2 if use_rt else 0.0
    )
    pairs["score_2d"] = np.sqrt(
        (pairs["ppm_error"] / ppm_tol) ** 2 + rt_component
    )
    needed = {"ref_ccs", "charge", "ref_mz", "feat_mobility"}
    if needed.issubset(pairs.columns):
        pairs["feat_ccs_raw"] = one_over_k0_to_ccs(
            pairs["feat_mobility"], pairs["ref_mz"], pairs["charge"])
        pairs["feat_ccs"] = apply_internal_standard_ccs_calibration(
            pairs["feat_ccs_raw"], calibration
        )
        pairs["ccs_error_raw"] = pairs["feat_ccs_raw"] - pairs["ref_ccs"]
        pairs["ccs_error_percent_raw"] = (
            100 * pairs["ccs_error_raw"] / pairs["ref_ccs"]
        )
        raw_expected_ccs = (
            (pairs["ref_ccs"] - calibration["intercept"])
            / calibration["slope"]
        )
        raw_expected_ccs = raw_expected_ccs.where(raw_expected_ccs > 0)
        pairs["mobility_expected"] = ccs_to_one_over_k0(
            raw_expected_ccs, pairs["ref_mz"], pairs["charge"])
        pairs["mobility_error"] = pairs["feat_mobility"] - pairs["mobility_expected"]
        pairs["ccs_error"] = pairs["feat_ccs"] - pairs["ref_ccs"]
        pairs["ccs_error_percent"] = 100 * pairs["ccs_error"] / pairs["ref_ccs"]
    else:
        pairs["feat_ccs_raw"] = np.nan
        pairs["feat_ccs"] = np.nan
        pairs["mobility_expected"] = np.nan
        pairs["mobility_error"] = np.nan
        pairs["ccs_error_raw"] = np.nan
        pairs["ccs_error_percent_raw"] = np.nan
        pairs["ccs_error"] = np.nan
        pairs["ccs_error_percent"] = np.nan
    pairs["ccs_calibration_slope"] = calibration["slope"]
    pairs["ccs_calibration_intercept"] = calibration["intercept"]
    pairs["ccs_calibrated"] = calibration["kind"] == "linear"
    pairs["has_mobility_evidence"] = use_mobility & pairs["ccs_error_percent"].notna()
    mobility_component = np.where(
        pairs["has_mobility_evidence"],
        (pairs["ccs_error_percent"] / ccs_tol_percent) ** 2,
        0.0,
    )
    pairs["score"] = np.sqrt(pairs["score_2d"] ** 2 + mobility_component)

    mobility_pass = (~pairs["has_mobility_evidence"]
                     | (pairs["ccs_error_percent"].abs() <= ccs_tol_percent))
    rt_pass = (
        pairs["rt_error"].abs() <= rt_tol
        if use_rt else pd.Series(True, index=pairs.index)
    )
    passing = pairs[
        (pairs["ppm_error"].abs() <= ppm_tol)
        & rt_pass
        & mobility_pass
    ].copy()

    # Count alternatives per feature (how many lipids it could be).
    alt = passing.groupby("feat_feature_id")["name"].transform("nunique")
    passing["n_alternatives"] = alt

    # Best annotation per feature and per lipid.
    passing = passing.sort_values("score")
    passing["rank_for_feature"] = passing.groupby("feat_feature_id").cumcount()
    passing["rank_for_lipid"] = passing.groupby(["name", "adduct"]).cumcount()

    return {
        "annotations": passing.reset_index(drop=True),
        "alignment": (slope, intercept, n_anchors),
        "ccs_evidence_n": int(pairs["has_mobility_evidence"].sum()),
        "ccs_calibration": calibration,
        "rt_calibration": frozen_rt_calibration,
        "all_pairs": pairs.reset_index(drop=True),
    }
