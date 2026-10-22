"""Label-free peak evidence for experimental standard-lipidomics selectors.

Uses supplied intensities and validated sample-group metadata, never curation.
Mass neighbours describe possible ion relationships, not confirmed identities.
XCMS group RT bounds are deliberately not interpreted as chromatogram shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import warnings

import numpy as np
import pandas as pd


# NIST 13C - 12C; both carbon isotopes have the same electronic charge.
C13_SPACING = 1.00335483507
POSITIVE_ADDUCT_DELTAS = (17.026549086379, 21.981941533379)


@dataclass(frozen=True)
class PeakEvidenceSettings:
    ppm: float = 3.0
    coelution_seconds: float = 2.0
    minimum_profile_pairs: int = 4


def sample_roles(features: pd.DataFrame) -> tuple[list[str], dict[str, str], str]:
    """Use explicit roles or validated group labels; never infer specimen pairing."""
    columns = [
        str(c) for c in features.attrs.get("sample_columns", ()) if str(c) in features
    ]
    explicit = features.attrs.get("sample_roles", {})
    if explicit:
        allowed = {"biological", "qc", "blank", "standard", "unknown"}
        roles = {c: str(explicit.get(c, "unknown")).casefold() for c in columns}
        if set(roles.values()) - allowed:
            raise ValueError("Unsupported explicit sample role")
        return columns, roles, "explicit_sample_roles"
    if features.attrs.get("sample_group_mapping_status") != "validated":
        return columns, dict.fromkeys(columns, "unknown"), "unknown"
    groups = features.attrs.get("sample_group_map", {})
    roles = {}
    for column in columns:
        label = re.sub(
            r"^group[_ -]*\d+[_ -]*", "", str(groups.get(column, "")), flags=re.I
        )
        tokens = set(re.findall(r"[a-z0-9]+", label.casefold()))
        if not tokens:
            role = "unknown"
        elif tokens & {
            "blank",
            "blanco",
            "blankis",
            "blancois",
            "solventblank",
            "blanks",
        }:
            role = "blank"
        elif (
            tokens & {"qc", "pool", "pooled", "qualitycontrol"}
            or {"quality", "control"} <= tokens
        ):
            role = "qc"
        elif tokens & {"is", "standard", "calibrant", "calibration", "reference"}:
            role = "standard"
        else:
            role = "biological"
        roles[column] = role
    return columns, roles, "validated_sample_group_labels"


def _matrix(features: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(features), 0))
    values = features[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return np.where(np.isfinite(values) & (values >= 0), values, np.nan)


def _median(values: np.ndarray) -> np.ndarray:
    if not values.shape[1]:
        return np.full(len(values), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=1)


def _profile_correlation(
    a: np.ndarray, b: np.ndarray, minimum: int
) -> tuple[float, int]:
    valid = np.isfinite(a) & np.isfinite(b) & ((a > 0) | (b > 0))
    n = int(valid.sum())
    if n < minimum:
        return np.nan, n
    a, b = np.log1p(a[valid]), np.log1p(b[valid])
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan, n
    return float(np.corrcoef(a, b)[0, 1]), n


def peak_evidence(
    features: pd.DataFrame,
    feature_ids,
    *,
    polarity: str,
    charge: int = 1,
    settings: PeakEvidenceSettings = PeakEvidenceSettings(),
) -> pd.DataFrame:
    """Return one evidence row per requested feature, including unknown values.

    Ratios use robust sample medians and profile correlations require at least
    four usable measurements. No raw EIC, cross-mode sample pairing or isotope
    identity is fabricated. Feature IDs never enter numerical model evidence.
    """
    if polarity not in {"Pos", "Neg"}:
        raise ValueError("Peak evidence polarity must be Pos or Neg")
    if (
        isinstance(charge, bool)
        or not isinstance(charge, (int, np.integer))
        or charge not in {1, 2, 3, 4}
    ):
        raise ValueError("Ion evidence charge must be an integer from 1 to 4")
    if not (0 < settings.ppm <= 3 and 0 < settings.coelution_seconds <= 5):
        raise ValueError("Ion evidence requires <=3 ppm and <=5 second coelution")
    if settings.minimum_profile_pairs < 4:
        raise ValueError("Profile correlations require at least four paired samples")
    required = {"feature_id", "mz", "rt"}
    if not required.issubset(features):
        raise ValueError("Peak evidence requires original feature IDs, m/z and RT")
    if features.feature_id.astype(str).duplicated().any():
        raise ValueError("Original feature IDs must be unique within one stream")
    ids = features.feature_id.astype(str).to_numpy()
    positions = {v: i for i, v in enumerate(ids)}
    requested = sorted(set(map(str, feature_ids)))
    missing = set(requested) - set(positions)
    if missing:
        raise ValueError(
            f"Requested original features are unavailable: {sorted(missing)[:5]}"
        )
    samples, roles, role_source = sample_roles(features)
    all_values = _matrix(features, samples)
    bio_columns = [i for i, c in enumerate(samples) if roles[c] == "biological"]
    # Unknown-role measurements can supply descriptive within-stream profiles,
    # but cannot supply a biological/blank comparison or a claimed QC result.
    profile_columns = bio_columns or [
        i for i, c in enumerate(samples) if roles[c] == "unknown"
    ]
    profiles = all_values[:, profile_columns]
    biological = all_values[:, bio_columns]
    blanks = all_values[:, [i for i, c in enumerate(samples) if roles[c] == "blank"]]
    qcs = all_values[:, [i for i, c in enumerate(samples) if roles[c] == "qc"]]
    median = _median(profiles)
    bio_median, blank_median = _median(biological), _median(blanks)
    qc_median = _median(qcs)
    mz = pd.to_numeric(features.mz, errors="coerce").to_numpy(float)
    rt = pd.to_numeric(features.rt, errors="coerce").to_numpy(float)
    order = np.flatnonzero(np.isfinite(mz) & (mz > 0) & np.isfinite(rt))
    order = order[np.argsort(mz[order], kind="stable")]
    sorted_mass = mz[order]
    rows = []
    relations = [
        ("isotope_lower", -C13_SPACING / charge),
        ("isotope_upper", C13_SPACING / charge),
        ("isotope_lower2", -2 * C13_SPACING / charge),
    ]
    if polarity == "Pos" and charge == 1:
        relations += [
            ("ammonium", POSITIVE_ADDUCT_DELTAS[0]),
            ("sodium", POSITIVE_ADDUCT_DELTAS[1]),
        ]
    for fid in requested:
        i = positions[fid]
        profile = profiles[i]
        usable = np.isfinite(profile)
        detected = profile[usable] > 0
        row = dict(
            feature_id=fid,
            evidence_role_source=role_source,
            q_profile_samples=int(usable.sum()),
            q_biological_samples=len(bio_columns),
            q_blank_samples=blanks.shape[1],
            q_qc_samples=qcs.shape[1],
            q_detection_fraction=float(detected.mean()) if len(detected) else np.nan,
            q_log_median=float(np.log1p(median[i])),
            q_log_biological_median=float(np.log1p(bio_median[i])),
            q_log_sample_blank_ratio=float(
                np.log1p(bio_median[i]) - np.log1p(blank_median[i])
            ),
            q_log_max_median_ratio=float(
                np.log1p(np.nanmax(profile)) - np.log1p(median[i])
            )
            if usable.any()
            else np.nan,
            q_qc_relative_mad=float(
                np.nanmedian(np.abs(qcs[i] - qc_median[i])) / qc_median[i]
            )
            if qcs.shape[1] >= 3 and np.isfinite(qc_median[i]) and qc_median[i] > 0
            else np.nan,
        )
        for name, delta in relations:
            target = mz[i] + delta
            lo = np.searchsorted(sorted_mass, target * (1 - settings.ppm * 1e-6))
            hi = np.searchsorted(
                sorted_mass, target * (1 + settings.ppm * 1e-6), side="right"
            )
            hits = order[lo:hi]
            hits = hits[np.abs(rt[hits] - rt[i]) * 60 <= settings.coelution_seconds]
            best = None
            for j in hits:
                corr, count = _profile_correlation(
                    profile, profiles[j], settings.minimum_profile_pairs
                )
                if not np.isfinite(corr):
                    continue
                rank = (
                    corr,
                    -abs(rt[j] - rt[i]),
                    median[j] if np.isfinite(median[j]) else -1,
                )
                if best is None or rank > best[0]:
                    best = (rank, j, count)
            row[f"q_{name}_nearby_count"] = len(hits)
            row[f"q_{name}_correlation"] = best[0][0] if best is not None else np.nan
            row[f"q_{name}_log_ratio"] = (
                float(np.log1p(median[best[1]]) - np.log1p(median[i]))
                if best is not None
                else np.nan
            )
            row[f"q_{name}_rt_seconds"] = (
                float(abs(rt[best[1]] - rt[i]) * 60) if best is not None else np.nan
            )
            row[f"{name}_feature_id"] = str(ids[best[1]]) if best is not None else ""
        rows.append(row)
    return pd.DataFrame(rows)
