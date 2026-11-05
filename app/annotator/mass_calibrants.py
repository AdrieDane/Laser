"""Reference panels and candidate search for optional precursor-mass calibration.

Mass calibration is deliberately independent from CCS calibration.  A user can
search one of the documented calibrant panels with explicit m/z and RT windows,
inspect every candidate, and only then fit a correction.  Isotope-labelled
standards use their measured neutral masses from the Amsterdam UMC standard
table rather than an unlabelled sum-composition formula.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import re

import numpy as np
import pandas as pd
from resource_paths import (
    REFERENCE_DATA_DIRECTORY,
    STANDARD_CALIBRATION_LIBRARY_DIRECTORY,
)

from .matching import (
    AGILENT_TUNE_MIX_REFERENCE,
    apply_mass_calibration,
    highest_abundance_internal_standard_indices,
    is_mass_calibration_model,
    one_over_k0_to_ccs,
)


MASS_REFERENCE_PANELS = (
    "Labelled IS",
    "Labelled IS + Agilent tune mix",
    "Unlabelled IS",
    "Unlabelled IS + Agilent tune mix",
    "Agilent tune mix",
)

_LIPID_VALIDATION_DIRECTORY = STANDARD_CALIBRATION_LIBRARY_DIRECTORY
_INTERNAL_STANDARD_RT_REFERENCE = (
    REFERENCE_DATA_DIRECTORY
    / "matlab_pipeline"
    / "DataBase"
    / "internal_standards_all.tsv"
)
_LIPID_VALIDATION_FILENAMES = {
    "pos": "mzmine_lipid_validation_pos.csv",
    "neg": "mzmine_lipid_validation_neg.csv",
}
LIPID_VALIDATION_MEDIAN_LIMIT_PPM = 2.0
LIPID_VALIDATION_P95_LIMIT_PPM = 5.0
LIPID_VALIDATION_MINIMUM_TARGETS = 3

_PROTON = 1.007276466621
_AMMONIUM = 18.033823
_FORMATE = 44.998201

# standard label, database identity, exact neutral mass, RP RT (min),
# positive adduct, negative adduct.  These values are the isotope-labelled
# entries in DataBase/internal_standards_all.tsv.
_LABELLED_INTERNAL_STANDARDS = (
    ("d4-LPC(26:0)", "1-acyl LPC(26:0)", 639.514099, 10.48037935, "[M+H]+", "[M+HCOO]-"),
    ("d4-AC(26:0)", "AC(26:0)", 543.516467, 10.97492007, "[M+H]+", None),
    ("CE(16:0)-D7", "CE(16:0)", 631.628469, 16.23190441, "[M+NH4]+", None),
    ("PC(33:1)-D7", "PC(33:1)", 752.606095, 11.79, "[M+H]+", "[M+HCOO]-"),
    ("PE(33:1)-D7", "PE(33:1)", 710.559145, 11.90, "[M+H]+", "[M-H]-"),
    ("PS(33:1)-D7", "PS(33:1)", 754.548975, 11.65, "[M+H]+", "[M-H]-"),
    ("PG(33:1)-D7", "PG(33:1)", 741.553726, 11.29, "[M+H]+", "[M-H]-"),
    ("PI(33:1)-D7", "PI(33:1)", 829.569771, 11.12, "[M+NH4]+", "[M-H]-"),
    ("PA(33:1)-D7", "PA(33:1)", 667.516946, 12.40, "[M+H]+", "[M-H]-"),
    ("LPC(18:1)-D7", "1-acyl LPC(18:1)", 528.392080, 7.17, "[M+H]+", "[M+HCOO]-"),
    ("LPE(18:1)-D7", "1-acyl LPE(18:1)", 486.345130, 7.28, "[M+H]+", "[M-H]-"),
    ("CE(18:1)-D7", "CE(18:1)", 657.644119, 16.21, "[M+NH4]+", None),
    ("MG(18:1)-D7", "MG(18:1)", 363.336599, 8.46, "[M+NH4]+", None),
    ("DG(33:1)-D7", "DG(33:1)", 587.550614, 13.11, "[M+NH4]+", None),
    ("TG(48:1)-D7", "TG(48:1)", 811.764629, 15.54, "[M+NH4]+", None),
    ("SM(d36:2)-D9", "SM(d36:2)", 737.639718, 11.43, "[M+H]+", "[M+HCOO]-"),
    ("Cholesterol-D7", "Cholesterol", 393.3988034, 11.63, "[M+NH4]+", None),
)


def _ion_mz(neutral_mass: float, adduct: str) -> float:
    shifts = {
        "[M+H]+": _PROTON,
        "[M+NH4]+": _AMMONIUM,
        "[M-H]-": -_PROTON,
        "[M+HCOO]-": _FORMATE,
    }
    return float(neutral_mass + shifts[adduct])


@lru_cache(maxsize=1)
def _internal_standard_rt_table() -> pd.DataFrame:
    """Load the authoritative chromatography-specific internal-standard RTs."""
    table = pd.read_csv(
        _INTERNAL_STANDARD_RT_REFERENCE, sep="\t", encoding="cp1252"
    )
    required = {"trait", "mass", "RT_NP"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            "Internal-standard RT library is missing columns: "
            + ", ".join(sorted(missing))
        )
    table = table.copy()
    table["mass"] = pd.to_numeric(table["mass"], errors="coerce")
    table["RT_NP"] = pd.to_numeric(table["RT_NP"], errors="coerce")
    return table


def _np_identity(value: object) -> str:
    identity = re.sub(r"\s+\(IS\)$", "", str(value).strip())
    return re.sub(r"^(?:1-acyl|2-acyl)\s+", "", identity)


def _unique_np_rt(rows: pd.DataFrame) -> float:
    values = pd.to_numeric(rows.get("RT_NP"), errors="coerce").dropna().unique()
    if not len(values):
        return np.nan
    if len(values) > 1:
        raise ValueError("Internal-standard RT library has conflicting NP RT values")
    return float(values[0])


def _labelled_np_rt(neutral_mass: float) -> float:
    table = _internal_standard_rt_table()
    matches = table[
        np.isclose(table["mass"], float(neutral_mass), rtol=0.0, atol=1e-6)
    ]
    return _unique_np_rt(matches)


def _unlabelled_np_rt(identity: str) -> float:
    table = _internal_standard_rt_table()
    normalized = table["trait"].map(_np_identity)
    return _unique_np_rt(table[normalized.eq(_np_identity(identity))])


def _labelled_reference_panel(
    polarity: str, *, chromatography: str = "RP",
) -> pd.DataFrame:
    rows = []
    for standard, name, neutral_mass, ref_rt, pos_adduct, neg_adduct in (
        _LABELLED_INTERNAL_STANDARDS
    ):
        adduct = pos_adduct if polarity == "pos" else neg_adduct
        if adduct is None:
            continue
        rows.append({
            "standard": standard,
            "name": name,
            "adduct": adduct,
            "ref_mz": _ion_mz(neutral_mass, adduct),
            "ref_rt": (
                _labelled_np_rt(neutral_mass)
                if chromatography == "NP" else float(ref_rt)
            ),
            "source": "labelled internal standard",
        })
    return pd.DataFrame(rows)


def _unlabelled_reference_panel(
    reference: pd.DataFrame,
    polarity: str,
    standards: Mapping[str, str],
    *,
    chromatography: str = "RP",
) -> pd.DataFrame:
    if not standards:
        return pd.DataFrame()
    ref = (
        reference[reference["polarity"].astype(str).eq(polarity)].copy()
        if not reference.empty and "polarity" in reference
        else pd.DataFrame()
    )
    reverse = {database_name: label for label, database_name in standards.items()}
    if not ref.empty and "name" in ref:
        ref = ref[ref["name"].isin(reverse)].copy()
        ref["standard"] = ref["name"].map(reverse)
        ref["source"] = "unlabelled internal standard"
        if chromatography == "NP":
            ref["ref_rt"] = ref["name"].map(_unlabelled_np_rt)
        matched = ref[
            ["standard", "name", "adduct", "ref_mz", "ref_rt", "source"]
        ].drop_duplicates(["standard", "adduct", "ref_mz"])
    else:
        matched = pd.DataFrame()
    matched_labels = set(
        matched.get("standard", pd.Series(dtype=str)).astype(str)
    )
    # Preserve the full declared standard list in the audit even when a
    # polarity-specific library definition is absent.  Such rows cannot match
    # a feature, but they must be reported as missing instead of silently
    # disappearing from the chosen reference panel.
    unavailable = pd.DataFrame([
        {
            "standard": label,
            "name": database_name,
            "adduct": "(reference unavailable)",
            "ref_mz": np.nan,
            "ref_rt": np.nan,
            "source": "unlabelled internal standard",
        }
        for label, database_name in standards.items()
        if label not in matched_labels
    ])
    populated = [frame for frame in (matched, unavailable) if not frame.empty]
    return pd.concat(populated, ignore_index=True, sort=False)


def _tune_mix_reference_panel(polarity: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "standard": standard,
            "name": standard,
            "adduct": "[M]+" if polarity == "pos" else "[M]-",
            "ref_mz": float(ref_mz),
            "ref_rt": np.nan,
            "source": "Agilent tune mix",
        }
        for standard, ref_mz, _ in AGILENT_TUNE_MIX_REFERENCE.get(polarity, ())
    ])


def build_mass_reference_panel(
    reference: pd.DataFrame,
    *,
    polarity: str,
    panel_name: str,
    unlabelled_standards: Mapping[str, str],
    chromatography: str = "RP",
) -> pd.DataFrame:
    """Build one explicit, polarity-specific mass-calibrant panel."""
    if panel_name not in MASS_REFERENCE_PANELS:
        raise ValueError(f"Unknown mass-reference panel: {panel_name}")
    chromatography = str(chromatography).strip().upper()
    if chromatography not in {"NP", "RP"}:
        raise ValueError("Mass-reference chromatography must be NP or RP")
    frames = []
    if panel_name.startswith("Labelled"):
        frames.append(_labelled_reference_panel(
            polarity, chromatography=chromatography
        ))
    elif panel_name.startswith("Unlabelled"):
        frames.append(
            _unlabelled_reference_panel(
                reference,
                polarity,
                unlabelled_standards,
                chromatography=chromatography,
            )
        )
    if "Agilent tune mix" in panel_name:
        frames.append(_tune_mix_reference_panel(polarity))
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=[
            "standard", "name", "adduct", "ref_mz", "ref_rt", "source",
        ])
    return pd.concat(frames, ignore_index=True, sort=False)


def build_lipid_validation_reference_panel(polarity: str) -> pd.DataFrame:
    """Load the independent lipid panel used to evaluate mass correction."""
    normalized = str(polarity).strip().casefold()
    filename = _LIPID_VALIDATION_FILENAMES.get(normalized)
    if filename is None:
        raise ValueError("Lipid-validation polarity must be pos or neg.")
    path = _LIPID_VALIDATION_DIRECTORY / filename
    panel = pd.read_csv(
        path,
        header=None,
        names=("ref_rt", "formula", "standard", "ref_mz"),
    )
    panel["ref_rt"] = pd.to_numeric(panel["ref_rt"], errors="raise")
    panel["ref_mz"] = pd.to_numeric(panel["ref_mz"], errors="raise")
    panel["name"] = panel["standard"].astype(str)
    panel["adduct"] = "independent exact-mass target"
    panel["source"] = "independent lipid validation"
    return panel[[
        "standard", "name", "formula", "adduct", "ref_mz", "ref_rt", "source",
    ]].sort_values("ref_mz", kind="stable").reset_index(drop=True)


def evaluate_lipid_validation_panel(
    features: pd.DataFrame,
    calibration: dict | None,
    *,
    polarity: str,
    ppm_tol: float = 30.0,
    rt_tol: float = 0.20,
) -> dict:
    """Evaluate a LASER mass model on independent aligned lipid features."""
    references = build_lipid_validation_reference_panel(polarity)
    search_features = features.copy()
    if "mz_raw" in search_features:
        search_features["mz"] = pd.to_numeric(
            search_features["mz_raw"], errors="coerce"
        )
    detection = detect_mass_calibration_candidates(
        search_features,
        references,
        ppm_tol=float(ppm_tol),
        rt_tol=float(rt_tol),
    )
    candidates = detection["candidates"]
    if candidates.empty or not is_mass_calibration_model(calibration):
        return {
            **detection,
            "polarity": str(polarity).strip().casefold(),
            "details": pd.DataFrame(),
            "status": "UNAVAILABLE",
            "median_absolute_ppm_before": np.nan,
            "median_absolute_ppm_after": np.nan,
            "p95_absolute_ppm_after": np.nan,
            "within_5ppm_percent": np.nan,
        }
    assert calibration is not None
    chosen = candidates.loc[
        highest_abundance_internal_standard_indices(candidates)
    ].copy()
    chosen = chosen.merge(
        references[["standard", "ref_mz", "formula"]],
        on=["standard", "ref_mz"],
        how="left",
        validate="many_to_one",
    )
    observed = pd.to_numeric(chosen["feat_mz"], errors="coerce")
    theoretical = pd.to_numeric(chosen["ref_mz"], errors="coerce")
    corrected = pd.Series(
        apply_mass_calibration(observed, calibration), index=chosen.index
    )
    chosen["measured_mz_before"] = observed
    chosen["measured_mz_after"] = corrected
    chosen["ppm_before"] = (observed - theoretical) / theoretical * 1e6
    chosen["ppm_after"] = (corrected - theoretical) / theoretical * 1e6
    details = chosen.rename(columns={
        "standard": "target",
        "ref_mz": "theoretical_mz",
        "ref_rt": "target_rt_min",
        "feat_rt": "observed_rt_min",
        "feat_intensity": "intensity",
        "feat_feature_id": "feature_id",
    })[[
        "target", "formula", "feature_id", "theoretical_mz",
        "measured_mz_before", "measured_mz_after", "ppm_before", "ppm_after",
        "target_rt_min", "observed_rt_min", "rt_error", "intensity",
    ]].sort_values("theoretical_mz", kind="stable").reset_index(drop=True)
    absolute_before = details["ppm_before"].abs().dropna()
    absolute_after = details["ppm_after"].abs().dropna()
    median_after = float(absolute_after.median()) if len(absolute_after) else np.nan
    p95_after = (
        float(np.percentile(absolute_after, 95)) if len(absolute_after) else np.nan
    )
    status = (
        "PASS"
        if len(details) >= LIPID_VALIDATION_MINIMUM_TARGETS
        and median_after <= LIPID_VALIDATION_MEDIAN_LIMIT_PPM
        and p95_after <= LIPID_VALIDATION_P95_LIMIT_PPM
        else "REVIEW"
    )
    return {
        **detection,
        "polarity": str(polarity).strip().casefold(),
        "details": details,
        "status": status,
        "median_absolute_ppm_before": (
            float(absolute_before.median()) if len(absolute_before) else np.nan
        ),
        "median_absolute_ppm_after": median_after,
        "p95_absolute_ppm_after": p95_after,
        "within_5ppm_percent": (
            float(absolute_after.le(5.0).mean() * 100.0)
            if len(absolute_after) else np.nan
        ),
    }


def build_ccs_reference_panel(
    reference: pd.DataFrame,
    *,
    polarity: str,
    panel_name: str,
    unlabelled_standards: Mapping[str, str],
) -> pd.DataFrame:
    """Build the CCS panel corresponding to a selectable mass panel.

    Isotope labels do not materially alter gas-phase collision cross section,
    so labelled standards use the CCS of the matching unlabelled lipid and
    adduct. Tune-mix CCS values are calculated from the instrument compendium
    1/K0 values.
    """
    panel = build_mass_reference_panel(
        reference,
        polarity=polarity,
        panel_name=panel_name,
        unlabelled_standards=unlabelled_standards,
    )
    if panel.empty:
        return panel
    ref = reference[reference["polarity"].astype(str).eq(polarity)].copy()
    lookup_columns = ["name", "adduct", "ref_ccs", "charge"]
    for column in lookup_columns:
        if column not in ref.columns:
            ref[column] = np.nan
    ref_lookup = ref[lookup_columns].drop_duplicates(["name", "adduct"])
    panel = panel.merge(ref_lookup, on=["name", "adduct"], how="left")
    tune_mask = panel["source"].astype(str).eq("Agilent tune mix")
    tune_mobility = {
        standard: float(ref_mobility)
        for standard, _, ref_mobility in AGILENT_TUNE_MIX_REFERENCE.get(
            polarity, ()
        )
    }
    panel["ref_mobility"] = panel["standard"].map(tune_mobility)
    panel.loc[tune_mask, "charge"] = 1 if polarity == "pos" else -1
    panel.loc[tune_mask, "ref_ccs"] = one_over_k0_to_ccs(
        panel.loc[tune_mask, "ref_mobility"],
        panel.loc[tune_mask, "ref_mz"],
        panel.loc[tune_mask, "charge"],
    )
    # Keep mass-matched standards without a CCS reference in the panel.  The
    # detector reports these separately as ``ccs_unavailable`` instead of
    # incorrectly classifying them as absent from the data.
    return panel.reset_index(drop=True)


def detect_mass_calibration_candidates(
    features: pd.DataFrame,
    references: pd.DataFrame,
    *,
    ppm_tol: float,
    rt_tol: float,
    tune_mix_rt_max: float = 3.0,
    use_lipid_rt: bool = True,
) -> dict:
    """Search features against a mass-reference panel with explicit windows."""
    required = {"feature_id", "mz", "rt"}
    if features.empty or references.empty or not required.issubset(features.columns):
        return {
            "candidates": pd.DataFrame(),
            "missing": references.get(
                "standard", pd.Series(dtype=str)
            ).astype(str).drop_duplicates().tolist(),
            "reference_count": int(references.get(
                "standard", pd.Series(dtype=str)
            ).nunique()),
            "ppm_tol": float(ppm_tol),
            "rt_tol": float(rt_tol),
            "tune_mix_rt_max": float(tune_mix_rt_max),
        }
    feature_mz = pd.to_numeric(features["mz"], errors="coerce")
    feature_rt = pd.to_numeric(features["rt"], errors="coerce")
    rows = []
    found = set()
    for ref_row in references.itertuples(index=False):
        ppm_error = (feature_mz - float(ref_row.ref_mz)) / float(ref_row.ref_mz) * 1e6
        keep = ppm_error.abs().le(float(ppm_tol))
        if str(ref_row.source) == "Agilent tune mix":
            keep &= feature_rt.ge(0) & feature_rt.le(float(tune_mix_rt_max))
            expected_rt = np.nan
        elif use_lipid_rt:
            keep &= (feature_rt - float(ref_row.ref_rt)).abs().le(float(rt_tol))
            expected_rt = float(ref_row.ref_rt)
        else:
            expected_rt = np.nan
        matches = features.loc[keep].copy()
        if matches.empty:
            continue
        found.add(str(ref_row.standard))
        for _, feature in matches.iterrows():
            observed_mz = float(feature["mz"])
            raw_mz = pd.to_numeric(feature.get("mz_raw", observed_mz), errors="coerce")
            if not np.isfinite(raw_mz):
                raw_mz = observed_mz
            observed_rt = float(feature["rt"])
            rt_error = observed_rt - expected_rt if np.isfinite(expected_rt) else np.nan
            rows.append({
                "standard": str(ref_row.standard),
                "internal_standard_concentration": getattr(ref_row, "concentration", np.nan),
                "internal_standard_unit": getattr(ref_row, "unit", ""),
                "internal_standard_file": getattr(ref_row, "is_file", ""),
                "name": str(ref_row.name),
                "adduct": str(ref_row.adduct),
                "source": str(ref_row.source),
                "feat_feature_id": feature["feature_id"],
                "feat_intensity": pd.to_numeric(feature.get("intensity", np.nan), errors="coerce"),
                "feat_mz_raw": float(raw_mz),
                "feat_mz": observed_mz,
                "ref_mz": float(ref_row.ref_mz),
                "ppm_error": (observed_mz - float(ref_row.ref_mz)) / float(ref_row.ref_mz) * 1e6,
                "feat_rt": observed_rt,
                "ref_rt": expected_rt,
                "rt_error": rt_error,
            })
    candidates = pd.DataFrame(rows)
    if not candidates.empty:
        candidates["candidate_key"] = (
            candidates["standard"].astype(str) + "|"
            + candidates["feat_feature_id"].astype(str)
        )
        candidates["search_score"] = (
            candidates["ppm_error"].abs() / float(ppm_tol)
            + candidates["rt_error"].abs().fillna(0) / float(rt_tol)
        )
        candidates = candidates.sort_values(
            ["standard", "search_score", "feat_intensity"],
            ascending=[True, True, False],
            na_position="last",
        ).reset_index(drop=True)
    standards = references["standard"].astype(str).drop_duplicates().tolist()
    return {
        "candidates": candidates,
        "missing": [standard for standard in standards if standard not in found],
        "reference_count": len(standards),
        "ppm_tol": float(ppm_tol),
        "rt_tol": float(rt_tol),
        "tune_mix_rt_max": float(tune_mix_rt_max),
    }


def detect_ccs_calibration_candidates(
    features: pd.DataFrame,
    references: pd.DataFrame,
    *,
    ppm_tol: float,
    rt_tol: float,
    tune_mix_rt_max: float = 3.0,
    use_ccs_criterion: bool = False,
    ccs_tol_percent: float = 5.0,
) -> dict:
    """Search a selected calibrant panel and attach measured/reference CCS."""
    searched = detect_mass_calibration_candidates(
        features,
        references,
        ppm_tol=ppm_tol,
        rt_tol=rt_tol,
        tune_mix_rt_max=tune_mix_rt_max,
    )
    candidates = searched["candidates"].copy()
    if candidates.empty:
        return {
            **searched,
            "candidates": candidates,
            "rt_candidates": candidates,
            "tune_candidates": candidates,
            "ccs_unavailable": [],
        }
    panel_columns = [
        "standard", "name", "adduct", "ref_mz", "ref_ccs",
        "ref_mobility", "charge",
    ]
    candidates = candidates.merge(
        references[panel_columns].drop_duplicates(
            ["standard", "name", "adduct", "ref_mz"]
        ),
        on=["standard", "name", "adduct", "ref_mz"],
        how="left",
    )
    feature_lookup = (
        features.assign(_feature_key=features["feature_id"].astype(str))
        .drop_duplicates("_feature_key", keep="first")
        .set_index("_feature_key")
    )
    candidate_ids = candidates["feat_feature_id"].astype(str)
    mobility = pd.to_numeric(
        feature_lookup.reindex(candidate_ids)["mobility"].to_numpy()
        if "mobility" in feature_lookup
        else np.nan,
        errors="coerce",
    )
    candidates["feat_mobility"] = mobility
    candidates["feat_ccs_raw"] = one_over_k0_to_ccs(
        candidates["feat_mobility"],
        candidates["ref_mz"],
        candidates["charge"],
    )
    candidates["raw_ccs_error_percent"] = 100.0 * (
        candidates["feat_ccs_raw"] / candidates["ref_ccs"] - 1.0
    )
    candidates["rt_expected"] = candidates["ref_rt"]
    rt_candidates = candidates.copy()
    ccs_available = (
        candidates["feat_ccs_raw"].notna() & candidates["ref_ccs"].notna()
    )
    ccs_unavailable = (
        candidates.loc[~ccs_available, "standard"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    keep = ccs_available
    if use_ccs_criterion:
        keep &= candidates["raw_ccs_error_percent"].abs().le(ccs_tol_percent)
    candidates = candidates.loc[keep].reset_index(drop=True)
    searched["candidates"] = candidates
    searched["rt_candidates"] = rt_candidates
    searched["tune_candidates"] = candidates.loc[
        candidates["source"].astype(str).eq("Agilent tune mix")
    ].copy()
    searched["ccs_unavailable"] = ccs_unavailable
    unavailable = set(ccs_unavailable)
    searched["missing"] = [
        standard for standard in searched["missing"] if standard not in unavailable
    ]
    return searched
