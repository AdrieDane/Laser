"""Strict RP-positive PE admission from independent, frozen stream choices.

The two NP streams corroborate identity presence, not matched specimens or
chromatographic structure. Labels from manual-curation workbooks are never
inputs. A preferred RP peak failing these checks is not replaced by a less
preferred alternative solely because that alternative passes admission.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _flags(frame: pd.DataFrame, column: str) -> pd.Series:
    return (
        frame.get(column, pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(str)
        .str.casefold()
        .isin({"true", "1", "1.0"})
    )


def _primary_pe(frame: pd.DataFrame, adduct: str) -> dict[str, str]:
    required = {"name", "class", "adduct", "feat_feature_id"}
    if frame.empty or not required.issubset(frame):
        return {}
    ordinary = frame["class"].eq("PE") & frame.adduct.eq(adduct)
    ordinary &= ~frame["name"].astype(str).str.endswith(" (IS)")
    ordinary &= ~_flags(frame, "is_is")
    ordinary &= (
        frame.get("internal_standard_name", pd.Series("", index=frame.index))
        .fillna("")
        .eq("")
    )
    ordinary &= (
        frame.get("isotopologue", pd.Series("", index=frame.index)).fillna("").eq("")
    )
    selected = frame[ordinary].drop_duplicates("name", keep=False)
    selected = selected.dropna(subset=["feat_feature_id"])
    return dict(zip(selected["name"].astype(str), selected.feat_feature_id.astype(str)))


def add_rp_consensus_evidence(
    pool: pd.DataFrame, preceding_annotations: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Record original NP feature provenance from pre-experiment choices only."""
    result = pool.copy()
    for stream, adduct, suffix in (
        ("NP_Pos", "[M+H]+", "np_positive"),
        ("NP_Neg", "[M-H]-", "np_negative"),
    ):
        lookup = _primary_pe(preceding_annotations.get(stream, pd.DataFrame()), adduct)
        ids = result["name"].map(lookup).fillna("")
        result[f"pe_rp_{suffix}_feature_id"] = ids
        result[f"pe_rp_{suffix}_confirmed"] = ids.ne("")
    return result


def rp_consensus_checks(
    ranked: pd.DataFrame,
    *,
    ppm_tolerance: float,
    rt_seconds: float,
    minimum_rank_score: float,
) -> pd.DataFrame:
    """Independent admission checks; missing evidence never satisfies a check."""

    def numeric(column: str) -> pd.Series:
        return pd.to_numeric(
            ranked.get(column, pd.Series(np.nan, index=ranked.index)),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

    expected = np.log((numeric("carbon") + 5) * 0.0107 / 0.9893)
    isotope = numeric("q_isotope_upper_correlation").ge(0.90)
    isotope &= (numeric("q_isotope_upper_log_ratio") - expected).abs().le(0.50)
    isotope &= (
        ranked.get("isotope_upper_feature_id", pd.Series("", index=ranked.index))
        .fillna("")
        .ne("")
    )
    checks = {
        "mass": numeric("pe_corrected_mass_error_ppm")
        .abs()
        .lt(min(2.5, ppm_tolerance)),
        "local_rt": numeric("pe_local_rt_error_seconds").abs().le(min(5.0, rt_seconds)),
        "rank": numeric("pe_experimental_rank_score").ge(max(0.80, minimum_rank_score)),
        "sample_detection": (
            numeric("q_detection_fraction") * numeric("q_profile_samples")
        ).ge(0.80 * numeric("q_biological_samples"))
        & numeric("q_biological_samples").ge(4),
        "blank": numeric("q_log_sample_blank_ratio").ge(np.log(3)),
        "isotope": isotope,
        "rp_negative": (numeric("feat_rt") - numeric("pe_negative_rt"))
        .abs()
        .mul(60)
        .le(5),
        "np_positive": _flags(ranked, "pe_rp_np_positive_confirmed")
        & ranked.get("pe_rp_np_positive_feature_id", pd.Series("", index=ranked.index))
        .fillna("")
        .ne(""),
        "np_negative": _flags(ranked, "pe_rp_np_negative_confirmed")
        & ranked.get("pe_rp_np_negative_feature_id", pd.Series("", index=ranked.index))
        .fillna("")
        .ne(""),
    }
    return pd.DataFrame(
        {
            f"pe_rp_consensus_{key}_pass": value.fillna(False)
            for key, value in checks.items()
        },
        index=ranked.index,
    )
