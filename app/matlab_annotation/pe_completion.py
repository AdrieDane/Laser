"""Add independently supported RP-positive PE without displacing existing calls.

This is a class-specific evidence check, not a probability or a relaxed mass/RT
window. Strong observed PE candidates define bounded local residuals; a target
composition never supports itself. Curation is not an input to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from resource_paths import REFERENCE_DATA_DIRECTORY
from .confidence import (
    _protected_rows,
    _row_keys,
    _strict_candidate_mask,
    _score_features,
)
from .composition_plausibility import composition_features


@dataclass(frozen=True)
class PECompletionSettings:
    enabled: bool = True
    ppm_tolerance: float = 3.0
    rt_tolerance_seconds: float = 10.0
    anchor_ppm: float = 1.5
    anchor_rt_seconds: float = 5.0
    minimum_support: int = 4
    maximum_anchors: int = 20
    maximum_composition_distance: float = 8.0
    local_rt_seconds: float = 5.0
    local_ppm: float = 1.0
    feature_margin: float = 0.25
    cross_polarity_seconds: float = 5.0
    confidence_threshold: float = 0.90
    model_path: Path = REFERENCE_DATA_DIRECTORY / "standard_pe_confidence.json"


def _validate(s: PECompletionSettings) -> None:
    for value in (
        s.ppm_tolerance,
        s.rt_tolerance_seconds,
        s.anchor_ppm,
        s.anchor_rt_seconds,
        s.maximum_composition_distance,
        s.local_rt_seconds,
        s.local_ppm,
        s.cross_polarity_seconds,
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("PE tolerances must be finite and positive")
    if s.ppm_tolerance > 3 or s.rt_tolerance_seconds > 10:
        raise ValueError("PE completion cannot widen the 3 ppm / 10 s envelope")
    if s.anchor_ppm > s.ppm_tolerance or s.anchor_rt_seconds > s.rt_tolerance_seconds:
        raise ValueError("PE anchors cannot use broader windows than candidates")
    if s.local_rt_seconds > s.rt_tolerance_seconds or s.local_ppm > s.ppm_tolerance:
        raise ValueError("PE local windows cannot exceed the candidate windows")
    if s.minimum_support < 4 or s.maximum_anchors <= s.minimum_support:
        raise ValueError("PE completion requires at least four other compositions")
    if not np.isfinite(s.feature_margin) or s.feature_margin < 0:
        raise ValueError("PE feature margin must be finite and nonnegative")
    if not 0 <= s.confidence_threshold <= 1:
        raise ValueError("PE confidence threshold must be in [0, 1]")


def pe_evidence_features(rows: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """PE evidence independent of the old globally calibrated class score."""
    x = pd.DataFrame(index=rows.index)
    x["local_rt_error"] = rows.pe_local_rt_error_seconds.abs()
    x["local_mass_error"] = rows.pe_local_mass_error_ppm.abs()
    x["original_rt_error"] = rows.rt_error.abs() * 60
    x["coherence"] = rows.grid_coherence_score
    x["intensity_z"] = rows.pe_intensity_z
    x["cross_polarity_supported"] = rows.pe_cross_polarity_supported.astype(float)
    x["cross_polarity_error"] = rows.get(
        "pe_polarity_error_seconds", pd.Series(np.nan, index=rows.index)
    ).abs()
    composition = composition_features(
        rows.rename(columns={"class": "lipid_class"}), priors
    )
    for column in composition:
        x[column] = composition[column]
    return x


def _pe_rows(frame: pd.DataFrame, adduct: str) -> pd.DataFrame:
    if frame.empty or not {"class", "adduct", "name"}.issubset(frame):
        return frame.iloc[:0].copy()
    mask = frame["class"].eq("PE") & frame.adduct.eq(adduct)
    mask &= ~frame["name"].astype(str).str.endswith(" (IS)")
    mask &= ~frame.get("is_is", pd.Series(False, index=frame.index)).fillna(
        False
    ).astype(bool)
    mask &= (
        frame.get("internal_standard_name", pd.Series("", index=frame.index))
        .fillna("")
        .eq("")
    )
    return frame[mask].copy()


def _clear_features(pool: pd.DataFrame, margin: float) -> pd.DataFrame:
    """One clearly preferred observed peak per composition, or no choice."""
    rows = []
    for _, group in pool.groupby(["carbon", "db"], sort=True):
        group = group.sort_values(
            ["_pe_cost", "feat_feature_id"], kind="stable"
        ).drop_duplicates("feat_feature_id")
        if (
            len(group) > 1
            and float(group._pe_cost.iloc[1] - group._pe_cost.iloc[0]) < margin
        ):
            continue
        rows.append(group.iloc[[0]])
    return pd.concat(rows) if rows else pool.iloc[:0].copy()


def _local_support(
    anchors: pd.DataFrame, row: pd.Series, s: PECompletionSettings
) -> pd.DataFrame:
    other = anchors[~(anchors.carbon.eq(row.carbon) & anchors.db.eq(row.db))].copy()
    distance = (other.carbon - row.carbon).abs() + 2 * (other.db - row.db).abs()
    other = other[distance.le(s.maximum_composition_distance)]
    if len(other) < s.minimum_support:
        return other.iloc[:0]
    if (
        other.carbon.max() - other.carbon.min() < 4
        and other.db.max() - other.db.min() < 2
    ):
        return other.iloc[:0]
    return other


def _negative_pe(annotations: pd.DataFrame, s: PECompletionSettings) -> pd.DataFrame:
    x = _pe_rows(annotations, "[M-H]-")
    if x.empty:
        return pd.DataFrame(columns=["carbon", "db", "negative_rt"])
    mask = _strict_candidate_mask(
        x,
        "RP_Neg",
        {
            "ppm": s.ppm_tolerance,
            "rt_seconds": s.rt_tolerance_seconds,
            "require_grid_coherence_pass": True,
        },
    )
    x = x[mask].drop_duplicates(["carbon", "db"], keep=False)
    return x[["carbon", "db", "feat_rt"]].rename(columns={"feat_rt": "negative_rt"})


def complete_rp_positive_pe(
    annotations: pd.DataFrame,
    all_pairs: pd.DataFrame,
    negative_annotations: pd.DataFrame | None = None,
    *,
    stream: str = "RP_Pos",
    settings: PECompletionSettings = PECompletionSettings(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return preserved baseline plus new, unambiguous, locally supported PE."""
    _validate(settings)
    audit = {
        "stream": stream,
        "enabled": settings.enabled,
        "baseline_annotations": len(annotations),
        "added_pe": 0,
    }
    empty = pd.DataFrame()
    if stream != "RP_Pos" or not settings.enabled or all_pairs.empty:
        status = (
            "not_applicable"
            if stream != "RP_Pos"
            else "disabled"
            if not settings.enabled
            else "no_candidates"
        )
        return (
            annotations.copy(),
            all_pairs.copy(),
            pd.DataFrame([{**audit, "status": status}]),
            empty,
            empty,
        )
    pe = _pe_rows(all_pairs, "[M+H]+")
    if pe.empty:
        return (
            annotations.copy(),
            all_pairs.copy(),
            pd.DataFrame([{**audit, "status": "no_pe_candidates"}]),
            empty,
            empty,
        )
    limits = {
        "ppm": settings.ppm_tolerance,
        "rt_seconds": settings.rt_tolerance_seconds,
        "require_grid_coherence_pass": True,
    }
    strict = _strict_candidate_mask(pe, stream, limits)
    pe = (
        pe[strict]
        .dropna(subset=["carbon", "db", "ppm_error", "rt_error", "feat_intensity"])
        .copy()
    )
    # Existing accepted rows and declared spike features are never repurposed.
    reserved = (
        set(annotations.feat_feature_id.astype(str)) if not annotations.empty else set()
    )
    declared = _protected_rows(all_pairs)
    reserved |= set(declared.feat_feature_id.astype(str))
    occupied_names = set(_pe_rows(annotations, "[M+H]+")["name"].astype(str))
    pe = pe[
        ~pe.feat_feature_id.astype(str).isin(set(declared.feat_feature_id.astype(str)))
    ].copy()
    pe["feat_feature_id"] = pe.feat_feature_id.astype(str)
    pe = pe.sort_values(["name", "feat_feature_id"], kind="stable").drop_duplicates(
        ["name", "adduct", "feat_feature_id"]
    )
    pe["_pe_cost"] = (
        pe.rt_error.abs() * 60 / settings.rt_tolerance_seconds
        + pe.ppm_error.abs() / settings.ppm_tolerance
    )
    logged = np.log1p(pe.feat_intensity.clip(lower=0))
    pe["pe_intensity_z"] = (logged - logged.median()) / max(float(logged.std()), 1e-9)
    clear = _clear_features(pe, settings.feature_margin)
    anchors = (
        clear[
            clear.ppm_error.abs().le(settings.anchor_ppm)
            & clear.rt_error.abs().mul(60).le(settings.anchor_rt_seconds)
            & ~clear.feat_feature_id.isin(reserved)
        ]
        .sort_values(["feat_intensity", "name"], ascending=[False, True], kind="stable")
        .head(settings.maximum_anchors)
        .copy()
    )
    anchors["pe_anchor_rt_residual_seconds"] = anchors.rt_error * 60
    anchors["pe_anchor_mass_residual_ppm"] = anchors.ppm_error
    # Cross-polarity support is calibrated only from strong positive anchors.
    # Target composition is excluded again from its alignment calculation.
    neg = _negative_pe(
        negative_annotations if negative_annotations is not None else empty, settings
    )
    offsets = anchors.merge(neg, on=["carbon", "db"], how="inner")
    offsets["polarity_offset_seconds"] = (offsets.feat_rt - offsets.negative_rt) * 60
    negative_lookup = {(r.carbon, r.db): r.negative_rt for r in neg.itertuples()}
    diagnostics = []
    for _, candidate in pe.iterrows():
        support = _local_support(anchors, candidate, settings)
        record = candidate.to_dict()
        record.update(
            pe_support_count=len(support),
            pe_support_compositions="",
            pe_cross_polarity_supported=False,
            pe_decision="insufficient_independent_anchors",
            pe_local_rt_shift_seconds=np.nan,
            pe_local_rt_error_seconds=np.nan,
            pe_local_mass_center_ppm=np.nan,
            pe_local_mass_error_ppm=np.nan,
        )
        if not support.empty:
            shift = float(support.pe_anchor_rt_residual_seconds.median())
            mass_center = float(support.pe_anchor_mass_residual_ppm.median())
            rt_error = float(candidate.rt_error * 60 - shift)
            ppm_error = float(candidate.ppm_error - mass_center)
            record.update(
                pe_local_rt_shift_seconds=shift,
                pe_local_rt_error_seconds=rt_error,
                pe_local_mass_center_ppm=mass_center,
                pe_local_mass_error_ppm=ppm_error,
                pe_support_compositions=";".join(
                    f"{int(r.carbon)}:{int(r.db)}" for r in support.itertuples()
                ),
                pe_decision="local_pattern_mismatch",
            )
            aligned = _local_support(offsets, candidate, settings)
            neg_rt = negative_lookup.get((candidate.carbon, candidate.db))
            polarity_error = np.nan
            if neg_rt is not None and not aligned.empty:
                offset = float(aligned.polarity_offset_seconds.median())
                spread = float(
                    (aligned.polarity_offset_seconds - offset).abs().median()
                )
                if spread <= settings.cross_polarity_seconds:
                    polarity_error = (float(candidate.feat_rt) - neg_rt) * 60 - offset
                    record.update(
                        pe_polarity_offset_seconds=offset,
                        pe_polarity_error_seconds=polarity_error,
                        pe_polarity_support_count=len(aligned),
                        pe_cross_polarity_supported=abs(polarity_error)
                        <= settings.cross_polarity_seconds,
                    )
            if (
                abs(rt_error) <= settings.local_rt_seconds
                and abs(ppm_error) <= settings.local_ppm
            ):
                record["pe_decision"] = "supported"
            record["_pe_cost"] = (
                abs(rt_error) / settings.local_rt_seconds
                + abs(ppm_error) / settings.local_ppm
            )
            if np.isfinite(polarity_error):
                record["_pe_cost"] += (
                    min(abs(polarity_error) / settings.cross_polarity_seconds, 2.0)
                    * 0.25
                )
        diagnostics.append(record)
    detail = pd.DataFrame(diagnostics)
    if detail.empty:
        return (
            annotations.copy(),
            all_pairs.copy(),
            pd.DataFrame([{**audit, "status": "no_eligible_pe"}]),
            anchors,
            detail,
        )
    proposed = detail[detail.pe_decision.eq("supported")].copy()
    try:
        model = json.loads(settings.model_path.read_text(encoding="utf-8"))
        if model.get("format") != "laser-pe-confidence-v1":
            raise ValueError("Unsupported PE confidence model")
    except (OSError, ValueError) as error:
        return (
            annotations.copy(),
            all_pairs.copy(),
            pd.DataFrame(
                [{**audit, "status": "model_unavailable", "detail": str(error)}]
            ),
            anchors,
            detail.drop(columns=[c for c in detail if c.startswith("_pe_")]),
        )
    detail["pe_completion_score"] = _score_features(
        pe_evidence_features(detail, model["composition_priors"]), model
    )
    proposed = _clear_features(
        detail[detail.pe_decision.eq("supported")], settings.feature_margin
    )
    proposed = proposed[
        proposed.pe_completion_score.ge(settings.confidence_threshold)
    ].copy()
    # Do not override another class at a shared feature, even if it was not
    # selected: multiple valid MS1 identities require review, not a PE bias.
    others = all_pairs[~all_pairs["class"].eq("PE")]
    if not others.empty:
        competing = others[_strict_candidate_mask(others, stream, limits)]
        proposed = proposed[
            ~proposed.feat_feature_id.isin(set(competing.feat_feature_id.astype(str)))
        ]
    proposed = (
        proposed[
            ~proposed.feat_feature_id.isin(reserved)
            & ~proposed["name"].astype(str).isin(occupied_names)
        ]
        .sort_values(["_pe_cost", "feat_feature_id"], kind="stable")
        .drop_duplicates("feat_feature_id")
    )
    proposed = proposed.drop(columns=[c for c in proposed if c.startswith("_pe_")])
    proposed["pe_completion_supported"] = True
    proposed["pe_evidence_score"] = (
        1.0
        - 0.25 * proposed.pe_local_rt_error_seconds.abs() / settings.local_rt_seconds
        - 0.25 * proposed.pe_local_mass_error_ppm.abs() / settings.local_ppm
    )
    proposed["automatic_accepted"] = True
    proposed["automatic_candidate"] = True
    proposed["auto_annotation_status"] = "pe_independent_anchor_supported"
    proposed["status"] = "pe_independent_anchor_supported"
    proposed["rt_evidence_mode"] = "pe_independent_local_pattern"
    proposed["standard_stream"] = stream
    proposed["chromatography"] = "RP"
    proposed["post_regression_count_retained"] = False
    accepted = (
        pd.concat([annotations, proposed], ignore_index=True, sort=False)
        if not proposed.empty
        else annotations.copy()
    )
    candidates = all_pairs.copy()
    chosen = set(_row_keys(proposed))
    mask = _row_keys(candidates).isin(chosen)
    for col in [
        "pe_completion_supported",
        "pe_completion_score",
        "pe_evidence_score",
        "pe_support_count",
        "pe_support_compositions",
        "pe_cross_polarity_supported",
        "pe_local_rt_shift_seconds",
        "pe_local_rt_error_seconds",
        "pe_local_mass_error_ppm",
    ]:
        lookup = proposed.set_index(_row_keys(proposed))[col].to_dict()
        candidates.loc[mask, col] = _row_keys(candidates[mask]).map(lookup).to_numpy()
    candidates.loc[mask, "automatic_accepted"] = True
    candidates.loc[mask, "automatic_candidate"] = True
    candidates.loc[mask, "auto_annotation_status"] = "pe_independent_anchor_supported"
    audit.update(
        status="completed",
        eligible_pe=len(pe),
        strong_anchors=len(anchors),
        cross_polarity_anchors=len(offsets),
        supported_candidates=int(detail.pe_decision.eq("supported").sum()),
        added_pe=len(proposed),
        final_annotations=len(accepted),
    )
    return (
        accepted,
        candidates,
        pd.DataFrame([audit]),
        anchors,
        detail.drop(columns=[c for c in detail if c.startswith("_pe_")]),
    )
