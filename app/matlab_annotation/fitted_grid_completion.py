"""Recover omitted fitted targets without widening windows or call counts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from resource_paths import REFERENCE_DATA_DIRECTORY
from scientific_defaults import SCIENTIFIC_DEFAULTS

from .automatic import matlab_annotation_adducts
from .confidence import (
    PostRegressionCompletionSettings, _candidate_features, _load_model,
    _preserve_count, _protected_rows, _row_keys, _score_features,
    _select_confident, _strict_candidate_mask,
)


@dataclass(frozen=True)
class FittedGridCompletionSettings:
    enabled: bool = True
    confidence_threshold: float = 0.50
    ambiguity_margin: float = 0.10
    feature_ambiguity_margin: float = 0.10
    grid_threshold: float = SCIENTIFIC_DEFAULTS.grid_coherence_threshold
    model_path: Path = (
        REFERENCE_DATA_DIRECTORY / "standard_fitted_grid_confidence.json"
    )


def propose_missing_pairs(
    existing: pd.DataFrame, reference: pd.DataFrame, features: pd.DataFrame,
    *, stream: str, limits: Mapping,
) -> pd.DataFrame:
    """Join unused feature/target pairs inside existing fitted mass/RT windows."""
    if reference.empty or features.empty:
        return pd.DataFrame()
    reference = reference[
        reference["ref_rt"].notna() & reference["ref_mz"].notna()
        & ~reference["class"].isin({
            "1-acylLPC", "2-acylLPC", "1-acylLPE", "2-acylLPE",
        })
        & ~reference["name"].astype(str).str.endswith(" (IS)")
        & reference["adduct"].isin(
            matlab_annotation_adducts(str(stream).split("_")[0])
        )
    ]
    masses = pd.to_numeric(features["mz"], errors="coerce").to_numpy(float)
    rts = pd.to_numeric(features["rt"], errors="coerce").to_numpy(float)
    valid = np.flatnonzero(np.isfinite(masses) & np.isfinite(rts) & (masses > 0))
    order = valid[np.argsort(masses[valid], kind="stable")]
    sorted_masses = masses[order]
    known = set(zip(
        existing["feat_feature_id"].astype(str), existing["name"],
        existing["adduct"], strict=True,
    ))
    ppm_limit = float(limits["ppm"])
    rt_limit = float(limits["rt_seconds"])
    rows = []
    for target in reference.to_dict("records"):
        mz = float(target["ref_mz"])
        rt = float(target["ref_rt"])
        if not np.isfinite(mz) or mz <= 0 or not np.isfinite(rt):
            continue
        left = np.searchsorted(sorted_masses, mz * (1 - ppm_limit * 1e-6))
        right = np.searchsorted(
            sorted_masses, mz * (1 + ppm_limit * 1e-6), side="right"
        )
        for position in order[left:right]:
            if abs(rts[position] - rt) * 60.0 > rt_limit:
                continue
            feature = features.iloc[position]
            key = (str(feature["feature_id"]), target["name"], target["adduct"])
            if key in known:
                continue
            ppm = (masses[position] - mz) / mz * 1e6
            if abs(ppm) >= ppm_limit:
                continue
            row = dict(target)
            row.update({f"feat_{column}": value for column, value in feature.items()})
            row.update({
                "feat_feature_id": str(feature["feature_id"]),
                "rt_expected": rt, "ppm_error": ppm,
                "rt_error": rts[position] - rt,
                "isotopologue": "", "is_is": False,
                "display_name": str(target["name"]),
                "fitted_grid_proposal": True,
                "matlab_candidate_stage": "fitted_grid_proposal",
                "positional_isomer_selected": False,
                "automatic_accepted": False, "automatic_candidate": False,
            })
            rows.append(row)
            known.add(key)
    return pd.DataFrame(rows)


def score_independent_grid_support(
    proposals: pd.DataFrame, accepted: pd.DataFrame,
    *, limits: Mapping, threshold: float,
) -> pd.DataFrame:
    """Score proposals using only pre-existing accepted, distinct grid nodes.

    New proposals never support one another. Sparse/isolated proposals remain
    review-only rather than passing automatically as unassessed edge nodes.
    """
    scored = proposals.copy()
    scored["grid_coherence_score"] = np.nan
    scored["grid_coherence_neighbors"] = 0
    scored["grid_coherence_dimensions"] = 2
    scored["grid_coherence_assessed"] = False
    scored["grid_coherence_pass"] = False
    scored["grid_coherence_threshold"] = float(threshold)
    if proposals.empty or accepted.empty:
        return scored
    for (lipid_class, adduct), group in proposals.groupby(["class", "adduct"]):
        support = accepted[
            accepted["class"].eq(lipid_class) & accepted["adduct"].eq(adduct)
        ].copy()
        support = support.dropna(subset=["carbon", "db", "ppm_error", "rt_error"])
        support = support.sort_values(
            "feat_intensity", ascending=False, kind="stable"
        ).drop_duplicates(["carbon", "db"])
        if len(support) < 2:
            continue
        distance = np.sqrt(
            ((group["carbon"].to_numpy(float)[:, None]
              - support["carbon"].to_numpy(float)[None, :]) / 2.0) ** 2
            + (group["db"].to_numpy(float)[:, None]
               - support["db"].to_numpy(float)[None, :]) ** 2
        )
        distance[(distance == 0) | (distance > 2.5)] = np.inf
        nearest = np.argsort(distance, axis=1, kind="stable")[:, :6]
        selected_distance = distance[np.arange(len(group))[:, None], nearest]
        valid = np.isfinite(selected_distance)
        mass_difference = (
            group["ppm_error"].to_numpy(float)[:, None]
            - support["ppm_error"].to_numpy(float)[nearest]
        ) / float(limits["ppm"])
        rt_difference = (
            group["rt_error"].to_numpy(float)[:, None]
            - support["rt_error"].to_numpy(float)[nearest]
        ) * 60.0 / float(limits["rt_seconds"])
        similarity = np.exp(-0.5 * (mass_difference ** 2 + rt_difference ** 2))
        weights = np.where(
            valid, np.exp(-0.5 * np.where(valid, selected_distance, 0.0) ** 2), 0.0
        )
        counts = valid.sum(axis=1)
        totals = weights.sum(axis=1)
        coherence = np.divide(
            (similarity * weights).sum(axis=1), totals,
            out=np.zeros(len(group)), where=totals > 0,
        )
        assessed = counts >= 2
        scored.loc[group.index, "grid_coherence_score"] = np.where(
            assessed, coherence, np.nan
        )
        scored.loc[group.index, "grid_coherence_neighbors"] = counts
        scored.loc[group.index, "grid_coherence_assessed"] = assessed
        scored.loc[group.index, "grid_coherence_pass"] = (
            assessed & (coherence >= threshold)
        )
    scored["grid_coherence_outlier"] = (
        scored["grid_coherence_assessed"] & ~scored["grid_coherence_pass"]
    )
    return scored


def filter_uncertain_new_features(
    selected: pd.DataFrame, pool: pd.DataFrame, features: pd.DataFrame,
    baseline: pd.DataFrame, *, margin: float,
) -> pd.DataFrame:
    """A new identity must also have a clear choice of XCMS feature."""
    baseline_identities = set(_candidate_features(baseline, "")["identity_key"])
    ranked = pool.assign(_identity=features["identity_key"]).sort_values(
        "post_regression_confidence", ascending=False, kind="stable"
    )
    rank = ranked.groupby("_identity").cumcount()
    second = ranked[rank.eq(1)].set_index("_identity")["post_regression_confidence"]
    identity = features.loc[selected.index, "identity_key"]
    gap = selected["post_regression_confidence"] - identity.map(second)
    return selected[
        identity.isin(baseline_identities) | gap.isna() | gap.ge(margin)
    ].copy()


def refine_fitted_grid_annotations(
    annotations: pd.DataFrame, all_pairs: pd.DataFrame,
    reference: pd.DataFrame, features: pd.DataFrame, *, stream: str,
    settings: FittedGridCompletionSettings = FittedGridCompletionSettings(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refine at the frozen baseline count, reserving all declared standards."""
    audit = {"stream": stream, "enabled": settings.enabled,
             "baseline_annotations": len(annotations)}
    if not settings.enabled or annotations.empty:
        return annotations, all_pairs, pd.DataFrame([{
            **audit, "status": "disabled" if not settings.enabled else "no_baseline",
        }])
    try:
        model = _load_model(settings.model_path)
    except (OSError, ValueError) as error:
        return annotations, all_pairs, pd.DataFrame([{
            **audit, "status": "model_unavailable", "detail": str(error),
        }])
    limits = model["candidate_limits"]
    proposals = propose_missing_pairs(
        all_pairs, reference, features, stream=stream, limits=limits,
    )
    proposals = score_independent_grid_support(
        proposals, annotations, limits=limits, threshold=settings.grid_threshold,
    )
    candidates = all_pairs.copy()
    candidates["fitted_grid_proposal"] = False
    if not proposals.empty:
        candidates = pd.concat([candidates, proposals], ignore_index=True, sort=False)
    protected = _protected_rows(annotations)
    protected_keys = set(_row_keys(protected))
    baseline = annotations[~_row_keys(annotations).isin(protected_keys)]
    eligible = _strict_candidate_mask(candidates, stream, limits)
    if not protected.empty:
        eligible &= ~candidates["feat_feature_id"].astype(str).isin(
            protected["feat_feature_id"].astype(str)
        )
    pool = candidates[eligible].copy()
    baseline_keys = set(_row_keys(baseline))
    original = pool[_row_keys(pool).isin(baseline_keys)].copy()
    if len(original) != len(baseline):
        return annotations, all_pairs, pd.DataFrame([{
            **audit, "status": "baseline_pair_unavailable",
        }])
    pool_features = _candidate_features(pool, stream, model)
    pool["post_regression_confidence"] = _score_features(pool_features, model)
    pool["post_regression_model"] = str(model["format"])
    selection_settings = PostRegressionCompletionSettings(
        confidence_threshold=settings.confidence_threshold,
        ambiguity_margin=settings.ambiguity_margin,
    )
    selected, _passing, _ambiguous = _select_confident(
        pool, pool_features, pool["post_regression_confidence"], selection_settings,
    )
    selected = filter_uncertain_new_features(
        selected, pool, pool_features, original,
        margin=settings.feature_ambiguity_margin,
    )
    selected, retained = _preserve_count(selected, original, pool, pool_features)
    selected["automatic_accepted"] = True
    selected["automatic_candidate"] = True
    selected["auto_annotation_status"] = np.where(
        selected["fitted_grid_proposal"], "fitted_grid_completed",
        "fitted_grid_confidence_supported",
    )
    selected.loc[selected["post_regression_count_retained"],
                 "auto_annotation_status"] = "baseline_count_retained"
    selected["status"] = selected["auto_annotation_status"]
    selected["rt_evidence_mode"] = "matlab_fitted_grid_refinement"
    accepted = pd.concat([selected, protected], ignore_index=True, sort=False)
    accepted["rank_for_feature"] = accepted.groupby("feat_feature_id").cumcount()
    if len(accepted) != len(annotations):
        raise ValueError("Fitted-grid refinement changed the baseline call count")
    selected_status = dict(zip(
        _row_keys(accepted), accepted["auto_annotation_status"], strict=True,
    ))
    keys = _row_keys(candidates)
    candidates["automatic_accepted"] = keys.isin(selected_status)
    candidates["automatic_candidate"] = candidates["automatic_accepted"]
    candidates["auto_annotation_status"] = keys.map(selected_status).fillna(
        "review_only_blue_candidate"
    )
    candidates.loc[pool.index, "post_regression_confidence"] = pool["post_regression_confidence"]
    candidates.loc[pool.index, "post_regression_model"] = str(model["format"])
    candidates["post_regression_eligible"] = candidates.get(
        "post_regression_eligible", pd.Series(False, index=candidates.index)
    ).fillna(False).astype(bool)
    candidates.loc[pool.index, "post_regression_eligible"] = True
    retained_keys = set(_row_keys(selected[selected["post_regression_count_retained"]]))
    candidates["post_regression_count_retained"] = keys.isin(retained_keys)
    return accepted, candidates, pd.DataFrame([{
        **audit, "status": "completed", "missing_pairs_proposed": len(proposals),
        "independently_supported_proposals": int(proposals["grid_coherence_pass"].sum()),
        "selected_completed_pairs": int(selected["fitted_grid_proposal"].sum()),
        "baseline_calls_retained": retained, "final_annotations": len(accepted),
        "feature_ambiguity_margin": settings.feature_ambiguity_margin,
        "model": str(model["format"]),
    }])


__all__ = ["FittedGridCompletionSettings", "refine_fitted_grid_annotations"]
