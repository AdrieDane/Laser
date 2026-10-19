"""Curated-cohort calibration for conservative post-regression completion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import exp
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from resource_paths import REFERENCE_DATA_DIRECTORY

from .automatic import matlab_annotation_adducts
from .composition_plausibility import composition_features


DEFAULT_CONFIDENCE_MODEL_PATH = (
    REFERENCE_DATA_DIRECTORY / "standard_post_regression_confidence.json"
)
_POSITIONAL_CLASSES = frozenset({
    "1-acylLPC", "2-acylLPC", "1-acylLPE", "2-acylLPE",
})


@dataclass(frozen=True)
class PostRegressionCompletionSettings:
    enabled: bool = True
    confidence_threshold: float = 0.50
    ambiguity_margin: float = 0.10
    model_path: Path = DEFAULT_CONFIDENCE_MODEL_PATH


def _load_model(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") not in {
        "laser-post-regression-confidence-v1",
        "laser-post-regression-confidence-v2",
    }:
        raise ValueError(f"Unsupported post-regression confidence model: {source}")
    return payload


def _zscore_by_group(
    values: pd.Series, groups: list[pd.Series],
) -> pd.Series:
    frame = pd.DataFrame({"value": values})
    names = []
    for index, group in enumerate(groups):
        name = f"group_{index}"
        frame[name] = group.to_numpy()
        names.append(name)
    grouped = frame.groupby(names, dropna=False)["value"]
    median = grouped.transform("median")
    standard_deviation = grouped.transform("std").fillna(0.0)
    return (values - median) / (standard_deviation + 1e-9)


def _candidate_features(
    candidates: pd.DataFrame, stream: str, model: Mapping | None = None,
) -> pd.DataFrame:
    features = pd.DataFrame(index=candidates.index)
    features["stream"] = str(stream)
    features["lipid_class"] = candidates["class"].fillna("").astype(str)
    features["adduct"] = candidates["adduct"].fillna("").astype(str)
    features["abs_ppm"] = pd.to_numeric(
        candidates["ppm_error"], errors="coerce"
    ).abs()
    features["abs_rt_seconds"] = (
        pd.to_numeric(candidates["rt_error"], errors="coerce").abs() * 60.0
    )
    features["coherence"] = pd.to_numeric(
        candidates.get("grid_coherence_score"), errors="coerce"
    )
    intensity = np.log1p(
        pd.to_numeric(candidates["feat_intensity"], errors="coerce").clip(lower=0)
    )
    stream_series = features["stream"]
    features["stream_intensity_z"] = _zscore_by_group(
        intensity, [stream_series]
    )
    features["class_intensity_z"] = _zscore_by_group(
        intensity, [stream_series, features["lipid_class"]]
    )
    feature_key = candidates["feat_feature_id"].fillna("").astype(str)
    identity_key = (
        features["lipid_class"] + "|"
        + candidates["name"].fillna("").astype(str) + "|"
        + features["adduct"]
    )
    features["feature_candidates"] = feature_key.groupby(feature_key).transform(
        "size"
    )
    features["identity_features"] = identity_key.groupby(identity_key).transform(
        "size"
    )
    features["identity_key"] = identity_key
    features["stream_class"] = features["stream"] + "|" + features["lipid_class"]
    if model and "composition_priors" in model:
        composition = pd.DataFrame({
            "lipid_class": features["lipid_class"],
            "carbon": candidates.get("carbon", pd.Series(np.nan, index=candidates.index)),
            "db": candidates.get("db", pd.Series(np.nan, index=candidates.index)),
        })
        for column, values in composition_features(
            composition, model["composition_priors"]
        ).items():
            features[column] = values
    return features


def _select_confident(
    pool: pd.DataFrame, features: pd.DataFrame, scores: pd.Series,
    settings: PostRegressionCompletionSettings,
) -> tuple[pd.DataFrame, int, int]:
    ranked = pool.assign(post_regression_confidence=scores).sort_values(
        ["feat_feature_id", "post_regression_confidence", "feat_intensity"],
        ascending=[True, False, False], kind="stable",
    )
    rank = ranked.groupby("feat_feature_id").cumcount()
    next_score = ranked.groupby("feat_feature_id")[
        "post_regression_confidence"
    ].shift(-1)
    confident = ranked["post_regression_confidence"].ge(settings.confidence_threshold)
    unambiguous = next_score.isna() | (
        ranked["post_regression_confidence"] - next_score
    ).ge(settings.ambiguity_margin)
    selected = ranked[confident & rank.eq(0) & unambiguous].copy()
    selected["_identity_key"] = features.loc[selected.index, "identity_key"]
    selected = selected.sort_values(
        ["_identity_key", "post_regression_confidence", "feat_intensity"],
        ascending=[True, False, False], kind="stable",
    ).drop_duplicates("_identity_key", keep="first")
    selected = selected.drop(columns="_identity_key")
    return selected, int(confident.sum()), int((confident & ~unambiguous).sum())


def _preserve_count(
    selected: pd.DataFrame, baseline: pd.DataFrame,
    pool: pd.DataFrame, features: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Reselect at the baseline count; never force a new subthreshold call."""
    target = len(baseline)
    selected = selected.sort_values(
        ["post_regression_confidence", "feat_intensity"],
        ascending=[False, False], kind="stable",
    ).head(target).copy()
    selected["post_regression_count_retained"] = False
    used_features = set(selected["feat_feature_id"].astype(str))
    used_identities = set(features.loc[selected.index, "identity_key"])
    fallback = pool.loc[baseline.index].sort_values(
        ["post_regression_confidence", "feat_intensity"],
        ascending=[False, False], kind="stable",
    )
    retained_indices = []
    for index, row in fallback.iterrows():
        if len(selected) + len(retained_indices) >= target:
            break
        feature = str(row["feat_feature_id"])
        identity = features.loc[index, "identity_key"]
        if feature in used_features or identity in used_identities:
            continue
        retained_indices.append(index)
        used_features.add(feature)
        used_identities.add(identity)
    if len(selected) + len(retained_indices) < target:
        # Conflicting assignments must not be duplicated to satisfy the count.
        restored = pool.loc[baseline.index].copy()
        restored["post_regression_count_retained"] = True
        return restored, len(restored)
    retained = pool.loc[retained_indices].copy()
    retained["post_regression_count_retained"] = True
    return pd.concat([selected, retained]), len(retained)


def _score_features(features: pd.DataFrame, model: Mapping) -> pd.Series:
    linear = np.full(len(features), float(model["intercept"]), dtype=float)
    for column, settings in model["numeric"].items():
        values = pd.to_numeric(features[column], errors="coerce").fillna(
            float(settings.get("impute", settings["mean"]))
        )
        linear += (
            (values.to_numpy(float) - float(settings["mean"]))
            / float(settings["scale"])
            * float(settings["coefficient"])
        )
    for column, coefficients in model["categorical"].items():
        linear += features[column].fillna("").astype(str).map(
            coefficients
        ).fillna(0.0).to_numpy(float)
    return pd.Series(
        [1.0 / (1.0 + exp(-float(np.clip(value, -60.0, 60.0)))) for value in linear],
        index=features.index,
        dtype=float,
    )


def _row_keys(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series("", index=frame.index, dtype=object)
    columns = [
        column for column in (
            "feat_feature_id", "name", "adduct", "matlab_code",
        ) if column in frame
    ]
    if not columns:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[columns].fillna("").astype(str).agg("\x1f".join, axis=1)


def _protected_rows(annotations: pd.DataFrame) -> pd.DataFrame:
    protected = annotations.get(
        "is_is", pd.Series(False, index=annotations.index)
    ).fillna(False).astype(bool)
    protected |= annotations.get(
        "internal_standard_name", pd.Series("", index=annotations.index)
    ).fillna("").astype(str).ne("")
    return annotations.loc[protected].copy()


def complete_post_regression_annotations(
    annotations: pd.DataFrame,
    all_pairs: pd.DataFrame,
    *,
    stream: str,
    settings: PostRegressionCompletionSettings = (
        PostRegressionCompletionSettings()
    ),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Promote calibrated, unambiguous fitted-grid candidates.

    The completion step never widens the MATLAB mass/RT windows, bypasses grid
    coherence, invents positional-isomer assignments, or reads a study's
    manually curated workbook. The bundled model only calibrates confidence
    inside the already strict candidate envelope.
    """
    candidates = all_pairs.copy()
    accepted = annotations.copy()
    base_audit = {
        "stream": str(stream),
        "enabled": bool(settings.enabled),
        "confidence_threshold": float(settings.confidence_threshold),
        "ambiguity_margin": float(settings.ambiguity_margin),
        "input_candidates": int(len(candidates)),
        "mapper_annotations": int(len(accepted)),
    }
    if not settings.enabled or candidates.empty:
        status = "disabled" if not settings.enabled else "no_candidates"
        return accepted, candidates, pd.DataFrame([{**base_audit, "status": status}])
    try:
        model = _load_model(settings.model_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return accepted, candidates, pd.DataFrame([{
            **base_audit,
            "status": "model_unavailable",
            "detail": str(error),
        }])

    phase = str(stream).split("_", 1)[0].upper()
    limits = model["candidate_limits"]
    isotopologue = candidates.get(
        "isotopologue", pd.Series("", index=candidates.index)
    ).fillna("").astype(str)
    coherence_pass = candidates.get(
        "grid_coherence_pass", pd.Series(True, index=candidates.index)
    ).fillna(True).astype(bool)
    eligible = (
        isotopologue.eq("")
        & pd.to_numeric(candidates["ppm_error"], errors="coerce").abs().lt(
            float(limits["ppm"])
        )
        & (
            pd.to_numeric(candidates["rt_error"], errors="coerce").abs() * 60.0
        ).le(float(limits["rt_seconds"]))
        & candidates["adduct"].astype(str).isin(matlab_annotation_adducts(phase))
    )
    if bool(limits.get("require_grid_coherence_pass", True)):
        eligible &= coherence_pass
    positional = candidates["class"].astype(str).isin(_POSITIONAL_CLASSES)
    positional_selected = candidates.get(
        "positional_isomer_selected", pd.Series(False, index=candidates.index)
    ).fillna(False).astype(bool)
    eligible &= ~positional | positional_selected

    pool = candidates.loc[eligible].copy()
    if pool.empty:
        return accepted, candidates, pd.DataFrame([{
            **base_audit, "status": "no_eligible_candidates",
            "eligible_candidates": 0, "promoted_annotations": 0,
        }])
    features = _candidate_features(pool, stream, model)
    pool["post_regression_confidence"] = _score_features(features, model)
    pool["post_regression_model"] = str(model["format"])
    pool["post_regression_eligible"] = True

    selected_rows, confidence_passing, ambiguous_count = _select_confident(
        pool, features, pool["post_regression_confidence"], settings,
    )
    reference_model = model.get("count_reference_model")
    protected = _protected_rows(accepted)
    count_target = len(selected_rows)
    count_retained = 0
    if reference_model:
        baseline_scores = _score_features(features, reference_model)
        baseline, _passing, _ambiguous = _select_confident(
            pool, features, baseline_scores, settings,
        )
        # Reserve declared standards before reselection. Counting them twice
        # when the new model ranks one lower would otherwise inflate totals.
        protected_keys = set(_row_keys(protected))
        baseline = baseline[~_row_keys(baseline).isin(protected_keys)]
        selected_rows = selected_rows[
            ~_row_keys(selected_rows).isin(protected_keys)
        ]
        if not protected.empty:
            selected_rows = selected_rows[
                ~selected_rows["feat_feature_id"].astype(str).isin(
                    protected["feat_feature_id"].astype(str)
                )
            ]
        count_target = len(baseline) + len(protected)
        selected_rows, count_retained = _preserve_count(
            selected_rows, baseline, pool, features,
        )
    else:
        selected_rows["post_regression_count_retained"] = False

    existing_keys = set(_row_keys(accepted))
    selected_keys = _row_keys(selected_rows)
    mapper_supported = selected_keys.isin(existing_keys)
    selected_rows["automatic_accepted"] = True
    selected_rows["automatic_candidate"] = True
    selected_rows["auto_annotation_status"] = np.where(
        mapper_supported,
        "matlab_mapper_confidence_supported",
        "curated_cohort_supported",
    )
    selected_rows.loc[
        selected_rows["post_regression_count_retained"], "auto_annotation_status"
    ] = "baseline_count_retained"
    selected_rows["status"] = selected_rows["auto_annotation_status"]
    selected_rows["rt_evidence_mode"] = "matlab_rtdb_spline_post_regression"
    selected_rows["standard_stream"] = str(stream)
    selected_rows["chromatography"] = phase

    # Declared spikes retain their protected identity even if their cohort
    # abundance falls outside the endogenous calibration model.
    if not protected.empty:
        protected = protected[
            ~_row_keys(protected).isin(set(_row_keys(selected_rows)))
        ]
        protected["post_regression_confidence"] = np.nan
        protected["post_regression_model"] = "protected_internal_standard"
        protected["post_regression_count_retained"] = False
    accepted = pd.concat(
        [selected_rows, protected], ignore_index=True, sort=False
    )
    accepted = accepted.drop_duplicates(
        [
            column for column in (
                "feat_feature_id", "name", "adduct", "matlab_code",
            ) if column in accepted
        ],
        keep="first",
    ).reset_index(drop=True)
    accepted["rank_for_feature"] = accepted.groupby("feat_feature_id").cumcount()

    candidates["post_regression_confidence"] = np.nan
    candidates["post_regression_model"] = ""
    candidates["post_regression_eligible"] = False
    candidates["post_regression_count_retained"] = False
    candidates.loc[pool.index, "post_regression_confidence"] = pool[
        "post_regression_confidence"
    ]
    candidates.loc[pool.index, "post_regression_model"] = pool[
        "post_regression_model"
    ]
    candidates.loc[pool.index, "post_regression_eligible"] = True
    selected_status = accepted.set_index(_row_keys(accepted))[
        "auto_annotation_status"
    ].to_dict()
    candidate_keys = _row_keys(candidates)
    selected = candidate_keys.isin(selected_status)
    candidates["automatic_accepted"] = selected
    candidates["automatic_candidate"] = selected
    candidates["auto_annotation_status"] = candidate_keys.map(
        selected_status
    ).fillna("review_only_blue_candidate")
    candidates.loc[
        candidate_keys.isin(set(_row_keys(accepted[
            accepted["post_regression_count_retained"].fillna(False).astype(bool)
        ]))), "post_regression_count_retained"
    ] = True

    audit = pd.DataFrame([{
        **base_audit,
        "status": "completed",
        "eligible_candidates": int(len(pool)),
        "confidence_passing": confidence_passing,
        "ambiguous_features_withheld": ambiguous_count,
        "baseline_count_target": count_target,
        "baseline_calls_retained": count_retained,
        "mapper_confidence_supported": int(mapper_supported.sum()),
        "mapper_annotations_withheld": int(
            max(len(annotations) - mapper_supported.sum() - len(protected), 0)
        ),
        "promoted_annotations": int((~mapper_supported).sum()),
        "final_annotations": int(len(accepted)),
        "model": str(model["format"]),
    }])
    return accepted, candidates, audit


__all__ = [
    "DEFAULT_CONFIDENCE_MODEL_PATH",
    "PostRegressionCompletionSettings",
    "complete_post_regression_annotations",
]
