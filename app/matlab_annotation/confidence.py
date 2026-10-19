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
    if payload.get("format") != "laser-post-regression-confidence-v1":
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


def _candidate_features(candidates: pd.DataFrame, stream: str) -> pd.DataFrame:
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
    return features


def _score_features(features: pd.DataFrame, model: Mapping) -> pd.Series:
    linear = np.full(len(features), float(model["intercept"]), dtype=float)
    for column, settings in model["numeric"].items():
        values = pd.to_numeric(features[column], errors="coerce").fillna(
            float(settings["mean"])
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
    columns = [
        column for column in (
            "feat_feature_id", "name", "adduct", "matlab_code",
        ) if column in frame
    ]
    if not columns:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[columns].fillna("").astype(str).agg("\x1f".join, axis=1)


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
    features = _candidate_features(pool, stream)
    pool["post_regression_confidence"] = _score_features(features, model)
    pool["post_regression_model"] = str(model["format"])
    pool["post_regression_eligible"] = True

    pool = pool.sort_values(
        ["feat_feature_id", "post_regression_confidence", "feat_intensity"],
        ascending=[True, False, False], kind="stable",
    )
    pool["_feature_rank"] = pool.groupby("feat_feature_id").cumcount()
    pool["_next_confidence"] = pool.groupby("feat_feature_id")[
        "post_regression_confidence"
    ].shift(-1)
    confident = pool["post_regression_confidence"].ge(
        float(settings.confidence_threshold)
    )
    unambiguous = pool["_next_confidence"].isna() | (
        pool["post_regression_confidence"] - pool["_next_confidence"]
    ).ge(float(settings.ambiguity_margin))
    selected_rows = pool[
        confident & pool["_feature_rank"].eq(0) & unambiguous
    ].copy()
    selected_rows["_identity_key"] = features.loc[
        selected_rows.index, "identity_key"
    ]
    selected_rows = selected_rows.sort_values(
        ["_identity_key", "post_regression_confidence", "feat_intensity"],
        ascending=[True, False, False], kind="stable",
    ).drop_duplicates("_identity_key", keep="first")

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
    selected_rows["status"] = selected_rows["auto_annotation_status"]
    selected_rows["rt_evidence_mode"] = "matlab_rtdb_spline_post_regression"
    selected_rows["standard_stream"] = str(stream)
    selected_rows["chromatography"] = phase
    selected_rows = selected_rows.drop(
        columns=["_feature_rank", "_next_confidence", "_identity_key"],
        errors="ignore",
    )

    # Declared spikes retain their protected identity even if their cohort
    # abundance falls outside the endogenous calibration model.
    protected = accepted.iloc[0:0].copy()
    if not accepted.empty:
        protected_mask = accepted.get(
            "is_is", pd.Series(False, index=accepted.index)
        ).fillna(False).astype(bool)
        protected_mask |= accepted.get(
            "internal_standard_name", pd.Series("", index=accepted.index)
        ).fillna("").astype(str).ne("")
        protected = accepted.loc[protected_mask].copy()
        protected = protected[
            ~_row_keys(protected).isin(set(_row_keys(selected_rows)))
        ]
        protected["post_regression_confidence"] = np.nan
        protected["post_regression_model"] = "protected_internal_standard"
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
    candidates.loc[selected, "auto_annotation_status"] = candidate_keys[
        selected
    ].map(selected_status)

    audit = pd.DataFrame([{
        **base_audit,
        "status": "completed",
        "eligible_candidates": int(len(pool)),
        "confidence_passing": int(confident.sum()),
        "ambiguous_features_withheld": int((confident & ~unambiguous).sum()),
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
