"""Preparation of reference-grid and candidate frames for standard review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from annotator.grid_coherence import GRID_COHERENCE_COLUMNS


@dataclass(frozen=True)
class StandardGridDataServices:
    """Identity and policy helpers needed by grid-data preparation."""

    empty_candidate_columns: Sequence[str]
    grid_class: Callable
    annotation_identity: Callable
    normalise_benchmark_species: Callable
    benchmark_identity_series: Callable
    accepted_annotation_keys: Callable
    status_lookup: Callable
    annotation_column_lookup: Callable



def _candidate_frame(
    stream_result: dict, services: StandardGridDataServices,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations = stream_result.get("annotations", pd.DataFrame())
    all_pairs = stream_result.get("all_pairs")
    candidates = (
        all_pairs.copy()
        if all_pairs is not None and not all_pairs.empty
        else annotations.copy()
        if annotations is not None and not annotations.empty
        else pd.DataFrame(columns=services.empty_candidate_columns)
    )
    return annotations, candidates


def _prepare_reference(
    reference: pd.DataFrame, models: pd.DataFrame, services: StandardGridDataServices,
) -> pd.DataFrame:
    reference["grid_class"] = services.grid_class(reference)
    reference["grid_rt"] = pd.to_numeric(reference["ref_rt"], errors="coerce")
    if models is None or models.empty:
        reference["grid_model_validated"] = False
        reference["grid_model_type"] = "historical_reference"
        return reference
    model_lookup = (
        models.drop_duplicates(["class", "adduct"], keep="first")
        .set_index(["class", "adduct"])
    )
    keys = pd.MultiIndex.from_frame(reference[["class", "adduct"]].astype(str))
    aligned = model_lookup.reindex(keys)
    offsets = pd.to_numeric(aligned["offset_min"], errors="coerce").fillna(0.0)
    validated = aligned["validated"].fillna(False).astype(bool)
    reference["grid_rt"] += np.where(
        validated.to_numpy(), offsets.to_numpy(), 0.0
    )
    reference["grid_model_validated"] = validated.to_numpy()
    reference["grid_model_type"] = aligned["model_type"].fillna(
        "historical_reference"
    ).to_numpy()
    return reference


def _anchor_keys(
    stream_result: dict, services: StandardGridDataServices,
) -> set[tuple[str, str, str]]:
    anchors = stream_result.get("automatic_anchors", pd.DataFrame())
    required = {"feat_feature_id", "species", "adduct"}
    if not isinstance(anchors, pd.DataFrame) or anchors.empty:
        return set()
    if not required.issubset(anchors.columns):
        return set()
    return {
        (
            str(feature_id),
            services.normalise_benchmark_species(species),
            str(adduct).strip(),
        )
        for feature_id, species, adduct in anchors[
            ["feat_feature_id", "species", "adduct"]
        ].itertuples(index=False, name=None)
    }


def _prepare_candidate_identity(
    candidates: pd.DataFrame, annotations: pd.DataFrame,
    stream_result: dict, services: StandardGridDataServices,
) -> None:
    candidates["grid_class"] = services.grid_class(candidates)
    numeric = (
        "feat_rt", "feat_mz", "ref_mz", "ref_rt", "rt_expected",
        "positional_isomer_rt_expected", "positional_isomer_rt_error_seconds",
        "ppm_error", "carbon", "db",
    )
    for column in numeric:
        if column in candidates:
            candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    candidates["key"] = services.annotation_identity(candidates)
    candidates["deletion_key"] = (
        candidates["grid_class"].astype(str) + "|"
        + candidates["adduct"].astype(str) + "|"
        + candidates["name"].astype(str)
    )
    anchor_keys = _anchor_keys(stream_result, services)
    candidates["automatic_rt_anchor"] = [
        (
            str(feature_id),
            services.normalise_benchmark_species(name),
            str(adduct).strip(),
        ) in anchor_keys
        for feature_id, name, adduct in candidates[
            ["feat_feature_id", "name", "adduct"]
        ].itertuples(index=False, name=None)
    ]
    benchmark = stream_result.get("benchmark_status_by_identity", {})
    if isinstance(benchmark, dict) and benchmark:
        candidates["benchmark_identity_key"] = services.benchmark_identity_series(
            candidates
        )
        candidates["benchmark_status"] = candidates["benchmark_identity_key"].map(
            benchmark
        ).fillna("")
    candidates["automatic_candidate"] = candidates["key"].isin(
        services.accepted_annotation_keys(annotations)
    )
    candidates["auto_annotation_status"] = candidates["key"].map(
        services.status_lookup(annotations)
    ).fillna("not_reviewed_by_the_run")


def _prepare_coherence(
    candidates: pd.DataFrame, annotations: pd.DataFrame,
    services: StandardGridDataServices,
) -> None:
    for column in (*GRID_COHERENCE_COLUMNS, "grid_coherence_threshold"):
        mapped = candidates["key"].map(
            services.annotation_column_lookup(annotations, column)
        )
        if column in candidates:
            candidates[column] = candidates[column].where(
                candidates[column].notna(), mapped
            )
        else:
            candidates[column] = mapped
    for column in ("grid_coherence_neighbors", "grid_coherence_dimensions"):
        candidates[column] = pd.to_numeric(
            candidates[column], errors="coerce"
        ).fillna(0).astype(int)
    for column in ("grid_coherence_assessed", "grid_coherence_outlier"):
        candidates[column] = candidates[column].fillna(False).astype(bool)
    candidates["grid_coherence_pass"] = (
        candidates["grid_coherence_pass"].fillna(True).astype(bool)
    )
    if "np_rt_identity" not in candidates:
        candidates["np_rt_identity"] = pd.NA


def prepare_standard_grid_data(
    stream_result: dict, *, services: StandardGridDataServices,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the shifted reference grid and complete m/z candidate superset."""
    reference = stream_result.get("reference", pd.DataFrame()).copy()
    annotations, candidates = _candidate_frame(stream_result, services)
    if reference.empty:
        return reference, candidates
    reference = _prepare_reference(
        reference, stream_result.get("rt_models", pd.DataFrame()), services
    )
    if candidates.empty:
        return reference, candidates
    _prepare_candidate_identity(candidates, annotations, stream_result, services)
    _prepare_coherence(candidates, annotations, services)
    return reference, candidates
