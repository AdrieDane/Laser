"""Model-independent assembly of MATLAB annotation regression stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidate_engine import (
    StageCandidateBatch,
    apply_interpolation_gate,
    build_regression_observations,
)
from .mapper import SecondaryAdductResult, propagate_secondary_adducts
from .spline import (
    OneDimensionalSplineTerm,
    TwoDimensionalSplineTerm,
    predict_compound_cluster_rt,
    predict_unique_composition_rt,
)


@dataclass(frozen=True)
class PrimaryStageFit:
    """Primary-adduct regression output aligned to the input candidate batch."""

    selected: np.ndarray
    predicted_rt_seconds: np.ndarray
    weights: np.ndarray
    coefficients: np.ndarray
    sibling_offsets: np.ndarray
    unique_compositions: np.ndarray
    observation_to_composition_matlab: np.ndarray


@dataclass(frozen=True)
class CompleteStageFit:
    """Primary and secondary selection aligned to the input candidate batch."""

    selected: np.ndarray
    predicted_rt_seconds: np.ndarray
    adduct_offsets: np.ndarray


@dataclass(frozen=True)
class StageHistory:
    """Selected observations carried into MATLAB's next regression stage."""

    codes: np.ndarray
    viewer_rows: np.ndarray
    theoretical_mz: np.ndarray
    observed_rt_seconds: np.ndarray
    predicted_rt_seconds: np.ndarray
    log_intensity: np.ndarray
    weights: np.ndarray
    compositions: np.ndarray


@dataclass(frozen=True)
class RegressionStageData:
    """Current sorted candidates followed by selected prior-stage history."""

    current_codes: np.ndarray
    current_viewer_rows: np.ndarray
    current_order: np.ndarray
    observation_codes: np.ndarray
    observation_viewer_rows: np.ndarray
    unique_compositions: np.ndarray
    observation_to_composition_matlab: np.ndarray
    weights: np.ndarray
    observed_rt_seconds: np.ndarray
    prior_predicted_rt_seconds: np.ndarray
    log_intensity: np.ndarray
    theoretical_mz: np.ndarray

    @property
    def current_candidate_count(self) -> int:
        return len(self.current_codes)


def empty_stage_history() -> StageHistory:
    """Return the empty state used before MATLAB stage 1."""
    return StageHistory(
        codes=np.empty(0, dtype=np.int64),
        viewer_rows=np.empty(0, dtype=np.int64),
        theoretical_mz=np.empty(0, dtype=float),
        observed_rt_seconds=np.empty(0, dtype=float),
        predicted_rt_seconds=np.empty(0, dtype=float),
        log_intensity=np.empty(0, dtype=float),
        weights=np.empty(0, dtype=float),
        compositions=np.empty((0, 6), dtype=np.int64),
    )


def stage_mass_offset(
    history: StageHistory,
    observed_mz_by_viewer: dict[int, float],
) -> float:
    """Return MATLAB's mean observed-minus-theoretical m/z correction."""
    if not len(history.viewer_rows):
        raise ValueError("mass offset requires selected prior-stage observations")
    observed = np.asarray([
        observed_mz_by_viewer[int(viewer)] for viewer in history.viewer_rows
    ], dtype=float)
    return float(np.mean(observed - history.theoretical_mz))


def assemble_regression_stage(
    batch: StageCandidateBatch,
    *,
    observed_rt_by_viewer: dict[int, float],
    log_intensity_by_viewer: dict[int, float],
    composition_bases: tuple[int, ...] | np.ndarray,
    sibling_compound_ids: np.ndarray,
    interpolation_enabled: bool,
    stage: int,
    stage_count: int,
    history: StageHistory | None = None,
) -> RegressionStageData:
    """Recreate MATLAB's current-candidate plus ``X1/w1/y1`` assembly."""
    previous = empty_stage_history() if history is None else history
    regression = build_regression_observations(
        batch.codes,
        batch.viewer_rows,
        batch.sibling_positions,
        composition_bases,
        sibling_compound_ids=sibling_compound_ids,
    )
    current_weights = apply_interpolation_gate(
        batch.weights,
        batch.codes,
        composition_bases,
        interpolation_enabled=interpolation_enabled,
        stage=stage,
        stage_count=stage_count,
    )[regression.order]
    current_rt = np.asarray([
        observed_rt_by_viewer[int(viewer)] for viewer in regression.viewer_rows
    ], dtype=float)
    current_intensity = np.asarray([
        log_intensity_by_viewer[int(viewer)] for viewer in regression.viewer_rows
    ], dtype=float)
    current_mz = np.asarray(batch.theoretical_mz, dtype=float)[regression.order]
    unique = np.vstack((regression.unique_compositions, previous.compositions))
    history_mapping = np.arange(
        len(regression.unique_compositions) + 1,
        len(regression.unique_compositions) + len(previous.compositions) + 1,
        dtype=np.int64,
    )
    mapping = np.concatenate((
        regression.observation_to_composition_matlab, history_mapping
    ))
    return RegressionStageData(
        current_codes=regression.codes,
        current_viewer_rows=regression.viewer_rows,
        current_order=regression.order,
        observation_codes=np.concatenate((regression.codes, previous.codes)),
        observation_viewer_rows=np.concatenate((
            regression.viewer_rows, previous.viewer_rows
        )),
        unique_compositions=unique,
        observation_to_composition_matlab=mapping,
        weights=np.concatenate((current_weights, previous.weights)),
        observed_rt_seconds=np.concatenate((current_rt, previous.observed_rt_seconds)),
        prior_predicted_rt_seconds=np.concatenate((
            np.full(len(current_rt), np.nan), previous.predicted_rt_seconds
        )),
        log_intensity=np.concatenate((current_intensity, previous.log_intensity)),
        theoretical_mz=np.concatenate((current_mz, previous.theoretical_mz)),
    )


def fit_regression_stage_primary(
    data: RegressionStageData,
    *,
    stage: int,
    ranges: np.ndarray,
    thresholds_seconds: np.ndarray,
    rt_preprocessing: np.ndarray,
    term: OneDimensionalSplineTerm | TwoDimensionalSplineTerm,
) -> PrimaryStageFit:
    """Fit the primary adduct for a fully assembled MATLAB stage."""
    primary_unique = data.unique_compositions[:, 1] == 1
    primary_observation = primary_unique[
        data.observation_to_composition_matlab - 1
    ]
    unique_primary = data.unique_compositions[primary_unique]
    source_mapping = data.observation_to_composition_matlab[primary_observation]
    primary_mapping = np.cumsum(
        np.r_[True, np.diff(source_mapping) != 0]
    ).astype(np.int64)
    keep, predicted, coefficients, offsets = predict_compound_cluster_rt(
        y=data.observed_rt_seconds[primary_observation],
        x_in=unique_primary,
        identity=np.where(
            np.arange(len(data.weights)) < data.current_candidate_count,
            data.observation_viewer_rows,
            0,
        )[primary_observation],
        design_rows_matlab=primary_mapping,
        weights_in=data.weights[primary_observation],
        ranges=np.asarray(ranges),
        thresholds=np.asarray(thresholds_seconds),
        stage=stage,
        y0=np.asarray(rt_preprocessing),
        term=term,
    )
    selected = np.zeros(len(data.weights), dtype=bool)
    predicted_aligned = np.full(len(data.weights), np.nan, dtype=float)
    selected[primary_observation] = keep
    predicted_aligned[primary_observation] = predicted
    return PrimaryStageFit(
        selected=selected,
        predicted_rt_seconds=predicted_aligned,
        weights=data.weights,
        coefficients=coefficients,
        sibling_offsets=offsets,
        unique_compositions=data.unique_compositions,
        observation_to_composition_matlab=data.observation_to_composition_matlab,
    )


def propagate_regression_stage_secondary(
    data: RegressionStageData,
    primary: PrimaryStageFit,
    *,
    composition_bases: tuple[int, ...] | np.ndarray,
    ranges: np.ndarray,
    rt_preprocessing: np.ndarray,
    term: OneDimensionalSplineTerm | TwoDimensionalSplineTerm,
) -> CompleteStageFit:
    """Propagate secondary adducts across current and historical observations."""
    primary_unique = data.unique_compositions[:, 1] == 1
    primary_observation = primary_unique[
        data.observation_to_composition_matlab - 1
    ]
    result = propagate_secondary_adducts(
        current_candidates=np.column_stack((
            data.current_codes, data.current_viewer_rows
        )),
        unique_compositions=data.unique_compositions,
        observation_to_composition_matlab=data.observation_to_composition_matlab,
        weights=data.weights,
        observed_rt=data.observed_rt_seconds,
        log_intensity=data.log_intensity,
        primary_selected=primary.selected[primary_observation],
        primary_predicted_rt=primary.predicted_rt_seconds[primary_observation],
        primary_coefficients=primary.coefficients,
        primary_sibling_offsets=primary.sibling_offsets,
        ranges=np.asarray(ranges),
        y0=np.asarray(rt_preprocessing),
        term=term,
        adduct_count=int(np.asarray(composition_bases).reshape(-1)[0]),
        current_candidate_count=data.current_candidate_count,
    )
    return CompleteStageFit(result.selected, result.predicted_rt, result.adduct_offsets)


def advance_stage_history(
    data: RegressionStageData,
    complete: CompleteStageFit,
) -> StageHistory:
    """Replace MATLAB's ``X1/w1/y1/z1/m1/ii1`` with stage survivors."""
    selected = np.asarray(complete.selected, dtype=bool)
    if selected.shape != data.weights.shape:
        raise ValueError("complete stage selection does not align with observations")
    return StageHistory(
        codes=data.observation_codes[selected],
        viewer_rows=data.observation_viewer_rows[selected],
        theoretical_mz=data.theoretical_mz[selected],
        observed_rt_seconds=data.observed_rt_seconds[selected],
        predicted_rt_seconds=complete.predicted_rt_seconds[selected],
        log_intensity=data.log_intensity[selected],
        weights=(data.weights[selected] + (data.weights[selected] == 0)).astype(float),
        compositions=data.unique_compositions[
            data.observation_to_composition_matlab[selected] - 1
        ],
    )


def select_final_natural_isotopes(
    data: RegressionStageData,
    history: StageHistory,
    *,
    natural_isotope_abundance: np.ndarray,
    composition_bases: tuple[int, ...] | np.ndarray,
    primary_coefficients: np.ndarray,
    adduct_offsets: np.ndarray,
    ranges: np.ndarray,
    rt_preprocessing: np.ndarray,
    term: OneDimensionalSplineTerm | TwoDimensionalSplineTerm,
) -> CompleteStageFit:
    """Port MATLAB's final-stage isotope abundance, RT, and prune branch."""
    bases = np.asarray(composition_bases, dtype=np.int64).reshape(-1)
    abundance = np.asarray(natural_isotope_abundance, dtype=float).reshape(-1)
    if bases.shape != (5,) or len(abundance) < bases[2]:
        raise ValueError("natural-isotope model does not cover the composition radix")
    if np.any(abundance[: bases[2]] <= 0):
        raise ValueError("natural-isotope abundances must be positive")
    current_count = data.current_candidate_count
    current_compositions = data.unique_compositions[
        data.observation_to_composition_matlab[:current_count] - 1
    ]
    parent_rt: dict[tuple[int, ...], float] = {}
    parent_intensity: dict[tuple[int, ...], float] = {}
    for composition, rt, intensity in zip(
        history.compositions,
        history.observed_rt_seconds,
        history.log_intensity,
        strict=True,
    ):
        parent = np.asarray(composition, dtype=np.int64).copy()
        parent[3] = 1
        if int(composition[3]) == 1:
            key = tuple(parent.tolist())
            parent_rt[key] = float(rt)
            parent_intensity[key] = float(intensity)

    isotope = current_compositions[:, 3].astype(np.int64) - 1
    log_ratio = np.log10(abundance[: bases[2]])
    log_ratio -= log_ratio[0]
    status = np.zeros(current_count, dtype=np.int8)
    for index, composition in enumerate(current_compositions):
        parent = composition.copy()
        parent[3] = 1
        key = tuple(parent.tolist())
        if key in parent_rt:
            delta = (
                parent_intensity[key]
                + log_ratio[isotope[index]]
                - data.log_intensity[index]
            )
            if (
                -0.4 <= delta < 0.8
                and abs(parent_rt[key] - data.observed_rt_seconds[index]) < 2.0
                and data.weights[index] != -1
            ):
                status[index] = 1
        if data.weights[index] > 0:
            status[index] = 2

    order = np.lexsort((np.arange(current_count), isotope, -status))
    eligible = order[status[order] > 0]
    seen_composition: set[tuple[int, int]] = set()
    seen_viewer: set[tuple[int, int]] = set()
    selected_current = np.zeros(current_count, dtype=bool)
    for index in eligible:
        sibling = int(current_compositions[index, 0])
        composition_key = (sibling, int(data.current_codes[index]))
        viewer_key = (sibling, int(data.current_viewer_rows[index]))
        if composition_key in seen_composition or viewer_key in seen_viewer:
            continue
        selected_current[index] = True
        seen_composition.add(composition_key)
        seen_viewer.add(viewer_key)

    base_prediction = predict_unique_composition_rt(
        data.unique_compositions,
        np.asarray(ranges),
        np.asarray(rt_preprocessing),
        term,
        primary_coefficients,
    )
    predicted_current = base_prediction[
        data.observation_to_composition_matlab[:current_count] - 1
    ]
    adduct = current_compositions[:, 1].astype(np.int64) - 1
    offsets = np.asarray(adduct_offsets, dtype=float)
    if offsets.ndim == 2:
        sibling = current_compositions[:, 0].astype(np.int64) - 1
        predicted_current += offsets[adduct, sibling]
    else:
        predicted_current += offsets[adduct]
    selected = np.concatenate((
        selected_current,
        np.ones(len(history.weights), dtype=bool),
    ))
    predicted = np.concatenate((
        predicted_current,
        history.predicted_rt_seconds,
    ))
    return CompleteStageFit(selected, predicted, np.asarray(adduct_offsets, dtype=float))


def fit_primary_stage(
    batch: StageCandidateBatch,
    *,
    observed_rt_by_viewer: dict[int, float],
    composition_bases: tuple[int, ...] | np.ndarray,
    sibling_compound_ids: np.ndarray,
    interpolation_enabled: bool,
    stage: int,
    stage_count: int,
    ranges: np.ndarray,
    thresholds_seconds: np.ndarray,
    rt_preprocessing: np.ndarray,
    term: OneDimensionalSplineTerm | TwoDimensionalSplineTerm,
) -> PrimaryStageFit:
    """Run the shared primary-adduct regression portion of one MATLAB stage."""
    gated_weights = apply_interpolation_gate(
        batch.weights,
        batch.codes,
        composition_bases,
        interpolation_enabled=interpolation_enabled,
        stage=stage,
        stage_count=stage_count,
    )
    regression = build_regression_observations(
        batch.codes,
        batch.viewer_rows,
        batch.sibling_positions,
        composition_bases,
        sibling_compound_ids=sibling_compound_ids,
    )
    sorted_weights = gated_weights[regression.order]
    sorted_rt = np.asarray([
        observed_rt_by_viewer[int(viewer)] for viewer in regression.viewer_rows
    ], dtype=float)
    primary_unique = regression.unique_compositions[:, 1] == 1
    primary_observation = primary_unique[
        regression.observation_to_composition_matlab - 1
    ]
    unique_primary = regression.unique_compositions[primary_unique]
    source_mapping = regression.observation_to_composition_matlab[primary_observation]
    primary_mapping = np.cumsum(
        np.r_[True, np.diff(source_mapping) != 0]
    ).astype(np.int64)
    keep, predicted, coefficients, offsets = predict_compound_cluster_rt(
        y=sorted_rt[primary_observation],
        x_in=unique_primary,
        identity=regression.viewer_rows[primary_observation],
        design_rows_matlab=primary_mapping,
        weights_in=sorted_weights[primary_observation],
        ranges=np.asarray(ranges),
        thresholds=np.asarray(thresholds_seconds),
        stage=stage,
        y0=np.asarray(rt_preprocessing),
        term=term,
    )
    selected = np.zeros(len(batch.codes), dtype=bool)
    predicted_aligned = np.full(len(batch.codes), np.nan, dtype=float)
    sorted_primary_positions = np.flatnonzero(primary_observation)
    original_positions = regression.order[sorted_primary_positions]
    selected[original_positions] = keep
    predicted_aligned[original_positions] = predicted
    return PrimaryStageFit(
        selected=selected,
        predicted_rt_seconds=predicted_aligned,
        weights=gated_weights,
        coefficients=coefficients,
        sibling_offsets=offsets,
        unique_compositions=regression.unique_compositions,
        observation_to_composition_matlab=(
            regression.observation_to_composition_matlab
        ),
    )


def propagate_stage_secondary_adducts(
    batch: StageCandidateBatch,
    primary: PrimaryStageFit,
    *,
    observed_rt_by_viewer: dict[int, float],
    log_intensity_by_viewer: dict[int, float],
    composition_bases: tuple[int, ...] | np.ndarray,
    sibling_compound_ids: np.ndarray,
    ranges: np.ndarray,
    rt_preprocessing: np.ndarray,
    term: OneDimensionalSplineTerm | TwoDimensionalSplineTerm,
) -> CompleteStageFit:
    """Run MATLAB's secondary-adduct propagation for the current stage rows."""
    regression = build_regression_observations(
        batch.codes,
        batch.viewer_rows,
        batch.sibling_positions,
        composition_bases,
        sibling_compound_ids=sibling_compound_ids,
    )
    sorted_weights = primary.weights[regression.order]
    sorted_rt = np.asarray([
        observed_rt_by_viewer[int(viewer)] for viewer in regression.viewer_rows
    ], dtype=float)
    sorted_intensity = np.asarray([
        log_intensity_by_viewer[int(viewer)] for viewer in regression.viewer_rows
    ], dtype=float)
    primary_unique = regression.unique_compositions[:, 1] == 1
    primary_observation = primary_unique[
        regression.observation_to_composition_matlab - 1
    ]
    primary_selected = primary.selected[regression.order][primary_observation]
    primary_predicted = primary.predicted_rt_seconds[regression.order][
        primary_observation
    ]
    result: SecondaryAdductResult = propagate_secondary_adducts(
        current_candidates=np.column_stack((regression.codes, regression.viewer_rows)),
        unique_compositions=regression.unique_compositions,
        observation_to_composition_matlab=(
            regression.observation_to_composition_matlab
        ),
        weights=sorted_weights,
        observed_rt=sorted_rt,
        log_intensity=sorted_intensity,
        primary_selected=primary_selected,
        primary_predicted_rt=primary_predicted,
        primary_coefficients=primary.coefficients,
        primary_sibling_offsets=primary.sibling_offsets,
        ranges=np.asarray(ranges),
        y0=np.asarray(rt_preprocessing),
        term=term,
        adduct_count=int(np.asarray(composition_bases).reshape(-1)[0]),
        current_candidate_count=len(batch.codes),
    )
    selected = np.zeros(len(batch.codes), dtype=bool)
    predicted = np.full(len(batch.codes), np.nan, dtype=float)
    selected[regression.order] = result.selected[: len(batch.codes)]
    predicted[regression.order] = result.predicted_rt[: len(batch.codes)]
    return CompleteStageFit(selected, predicted, result.adduct_offsets)
