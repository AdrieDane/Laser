"""Captured TG mapper propagation primitives outside the spline kernel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spline import (
    OneDimensionalSplineTerm,
    TwoDimensionalSplineTerm,
    predict_unique_composition_rt,
)


@dataclass(frozen=True)
class SecondaryAdductResult:
    selected: np.ndarray
    predicted_rt: np.ndarray
    adduct_offsets: np.ndarray


def _sort_rows(values: np.ndarray, columns: list[int] | None = None) -> np.ndarray:
    if columns is None:
        columns = list(range(values.shape[1]))
    return np.lexsort(tuple(values[:, column] for column in reversed(columns)))


def _iterative_absolute_regression(
    y: np.ndarray,
    design: np.ndarray,
    cutoff: float,
    initial: np.ndarray | None = None,
    iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Relevant ``iterative_regression`` branch with an absolute cutoff."""

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    design = np.asarray(design, dtype=np.float64)
    included = np.isfinite(y)
    previous = ~included
    iteration = 0
    use_initial = initial is not None
    repeat = initial is not None
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    while (iteration < iterations and not np.array_equal(included, previous)) or repeat:
        repeat = False
        previous = included.copy()
        if use_initial:
            coefficients = np.asarray(initial, dtype=np.float64).reshape(-1)
            use_initial = False
            repeat = True
        elif int(np.sum(included)) <= design.shape[1]:
            # MATLAB's iterative_regression.m only solves the unregularized
            # system when retained observations outnumber coefficients.
            coefficients = np.full(design.shape[1], np.nan, dtype=np.float64)
        else:
            coefficients = np.linalg.lstsq(design[included], y[included], rcond=None)[0]
        residual = y - design @ coefficients
        included = (np.abs(residual) <= cutoff) & np.isfinite(y)
        iteration += 1
    return coefficients, y - design @ coefficients


def propagate_secondary_adducts(
    *,
    current_candidates: np.ndarray,
    unique_compositions: np.ndarray,
    observation_to_composition_matlab: np.ndarray,
    weights: np.ndarray,
    observed_rt: np.ndarray,
    log_intensity: np.ndarray,
    primary_selected: np.ndarray,
    primary_predicted_rt: np.ndarray,
    primary_coefficients: np.ndarray,
    primary_sibling_offsets: np.ndarray | None = None,
    ranges: np.ndarray,
    y0: np.ndarray,
    term: OneDimensionalSplineTerm | TwoDimensionalSplineTerm,
    adduct_count: int,
    current_candidate_count: int,
) -> SecondaryAdductResult:
    """Port the TG secondary-adduct branch at MS_viewer.m:2955–3027."""

    candidates = np.asarray(current_candidates)
    compositions = np.asarray(unique_compositions, dtype=np.int64)
    mapping = np.asarray(observation_to_composition_matlab, dtype=np.int64).reshape(-1) - 1
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    observed_rt = np.asarray(observed_rt, dtype=np.float64).reshape(-1)
    intensity = np.asarray(log_intensity, dtype=np.float64).reshape(-1)
    primary_selected = np.asarray(primary_selected, dtype=bool).reshape(-1)
    primary_predicted_rt = np.asarray(primary_predicted_rt, dtype=np.float64).reshape(-1)
    primary_observations = compositions[mapping, 1] == 1
    if int(np.sum(primary_observations)) != len(primary_selected):
        raise ValueError("primary kernel output does not align with mapper observations")

    selected = np.zeros(len(mapping), dtype=bool)
    predicted = np.full(len(mapping), np.nan, dtype=np.float64)
    selected[primary_observations] = primary_selected
    predicted[primary_observations] = primary_predicted_rt

    primary_unique_by_family: dict[tuple[int, ...], int] = {}
    for unique_index, row in enumerate(compositions):
        if row[1] == 1:
            primary_unique_by_family[tuple(np.delete(row, 1))] = unique_index
    primary_for_unique = np.full(len(compositions), -1, dtype=np.int64)
    for unique_index, row in enumerate(compositions):
        primary_for_unique[unique_index] = primary_unique_by_family.get(
            tuple(np.delete(row, 1)), -1
        )

    selected_primary_observation = np.full(len(compositions), -1, dtype=np.int64)
    for observation in np.flatnonzero(selected):
        unique_index = mapping[observation]
        if selected_primary_observation[unique_index] != -1:
            raise ValueError("multiple selected primary observations map to one composition")
        selected_primary_observation[unique_index] = observation
    paired_primary = np.full(len(mapping), -1, dtype=np.int64)
    valid_unique = primary_for_unique[mapping] >= 0
    paired_primary[valid_unique] = selected_primary_observation[
        primary_for_unique[mapping[valid_unique]]
    ]
    secondary = np.flatnonzero(
        (paired_primary >= 0)
        & (np.abs(observed_rt - observed_rt[np.maximum(paired_primary, 0)]) < 20)
        & (compositions[mapping, 1] > 1)
        & (weights >= 0)
    )

    offsets = np.zeros(adduct_count, dtype=np.float64)
    kept = secondary.copy()
    for adduct in range(2, adduct_count + 1):
        positions = np.flatnonzero(compositions[mapping[kept], 1] == adduct)
        if not len(positions):
            continue
        observations = kept[positions]
        primary = paired_primary[observations]
        differences = observed_rt[observations] - observed_rt[primary]
        positive = weights[observations] > 0
        offset = float(np.mean(differences[positive])) if np.any(positive) else 0.0
        if len(positions) > 5:
            fitted, residual = _iterative_absolute_regression(
                differences,
                np.ones((len(differences), 1)),
                cutoff=20.0,
                initial=np.asarray([offset]),
            )
            residual = np.where(np.isnan(residual), np.inf, residual)
            offset = float(fitted[0])
            if np.isnan(offset):
                offset = 0.0
            remove = (np.abs(residual) > 15.0) & ~positive
            kept = np.delete(kept, positions[remove])
        else:
            remove = (np.abs(differences - offset) > 15.0) & ~positive
            kept = np.delete(kept, positions[remove])
        offsets[adduct - 1] = offset

        positions = np.flatnonzero(compositions[mapping[kept], 1] == adduct)
        observations = kept[positions]
        primary = paired_primary[observations]
        if len(positions) > 20:
            differences = intensity[observations] - intensity[primary]
            initial = np.asarray([np.quantile(differences, 0.5), 1.0])
            _, residual = _iterative_absolute_regression(
                intensity[observations],
                np.column_stack((np.ones(len(observations)), intensity[primary])),
                cutoff=1.0,
                initial=initial,
            )
            residual = np.where(np.isnan(residual), np.inf, residual)
            remove = (np.abs(residual) > 1.0) & (weights[observations] <= 0)
            kept = np.delete(kept, positions[remove])
        elif len(positions) >= 5:
            differences = intensity[observations] - intensity[primary]
            _, residual = _iterative_absolute_regression(
                differences, np.ones((len(differences), 1)), cutoff=1.0
            )
            residual = np.where(np.isnan(residual), np.inf, residual)
            remove = (np.abs(residual) > 1.0) & (weights[observations] <= 0)
            kept = np.delete(kept, positions[remove])

    optional = kept[weights[kept] <= 0]
    forced = np.flatnonzero(
        (weights[: len(mapping)] > 0)
        & ~selected[: len(mapping)]
        & (compositions[mapping, 1] > 1)
    )
    choose = np.concatenate((optional, forced))
    if len(choose):
        distance = np.abs(
            observed_rt[choose]
            - observed_rt[np.where(paired_primary[choose] >= 0, paired_primary[choose], 0)]
        )
        order = np.lexsort((distance, weights[choose] <= 0, mapping[choose]))
        choose = choose[order]
        choose = choose[np.r_[True, np.diff(mapping[choose]) != 0]]
        selected[choose] = True

    base_prediction = predict_unique_composition_rt(
        compositions, ranges, y0, term, primary_coefficients
    )
    sibling_offsets = np.asarray(
        [0.0] if primary_sibling_offsets is None else primary_sibling_offsets,
        dtype=float,
    ).reshape(-1)
    if len(sibling_offsets) > 1:
        offset_model = offsets[:, None] + sibling_offsets[None, :]
        offset_model[0, :] = 0.0
    else:
        offset_model = offsets
    selected_secondary = np.flatnonzero(selected & (compositions[mapping, 1] > 1))
    selected_adduct = compositions[mapping[selected_secondary], 1] - 1
    if np.asarray(offset_model).ndim == 2:
        selected_sibling = compositions[mapping[selected_secondary], 0] - 1
        selected_offsets = offset_model[selected_adduct, selected_sibling]
    else:
        selected_offsets = offset_model[selected_adduct]
    predicted[selected_secondary] = (
        base_prediction[mapping[selected_secondary]] + selected_offsets
    )

    current_count = min(int(current_candidate_count), len(candidates), len(selected))
    if current_count:
        order = np.lexsort((
            candidates[:current_count, 0],
            ~selected[:current_count],
            candidates[:current_count, 1],
        ))
        ordered_viewer = candidates[order, 1]
        duplicates = np.r_[False, np.diff(ordered_viewer) == 0]
        reject = order[duplicates & selected[order]]
        selected[reject] = False
    return SecondaryAdductResult(selected, predicted, offset_model)
