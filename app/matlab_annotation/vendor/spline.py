"""MATLAB-equivalent spline/refit primitives for annotation review.

The initial contract is intentionally narrow: it implements the exact
two-dimensional interaction model exercised by ProjectData2 RP/TG.  The API is
kept independent of Qt so numerical parity can be established before edits are
wired into the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class TwoDimensionalSplineTerm:
    """Precalculated MATLAB spline term used by ``get_design_matrix``.

    ``dimensions_matlab`` refers to columns of the matrix passed to MATLAB's
    ``get_design_matrix`` and therefore remains one-based.
    """

    dimensions_matlab: tuple[int, int]
    smoothing: tuple[float, float]
    basis: tuple[np.ndarray, np.ndarray]
    penalty_0: tuple[np.ndarray, np.ndarray]
    penalty_1: tuple[np.ndarray, np.ndarray]
    penalty_2: tuple[np.ndarray, np.ndarray]
    strength: float = 1.0


@dataclass(frozen=True)
class OneDimensionalSplineTerm:
    """Precalculated one-variable spline term used by ``get_design_matrix``."""

    dimension_matlab: int
    smoothing: float
    basis: np.ndarray
    penalty: np.ndarray
    strength: float = 1.0


def _open_uniform_bspline_basis(
    values: np.ndarray, knot_count: int, degree: int
) -> np.ndarray:
    """Evaluate the open-uniform basis constructed by MATLAB B_spline_basis."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or knot_count < 1 or degree < 0:
        raise ValueError("invalid B-spline basis request")
    knots = np.linspace(values.min(), values.max(), knot_count + 1)
    augmented = np.concatenate((
        np.repeat(knots[0], degree), knots, np.repeat(knots[-1], degree)
    ))
    return BSpline.design_matrix(
        values, augmented, degree, extrapolate=False
    ).toarray()


def _precalculate_spline_components(
    size: int, smoothing: float, resolution: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = (size - 1) / smoothing
    k = int(np.floor(raw + 0.5)) - 2
    knot_count = max(k, 1)
    degree = 3 + k - knot_count
    grid = np.arange((size - 1) * resolution + 1, dtype=np.float64) / resolution
    dense_basis = _open_uniform_bspline_basis(grid, knot_count, degree)
    basis = dense_basis[::resolution]
    penalty_0 = dense_basis.T @ dense_basis / resolution
    first = np.diff(dense_basis, axis=0) * resolution
    penalty_1 = first.T @ first / resolution
    second = np.diff(first, axis=0) * resolution
    penalty_2 = second.T @ second / resolution
    return basis, penalty_0, penalty_1, penalty_2


def precalculate_one_dimensional_spline(
    size: int,
    smoothing: float,
    *,
    dimension_matlab: int = 1,
    integration_steps_per_unit: int = 1000,
) -> OneDimensionalSplineTerm:
    """Port the active 1D branch of MATLAB ``precalculate_B_splines``."""
    if size < 1 or smoothing <= 0 or integration_steps_per_unit < 1:
        raise ValueError("spline size, smoothing and integration resolution must be positive")
    resolution = int(integration_steps_per_unit)
    basis, _, _, penalty = _precalculate_spline_components(
        size, smoothing, resolution
    )
    if dimension_matlab < 1:
        raise ValueError("MATLAB spline dimensions are one-based")
    return OneDimensionalSplineTerm(
        int(dimension_matlab), float(smoothing), basis, penalty
    )


def precalculate_two_dimensional_spline(
    sizes: tuple[int, int],
    smoothing: tuple[float, float],
    *,
    dimensions_matlab: tuple[int, int],
    integration_steps_per_unit: int = 1000,
) -> TwoDimensionalSplineTerm:
    """Port MATLAB's precalculation for an active two-variable interaction."""
    if any(size < 1 for size in sizes) or any(value <= 0 for value in smoothing):
        raise ValueError("spline sizes and smoothing values must be positive")
    if any(value < 1 for value in dimensions_matlab):
        raise ValueError("MATLAB spline dimensions are one-based")
    resolution = int(integration_steps_per_unit)
    if resolution < 1:
        raise ValueError("integration resolution must be positive")
    components = tuple(
        _precalculate_spline_components(size, strength, resolution)
        for size, strength in zip(sizes, smoothing, strict=True)
    )
    return TwoDimensionalSplineTerm(
        dimensions_matlab=dimensions_matlab,
        smoothing=smoothing,
        basis=(components[0][0], components[1][0]),
        penalty_0=(components[0][1], components[1][1]),
        penalty_1=(components[0][2], components[1][2]),
        penalty_2=(components[0][3], components[1][3]),
    )


def build_one_dimensional_design(
    x_in: np.ndarray, term: OneDimensionalSplineTerm
) -> tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix]:
    """Port a one-variable continuous term from MATLAB ``get_design_matrix``."""
    values = np.asarray(x_in)
    if values.ndim != 2:
        raise ValueError("design input must be a matrix")
    column = term.dimension_matlab - 1
    if column < 0 or column >= values.shape[1]:
        raise ValueError("spline dimension lies outside the design input")
    rows = values[:, column].astype(np.int64) - 1
    basis = np.asarray(term.basis, dtype=np.float64)
    if np.any(rows < 0) or np.any(rows >= basis.shape[0]):
        raise ValueError("spline coordinate lies outside its basis")
    design = sparse.csr_matrix(basis[rows])
    groups = np.ones(design.shape[1], dtype=np.int64)
    return design, groups, sparse.csr_matrix(term.penalty, dtype=np.float64)


def build_two_dimensional_design(
    x_in: np.ndarray, term: TwoDimensionalSplineTerm
) -> tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix]:
    """Port the two-variable branch of MATLAB ``get_design_matrix``."""

    x_in = np.asarray(x_in)
    d0, d1 = (dimension - 1 for dimension in term.dimensions_matlab)
    row0 = x_in[:, d0].astype(np.int64) - 1
    row1 = x_in[:, d1].astype(np.int64) - 1
    basis0 = np.asarray(term.basis[0], dtype=np.float64)
    basis1 = np.asarray(term.basis[1], dtype=np.float64)
    if np.any(row0 < 0) or np.any(row0 >= basis0.shape[0]):
        raise ValueError("first spline coordinate lies outside its basis")
    if np.any(row1 < 0) or np.any(row1 >= basis1.shape[0]):
        raise ValueError("second spline coordinate lies outside its basis")

    n0, n1 = basis0.shape[1], basis1.shape[1]
    expanded0 = np.tile(basis0, (1, n1))
    expanded1 = np.repeat(basis1, n0, axis=1)
    design = expanded0[row0, :] * expanded1[row1, :]

    map0 = np.tile(np.arange(n0), n1)
    map1 = np.repeat(np.arange(n1), n0)

    def expand(matrix: np.ndarray, mapping: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        return matrix[np.ix_(mapping, mapping)]

    penalty = (
        expand(term.penalty_0[0], map0) * expand(term.penalty_2[1], map1)
        + 2.0
        * expand(term.penalty_1[0], map0)
        * expand(term.penalty_1[1], map1)
        + expand(term.penalty_2[0], map0) * expand(term.penalty_0[1], map1)
    )
    # A single interaction term gives every design column the same group id.
    groups = np.ones(design.shape[1], dtype=np.int64)
    return sparse.csr_matrix(design), groups, sparse.csr_matrix(penalty)


def weighted_penalized_regression(
    y: np.ndarray,
    design: sparse.spmatrix,
    weights_in: np.ndarray,
    penalty_in: sparse.spmatrix,
    smoothing: np.ndarray,
) -> np.ndarray:
    """Port MATLAB ``weighted_penalized_regression``."""

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    weights_in = np.asarray(weights_in, dtype=np.float64).reshape(-1)
    smoothing = np.asarray(smoothing, dtype=np.float64).reshape(-1)
    design = sparse.csr_matrix(design, dtype=np.float64)
    penalty_in = sparse.csr_matrix(penalty_in, dtype=np.float64)

    scale_true = 100.0 / max(int(np.sum(weights_in == 1)), 1)
    scale_zero = 1.0 / max(int(np.sum(weights_in == 0)), 1)
    weights = (
        weights_in * (scale_true - scale_zero) + scale_zero
    ) * (weights_in >= 0)
    weighted_design = design.multiply(weights[:, None])
    smoothing_diagonal = sparse.diags(smoothing)
    normal = (
        design.T @ weighted_design
        + smoothing_diagonal @ penalty_in @ smoothing_diagonal
        + sparse.diags(smoothing * smoothing * 1e-8)
    )
    normal = (normal + normal.T) * 0.5
    rhs = np.asarray(design.T @ (weights * y)).reshape(-1)
    return np.asarray(spsolve(normal.tocsc(), rhs)).reshape(-1)


def _sort_rows(values: np.ndarray, columns: list[int]) -> np.ndarray:
    keys = tuple(values[:, column] for column in reversed(columns))
    return np.lexsort(keys)


def _linearized_y0(x_in: np.ndarray, ranges: np.ndarray, y0: np.ndarray) -> np.ndarray:
    if np.asarray(y0).shape[0] <= 1:
        return np.zeros(x_in.shape[0], dtype=np.float64)
    flat = np.asarray(y0, dtype=np.float64).reshape(-1, order="F")
    coordinates = x_in[:, [0, 1, 2, *range(4, x_in.shape[1])]].astype(np.int64) - 1
    dimensions = np.concatenate((ranges[[1, 2, *range(4, len(ranges))]], [1]))
    strides = np.cumprod(dimensions[::-1])[::-1]
    indices = coordinates @ strides
    return flat[indices.astype(np.int64)]


def predict_unique_composition_rt(
    x_in: np.ndarray,
    ranges: np.ndarray,
    y0: np.ndarray,
    term: OneDimensionalSplineTerm | TwoDimensionalSplineTerm,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Evaluate a fitted captured spline for each unique composition row."""

    x_in = np.asarray(x_in)
    if isinstance(term, OneDimensionalSplineTerm):
        design, _, _ = build_one_dimensional_design(x_in[:, 1:], term)
    else:
        design, _, _ = build_two_dimensional_design(x_in[:, 1:], term)
    baseline = _linearized_y0(x_in, np.asarray(ranges), np.asarray(y0))
    coefficient_array = np.asarray(coefficients)
    fitted = np.asarray(design @ coefficient_array)
    if coefficient_array.ndim == 1:
        return fitted.reshape(-1) + baseline
    if coefficient_array.ndim != 2:
        raise ValueError("spline coefficients must be a vector or sibling matrix")
    siblings = x_in[:, 0].astype(np.int64) - 1
    if np.any(siblings < 0) or np.any(siblings >= coefficient_array.shape[1]):
        raise ValueError("composition sibling lies outside coefficient matrix")
    return fitted[np.arange(len(x_in)), siblings] + baseline


def _restrict_stage_three(
    x_in: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> None:
    positive = weights > 0
    mapped = x_in[indices]
    # MATLAB's ``min([])`` returns an empty array, so the first comparison is
    # an empty/no-op when stage three has no positive-weight observations.
    # NumPy raises on the reduction unless this case is handled explicitly.
    if np.any(positive):
        minimum = np.min(mapped[positive, 4])
        weights[mapped[:, 4] < minimum] = -1

    rows = mapped[weights > 0, 4:6]
    order = _sort_rows(rows, [0, 1])
    rows = rows[order]
    if len(rows) > 1:
        rows = np.delete(rows, np.flatnonzero(np.diff(rows[:, 0]) == 0), axis=0)
    lookup = np.zeros(int(np.max(x_in[:, 4])), dtype=np.float64)
    lookup[rows[:, 0].astype(np.int64) - 1] = rows[:, 1]
    lookup = np.maximum.accumulate(lookup)
    weights[
        mapped[:, 5] > lookup[mapped[:, 4].astype(np.int64) - 1] + 1
    ] = -1


def _propagate_connected_weights(
    x_in: np.ndarray,
    indices: np.ndarray,
    weights_in: np.ndarray,
    weights: np.ndarray,
) -> None:
    xx = x_in[indices]
    distances = np.full(xx.shape[1], 1e10, dtype=np.float64)
    if xx.shape[1] >= 6:
        distances[[4, 5]] = 2
    keep = np.any(xx > 1, axis=0)
    xx = xx[:, keep]
    distances = distances[keep]
    active = np.flatnonzero(weights >= 0)
    pairs: list[np.ndarray] = []
    for column in range(xx.shape[1]):
        others = [item for item in range(xx.shape[1]) if item != column]
        order = _sort_rows(xx[active], [*others, column])
        sorted_rows = xx[active][order]
        if len(sorted_rows) < 2:
            continue
        same = np.all(np.diff(sorted_rows[:, others], axis=0) == 0, axis=1)
        close = np.diff(sorted_rows[:, column]) <= distances[column]
        positions = np.flatnonzero(same & close)
        ordered_active = active[order]
        forward = np.column_stack(
            (ordered_active[positions], ordered_active[positions + 1])
        )
        pairs.extend((forward[:, ::-1], forward))
    if pairs:
        links = np.vstack(pairs)
        reachable = weights_in > 0
        previous = -1
        while previous < int(np.sum(reachable)):
            previous = int(np.sum(reachable))
            reachable[links[reachable[links[:, 0]], 1]] = True
        weights[~reachable] = -1
    else:
        weights[weights == 0] = -1


def predict_compound_cluster_rt(
    y: np.ndarray,
    x_in: np.ndarray,
    identity: np.ndarray,
    design_rows_matlab: np.ndarray,
    weights_in: np.ndarray,
    ranges: np.ndarray,
    thresholds: np.ndarray,
    stage: int,
    y0: np.ndarray,
    term: OneDimensionalSplineTerm | TwoDimensionalSplineTerm,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port MATLAB ``predict_compound_cluster_RT`` for the TG contract."""

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x_in = np.asarray(x_in)
    identity = np.asarray(identity).reshape(-1)
    indices = np.asarray(design_rows_matlab, dtype=np.int64).reshape(-1) - 1
    weights_in = np.asarray(weights_in, dtype=np.float64).reshape(-1).copy()
    ranges = np.asarray(ranges, dtype=np.int64).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    baseline = _linearized_y0(x_in, ranges, y0)

    positive_identity = identity[identity > 0].astype(np.int64)
    if len(positive_identity) == len(np.unique(positive_identity)):
        grouped_identity: np.ndarray | None = None
    else:
        order = np.argsort(identity, kind="stable")
        sorted_identity = identity[order]
        grouped = np.cumsum(
            np.r_[True, np.diff(sorted_identity) != 0] | (sorted_identity == 0)
        )
        grouped_identity = np.empty_like(grouped)
        grouped_identity[order] = grouped

    def prune_mapping_flags(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Port MATLAB ``prune_mapping`` method 3 for an ordered pair table."""
        used_first: set[int] = set()
        used_second: set[int] = set()
        flags = np.zeros(len(first), dtype=bool)
        for row, (left, right) in enumerate(zip(first, second, strict=True)):
            left = int(left)
            right = int(right)
            flags[row] = left in used_first or right in used_second
            if not flags[row]:
                used_first.add(left)
                used_second.add(right)
        return flags

    if isinstance(term, OneDimensionalSplineTerm):
        design, groups, penalty = build_one_dimensional_design(x_in[:, 1:], term)
    else:
        design, groups, penalty = build_two_dimensional_design(x_in[:, 1:], term)
    sibling_count = int(ranges[0])
    if sibling_count > 1:
        sibling = x_in[:, 0].astype(np.int64)
        if np.any(sibling < 1) or np.any(sibling > sibling_count):
            raise ValueError("sibling coordinate lies outside the regression range")
        rows = np.flatnonzero(sibling > 1)
        group_design = sparse.csr_matrix(
            (
                np.ones(len(rows), dtype=np.float64),
                (rows, sibling[rows] - 2),
            ),
            shape=(x_in.shape[0], sibling_count - 1),
        )
        penalty = sparse.block_diag((
            sparse.csr_matrix((sibling_count - 1, sibling_count - 1)), penalty
        ), format="csr")
    else:
        group_design = sparse.csr_matrix((x_in.shape[0], 0), dtype=np.float64)
    combined_design = sparse.hstack((group_design, design), format="csr")
    smoothing = np.full(combined_design.shape[1], term.strength, dtype=np.float64)

    if stage == 3 and x_in.shape[1] == 6:
        _restrict_stage_three(x_in, indices, weights_in)

    weights = weights_in.copy()
    if np.any(baseline != 0) and np.any(weights == 1):
        residual = y - baseline[indices]
        residual -= np.mean(residual[weights == 1])
        error = (np.abs(residual) + 1e30 * (weights_in < 0)) / (
            1 + 1e6 * (weights_in > 0)
        )
        if grouped_identity is not None:
            order = np.argsort(error, kind="stable")
            conflict = prune_mapping_flags(
                grouped_identity[order], indices[order]
            )
            error[order[conflict]] = 1e6
        weights = np.sign(weights_in - (error > thresholds[0]))
        weights[weights_in == 0.5] = 0.5

    previous = weights - 1
    xx = x_in[indices]
    iteration = 1
    stable = False
    coefficients = np.zeros(combined_design.shape[1], dtype=np.float64)
    predicted = np.zeros_like(y)
    smoothing_schedule = np.asarray([100.0, 10.0, 1.0, 0.1])
    if np.any(baseline != 0):
        smoothing_schedule[:] = 100.0

    while iteration < 100 and not stable:
        stable = bool(np.array_equal(previous, weights))
        previous = weights.copy()
        regression_weights = weights.copy()
        regression_weights[regression_weights == 0.5] -= 0.5
        scale = smoothing_schedule[min(iteration, len(smoothing_schedule) - 1)]
        coefficients = weighted_penalized_regression(
            y - baseline[indices],
            combined_design[indices],
            regression_weights,
            penalty,
            smoothing * scale,
        )
        all_predicted = np.asarray(combined_design @ coefficients).reshape(-1) + baseline
        predicted = all_predicted[indices]
        error = (np.abs(predicted - y) + 1e30 * (weights_in < 0)) / (
            1 + 1e6 * (weights_in > 0)
        )
        threshold = thresholds[2 if stable else 1]
        weights = np.sign(weights_in - (error > threshold))
        weights[weights_in == 0.5] = 0.5
        if grouped_identity is not None:
            conflict_error = np.abs(y - predicted) + 1e30 * (weights < 0)
            conflict_error /= 1 + 1e6 * (weights_in > 0)
            order = np.argsort(conflict_error, kind="stable")
            conflict = prune_mapping_flags(
                grouped_identity[order], indices[order]
            )
            conflict_error[order[conflict]] = 1e6
            weights[conflict_error > threshold] = -1
        else:
            order = np.lexsort((error, -weights_in, indices))
            duplicate = np.r_[False, np.diff(indices[order]) == 0]
            weights[order[duplicate]] = -1
        if np.any(weights_in > 0) and np.prod(ranges[4:]) > 1:
            _propagate_connected_weights(x_in, indices, weights_in, weights)
        iteration += 1

    if sibling_count > 1:
        offsets = np.concatenate((np.asarray([0.0]), coefficients[: sibling_count - 1]))
        coefficients = coefficients[sibling_count - 1 :, None] + offsets[None, :]
        return weights >= 0, predicted, coefficients, offsets
    return weights >= 0, predicted, coefficients, np.asarray(0.0)
