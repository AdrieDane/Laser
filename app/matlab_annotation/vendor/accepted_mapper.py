"""Native port of MATLAB's accepted branch in ``map_compound_cluster``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .annotation_runner import AnnotationRunResult
from .model_types import AnnotationModel
from .project import LegacyProject
from .rt_projection import design_row_from_dense_code
from .spline import (
    OneDimensionalSplineTerm,
    build_one_dimensional_design,
    build_two_dimensional_design,
)


@dataclass(frozen=True)
class AcceptedCandidateResult:
    data_rows: np.ndarray
    viewer_rows_matlab: np.ndarray
    sibling_positions: np.ndarray
    codes: np.ndarray
    selected: np.ndarray
    predicted_rt_seconds: np.ndarray
    stages: np.ndarray


def _full_design(model: AnnotationModel, bases: np.ndarray) -> np.ndarray:
    count = int(bases[0] * bases[1] * bases[3] * bases[4])
    coordinates = np.column_stack((
        np.repeat(np.arange(1, bases[0] + 1), bases[1] * bases[3] * bases[4]),
        np.tile(
            np.repeat(np.arange(1, bases[1] + 1), bases[3] * bases[4]),
            bases[0],
        ),
        np.ones(count, dtype=np.int64),
        np.tile(
            np.repeat(np.arange(1, bases[3] + 1), bases[4]),
            bases[0] * bases[1],
        ),
        np.tile(
            np.arange(1, bases[4] + 1), bases[0] * bases[1] * bases[3]
        ),
    ))
    adduct = np.eye(int(bases[0]), dtype=float)[coordinates[:, 0] - 1]
    # MATLAB's configured FA/isoprene model prepends the discrete adduct term
    # and a one-column label term.  The latter is identically zero for the
    # single unlabelled coordinate used by these project definitions.
    prefix = np.column_stack((adduct, np.zeros(count, dtype=float)))
    if isinstance(model.term, OneDimensionalSplineTerm):
        interaction = build_one_dimensional_design(
            coordinates, model.term
        )[0].toarray()
        return np.column_stack((prefix, interaction))
    if model.term is None:
        return np.ones((count, 1), dtype=float)
    interaction = build_two_dimensional_design(
        coordinates, model.term
    )[0].toarray()
    return np.column_stack((prefix, interaction))


def _fit_accepted_prediction(
    model: AnnotationModel,
    bases: np.ndarray,
    sibling_positions: np.ndarray,
    codes: np.ndarray,
    stored_rt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xp = _full_design(model, bases)
    rows = np.asarray([
        design_row_from_dense_code(int(code), bases) - 1 for code in codes
    ], dtype=np.int64)
    sibling_count = int(np.max(sibling_positions))
    group = np.zeros((len(rows), max(sibling_count - 1, 0)), dtype=float)
    if sibling_count > 1:
        other = sibling_positions > 1
        group[np.flatnonzero(other), sibling_positions[other] - 2] = 1.0
    design = np.column_stack((group, xp[rows]))
    baseline = np.asarray(model.rt_preprocessing, dtype=float)
    offset = (
        baseline[rows]
        if baseline.ndim == 1
        else baseline[rows, sibling_positions - 1]
    )
    normal = design.T @ design
    normal = (normal + normal.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(normal)
    floor = float(np.max(eigenvalues)) / 1e8
    beta = eigenvectors @ (
        (eigenvectors.T @ (design.T @ (stored_rt - offset)))
        / np.maximum(eigenvalues, floor)
    )
    spline = beta[sibling_count - 1 :, None]
    sibling_offsets = np.r_[0.0, beta[: sibling_count - 1]][None, :]
    coefficients = spline + sibling_offsets
    prediction = xp @ coefficients
    prediction += baseline[:, None] if baseline.ndim == 1 else baseline
    return prediction, xp


def _expand_primary_codes(codes: np.ndarray, bases: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(codes, dtype=np.int64)[codes > 0]) - 1
    if not len(values):
        return np.empty(0, dtype=np.int64)
    digits = np.empty((len(values), 5), dtype=np.int64)
    remainder = values.copy()
    for dimension in range(5):
        stride = int(np.prod(bases[dimension + 1 :], dtype=np.int64))
        digits[:, dimension] = remainder // stride
        remainder %= stride
    digits = digits[digits[:, 2] == 0]
    if not len(digits):
        return np.empty(0, dtype=np.int64)

    collapsed_bases = np.asarray(
        [bases[0] * bases[1] * bases[2], bases[3], bases[4]],
        dtype=np.int64,
    )
    collapsed = np.column_stack((
        digits[:, :3] @ np.asarray([bases[1] * bases[2], bases[2], 1]),
        digits[:, 3], digits[:, 4],
    ))
    expanded: list[np.ndarray] = []
    for varying, margin in ((1, 4), (2, 3)):
        other = [index for index in range(3) if index != varying]
        for key in np.unique(collapsed[:, other], axis=0):
            group = collapsed[np.all(collapsed[:, other] == key, axis=1)]
            lower = max(0, int(np.min(group[:, varying])) - margin)
            upper = min(
                int(collapsed_bases[varying]) - 1,
                int(np.max(group[:, varying])) + margin,
            )
            rows = np.zeros((upper - lower + 1, 3), dtype=np.int64)
            rows[:, other] = key
            rows[:, varying] = np.arange(lower, upper + 1)
            expanded.append(rows)
    values = np.unique(np.vstack(expanded), axis=0)
    # MATLAB builds ``X`` as the *absent* coordinates in each local range
    # (``X == 0`` in expand_compound_indices). Stored M+0 compositions are
    # already present in the accepted annotation table and must therefore not
    # be returned as nearby candidates a second time.
    observed = {tuple(int(value) for value in row) for row in collapsed}
    values = np.asarray([
        row for row in values
        if tuple(int(value) for value in row) not in observed
    ], dtype=np.int64)
    strides = np.asarray(
        [collapsed_bases[1] * collapsed_bases[2], collapsed_bases[2], 1]
    )
    return values @ strides + 1


def accepted_candidate_result(
    project: LegacyProject,
    run: AnnotationRunResult,
    model: AnnotationModel,
    *,
    phase: str,
    mass_tolerance_ppm: float,
    rtdb_state: int = 5,
) -> AcceptedCandidateResult:
    """Recreate the accepted stored rows plus fitted nearby candidates."""

    data = project.phases[phase]
    compounds = np.asarray(run.compounds, dtype=np.int64)
    source_lookup = {int(row): index + 1 for index, row in enumerate(run.source_rows)}
    selected_rows = np.flatnonzero(np.isin(
        data.annotation[:, 0].astype(np.int64), compounds
    ))
    selected_rows = np.asarray([
        row for row in selected_rows if int(row) in source_lookup
    ], dtype=np.int64)
    selected_siblings = np.asarray([
        int(np.flatnonzero(compounds == int(compound))[0]) + 1
        for compound in data.annotation[selected_rows, 0].astype(np.int64)
    ], dtype=np.int64)
    selected_codes = data.annotation[selected_rows, 1].astype(np.int64)
    selected_rt = data.predicted_rt[selected_rows]
    finite = np.isfinite(selected_rt)
    selected_rows = selected_rows[finite]
    selected_siblings = selected_siblings[finite]
    selected_codes = selected_codes[finite]
    selected_rt = selected_rt[finite]

    target = {
        (int(sibling), int(code)): (float(mz), int(stage), float(database))
        for sibling, code, mz, stage, database in zip(
            run.targets.sibling_positions,
            run.targets.codes,
            run.targets.theoretical_mz,
            run.targets.stages,
            run.database_rt_seconds,
            strict=True,
        )
    }
    selected_stages = np.asarray([
        target[(int(sibling), int(code))][1]
        for sibling, code in zip(
            selected_siblings, selected_codes, strict=True
        )
    ], dtype=np.int16)
    bases = np.asarray(run.targets.composition_bases, dtype=np.int64)

    candidate_rows = np.empty(0, dtype=np.int64)
    candidate_codes = np.empty(0, dtype=np.int64)
    candidate_rt = np.empty(0, dtype=float)
    candidate_stages = np.empty(0, dtype=np.int16)
    primary = selected_siblings == 1
    if model.term is not None and np.any(primary):
        prediction, _xp = _fit_accepted_prediction(
            model, bases, selected_siblings, selected_codes, selected_rt
        )
        expanded = _expand_primary_codes(selected_codes[primary], bases)
        valid = np.asarray([
            (1, int(code)) in target
            and np.isfinite(target[(1, int(code))][0])
            for code in expanded
        ])
        expanded = expanded[valid]
        target_mz = np.asarray([target[(1, int(code))][0] for code in expanded])
        design_rows = np.asarray([
            design_row_from_dense_code(int(code), bases) - 1 for code in expanded
        ])
        target_rt = prediction[design_rows, 0]

        simple = (
            (selected_codes > 0)
            & (selected_codes <= int(bases[3] * bases[4]))
        )
        if np.any(simple):
            selected_mz = np.asarray([
                target[(int(sibling), int(code))][0]
                for sibling, code in zip(
                    selected_siblings[simple], selected_codes[simple], strict=True
                )
            ])
            residual = np.column_stack((
                data.mz_grouped[selected_rows[simple]] - selected_mz,
                data.rt_grouped[selected_rows[simple]] - selected_rt[simple],
            ))
            center = np.quantile(residual, 0.5, axis=0)
            tolerance = np.max(np.abs(residual - center), axis=0)
            ppm = float(mass_tolerance_ppm)
            ppm_width = (
                ppm * np.max(np.abs(data.mz_grouped[selected_rows[simple]])) / 1e6
            )
            tolerance[0] = np.mean((2.0 * tolerance[0], ppm_width))
            # The legacy branch tests for a zero width *before* applying the
            # RTDB popup's minimum RT envelope. With only one usable RT
            # residual (ProjectData2 NP HexCer), the RT width is zero and
            # MATLAB intentionally emits no additional accepted candidates.
            has_nonzero_box = bool(np.all(tolerance != 0))
            # ``RTDB`` is a seven-position MATLAB popup. Position 1 keeps the
            # fitted residual envelope; positions 2..7 widen it to at least
            # 2/5/10/20/30/60 seconds. The legacy GUI opens on position 5.
            rt_floor = (None, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)
            if not 1 <= int(rtdb_state) <= len(rt_floor):
                raise ValueError("RTDB state must be in MATLAB's 1..7 range")
            if rt_floor[int(rtdb_state) - 1] is not None:
                tolerance[1] = max(
                    tolerance[1], float(rt_floor[int(rtdb_state) - 1])
                )
            target_mz = target_mz + center[0]
            target_rt = target_rt + center[1]
            order = np.lexsort((target_rt, target_mz))
            target_mz = target_mz[order]
            target_rt = target_rt[order]
            expanded = expanded[order]

            pairs: list[tuple[int, int]] = []
            source_mz = data.mz_grouped[run.source_rows]
            source_rt = data.rt_grouped[run.source_rows]
            if has_nonzero_box:
                for viewer, (mz, rt) in enumerate(
                    zip(source_mz, source_rt, strict=True), start=1
                ):
                    left = int(np.searchsorted(
                        target_mz, mz - tolerance[0], side="left"
                    ))
                    right = int(np.searchsorted(
                        target_mz, mz + tolerance[0], side="right"
                    ))
                    for target_index in range(left, right):
                        if abs(target_rt[target_index] - rt) <= tolerance[1]:
                            pairs.append((viewer, target_index))
            if pairs:
                viewers = np.asarray([item[0] for item in pairs], dtype=np.int64)
                indices = np.asarray([item[1] for item in pairs], dtype=np.int64)
                candidate_rows = run.source_rows[viewers - 1]
                candidate_codes = expanded[indices]
                candidate_rt = target_rt[indices]
                candidate_stages = np.asarray([
                    target[(1, int(code))][1] for code in candidate_codes
                ], dtype=np.int16)

    rows = np.concatenate((selected_rows, candidate_rows))
    viewers = np.asarray([source_lookup[int(row)] for row in rows], dtype=np.int64)
    siblings = np.concatenate((
        selected_siblings, np.ones(len(candidate_rows), dtype=np.int64)
    ))
    codes = np.concatenate((selected_codes, candidate_codes))
    selected = np.concatenate((
        np.ones(len(selected_rows), dtype=bool),
        np.zeros(len(candidate_rows), dtype=bool),
    ))
    predicted = np.concatenate((selected_rt, candidate_rt))
    stages = np.concatenate((selected_stages, candidate_stages))
    return AcceptedCandidateResult(
        rows, viewers, siblings, codes, selected, predicted, stages
    )
