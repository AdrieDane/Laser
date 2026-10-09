"""Pure candidate-enumeration primitives ported from MATLAB/C/MEX."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidateStagePolicy:
    """The mapping policy used by one ``map_compound_cluster`` stage."""

    stage: int
    target_stages: tuple[int, ...]
    ppm_tolerance: float
    rt_tolerance_seconds: float | None
    requires_prior_fit: bool


@dataclass(frozen=True)
class ForcedMembershipRows:
    """Stored positive inliers that MATLAB appends after normal enumeration."""

    sibling_positions: np.ndarray
    codes: np.ndarray
    viewer_rows: np.ndarray
    theoretical_mz: np.ndarray
    predicted_rt_seconds: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class StageCandidateBatch:
    """Candidate rows emitted immediately before MATLAB's regression step."""

    target_indices: np.ndarray
    observation_indices: np.ndarray
    sibling_positions: np.ndarray
    codes: np.ndarray
    viewer_rows: np.ndarray
    theoretical_mz: np.ndarray
    predicted_rt_seconds: np.ndarray
    weights: np.ndarray
    forced_inlier: np.ndarray


@dataclass(frozen=True)
class RegressionObservations:
    """MATLAB-sorted candidate observations and their unique compositions."""

    order: np.ndarray
    codes: np.ndarray
    viewer_rows: np.ndarray
    sibling_positions: np.ndarray
    unique_compositions: np.ndarray
    observation_to_composition_matlab: np.ndarray


def stage_policy(
    stage: int,
    stage_count: int,
    minimum_inlier_stage: float = np.inf,
    broad_ppm: float = 6.0,
    narrow_ppm: float = 3.0,
) -> CandidateStagePolicy | None:
    """Return the literal MATLAB candidate-enumeration policy for a stage.

    A stage before the earliest stored inlier is skipped.  With no stored
    inliers, each pass introduces only its own target rows.  At the earliest
    inlier stage, all targets up to that stage are introduced together.
    """
    if stage_count < 1 or stage < 1 or stage > stage_count:
        raise ValueError("stage must be within the available stage count")
    if minimum_inlier_stage < stage:
        target_stages = (stage,)
    elif minimum_inlier_stage == stage:
        target_stages = tuple(range(1, stage + 1))
    elif np.isinf(minimum_inlier_stage):
        target_stages = (stage,)
    else:
        return None
    if stage <= 2:
        return CandidateStagePolicy(stage, target_stages, broad_ppm, None, False)
    if stage == 3:
        rt_tolerance = 120.0
    elif stage == 4 and stage != stage_count:
        rt_tolerance = 60.0
    else:
        rt_tolerance = 10.0
    return CandidateStagePolicy(stage, target_stages, narrow_ppm, rt_tolerance, True)


def match_stage_targets(
    target_mz: np.ndarray,
    target_stage: np.ndarray,
    observed_mz: np.ndarray,
    observed_rt_seconds: np.ndarray,
    policy: CandidateStagePolicy,
    *,
    predicted_rt_seconds: np.ndarray | None = None,
    mass_offset: float = 0.0,
    prior_fit_available: bool = True,
) -> np.ndarray:
    """Enumerate one stage's target/peak pairs as ``map_compound_cluster`` does.

    Returned target indices refer to the original target arrays even though
    MATLAB sorts each stage's theoretical m/z values before mapping.
    """
    mz = np.asarray(target_mz, dtype=float).reshape(-1)
    stages = np.asarray(target_stage, dtype=int).reshape(-1)
    observed_mz = np.asarray(observed_mz, dtype=float).reshape(-1)
    observed_rt = np.asarray(observed_rt_seconds, dtype=float).reshape(-1)
    if mz.shape != stages.shape:
        raise ValueError("target m/z and stage arrays must align")
    if observed_mz.shape != observed_rt.shape:
        raise ValueError("observed m/z and RT arrays must align")
    chosen = np.flatnonzero(np.isin(stages, policy.target_stages) & np.isfinite(mz))
    if chosen.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    order = np.argsort(mz[chosen], kind="stable")
    chosen = chosen[order]
    if policy.rt_tolerance_seconds is None:
        pairs = match_targets(
            mz[chosen], observed_mz, np.asarray([policy.ppm_tolerance])
        )
    else:
        if policy.requires_prior_fit and not prior_fit_available:
            return np.empty((0, 2), dtype=np.int64)
        if predicted_rt_seconds is None:
            raise ValueError("predicted RT is required for RT-filtered stages")
        predicted = np.asarray(predicted_rt_seconds, dtype=float).reshape(-1)
        if predicted.shape != mz.shape:
            raise ValueError("predicted RT and target arrays must align")
        targets = np.column_stack((mz[chosen], predicted[chosen]))
        observations = np.column_stack((observed_mz - mass_offset, observed_rt))
        pairs = match_targets(
            targets,
            observations,
            np.asarray([policy.ppm_tolerance, policy.rt_tolerance_seconds]),
        )
    if pairs.size:
        pairs[:, 0] = chosen[pairs[:, 0]]
    return pairs


def assign_membership_weights(
    candidate_codes: np.ndarray,
    candidate_viewer_rows: np.ndarray,
    composition_bases: tuple[int, ...] | np.ndarray,
    memberships: np.ndarray,
    *,
    sibling_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Map stored MATLAB inlier/outlier decisions onto candidate rows.

    Membership rows are ``[sibling, signed composition, viewer row, weight]``.
    MATLAB removes the adduct digit from each candidate code before matching,
    but preserves sibling identity and the internal-standard sign.
    """
    codes = np.asarray(candidate_codes, dtype=np.int64).reshape(-1)
    viewers = np.asarray(candidate_viewer_rows, dtype=np.int64).reshape(-1)
    bases = np.asarray(composition_bases, dtype=np.int64).reshape(-1)
    if codes.shape != viewers.shape:
        raise ValueError("candidate codes and viewer rows must align")
    if bases.size < 2 or np.any(bases < 1):
        raise ValueError("composition bases must contain positive adduct and tail radices")
    if sibling_positions is None:
        siblings = np.ones(codes.shape, dtype=np.int64)
    else:
        siblings = np.asarray(sibling_positions, dtype=np.int64).reshape(-1)
    if siblings.shape != codes.shape or np.any(siblings < 1):
        raise ValueError("candidate sibling positions must align and be positive")
    table = np.asarray(memberships, dtype=float)
    if table.size == 0:
        return np.zeros(codes.shape, dtype=float)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.ndim != 2 or table.shape[1] != 4:
        raise ValueError("memberships must be an N x 4 table")

    lookup: dict[tuple[int, bool, int, int], float] = {}
    for sibling, signed_code, viewer, weight in table:
        key = (int(sibling), signed_code < 0, abs(int(signed_code)), int(viewer))
        if key in lookup:
            raise ValueError("duplicate stored membership key")
        lookup[key] = float(weight)
    tail_size = int(np.prod(bases[1:], dtype=np.int64))
    normalized = np.abs(codes) - ((np.abs(codes) - 1) // tail_size) * tail_size
    return np.asarray([
        lookup.get((int(sibling), code < 0, int(base_code), int(viewer)), 0.0)
        for sibling, code, base_code, viewer in zip(
            siblings, codes, normalized, viewers, strict=True
        )
    ])


def unmatched_positive_memberships(
    candidate_codes: np.ndarray,
    candidate_viewer_rows: np.ndarray,
    composition_bases: tuple[int, ...] | np.ndarray,
    memberships: np.ndarray,
    membership_theoretical_mz: np.ndarray,
    observed_mz_by_viewer: dict[int, float],
    *,
    membership_predicted_rt_seconds: np.ndarray | None = None,
    sibling_positions: np.ndarray | None = None,
    stage: int,
    narrow_ppm: float = 3.0,
) -> ForcedMembershipRows:
    """Port MATLAB's defensive inclusion of unmatched positive inliers.

    This branch runs only after stage 1. It keeps a stored positive inlier when
    its peak is within ``2 * narrow_ppm`` of the membership's theoretical m/z;
    outliers and already matched memberships are never appended.
    """
    table = np.asarray(memberships, dtype=float)
    if table.size == 0:
        table = np.empty((0, 4), dtype=float)
    elif table.ndim == 1:
        table = table.reshape(1, -1)
    if table.ndim != 2 or table.shape[1] != 4:
        raise ValueError("memberships must be an N x 4 table")
    theoretical = np.asarray(membership_theoretical_mz, dtype=float).reshape(-1)
    if theoretical.shape != (len(table),):
        raise ValueError("one theoretical m/z is required per membership")
    if membership_predicted_rt_seconds is None:
        predicted = np.full(len(table), np.nan, dtype=float)
    else:
        predicted = np.asarray(
            membership_predicted_rt_seconds, dtype=float
        ).reshape(-1)
        if predicted.shape != (len(table),):
            raise ValueError("one predicted RT is required per membership")
    if stage <= 1 or not len(table):
        empty_float = np.empty(0, dtype=float)
        empty_int = np.empty(0, dtype=np.int64)
        return ForcedMembershipRows(
            empty_int, empty_int, empty_int, empty_float, empty_float, empty_float
        )

    codes = np.asarray(candidate_codes, dtype=np.int64).reshape(-1)
    viewers = np.asarray(candidate_viewer_rows, dtype=np.int64).reshape(-1)
    bases = np.asarray(composition_bases, dtype=np.int64).reshape(-1)
    if codes.shape != viewers.shape:
        raise ValueError("candidate codes and viewer rows must align")
    if sibling_positions is None:
        siblings = np.ones(codes.shape, dtype=np.int64)
    else:
        siblings = np.asarray(sibling_positions, dtype=np.int64).reshape(-1)
    if siblings.shape != codes.shape:
        raise ValueError("candidate sibling positions must align")
    tail_size = int(np.prod(bases[1:], dtype=np.int64))
    normalized = np.abs(codes) - ((np.abs(codes) - 1) // tail_size) * tail_size
    candidate_keys = {
        (int(sibling), code < 0, int(base_code), int(viewer))
        for sibling, code, base_code, viewer in zip(
            siblings, codes, normalized, viewers, strict=True
        )
    }
    keep = []
    for index, (sibling, signed_code, viewer, weight) in enumerate(table):
        key = (int(sibling), signed_code < 0, abs(int(signed_code)), int(viewer))
        if weight <= 0 or key in candidate_keys:
            continue
        observed = observed_mz_by_viewer.get(int(viewer), np.nan)
        target = theoretical[index]
        ppm = abs(observed - target) / abs(target) * 1e6
        if np.isfinite(ppm) and ppm <= 2.0 * narrow_ppm:
            keep.append(index)
    indices = np.asarray(keep, dtype=np.int64)
    return ForcedMembershipRows(
        sibling_positions=table[indices, 0].astype(np.int64),
        codes=table[indices, 1].astype(np.int64),
        viewer_rows=table[indices, 2].astype(np.int64),
        theoretical_mz=theoretical[indices],
        predicted_rt_seconds=predicted[indices],
        weights=table[indices, 3].astype(float),
    )


def apply_candidate_exclusions(
    weights: np.ndarray,
    candidate_viewer_rows: np.ndarray,
    existing_compound_by_viewer: dict[int, int],
    current_compound: int | tuple[int, ...] | list[int] | np.ndarray,
    previously_selected_viewer_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Mark competing or already-used peaks as MATLAB mapper outliers."""
    result = np.asarray(weights, dtype=float).reshape(-1).copy()
    viewers = np.asarray(candidate_viewer_rows, dtype=np.int64).reshape(-1)
    if result.shape != viewers.shape:
        raise ValueError("weights and candidate viewer rows must align")
    used = set() if previously_selected_viewer_rows is None else set(
        np.asarray(previously_selected_viewer_rows, dtype=np.int64).reshape(-1).tolist()
    )
    current = np.asarray(current_compound, dtype=np.int64).reshape(-1)
    if not len(current):
        raise ValueError("at least one current compound is required")
    allowed_compounds = {0, *current.tolist()}
    for index, viewer in enumerate(viewers):
        existing = int(existing_compound_by_viewer.get(int(viewer), 0))
        if existing not in allowed_compounds or int(viewer) in used:
            result[index] = -1.0
    return result


def apply_interpolation_gate(
    weights: np.ndarray,
    candidate_codes: np.ndarray,
    composition_bases: tuple[int, ...] | np.ndarray,
    *,
    interpolation_enabled: bool,
    stage: int,
    stage_count: int,
) -> np.ndarray:
    """Port the pre-regression primary-adduct gate for interpolation OFF."""
    result = np.asarray(weights, dtype=float).reshape(-1).copy()
    codes = np.asarray(candidate_codes, dtype=np.int64).reshape(-1)
    bases = np.asarray(composition_bases, dtype=np.int64).reshape(-1)
    if result.shape != codes.shape:
        raise ValueError("weights and candidate codes must align")
    if interpolation_enabled or stage >= stage_count or not len(codes):
        return result
    tail_size = int(np.prod(bases[1:], dtype=np.int64))
    primary = np.abs(codes - (codes < 0).astype(np.int64)) <= tail_size
    eligible = primary | (result != 0)
    return (2.0 * (result > 0) - 1.0) * eligible


def enumerate_stage_candidates(
    *,
    target_codes: np.ndarray,
    target_mz: np.ndarray,
    target_stage: np.ndarray,
    target_sibling_positions: np.ndarray,
    observed_mz: np.ndarray,
    observed_rt_seconds: np.ndarray,
    observed_viewer_rows: np.ndarray,
    policy: CandidateStagePolicy,
    composition_bases: tuple[int, ...] | np.ndarray,
    memberships: np.ndarray,
    membership_theoretical_mz: np.ndarray,
    membership_predicted_rt_seconds: np.ndarray | None = None,
    predicted_target_rt_seconds: np.ndarray | None = None,
    mass_offset: float = 0.0,
    prior_fit_available: bool = True,
    existing_compound_by_viewer: dict[int, int] | None = None,
    current_compound: int | tuple[int, ...] | list[int] | np.ndarray | None = None,
    previously_selected_viewer_rows: np.ndarray | None = None,
) -> StageCandidateBatch:
    """Assemble one exact pre-regression ``map_compound_cluster`` stage."""
    codes = np.asarray(target_codes, dtype=np.int64).reshape(-1)
    mz = np.asarray(target_mz, dtype=float).reshape(-1)
    stages = np.asarray(target_stage, dtype=np.int64).reshape(-1)
    siblings = np.asarray(target_sibling_positions, dtype=np.int64).reshape(-1)
    if not (codes.shape == mz.shape == stages.shape == siblings.shape):
        raise ValueError("target code, m/z, stage and sibling arrays must align")
    observed_mz_array = np.asarray(observed_mz, dtype=float).reshape(-1)
    observed_rt = np.asarray(observed_rt_seconds, dtype=float).reshape(-1)
    viewer_rows = np.asarray(observed_viewer_rows, dtype=np.int64).reshape(-1)
    if not (observed_mz_array.shape == observed_rt.shape == viewer_rows.shape):
        raise ValueError("observed m/z, RT and viewer rows must align")
    pairs = match_stage_targets(
        mz,
        stages,
        observed_mz_array,
        observed_rt,
        policy,
        predicted_rt_seconds=predicted_target_rt_seconds,
        mass_offset=mass_offset,
        prior_fit_available=prior_fit_available,
    )
    target_indices = pairs[:, 0].copy()
    observation_indices = pairs[:, 1].copy()
    batch_codes = codes[target_indices]
    batch_siblings = siblings[target_indices]
    batch_viewers = viewer_rows[observation_indices]
    batch_mz = mz[target_indices]
    if predicted_target_rt_seconds is None:
        batch_predicted = np.full(len(pairs), np.nan, dtype=float)
    else:
        target_prediction = np.asarray(
            predicted_target_rt_seconds, dtype=float
        ).reshape(-1)
        batch_predicted = target_prediction[target_indices]
    weights = assign_membership_weights(
        batch_codes,
        batch_viewers,
        composition_bases,
        memberships,
        sibling_positions=batch_siblings,
    )
    observed_by_viewer = {
        int(viewer): float(value)
        for viewer, value in zip(viewer_rows, observed_mz_array, strict=True)
    }
    forced = unmatched_positive_memberships(
        batch_codes,
        batch_viewers,
        composition_bases,
        memberships,
        membership_theoretical_mz,
        observed_by_viewer,
        membership_predicted_rt_seconds=membership_predicted_rt_seconds,
        sibling_positions=batch_siblings,
        stage=policy.stage,
        narrow_ppm=policy.ppm_tolerance,
    )
    forced_mask = np.zeros(len(batch_codes) + len(forced.codes), dtype=bool)
    forced_mask[len(batch_codes):] = True
    target_indices = np.concatenate((
        target_indices, np.full(len(forced.codes), -1, dtype=np.int64)
    ))
    viewer_lookup = {int(viewer): index for index, viewer in enumerate(viewer_rows)}
    forced_observations = np.asarray(
        [viewer_lookup[int(viewer)] for viewer in forced.viewer_rows], dtype=np.int64
    )
    observation_indices = np.concatenate((observation_indices, forced_observations))
    batch_codes = np.concatenate((batch_codes, forced.codes))
    batch_siblings = np.concatenate((batch_siblings, forced.sibling_positions))
    batch_viewers = np.concatenate((batch_viewers, forced.viewer_rows))
    batch_mz = np.concatenate((batch_mz, forced.theoretical_mz))
    batch_predicted = np.concatenate((batch_predicted, forced.predicted_rt_seconds))
    weights = np.concatenate((weights, forced.weights))
    if existing_compound_by_viewer is not None:
        if current_compound is None:
            raise ValueError("current compound is required for competing exclusions")
        weights = apply_candidate_exclusions(
            weights,
            batch_viewers,
            existing_compound_by_viewer,
            current_compound,
            previously_selected_viewer_rows,
        )
    return StageCandidateBatch(
        target_indices,
        observation_indices,
        batch_siblings,
        batch_codes,
        batch_viewers,
        batch_mz,
        batch_predicted,
        weights,
        forced_mask,
    )


def build_regression_observations(
    codes: np.ndarray,
    viewer_rows: np.ndarray,
    sibling_positions: np.ndarray,
    composition_bases: tuple[int, ...] | np.ndarray,
    *,
    sibling_compound_ids: np.ndarray | None = None,
) -> RegressionObservations:
    """Port candidate ordering/unique-row construction before RT regression."""
    code_array = np.asarray(codes, dtype=np.int64).reshape(-1)
    viewers = np.asarray(viewer_rows, dtype=np.int64).reshape(-1)
    siblings = np.asarray(sibling_positions, dtype=np.int64).reshape(-1)
    bases = np.asarray(composition_bases, dtype=np.int64).reshape(-1)
    if not (code_array.shape == viewers.shape == siblings.shape):
        raise ValueError("codes, viewer rows and siblings must align")
    if bases.shape != (5,) or np.any(bases < 1):
        raise ValueError("the MATLAB annotation model requires five positive bases")
    if sibling_compound_ids is None:
        compound_ids = siblings
    else:
        sibling_ids = np.asarray(sibling_compound_ids, dtype=np.int64).reshape(-1)
        if np.any(siblings < 1) or np.any(siblings > len(sibling_ids)):
            raise ValueError("sibling position is outside the compound-id table")
        compound_ids = sibling_ids[siblings - 1]
    order = np.lexsort((viewers, code_array, compound_ids))
    sorted_codes = code_array[order]
    sorted_viewers = viewers[order]
    sorted_siblings = siblings[order]
    if len(order) == 0:
        return RegressionObservations(
            order,
            sorted_codes,
            sorted_viewers,
            sorted_siblings,
            np.empty((0, 6), dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    new_composition = np.r_[
        True,
        (np.diff(sorted_siblings) != 0) | (np.diff(sorted_codes) != 0),
    ]
    mapping = np.cumsum(new_composition).astype(np.int64)
    unique_siblings = sorted_siblings[new_composition]
    unique_codes = np.abs(sorted_codes[new_composition]) - 1
    full_size = int(np.prod(bases, dtype=np.int64))
    if np.any(unique_codes >= full_size):
        raise ValueError("candidate code exceeds the composition radix")
    strides = np.asarray([
        np.prod(bases[index + 1 :], dtype=np.int64)
        for index in range(len(bases))
    ])
    digits = (unique_codes[:, None] // strides) % bases + 1
    unique = np.column_stack((unique_siblings, digits)).astype(np.int64)
    return RegressionObservations(
        order,
        sorted_codes,
        sorted_viewers,
        sorted_siblings,
        unique,
        mapping,
    )


def match_targets(
    targets: np.ndarray, observations: np.ndarray, tolerances: np.ndarray
) -> np.ndarray:
    """Port the one-to-four dimensional MATLAB ``MS_mapping`` primitive.

    The first coordinate is signed m/z and uses MATLAB's symmetric ppm
    comparison.  Remaining coordinates use inclusive absolute windows.  Both
    inputs must be sorted by their first coordinate; returned zero-based pairs
    preserve the target-major ordering of ``range_intersection_filter.c``.
    """
    target = np.asarray(targets, dtype=float)
    observed = np.asarray(observations, dtype=float)
    tolerance = np.asarray(tolerances, dtype=float).reshape(-1)
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    if observed.ndim == 1:
        observed = observed.reshape(-1, 1)
    if target.ndim != 2 or observed.ndim != 2:
        raise ValueError("targets and observations must be one- or two-dimensional")
    if target.shape[1] != observed.shape[1]:
        raise ValueError("targets and observations must have the same dimensions")
    if target.shape[1] < 1 or target.shape[1] > 4:
        raise ValueError("MS_mapping supports one to four dimensions")
    if tolerance.shape != (target.shape[1],):
        raise ValueError("one tolerance is required per coordinate")
    if np.any(tolerance <= 0):
        raise ValueError("tolerances must be positive")
    if np.any(np.diff(target[:, 0]) < 0) or np.any(np.diff(observed[:, 0]) < 0):
        raise ValueError("MS_mapping inputs must be ascending by signed m/z")
    if target.shape[0] == 0 or observed.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)

    maximum = max(
        -target[0, 0], target[-1, 0], -observed[0, 0], observed[-1, 0]
    )
    absolute_mz_window = tolerance[0] * maximum / 1e6
    exact_scale = 2e6 / tolerance[0]
    starts = np.searchsorted(
        observed[:, 0], target[:, 0] - absolute_mz_window, side="left"
    )
    stops = np.searchsorted(
        observed[:, 0], target[:, 0] + absolute_mz_window, side="right"
    )
    pairs: list[tuple[int, int]] = []
    # Most theoretical targets have no project peak inside even the coarse m/z
    # window. Vectorized bounds avoid a Python loop over the complete dense
    # composition grid while preserving MATLAB's target-major pair order.
    for target_index in np.flatnonzero(stops > starts):
        row = target[target_index]
        start = int(starts[target_index])
        stop = int(stops[target_index])
        candidates = observed[start:stop]
        keep = np.abs(exact_scale * (row[0] - candidates[:, 0])) <= np.abs(
            row[0] + candidates[:, 0]
        )
        if target.shape[1] > 1:
            keep &= np.all(
                (candidates[:, 1:] >= row[1:] - tolerance[1:])
                & (candidates[:, 1:] <= row[1:] + tolerance[1:]),
                axis=1,
            )
        pairs.extend(
            (target_index, start + offset) for offset in np.flatnonzero(keep)
        )
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def match_mass_targets(
    target_mz: np.ndarray, observed_mz: np.ndarray, ppm_tolerance: float
) -> np.ndarray:
    """Port the one-dimensional `MS_mapping`/`range_intersection_filter` path.

    Inputs must be ascending signed m/z arrays. Returned indices are zero-based
    `(target, observed)` pairs in MATLAB's target-major order.
    """
    return match_targets(target_mz, observed_mz, np.asarray([ppm_tolerance]))
