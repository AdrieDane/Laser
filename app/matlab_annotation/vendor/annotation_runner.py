"""Native orchestration of MATLAB's unaccepted multistage annotation loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidate_engine import StageCandidateBatch, enumerate_stage_candidates, stage_policy
from .membership_model import resolve_project_memberships
from .project import LegacyProject
from .reference import ReferenceData
from .model_types import AnnotationModel
from .stage_loop import (
    CompleteStageFit,
    PrimaryStageFit,
    RegressionStageData,
    StageHistory,
    advance_stage_history,
    assemble_regression_stage,
    empty_stage_history,
    fit_regression_stage_primary,
    propagate_regression_stage_secondary,
    select_final_natural_isotopes,
    stage_mass_offset,
)
from .target_model import (
    CompoundSetTargetTable,
    build_compound_set_target_table,
    predict_compound_set_target_rt,
)


@dataclass(frozen=True)
class AnnotationStageResult:
    stage: int
    candidates: StageCandidateBatch
    regression: RegressionStageData
    primary: PrimaryStageFit | None
    complete: CompleteStageFit
    history: StageHistory


@dataclass(frozen=True)
class AnnotationRunResult:
    phase: str
    compounds: tuple[int, ...]
    source_rows: np.ndarray
    viewer_rows: np.ndarray
    targets: CompoundSetTargetTable
    memberships: np.ndarray
    stages: tuple[AnnotationStageResult, ...]
    final_history: StageHistory
    source_rt_seconds: np.ndarray
    database_rt_seconds: np.ndarray


def _target_lookup(
    table: CompoundSetTargetTable,
) -> dict[tuple[int, int], tuple[float, int, int]]:
    lookup: dict[tuple[int, int], tuple[float, int, int]] = {}
    for index, (sibling, code, mz, stage) in enumerate(zip(
        table.sibling_positions,
        table.codes,
        table.theoretical_mz,
        table.stages,
        strict=True,
    )):
        key = (int(sibling), int(code))
        if key in lookup:
            raise ValueError(f"duplicate theoretical target {key}")
        lookup[key] = (float(mz), int(stage), index)
    return lookup


def _membership_inputs(
    memberships: np.ndarray,
    lookup: dict[tuple[int, int], tuple[float, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not len(memberships):
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype=np.int64),
            np.inf,
        )
    mz = np.empty(len(memberships), dtype=float)
    stages = np.empty(len(memberships), dtype=np.int16)
    indices = np.empty(len(memberships), dtype=np.int64)
    for row, (sibling, code, _viewer, _weight) in enumerate(memberships):
        key = (int(sibling), int(code))
        try:
            mz[row], stages[row], indices[row] = lookup[key]
        except KeyError as exc:
            raise ValueError(f"stored membership has no theoretical target {key}") from exc
    positive_inlier = memberships[:, 3] == 1
    minimum = (
        float(np.min(stages[positive_inlier]))
        if np.any(positive_inlier)
        else np.inf
    )
    return mz, stages, indices, minimum


def _stage_membership_mask(
    memberships: np.ndarray,
    membership_stages: np.ndarray,
    minimum_inlier_stage: float,
    stage: int,
) -> np.ndarray:
    if minimum_inlier_stage < stage:
        eligible = membership_stages == stage
    else:
        eligible = membership_stages <= stage
    return eligible | (memberships[:, 3] < 0)


def run_annotation(
    project: LegacyProject,
    reference: ReferenceData,
    *,
    phase: str,
    compounds: tuple[int, ...] | list[int] | np.ndarray,
    model: AnnotationModel,
    targets: CompoundSetTargetTable | None = None,
    database_rt_seconds: np.ndarray | None = None,
) -> AnnotationRunResult:
    """Run MATLAB's unaccepted annotation science without candidate fixtures.

    ``model.rt_preprocessing`` is explicit so callers can use either a captured
    oracle or the native project RTDB calibration. Everything beginning with
    theoretical targets and project-peak mapping is native here.
    """
    active = tuple(int(value) for value in np.asarray(compounds).reshape(-1))
    if not active:
        raise ValueError("at least one compound is required")
    data = project.phases[phase]
    source_rows = np.flatnonzero(
        data.grouped_peak_count / project.metadata.sample_count > 0.01
    )
    viewer_rows = np.arange(1, len(source_rows) + 1, dtype=np.int64)
    observed_mz = np.asarray(data.mz_grouped[source_rows], dtype=float)
    observed_rt = np.asarray(data.rt_grouped[source_rows], dtype=float)
    log_intensity = np.log10(
        np.asarray(data.representative_intensity[source_rows], dtype=float)
    ).astype(np.float32)
    observed_mz_by_viewer = dict(zip(viewer_rows, observed_mz, strict=True))
    observed_rt_by_viewer = dict(zip(viewer_rows, observed_rt, strict=True))
    log_intensity_by_viewer = dict(zip(viewer_rows, log_intensity, strict=True))
    existing = {
        int(viewer): int(compound)
        for viewer, compound in zip(
            viewer_rows, data.annotation[source_rows, 0], strict=True
        )
        if compound != 0
    }

    if targets is None:
        targets = build_compound_set_target_table(
            project, reference, phase, active
        )
    if database_rt_seconds is None:
        database_rt_seconds = _target_database_rt_seconds(
            project, reference, phase, active, targets, model.rt_preprocessing
        )
    ranges = np.asarray(model.ranges, dtype=np.int64).reshape(-1)
    expected_ranges = np.asarray([len(active), *targets.composition_bases])
    if not np.array_equal(ranges, expected_ranges):
        raise ValueError(
            f"model ranges {ranges.tolist()} do not match {expected_ranges.tolist()}"
        )
    memberships = resolve_project_memberships(
        project,
        phase,
        active,
        observed_mz=observed_mz,
        observed_rt_seconds=observed_rt,
        observed_viewer_rows=viewer_rows,
    )
    lookup = _target_lookup(targets)
    membership_mz, membership_stages, membership_target_indices, minimum = (
        _membership_inputs(memberships, lookup)
    )
    stage_count = int(np.max(targets.stages))
    interpolation = bool(project.compounds.interpolation[phase][active[0] - 1])
    sibling_flags = project.compounds.interpolation[phase][
        np.asarray(active, dtype=np.int64) - 1
    ]
    if np.any(sibling_flags != sibling_flags[0]):
        raise ValueError("active siblings disagree on interpolation state")

    history = empty_stage_history()
    primary: PrimaryStageFit | None = None
    complete: CompleteStageFit | None = None
    results: list[AnnotationStageResult] = []
    for stage in range(1, stage_count + 1):
        policy = stage_policy(stage, stage_count, minimum)
        if policy is None:
            continue
        predicted_target_rt = None
        mass_offset = 0.0
        if policy.requires_prior_fit:
            if primary is None or complete is None or not len(history.viewer_rows):
                continue
            predicted_target_rt = predict_compound_set_target_rt(
                targets,
                primary_coefficients=primary.coefficients,
                adduct_offsets=complete.adduct_offsets,
                ranges=ranges,
                rt_preprocessing=model.rt_preprocessing,
                term=model.term,
            )
            mass_offset = stage_mass_offset(history, observed_mz_by_viewer)

        member_mask = _stage_membership_mask(
            memberships, membership_stages, minimum, stage
        )
        stage_memberships = memberships[member_mask]
        stage_membership_mz = membership_mz[member_mask]
        if predicted_target_rt is None:
            stage_membership_rt = None
        else:
            stage_membership_rt = predicted_target_rt[
                membership_target_indices[member_mask]
            ]
        batch = enumerate_stage_candidates(
            target_codes=targets.codes,
            target_mz=targets.theoretical_mz,
            target_stage=targets.stages,
            target_sibling_positions=targets.sibling_positions,
            observed_mz=observed_mz,
            observed_rt_seconds=observed_rt,
            observed_viewer_rows=viewer_rows,
            policy=policy,
            composition_bases=targets.composition_bases,
            memberships=stage_memberships,
            membership_theoretical_mz=stage_membership_mz,
            membership_predicted_rt_seconds=stage_membership_rt,
            predicted_target_rt_seconds=predicted_target_rt,
            mass_offset=mass_offset,
            prior_fit_available=bool(len(history.viewer_rows)),
            existing_compound_by_viewer=existing,
            current_compound=active,
            previously_selected_viewer_rows=history.viewer_rows,
        )
        regression = assemble_regression_stage(
            batch,
            observed_rt_by_viewer=observed_rt_by_viewer,
            log_intensity_by_viewer=log_intensity_by_viewer,
            composition_bases=targets.composition_bases,
            sibling_compound_ids=np.asarray(active, dtype=np.int64),
            interpolation_enabled=interpolation,
            stage=stage,
            stage_count=stage_count,
            history=history,
        )
        if not len(regression.weights):
            continue
        if model.term is None:
            # MATLAB's empty-R-group path constructs a dummy identity design,
            # but without a stored positive inlier it cannot establish an RT
            # fit.  Its observable result is therefore the mass-stage
            # candidate table with no selections and no predicted RTs.
            if stage > 2 or np.any(regression.weights == 1):
                raise ValueError(
                    "an empty-R annotation model only supports unanchored "
                    "mass stages"
                )
            complete = CompleteStageFit(
                selected=np.zeros(len(regression.weights), dtype=bool),
                predicted_rt_seconds=np.full(
                    len(regression.weights), np.nan, dtype=float
                ),
                adduct_offsets=np.zeros(targets.composition_bases[0], dtype=float),
            )
            history = advance_stage_history(regression, complete)
            results.append(AnnotationStageResult(
                stage, batch, regression, None, complete, history
            ))
            continue
        if stage == stage_count:
            assert primary is not None and complete is not None
            abundance = np.asarray(
                project.compounds.legacy_fields["natural_isotopes_abundance"],
                dtype=object,
            ).reshape(-1)[active[0] - 1]
            final = select_final_natural_isotopes(
                regression,
                history,
                natural_isotope_abundance=np.asarray(abundance, dtype=float),
                composition_bases=targets.composition_bases,
                primary_coefficients=primary.coefficients,
                adduct_offsets=complete.adduct_offsets,
                ranges=ranges,
                rt_preprocessing=model.rt_preprocessing,
                term=model.term,
            )
            history = advance_stage_history(regression, final)
            results.append(AnnotationStageResult(
                stage, batch, regression, None, final, history
            ))
            continue

        primary = fit_regression_stage_primary(
            regression,
            stage=stage,
            ranges=ranges,
            thresholds_seconds=model.thresholds_seconds,
            rt_preprocessing=model.rt_preprocessing,
            term=model.term,
        )
        complete = propagate_regression_stage_secondary(
            regression,
            primary,
            composition_bases=targets.composition_bases,
            ranges=ranges,
            rt_preprocessing=model.rt_preprocessing,
            term=model.term,
        )
        # MATLAB only materializes stages 1-2 when at least one stored
        # inlier (w==1) anchors the regression. Mass-only candidates remain
        # visible but are not carried into RT-driven stages.
        if stage <= 2 and not np.any(regression.weights == 1):
            complete = CompleteStageFit(
                selected=np.zeros_like(complete.selected, dtype=bool),
                predicted_rt_seconds=complete.predicted_rt_seconds,
                adduct_offsets=complete.adduct_offsets,
            )
        history = advance_stage_history(regression, complete)
        results.append(AnnotationStageResult(
            stage, batch, regression, primary, complete, history
        ))
    return AnnotationRunResult(
        phase=phase,
        compounds=active,
        source_rows=source_rows,
        viewer_rows=viewer_rows,
        targets=targets,
        memberships=memberships,
        stages=tuple(results),
        final_history=history,
        source_rt_seconds=observed_rt,
        database_rt_seconds=database_rt_seconds,
    )


def _target_database_rt_seconds(
    project: LegacyProject,
    reference: ReferenceData,
    phase: str,
    compounds: tuple[int, ...],
    targets: CompoundSetTargetTable,
    rt_preprocessing: np.ndarray,
) -> np.ndarray:
    """Project MATLAB's synthetic ``all`` RTDB row onto available traits."""

    from .rtdb_model import map_retention_time_observations

    bases = np.asarray(targets.composition_bases, dtype=np.int64)
    zero = np.abs(targets.codes.astype(np.int64)) - 1
    strides = np.asarray([
        np.prod(bases[index + 1 :], dtype=np.int64)
        for index in range(len(bases))
    ], dtype=np.int64)
    digits = (zero[:, None] // strides) % bases
    retained = digits[:, [0, 1, 3, 4]]
    retained_bases = bases[[0, 1, 3, 4]]
    retained_strides = np.asarray([
        np.prod(retained_bases[index + 1 :], dtype=np.int64)
        for index in range(len(retained_bases))
    ], dtype=np.int64)
    baseline_index = np.sum(retained * retained_strides, axis=1).astype(np.int64)
    composition_size = int(np.prod(bases[3:], dtype=np.int64))
    composition_index = zero % composition_size + 1
    compound_positions = {compound: index + 1 for index, compound in enumerate(compounds)}
    available = {
        (compound_positions[item.compound], item.composition_index_matlab)
        for item in map_retention_time_observations(project, reference)
        if item.phase == phase and item.compound in compound_positions
    }
    valid = np.asarray([
        (int(sibling), int(composition)) in available
        for sibling, composition in zip(
            targets.sibling_positions, composition_index, strict=True
        )
    ])
    baseline = np.asarray(rt_preprocessing, dtype=float)
    if baseline.ndim == 0:
        values = np.full(len(baseline_index), float(baseline), dtype=float)
    elif baseline.ndim == 1:
        values = baseline[baseline_index]
    else:
        values = baseline[
            baseline_index, targets.sibling_positions.astype(np.int64) - 1
        ]
    return np.where(valid, values, np.nan)
