"""Native construction of MATLAB compound-set theoretical target tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compound_model import (
    build_positive_target_table,
    phase_composition,
    suppress_internal_standard_duplicates,
)
from .internal_standard_model import build_simple_internal_standard_targets
from .project import LegacyProject
from .reference import ReferenceData
from .rt_projection import project_dense_codes_to_rt
from .spline import (
    OneDimensionalSplineTerm,
    TwoDimensionalSplineTerm,
    predict_unique_composition_rt,
)


@dataclass(frozen=True)
class CompoundSetTargetTable:
    codes: np.ndarray
    theoretical_mz: np.ndarray
    stages: np.ndarray
    sibling_positions: np.ndarray
    composition_bases: tuple[int, ...]


def build_compound_set_target_table(
    project: LegacyProject,
    reference: ReferenceData,
    phase: str,
    compounds: tuple[int, ...] | list[int] | np.ndarray,
) -> CompoundSetTargetTable:
    """Build positive and simple-IS targets for one active MATLAB sibling set."""
    active = tuple(int(value) for value in np.asarray(compounds).reshape(-1))
    if not active:
        raise ValueError("compound set cannot be empty")
    r_groups = np.asarray(
        project.compounds.legacy_fields["R_group"], dtype=object
    ).reshape(-1)
    standards = build_simple_internal_standard_targets(
        project, reference, phase, active
    )
    codes: list[np.ndarray] = []
    mz: list[np.ndarray] = []
    stages: list[np.ndarray] = []
    siblings: list[np.ndarray] = []
    shared_bases: tuple[int, ...] | None = None
    for sibling, compound in enumerate(active, start=1):
        attr = project.compounds.composition_attr[compound - 1]
        model = phase_composition(attr, phase)
        if shared_bases is None:
            shared_bases = model.bases
        elif model.bases != shared_bases:
            raise ValueError("active siblings have incompatible composition radices")
        positive = build_positive_target_table(
            attr, phase, str(r_groups[compound - 1])
        )
        standard_mask = standards.compound_ids == compound
        standard_codes = standards.codes[standard_mask]
        if len(standard_codes):
            positive = suppress_internal_standard_duplicates(
                positive, standard_codes
            )
        codes.append(positive.codes)
        mz.append(positive.theoretical_mz)
        stages.append(positive.stages)
        siblings.append(np.full(len(positive.codes), sibling, dtype=np.int64))
        if len(standard_codes):
            codes.append(standard_codes)
            mz.append(standards.theoretical_mz[standard_mask])
            stages.append(standards.stages[standard_mask])
            siblings.append(np.full(len(standard_codes), sibling, dtype=np.int64))
    assert shared_bases is not None
    return CompoundSetTargetTable(
        codes=np.concatenate(codes),
        theoretical_mz=np.concatenate(mz),
        stages=np.concatenate(stages),
        sibling_positions=np.concatenate(siblings),
        composition_bases=shared_bases,
    )


def predict_compound_set_target_rt(
    table: CompoundSetTargetTable,
    *,
    primary_coefficients: np.ndarray,
    adduct_offsets: np.ndarray,
    ranges: np.ndarray,
    rt_preprocessing: np.ndarray,
    term: OneDimensionalSplineTerm | TwoDimensionalSplineTerm,
) -> np.ndarray:
    """Project a carried MATLAB fit onto every positive and IS target code."""
    bases = np.asarray(table.composition_bases, dtype=np.int64)
    sibling_count = int(np.max(table.sibling_positions, initial=0))
    retained_bases = bases[[0, 1, 3, 4]]
    row_count = int(np.prod(retained_bases, dtype=np.int64))
    zero_based = np.arange(row_count, dtype=np.int64)
    strides = np.asarray([
        np.prod(retained_bases[index + 1 :], dtype=np.int64)
        for index in range(len(retained_bases))
    ], dtype=np.int64)
    retained = (zero_based[:, None] // strides) % retained_bases + 1
    grid = np.column_stack((
        np.ones(row_count, dtype=np.int64),
        retained[:, 0],
        retained[:, 1],
        np.ones(row_count, dtype=np.int64),
        retained[:, 2],
        retained[:, 3],
    ))
    design_prediction = np.empty((row_count, sibling_count), dtype=float)
    for sibling in range(1, sibling_count + 1):
        grid[:, 0] = sibling
        design_prediction[:, sibling - 1] = predict_unique_composition_rt(
            grid,
            np.asarray(ranges),
            np.asarray(rt_preprocessing),
            term,
            primary_coefficients,
        )
    predicted = project_dense_codes_to_rt(
        design_prediction,
        table.codes,
        bases,
        table.sibling_positions,
    )
    tail = int(np.prod(bases[1:], dtype=np.int64))
    adduct = (np.abs(table.codes) - 1) // tail
    offsets = np.asarray(adduct_offsets, dtype=float)
    if offsets.ndim == 2:
        predicted += offsets[adduct, table.sibling_positions - 1]
    else:
        predicted += offsets[adduct]
    return predicted
