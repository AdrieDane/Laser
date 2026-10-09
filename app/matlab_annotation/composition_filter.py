"""Apply standard-mode composition limits to MATLAB target tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from annotator.composition_ranges import CompositionRangeSettings

from .decode import CodeDecoder
from .vendor.target_model import CompoundSetTargetTable


@dataclass(frozen=True)
class TargetCompositionAudit:
    targets_before: int
    targets_after: int
    endogenous_filtered: int
    internal_standards_protected: int


def filter_compound_set_targets(
    project,
    table: CompoundSetTargetTable,
    compounds: tuple[int, ...] | list[int] | np.ndarray,
    *,
    phase: str,
    settings: CompositionRangeSettings,
) -> tuple[CompoundSetTargetTable, TargetCompositionAudit]:
    """Remove disallowed endogenous targets without changing MATLAB radices.

    The composition bases remain untouched because they define MATLAB's dense
    code arithmetic and RT-design matrix. Only concrete target rows are
    removed. Negative codes are declared internal standards and are always
    retained regardless of user limits.
    """
    active = tuple(int(value) for value in np.asarray(compounds).reshape(-1))
    decoders = {
        compound: CodeDecoder(project, compound, phase) for compound in active
    }
    keep: np.ndarray = np.ones(len(table.codes), dtype=bool)
    protected = 0
    filtered = 0
    for index, (position, code) in enumerate(zip(
        table.sibling_positions, table.codes, strict=True
    )):
        if int(code) < 0:
            protected += 1
            continue
        compound = active[int(position) - 1]
        decoded = decoders[compound].decode(int(code))
        decision = settings.decide(
            decoded.lipid_class,
            decoded.carbon,
            decoded.double_bonds,
            species=decoded.species,
        )
        keep[index] = decision.allowed
        filtered += int(not decision.allowed)
    output = CompoundSetTargetTable(
        codes=table.codes[keep],
        theoretical_mz=table.theoretical_mz[keep],
        stages=table.stages[keep],
        sibling_positions=table.sibling_positions[keep],
        composition_bases=table.composition_bases,
    )
    return output, TargetCompositionAudit(
        targets_before=int(len(table.codes)),
        targets_after=int(keep.sum()),
        endogenous_filtered=int(filtered),
        internal_standards_protected=int(protected),
    )
