"""Project/database resolution of MATLAB internal-standard target rows."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from .compound_model import (
    DecodedCode,
    encode_dense_code,
    phase_composition,
    theoretical_mz,
)
from .project import LegacyProject
from .reference import ReferenceData


_TOTAL_COMPOSITION = re.compile(
    r"^(?P<label>[^()]*)\((?P<marker>[^\d()]*)(?P<carbon>\d+):"
    r"(?P<double_bonds>\d+)\)$"
)


@dataclass(frozen=True)
class InternalStandardTargetTable:
    compound_ids: np.ndarray
    codes: np.ndarray
    theoretical_mz: np.ndarray
    stages: np.ndarray
    names: tuple[str, ...]


def build_simple_internal_standard_targets(
    project: LegacyProject,
    reference: ReferenceData,
    phase: str,
    active_compounds: tuple[int, ...] | list[int] | np.ndarray,
) -> InternalStandardTargetTable:
    """Resolve simple ``CLASS(total-C:DB)`` standards used by MATLAB ``i_IS``.

    More complex chain-resolved traits are rejected rather than guessed; their
    MATLAB composition parser remains a separate extraction boundary.
    """
    active = {int(value) for value in np.asarray(active_compounds).reshape(-1)}
    database = reference.database_tables["internal_standards_all.tsv"]
    header = {name: index for index, name in enumerate(database.headers)}
    by_name: dict[str, list[tuple[str, ...]]] = {}
    for row in database.rows:
        by_name.setdefault(row[header["name"]].casefold(), []).append(row)
    composition = np.asarray(
        project.compounds.legacy_fields["composition"], dtype=object
    )
    catalog: dict[str, int] = {}
    duplicates: set[str] = set()
    for compound, (label, parts) in enumerate(
        zip(project.compounds.classes, composition, strict=True), start=1
    ):
        values = [
            "" if np.asarray(value).size == 0 else str(value)
            for value in parts
        ]
        key = f"{values[0]}{label}{values[1]}_{values[2]}"
        if key in catalog:
            duplicates.add(key)
        else:
            catalog[key] = compound
    for key in duplicates:
        catalog.pop(key, None)

    rows: list[tuple[int, int, float, int, str]] = []
    for requested in reference.internal_standards:
        for database_row in by_name.get(requested.name.casefold(), []):
            trait = database_row[header["trait"]]
            match = _TOTAL_COMPOSITION.fullmatch(trait)
            if match is None:
                continue
            compound = catalog.get(
                f"{match.group('label')}_{match.group('marker')}"
            )
            if compound is None or compound not in active:
                continue
            base_mass = float(database_row[header["mass"]])
            attr = project.compounds.composition_attr[compound - 1]
            model = phase_composition(attr, phase)
            try:
                carbon = model.text[3].index(match.group("carbon"))
                double_bond = model.text[4].index(match.group("double_bonds"))
            except ValueError as exc:
                raise ValueError(
                    f"{requested.name} composition is outside compound {compound} radix"
                ) from exc
            for adduct in range(model.bases[0]):
                for natural_isotope in range(model.bases[2]):
                    decoded = DecodedCode(
                        digits=(adduct, 0, natural_isotope, carbon, double_bond),
                        bases=model.bases,
                        adduct_index=adduct + 1,
                        isotope_index=natural_isotope,
                        is_internal_standard=True,
                    )
                    code = encode_dense_code(decoded)
                    stage = 1 if natural_isotope == 0 else model.bases[1] + 3
                    rows.append((
                        compound,
                        code,
                        theoretical_mz(
                            attr,
                            phase,
                            code,
                            internal_standard_mass=base_mass,
                            model=model,
                        ),
                        stage,
                        requested.name,
                    ))
    if not rows:
        return InternalStandardTargetTable(
            compound_ids=np.empty(0, dtype=np.int64),
            codes=np.empty(0, dtype=np.int64),
            theoretical_mz=np.empty(0, dtype=float),
            stages=np.empty(0, dtype=np.int16),
            names=(),
        )
    return InternalStandardTargetTable(
        compound_ids=np.asarray([row[0] for row in rows], dtype=np.int64),
        codes=np.asarray([row[1] for row in rows], dtype=np.int64),
        theoretical_mz=np.asarray([row[2] for row in rows], dtype=float),
        stages=np.asarray([row[3] for row in rows], dtype=np.int16),
        names=tuple(row[4] for row in rows),
    )
