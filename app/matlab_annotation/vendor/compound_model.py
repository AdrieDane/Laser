"""Pure MATLAB-compatible compound radix and exact-mass operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# MS_analysis.m derives this from stable_isotopes.tsv:
# mass.H1 (1.007825) - electron mass (0.0005485799).
PROTON_MASS = 1.0072764201


def _flat(value) -> list:
    array = np.asarray(value, dtype=object).reshape(-1)
    return array.tolist() if array.size else []


@dataclass(frozen=True)
class DecodedCode:
    digits: tuple[int, ...]
    bases: tuple[int, ...]
    adduct_index: int
    isotope_index: int
    is_internal_standard: bool


@dataclass(frozen=True)
class PhaseComposition:
    text: tuple[tuple[str, ...], ...]
    mass: tuple[tuple[float, ...], ...]
    charges: tuple[float, ...]

    @property
    def bases(self) -> tuple[int, ...]:
        return tuple(len(values) for values in self.text)


@dataclass(frozen=True)
class PositiveTargetTable:
    """All positive dense codes, exact m/z values, and MATLAB stage labels."""

    codes: np.ndarray
    theoretical_mz: np.ndarray
    stages: np.ndarray


def suppress_internal_standard_duplicates(
    table: PositiveTargetTable,
    internal_standard_codes: np.ndarray,
) -> PositiveTargetTable:
    """Set endogenous targets shadowed by MATLAB ``i_IS`` rows to ``NaN``."""
    codes = np.asarray(internal_standard_codes, dtype=np.int64).reshape(-1)
    if np.any(codes == 0):
        raise ValueError("zero is not an internal-standard dense code")
    positions = np.unique(np.abs(codes)) - 1
    if np.any(positions < 0) or np.any(positions >= len(table.codes)):
        raise ValueError("internal-standard code exceeds the positive target table")
    mz = np.asarray(table.theoretical_mz, dtype=float).copy()
    mz[positions] = np.nan
    return PositiveTargetTable(
        codes=table.codes.copy(),
        theoretical_mz=mz,
        stages=table.stages.copy(),
    )


def phase_composition(composition_attr: dict, phase: str) -> PhaseComposition:
    text = [tuple(str(item) for item in (_flat(value) or [""]))
            for value in _flat(composition_attr["text_comp"])]
    mass = [tuple(float(item) for item in (_flat(value) or [0.0]))
            for value in _flat(composition_attr["mass_comp"])]
    columns = tuple(str(value) for value in _flat(composition_attr.get("column", [])))
    charges = tuple(float(value) for value in _flat(composition_attr.get("charge", [])))
    if columns and len(columns) == len(text[0]):
        keep = tuple(index for index, column in enumerate(columns) if column == phase)
        text[0] = tuple(text[0][index] for index in keep)
        mass[0] = tuple(mass[0][index] for index in keep)
        charges = tuple(charges[index] for index in keep)
    if len(text) != len(mass):
        raise ValueError("composition text and mass dimensions differ")
    if not text or len(text[0]) != len(charges):
        raise ValueError(f"compound has no charge-aligned adducts for phase {phase}")
    return PhaseComposition(tuple(text), tuple(mass), charges)


def decode_dense_code(
    composition_attr: dict,
    phase: str,
    code: int,
    *,
    model: PhaseComposition | None = None,
) -> DecodedCode:
    if not code:
        raise ValueError("zero is not an annotated dense code")
    model = phase_composition(composition_attr, phase) if model is None else model
    remainder = abs(int(code)) - 1
    digits = []
    for index in range(len(model.bases)):
        stride = int(np.prod(model.bases[index + 1 :], dtype=np.int64))
        digits.append(remainder // stride)
        remainder %= stride
    if any(digit >= base for digit, base in zip(digits, model.bases, strict=True)):
        raise ValueError(f"dense code {code} exceeds the phase composition radix")
    return DecodedCode(
        digits=tuple(digits), bases=model.bases,
        adduct_index=digits[0] + 1,
        isotope_index=digits[2] if len(digits) > 2 else 0,
        is_internal_standard=code < 0,
    )


def encode_dense_code(decoded: DecodedCode) -> int:
    value = 0
    for digit, base in zip(decoded.digits, decoded.bases, strict=True):
        if digit < 0 or digit >= base:
            raise ValueError("composition digit is outside its radix")
        value = value * base + digit
    code = value + 1
    return -code if decoded.is_internal_standard else code


def theoretical_mz(
    composition_attr: dict,
    phase: str,
    code: int,
    *,
    internal_standard_mass: float | None = None,
    proton_mass: float = PROTON_MASS,
    model: PhaseComposition | None = None,
) -> float:
    model = phase_composition(composition_attr, phase) if model is None else model
    decoded = decode_dense_code(composition_attr, phase, code, model=model)
    if decoded.is_internal_standard:
        if internal_standard_mass is None:
            return float("nan")
        base_mass = float(internal_standard_mass)
    else:
        base_mass = float(np.asarray(composition_attr.get("mass", np.nan)).reshape(-1)[0])
    selected = tuple(
        values[digit] for values, digit in zip(model.mass, decoded.digits, strict=True)
    )
    # MATLAB's selected IS mass already contains carbon/double-bond composition.
    addition = (
        selected[0] + (selected[2] if len(selected) > 2 else 0.0)
        if decoded.is_internal_standard else sum(selected)
    )
    charge = model.charges[decoded.digits[0]]
    if not charge:
        return float("nan")
    return (base_mass + addition) / charge + proton_mass


def candidate_stage(
    composition_attr: dict,
    phase: str,
    code: int,
    r_group: str,
) -> int:
    """Port positive-code ``mz_filter`` construction from MATLAB."""
    if code <= 0:
        raise ValueError("candidate_stage requires a positive dense code")
    decoded = decode_dense_code(composition_attr, phase, code)
    digits = decoded.digits
    bases = decoded.bases
    if len(bases) != 5:
        raise ValueError("the MATLAB annotation stage model requires five dimensions")

    stage = bases[1] + 3
    if digits[2] == 0:
        stage = digits[1] + 3

    abundant = np.asarray(
        composition_attr.get("R_group_abundant", []), dtype=float
    ).reshape(-1)
    abundance_index = digits[3] * bases[4] + digits[4]
    if (
        digits[1] == 0
        and digits[2] == 0
        and abundance_index < abundant.size
        and abundant[abundance_index] != 0
    ):
        stage = 2

    if digits[1] == 0 and digits[2] == 0:
        if r_group in {"FA", "FAfull", "ST"}:
            range_start = {"FA": 8, "FAfull": 1, "ST": 17}[r_group]
            selected_parity = ((digits[3] + 1 + range_start) % 2) == 1
            if selected_parity and digits[4] == 1:
                stage = 1
        elif digits[4] == 0:
            stage = 1
    return int(stage)


def build_positive_target_table(
    composition_attr: dict,
    phase: str,
    r_group: str,
    *,
    proton_mass: float = PROTON_MASS,
) -> PositiveTargetTable:
    """Vectorize MATLAB's positive-code ``mz`` and ``mz_filter`` tables."""
    model = phase_composition(composition_attr, phase)
    bases = np.asarray(model.bases, dtype=np.int64)
    if bases.shape != (5,) or np.any(bases < 1):
        raise ValueError("the MATLAB annotation target model requires five dimensions")
    size = int(np.prod(bases, dtype=np.int64))
    zero_based = np.arange(size, dtype=np.int64)
    strides = np.asarray([
        np.prod(bases[index + 1 :], dtype=np.int64)
        for index in range(len(bases))
    ], dtype=np.int64)
    digits = (zero_based[:, None] // strides) % bases

    base_mass = float(np.asarray(composition_attr.get("mass", np.nan)).reshape(-1)[0])
    addition = np.zeros(size, dtype=float)
    for column, values in enumerate(model.mass):
        addition += np.asarray(values, dtype=float)[digits[:, column]]
    charge = np.asarray(model.charges, dtype=float)[digits[:, 0]]
    mz = np.full(size, np.nan, dtype=float)
    valid_charge = charge != 0
    mz[valid_charge] = (
        (base_mass + addition[valid_charge]) / charge[valid_charge] + proton_mass
    )

    stages = np.full(size, int(bases[1] + 3), dtype=np.int16)
    monoisotopic = digits[:, 2] == 0
    stages[monoisotopic] = digits[monoisotopic, 1] + 3
    primary_label = (digits[:, 1] == 0) & monoisotopic
    abundance = np.asarray(
        composition_attr.get("R_group_abundant", []), dtype=float
    ).reshape(-1)
    abundance_index = digits[:, 3] * bases[4] + digits[:, 4]
    abundant = np.zeros(size, dtype=bool)
    covered = abundance_index < len(abundance)
    abundant[covered] = abundance[abundance_index[covered]] != 0
    stages[primary_label & abundant] = 2

    if r_group in {"FA", "FAfull", "ST"}:
        range_start = {"FA": 8, "FAfull": 1, "ST": 17}[r_group]
        selected_parity = ((digits[:, 3] + 1 + range_start) % 2) == 1
        stages[primary_label & selected_parity & (digits[:, 4] == 1)] = 1
    else:
        stages[primary_label & (digits[:, 4] == 0)] = 1
    return PositiveTargetTable(
        codes=zero_based + 1,
        theoretical_mz=mz,
        stages=stages,
    )
