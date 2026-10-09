"""Initialize MATLAB's derived ``MS.composition_attr`` cache."""

from __future__ import annotations

import re
from typing import Any

import numpy as np


# Source: Current_version/DataBase/stable_isotopes.tsv. MATLAB selects the
# most-abundant stable isotope for each element in import_stable_isotopes.
_MASS = {
    "H": 1.007825,
    "C": 12.0,
    "N": 14.003074,
    "O": 15.994915,
    "P": 30.973762,
    "S": 31.972071,
}

# Source: MS_analysis.m import_R_group. Keep phase/class identity untouched.
_R_GROUPS = {
    "FA": {
        "range": ((8, 42), (0, 8)),
        "abundant": (
            (10, 0), (12, 0), (14, 0), (14, 1), (16, 0), (16, 1),
            (16, 2), (18, 0), (18, 1), (18, 2), (18, 3), (20, 0),
            (20, 1), (20, 2), (20, 3), (20, 4), (20, 5), (22, 0),
            (22, 1), (22, 2), (22, 3), (22, 4), (22, 5), (22, 6),
            (24, 0), (26, 0), (28, 0),
        ),
        "offset": -(_MASS["C"] + _MASS["H"]),
        "increments": (_MASS["C"] + 2 * _MASS["H"], -2 * _MASS["H"]),
    },
    "FAfull": {
        "range": ((1, 42), (0, 8)),
        "abundant": (
            (10, 0), (12, 0), (14, 0), (14, 1), (16, 0), (16, 1),
            (16, 2), (18, 0), (18, 1), (18, 2), (18, 3), (20, 0),
            (20, 1), (20, 2), (20, 3), (20, 4), (20, 5), (22, 0),
            (22, 1), (22, 2), (22, 3), (22, 4), (22, 5), (22, 6),
            (24, 0), (26, 0), (28, 0),
        ),
        "offset": -(_MASS["C"] + _MASS["H"]),
        "increments": (_MASS["C"] + 2 * _MASS["H"], -2 * _MASS["H"]),
    },
    "ST": {
        "range": ((17, 42), (0, 8)),
        "abundant": (
            (18, 1), (18, 2), (18, 3), (19, 1), (19, 2), (19, 3),
            (21, 1), (21, 2), (21, 3), (27, 0), (27, 1), (27, 2),
            (27, 3),
        ),
        "offset": -6 * _MASS["H"],
        "increments": (_MASS["C"] + 2 * _MASS["H"], -2 * _MASS["H"]),
    },
    "isoprene": {
        "range": ((10, 25),),
        "abundant": tuple((value,) for value in range(10, 26)),
        "offset": 0.0,
        "increments": (5 * _MASS["C"] + 8 * _MASS["H"],),
    },
}

# Exact results of MATLAB's import_adducts for every token used by the shipped
# 228-compound catalog. Keys omit the leading NP_/RP_ column name.
_ADDUCTS = {
    "neg2_0": ("[M-2H](2-)", 0.0, -2),
    "neg4_0": ("[M-4H](4-)", 0.0, -4),
    "neg_0": ("[M-H]-", 0.0, -1),
    "neg_CH2O2": ("[M+CHO2]-", 46.005480000000006, -1),
    "neg_HCl": ("[M+Cl]-", 35.976678, -1),
    "neg_Na": ("[M+Na-2H]-", 21.981945, -1),
    "pos2_0": ("[M+2H](2+)", 0.0, 2),
    "pos_0": ("[M+H]+", 0.0, 1),
    "pos_0-H2O": ("[M+H-H2O]+", -18.010565, 1),
    "pos_K": ("[M+K]+", 37.955882, 1),
    "pos_NH3": ("[M+NH4]+", 17.026549, 1),
    "pos_Na": ("[M+Na]+", 21.981945, 1),
}


def initialize_composition_attrs(ms: dict[str, Any], metadata: dict[str, Any]) -> tuple[dict, ...]:
    """Port ``MS_analysis('calculate_composition')`` for an uncached project."""
    required = {
        "chem", "nR", "R_group", "measurement", "natural_isotopes",
        "natural_isotopes_dmz", "natural_isotopes_label",
    }
    missing = sorted(required - ms.keys())
    if missing:
        raise ValueError(f"cannot initialize composition_attr; MS is missing: {', '.join(missing)}")

    count = np.asarray(ms["class"], dtype=object).size
    fields = {
        name: np.asarray(ms[name], dtype=object).reshape(-1)
        for name in ("chem", "nR", "R_group", "measurement", "natural_isotopes", "natural_isotopes_dmz")
    }
    if any(len(values) != count for values in fields.values()):
        raise ValueError("cannot initialize composition_attr; compound fields are not aligned")
    isotope_labels = tuple(str(value) for value in np.asarray(
        ms["natural_isotopes_label"], dtype=object
    ).reshape(-1))
    label_type = _matlab_string(metadata.get("label_type", ""))
    if label_type:
        raise ValueError(f"cannot initialize labelled composition_attr: {label_type!r}")

    return tuple(
        _initialize_one(
            chem=_matlab_string(fields["chem"][index]),
            n_r=int(np.asarray(fields["nR"][index]).reshape(-1)[0]),
            r_group=_matlab_string(fields["R_group"][index]),
            measurements=tuple(str(value) for value in np.atleast_1d(fields["measurement"][index])),
            isotope_counts=np.asarray(fields["natural_isotopes"][index]),
            isotope_masses=np.asarray(fields["natural_isotopes_dmz"][index], dtype=float).reshape(-1),
            isotope_labels=isotope_labels,
        )
        for index in range(count)
    )


def _initialize_one(
    *, chem: str, n_r: int, r_group: str, measurements: tuple[str, ...],
    isotope_counts: np.ndarray, isotope_masses: np.ndarray,
    isotope_labels: tuple[str, ...],
) -> dict[str, Any]:
    columns = []
    adduct_text = []
    adduct_mass = []
    charges = []
    for measurement in measurements:
        column, separator, token = measurement.partition("_")
        if not separator or token not in _ADDUCTS:
            raise ValueError(f"unknown measurement/adduct token: {measurement!r}")
        text, mass, charge = _ADDUCTS[token]
        columns.append(column)
        adduct_text.append(text)
        adduct_mass.append(mass)
        charges.append(charge)

    isotope_counts = np.asarray(isotope_counts)
    if isotope_counts.size == 0:
        isotope_counts = np.empty((0, len(isotope_labels)), dtype=int)
    elif isotope_counts.ndim == 1:
        isotope_counts = isotope_counts.reshape(1, -1)
    isotope_text = [
        "".join(
            f"({label})" + (str(int(number)) if int(number) > 1 else "")
            for label, number in zip(isotope_labels, row, strict=True) if number
        )
        for row in isotope_counts
    ]
    isotope_text_values = np.empty(len(isotope_text), dtype=object)
    for index, value in enumerate(isotope_text):
        # scipy.loadmat represents MATLAB's empty M+0 label as an empty char
        # array nested inside the cell vector, not as a scalar empty string.
        isotope_text_values[index] = np.asarray([], dtype=str) if not value else value

    group = _R_GROUPS.get(r_group) if r_group else None
    dimensions = []
    dimension_masses = []
    if group is not None:
        for bounds, increment in zip(group["range"], group["increments"], strict=True):
            values = np.arange(bounds[0] * n_r, bounds[1] * n_r + 1, dtype=int)
            dimensions.append(values)
            dimension_masses.append(values.astype(float) * increment)
    elif r_group:
        raise ValueError(f"unknown R_group: {r_group!r}")
    while len(dimensions) < 2:
        dimensions.append(np.asarray([0], dtype=int))
        dimension_masses.append(np.asarray([0.0]))

    base_mass = _formula_mass(chem) + (n_r * float(group["offset"]) if group else 0.0)
    abundance = _abundance_grid(group, n_r, dimensions) if group else np.asarray([True])
    return {
        "text_comp": (
            np.asarray(adduct_text, dtype=object), np.asarray([], dtype=str), isotope_text_values,
            np.asarray([str(value) for value in dimensions[0]], dtype=object),
            np.asarray([str(value) for value in dimensions[1]], dtype=object),
        ),
        "i_comp": (
            np.arange(1, len(adduct_text) + 1), 0,
            np.arange(1, len(isotope_masses) + 1), dimensions[0], dimensions[1],
        ),
        "mass_comp": (
            np.asarray(adduct_mass), 0.0, isotope_masses,
            dimension_masses[0], dimension_masses[1],
        ),
        "R_group_abundant": abundance,
        "charge": np.asarray(charges),
        "column": np.asarray(columns, dtype=object),
        "mass": base_mass,
    }


def _formula_mass(formula: str) -> float:
    total = 0.0
    for atom, count_text in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if atom == "R":
            continue
        if atom not in _MASS:
            raise ValueError(f"unknown atom in compound formula: {atom}")
        total += _MASS[atom] * (int(count_text) if count_text else 1)
    return total


def _abundance_grid(group: dict, n_r: int, dimensions: list[np.ndarray]) -> np.ndarray:
    sums = {tuple(0 for _ in group["range"])}
    for _ in range(n_r):
        sums = {
            tuple(left + right for left, right in zip(total, item, strict=True))
            for total in sums for item in group["abundant"]
        }
    output = np.zeros(tuple(len(values) for values in dimensions[:len(group["range"])]), dtype=bool)
    starts = tuple(int(values[0]) for values in dimensions)
    for total in sums:
        index = tuple(value - start for value, start in zip(total, starts, strict=False))
        if all(0 <= value < size for value, size in zip(index, output.shape, strict=True)):
            output[index] = True
    return output.reshape(-1)


def _matlab_string(value: Any) -> str:
    array = np.asarray(value, dtype=object)
    if array.size == 0:
        return ""
    if array.size == 1:
        return str(array.reshape(-1)[0])
    return "".join(str(item) for item in array.reshape(-1))
