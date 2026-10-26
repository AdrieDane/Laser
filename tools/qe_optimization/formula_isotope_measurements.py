"""Read-only native-formula compatibility descriptors; no acceptance rules.

Uses existing charge-aware C13-neighbour measurements. Nominal isotope totals
are upper compatibility bounds, NOT a claim that unresolved M+1/M+2 envelopes
were acquired. No upper-M+2 channel exists in the current raw measurements.
"""

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd

PREFIX = "q_formula_isotope_"
FIELDS = (
    "available",
    "carbon_atoms",
    "native_mass_error_da",
    "carbon_R1",
    "nominal_R1",
    "carbon_R2",
    "nominal_R2",
    "upper_log_ratio",
    "upper_carbon_error",
    "upper_interval_error",
    "lower_inverse_log_ratio",
    "lower_inverse_interval_error",
    "lower2_inverse_log_ratio",
    "lower2_inverse_interval_error",
)
ADDUCT_ATOMS = {
    "[M-H]-": ({"H": -1}, -1),
    "[M-2H](2-)": ({"H": -2}, -2),
    "[M-4H](4-)": ({"H": -4}, -4),
    "[M+H]+": ({"H": 1}, 1),
    "[M+2H](2+)": ({"H": 2}, 2),
    "[M+NH4]+": ({"N": 1, "H": 4}, 1),
    "[M+Na]+": ({"Na": 1}, 1),
    "[M+K]+": ({"K": 1}, 1),
    "[M+Cl]-": ({"Cl": 1}, -1),
    "[M+CHO2]-": ({"C": 1, "H": 1, "O": 2}, -1),
    "[M+Na-2H]-": ({"Na": 1, "H": -2}, -1),
    "[M+H-H2O]+": ({"H": -1, "O": -1}, 1),
}
TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
MASS_TOLERANCE_DA = 2e-5


def parse_formula(text):
    matches = list(TOKEN.finditer(text))
    if "".join(m.group() for m in matches) != text:
        raise ValueError("Unsupported formula syntax")
    atoms = defaultdict(int)
    for m in matches:
        atoms[m[1]] += int(m[2] or 1)
    return dict(atoms)


def isotope_table(root):
    table = pd.read_csv(Path(root) / "DataBase/stable_isotopes.tsv", sep="\t")
    result = {}
    for atom, group in table.assign(
        element=table.atom.str.extract(r"\d+([A-Z][a-z]?)", expand=False)
    ).groupby("element"):
        # For the elements used here the lightest stable isotope is also the
        # native monoisotopic reference. Do not generalize this to all metals.
        group = group.copy()
        group["number"] = group.atom.str.extract(r"(\d+)", expand=False).astype(int)
        group = group.sort_values("number")
        base = group.iloc[0]
        result[atom] = dict(
            mass=float(base.mass),
            r1=float(
                group.loc[group.number.eq(base.number + 1), "abundance"].sum()
                / base.abundance
            ),
            r2=float(
                group.loc[group.number.eq(base.number + 2), "abundance"].sum()
                / base.abundance
            ),
        )
    return result


def atom_counts(chem, r_group, n_r, carbon, db):
    atoms = parse_formula(chem)
    if atoms.pop("R", 0) != n_r:
        raise ValueError("R-group count mismatch")
    if r_group in {"FA", "FAfull"}:
        atoms["C"] = atoms.get("C", 0) + carbon - n_r
        atoms["H"] = atoms.get("H", 0) + 2 * carbon - 2 * db - n_r
    elif r_group == "ST":
        atoms["C"] = atoms.get("C", 0) + carbon
        atoms["H"] = atoms.get("H", 0) + 2 * carbon - 2 * db - 6 * n_r
    elif r_group == "isoprene":
        atoms["C"] = atoms.get("C", 0) + 5 * carbon
        atoms["H"] = atoms.get("H", 0) + 8 * carbon
    elif r_group or n_r:
        raise ValueError("Unsupported R-group")
    if any(v < 0 or int(v) != v for v in atoms.values()):
        raise ValueError("Invalid atom count")
    return {k: int(v) for k, v in atoms.items() if v}


def ion_formula(neutral, adduct):
    delta, charge = ADDUCT_ATOMS[adduct]
    atoms = dict(neutral)
    for atom, count in delta.items():
        atoms[atom] = atoms.get(atom, 0) + count
    if any(v < 0 for v in atoms.values()):
        raise ValueError("Invalid ion atom count")
    return {k: v for k, v in atoms.items() if v}, charge


def moments(atoms, isotopes):
    first = sum(n * isotopes[a]["r1"] for a, n in atoms.items())
    second = sum(n * isotopes[a]["r2"] for a, n in atoms.items())
    second += (first**2 - sum(n * isotopes[a]["r1"] ** 2 for a, n in atoms.items())) / 2
    n_c = atoms.get("C", 0)
    r = isotopes["C"]["r1"]
    return n_c * r, first, n_c * (n_c - 1) * r**2 / 2, second


@dataclass(frozen=True)
class Definition:
    template: str
    chem: str
    r_group: str
    n_r: int
    ions: tuple[str, ...]
    combined_p: bool


def definitions(root):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.decode import CodeDecoder
    from matlab_annotation.xcms_project import load_bundled_compound_catalog
    from matlab_annotation.vendor.compound_model import phase_composition

    catalog = load_bundled_compound_catalog(root)
    context = SimpleNamespace(compounds=catalog)
    lookup = defaultdict(list)
    for phase in ("RP", "NP"):
        for i, attr in enumerate(catalog.composition_attr):
            if phase not in tuple(attr.get("column", ())):
                continue
            model = phase_composition(attr, phase)
            if not model.charges:
                continue
            decoder = CodeDecoder(context, i + 1, phase)
            cls = decoder.lipid_class
            combined = phase == "NP" and cls.endswith(("-O", "-P"))
            reported = re.sub(r"-(?:O|P)$", "-O+P", cls) if combined else cls
            lookup[(phase, canonical_lipid_class(reported))].append(
                Definition(
                    decoder.template,
                    str(catalog.legacy_fields["chem"][i]),
                    str(catalog.legacy_fields["R_group"][i]),
                    int(catalog.legacy_fields["nR"][i]),
                    tuple(model.text[0]),
                    combined and cls.endswith("-P"),
                )
            )
    return lookup


def hypothesis_formula(row, lookup, isotopes):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.vendor.compound_model import PROTON_MASS

    if bool(row["is_is"]) or str(row.get("isotopologue", "")) not in {"", "nan"}:
        return None, "standard_or_labelled", np.nan
    values = [float(row[k]) for k in ("carbon", "db", "ref_mz")]
    c, db, mass = values
    if not all(np.isfinite(values)) or int(c) != c or int(db) != db or c < 0 or db < 0:
        return None, "invalid_coordinates", np.nan
    phase, ion = row["stream"].split("_")[0], str(row["adduct"])
    if ion not in ADDUCT_ATOMS:
        return None, "unsupported_adduct", np.nan
    candidates = {}
    errors = []
    for definition in lookup.get((phase, canonical_lipid_class(row["class"])), []):
        if ion not in definition.ions:
            continue
        d = int(db) - int(definition.combined_p)
        if d < 0:
            continue
        name = definition.template.replace("x:x", f"{int(c)}:{d}")
        if name.endswith("x)"):
            name = name[:-2] + f"{int(c)})"
        if phase == "NP":
            name = re.sub(r"\((?:O|P)-(\d+):(\d+)\)", f"(O+P-{int(c)}:{int(db)})", name)
        if name != str(row["name"]):
            continue
        try:
            atoms, charge = ion_formula(
                atom_counts(
                    definition.chem, definition.r_group, definition.n_r, int(c), d
                ),
                ion,
            )
            electron = isotopes["H"]["mass"] - PROTON_MASS
            computed = (
                sum(isotopes[a]["mass"] * n for a, n in atoms.items())
                - charge * electron
            ) / abs(charge)
        except (ValueError, KeyError):
            continue
        error = abs(computed - abs(mass))
        errors.append(error)
        if error <= MASS_TOLERANCE_DA:
            candidates[tuple(sorted(atoms.items()))] = error
    if len(candidates) != 1:
        return (
            None,
            "ambiguous_formula" if len(candidates) > 1 else "no_mass_valid_formula",
            min(errors, default=np.nan),
        )
    atoms, error = next(iter(candidates.items()))
    return dict(atoms), "available", error


def true_log_ratio(log_median, logged_ratio):
    if not np.isfinite(log_median) or not np.isfinite(logged_ratio):
        return np.nan
    partner = log_median + logged_ratio
    if log_median <= 0 or partner <= 0:
        return np.nan
    # log(expm1(x)) is stable at large x and for small positive intensities.
    return (
        partner
        + math.log(-math.expm1(-partner))
        - log_median
        - math.log(-math.expm1(-log_median))
    )


def interval_error(value, low, high):
    if not all(np.isfinite([value, low, high])) or low <= 0 or high <= 0:
        return np.nan
    lower, upper = sorted((math.log(low), math.log(high)))
    return max(lower - value, value - upper, 0.0)


def measure(pool, lookup, isotopes):
    evidence, audit = [], []
    cache = {}
    for idx, row in pool.iterrows():
        key = tuple(
            str(row.get(k, ""))
            for k in (
                "stream",
                "name",
                "class",
                "adduct",
                "carbon",
                "db",
                "ref_mz",
                "is_is",
                "isotopologue",
            )
        )
        if key not in cache:
            cache[key] = hypothesis_formula(row, lookup, isotopes)
        atoms, status, error = cache[key]
        result = dict.fromkeys(FIELDS, np.nan)
        result["available"] = float(atoms is not None)
        result["native_mass_error_da"] = error
        if atoms is not None:
            r1, n1, r2, n2 = moments(atoms, isotopes)
            upper, lower, lower2 = [
                true_log_ratio(
                    float(row.get("q_raw_log_median", np.nan)),
                    float(row.get("q_raw_isotope_" + ch + "_log_ratio", np.nan)),
                )
                for ch in ("upper", "lower", "lower2")
            ]
            result.update(
                carbon_atoms=atoms.get("C", 0),
                carbon_R1=r1,
                nominal_R1=n1,
                carbon_R2=r2,
                nominal_R2=n2,
                upper_log_ratio=upper,
                upper_carbon_error=abs(upper - math.log(r1)) if r1 > 0 else np.nan,
                upper_interval_error=interval_error(upper, r1, n1),
                lower_inverse_log_ratio=-lower,
                lower_inverse_interval_error=interval_error(-lower, r1, n1),
                lower2_inverse_log_ratio=-lower2,
                lower2_inverse_interval_error=interval_error(-lower2, r2, n2),
            )
        evidence.append(result)
        audit.append(
            dict(
                index=idx,
                status=status,
                atom_counts=tuple(sorted(atoms.items())) if atoms else (),
            )
        )
    return pd.DataFrame(evidence, index=pool.index).add_prefix(PREFIX), pd.DataFrame(
        audit
    ).set_index("index")
