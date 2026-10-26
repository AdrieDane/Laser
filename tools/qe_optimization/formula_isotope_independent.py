"""Independent TSV/native-affine/polynomial/expm1 isotope pilot checker.

Does not import the producer or call its formula, moments or ratio helpers.
"""

from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


def formula(text):
    parts = re.findall(r"[A-Z][a-z]?\d*", text)
    if "".join(parts) != text:
        raise ValueError("Unsupported independent formula")
    result = Counter()
    for part in parts:
        atom = part.rstrip("0123456789")
        result[atom] += int(part[len(atom) :] or "1")
    return result


def resources(root):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.vendor.composition_initializer import _ADDUCTS

    atoms = defaultdict(list)
    with (Path(root) / "DataBase/stable_isotopes.tsv").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            number = int(re.match(r"\d+", row["atom"])[0])
            atom = row["atom"][len(str(number)) :]
            atoms[atom].append((number, float(row["mass"]), float(row["abundance"])))
    isotope = {}
    for atom, rows in atoms.items():
        rows.sort()
        base, mass, abundance = rows[0]
        isotope[atom] = (
            mass,
            np.array(
                [
                    sum(p for n, _, p in rows if n == base + i) / abundance
                    for i in range(3)
                ]
            ),
        )
    result = defaultdict(list)
    active, group = "", ""
    with (Path(root) / "DataBase/lipids.tsv").open(encoding="latin-1") as f:
        rows = csv.reader(f, delimiter="\t")
        next(rows)
        for row in rows:
            if not any(s.strip() for s in row):
                continue
            if row[0].strip():
                active, group = row[0].strip(), row[2].strip()
            decoration = [row[i].strip() for i in (3, 4, 5)]
            decoration[1] = (
                "'"
                if decoration[1] == "*"
                else decoration[1].replace("_", "-", 1)
                if decoration[1].startswith("_")
                else decoration[1]
            )
            base = decoration[0] + active + decoration[1]
            template = base + (
                "(" + decoration[2] + ("x)" if group == "isoprene" else "x:x)")
                if group
                else decoration[2]
            )
            cls = base
            marker = re.match(r"[OPdtm]", decoration[2])
            if group and marker:
                # Literal native decoder convention: O+P template is class O.
                cls += "-" + marker[0]
            chemistry = formula(row[10].strip())
            for measurement in row[9].split(";"):
                measurement = measurement.strip()
                if not measurement:
                    continue
                phase, token = measurement.split("_", 1)
                adduct, addition, charge = _ADDUCTS[token]
                combined = phase == "NP" and cls.endswith(("-O", "-P"))
                reported = cls[:-2] + "-O+P" if combined else cls
                result[(phase, canonical_lipid_class(reported), adduct)].append(
                    (
                        template,
                        chemistry.copy(),
                        group,
                        addition,
                        charge,
                        combined and cls.endswith("-P"),
                    )
                )
    return result, isotope


def ion_delta(adduct):
    body = adduct[2 : adduct.index("]")]
    atoms = Counter()
    for sign, chunk in re.findall(r"([+-])([^+-]+)", body):
        multiplier = re.match(r"\d+", chunk)
        count = int(multiplier[0]) if multiplier else 1
        chunk = chunk[len(multiplier[0]) :] if multiplier else chunk
        for atom, n in formula(chunk).items():
            atoms[atom] += (1 if sign == "+" else -1) * count * n
    return atoms


def hypothesis(row, definitions, isotopes):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.vendor.compound_model import PROTON_MASS
    from matlab_annotation.vendor.composition_initializer import _MASS

    if bool(row["is_is"]) or str(row.get("isotopologue", "")) not in ("", "nan"):
        return None, "standard_or_labelled", np.nan
    carbon, db, target = (float(row[k]) for k in ("carbon", "db", "ref_mz"))
    if (
        not all(np.isfinite([carbon, db, target]))
        or carbon < 0
        or db < 0
        or carbon != int(carbon)
        or db != int(db)
    ):
        return None, "invalid_coordinates", np.nan
    ion = row["adduct"]
    known_ions = {key[2] for key in definitions}
    if ion not in known_ions:
        return None, "unsupported_adduct", np.nan
    phase = str(row["stream"]).split("_")[0]
    matches, errors = {}, []
    for template, base, group, addition, charge, p_combined in definitions.get(
        (phase, canonical_lipid_class(row["class"]), ion), []
    ):
        d = int(db) - p_combined
        if d < 0:
            continue
        name = template.replace("x:x", str(int(carbon)) + ":" + str(d)).replace(
            "x)", str(int(carbon)) + ")"
        )
        if phase == "NP":
            name = re.sub(
                r"\([OP]-\d+:\d+\)",
                "(O+P-" + str(int(carbon)) + ":" + str(int(db)) + ")",
                name,
            )
        if name != row["name"]:
            continue
        neutral = base.copy()
        n_r = neutral.pop("R", 0)
        base_mass = sum(_MASS[a] * n for a, n in neutral.items())
        if group in ("FA", "FAfull", "ST"):
            c_offset, h_offset = (-n_r, -n_r) if group != "ST" else (0, -6 * n_r)
            neutral["C"] += int(carbon) + c_offset
            neutral["H"] += int(2 * carbon - 2 * d) + h_offset
            affine_mass = (
                base_mass
                + c_offset * _MASS["C"]
                + h_offset * _MASS["H"]
                + carbon * (_MASS["C"] + 2 * _MASS["H"])
                - 2 * d * _MASS["H"]
            )
        elif group == "isoprene":
            neutral["C"] += int(5 * carbon)
            neutral["H"] += int(8 * carbon)
            affine_mass = base_mass + carbon * (5 * _MASS["C"] + 8 * _MASS["H"])
        elif not group and not n_r:
            affine_mass = base_mass
        else:
            continue
        for atom, count in ion_delta(ion).items():
            neutral[atom] += count
        if any(n < 0 or a not in isotopes for a, n in neutral.items()):
            continue
        computed = abs((affine_mass + addition) / charge + PROTON_MASS)
        error = abs(computed - abs(target))
        errors.append(error)
        if error <= 0.00002:
            matches[tuple(sorted((a, n) for a, n in neutral.items() if n))] = error
    if len(matches) != 1:
        return (
            None,
            "ambiguous_formula" if len(matches) > 1 else "no_mass_valid_formula",
            min(errors, default=np.nan),
        )
    atoms, error = next(iter(matches.items()))
    return dict(atoms), "available", error


def polynomial(atoms, isotopes, carbon_only=False):
    poly = np.array([1.0, 0.0, 0.0])
    for atom, n in atoms.items():
        if carbon_only and atom != "C":
            continue
        for _ in range(n):
            poly = np.convolve(poly, isotopes[atom][1])[:3]
    return float(poly[1]), float(poly[2])


def inverse_ratio(source, difference):
    if not np.isfinite(source) or not np.isfinite(difference):
        return np.nan
    if source <= 0 or source + difference <= 0:
        return np.nan
    with np.errstate(over="ignore", invalid="ignore"):
        first, second = np.expm1(source), np.expm1(source + difference)
    if not all(np.isfinite([first, second])):
        return np.nan
    return float(np.log(second / first))


def distance(value, low, high):
    if not all(np.isfinite([value, low, high])) or min(low, high) <= 0:
        return np.nan
    limits = sorted([np.log(low), np.log(high)])
    return float(abs(value - np.clip(value, *limits)))


def independent_measure(pool, definitions, isotopes, fields, prefix):
    results, audits, cache = [], [], {}
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
            cache[key] = hypothesis(row, definitions, isotopes)
        atoms, status, error = cache[key]
        record = dict.fromkeys(fields, np.nan)
        record.update(available=float(atoms is not None), native_mass_error_da=error)
        if atoms is not None:
            first, second = polynomial(atoms, isotopes)
            carbon_first, carbon_second = polynomial(atoms, isotopes, True)
            ratios = [
                inverse_ratio(
                    float(row.get("q_raw_log_median", np.nan)),
                    float(row.get("q_raw_isotope_" + channel + "_log_ratio", np.nan)),
                )
                for channel in ("upper", "lower", "lower2")
            ]
            upper, lower, lower2 = ratios
            record.update(
                carbon_atoms=atoms.get("C", 0),
                carbon_R1=carbon_first,
                nominal_R1=first,
                carbon_R2=carbon_second,
                nominal_R2=second,
                upper_log_ratio=upper,
                upper_carbon_error=abs(upper - math.log(carbon_first))
                if carbon_first > 0
                else np.nan,
                upper_interval_error=distance(upper, carbon_first, first),
                lower_inverse_log_ratio=-lower,
                lower_inverse_interval_error=distance(-lower, carbon_first, first),
                lower2_inverse_log_ratio=-lower2,
                lower2_inverse_interval_error=distance(-lower2, carbon_second, second),
            )
        results.append(record)
        audits.append(
            dict(
                index=idx,
                status=status,
                atom_counts=tuple(sorted(atoms.items())) if atoms else (),
            )
        )
    return pd.DataFrame(results, index=pool.index).add_prefix(prefix), pd.DataFrame(
        audits
    ).set_index("index")
