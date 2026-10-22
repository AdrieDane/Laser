"""Research-only explicit targets for declared spikes, independent of RT fits.

Uses the bundled validated composition/adduct decoder. No source features,
curated annotation, saved per-dataset RT fit or endogenous carbon limit is read.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .automatic import matlab_annotation_adducts
from .decode import CodeDecoder
from .vendor.internal_standard_model import build_simple_internal_standard_targets
from .vendor.reference import ProjectInternalStandard
from .vendor.reference_loader import _load_table
from .xcms_project import load_bundled_compound_catalog


@dataclass(frozen=True)
class _ChemicalContext:
    compounds: object


@dataclass(frozen=True)
class _DeclarationContext:
    database_tables: dict
    internal_standards: tuple


@lru_cache(maxsize=4)
def _catalog(root):
    root = Path(root)
    return load_bundled_compound_catalog(root), _load_table(
        root / "DataBase/internal_standards_all.tsv"
    )


def declared_standard_targets(reference_root, declarations, *, phases=("RP", "NP")):
    """Compile explicit chemical identities/masses for positive declared spikes.

    The inherited reference grid can omit these rows. A separate declaration
    catalog is therefore necessary, including standards that were NOT selected
    earlier. Negative dense codes retain the standard's actual labelled mass.
    No RT prediction or candidate selection is performed by this function.
    """
    catalog, database = _catalog(str(Path(reference_root).resolve()))
    requested = []
    for item in declarations:
        name = str(item["name"]).strip()
        quantity = float(item["quantity"])
        if not name or not np.isfinite(quantity) or quantity < 0:
            raise ValueError("Invalid declared standard name or quantity")
        if quantity == 0:
            continue
        requested.append(ProjectInternalStandard(name, quantity, str(item["unit"])))
    byname = {i.name: i for i in requested}
    if len(byname) != len(requested):
        raise ValueError("Declared standard names must be unique")
    project = _ChemicalContext(catalog)
    reference = _DeclarationContext(
        {"internal_standards_all.tsv": database}, tuple(requested)
    )
    header = {v: i for i, v in enumerate(database.headers)}
    masses = {}
    for row in database.rows:
        masses.setdefault(
            (row[header["name"]].casefold(), row[header["trait"]]), set()
        ).add(float(row[header["mass"]]))
    rows = []
    for phase in phases:
        if phase not in {"RP", "NP"}:
            raise ValueError("Standard targets require RP or NP")
        table = build_simple_internal_standard_targets(
            project, reference, phase, tuple(range(1, catalog.count + 1))
        )
        decoders = {}
        for compound, code, mass, name in zip(
            table.compound_ids,
            table.codes,
            table.theoretical_mz,
            table.names,
            strict=True,
        ):
            compound = int(compound)
            code = int(code)
            if compound not in decoders:
                decoders[compound] = CodeDecoder(project, compound, phase)
            decoded = decoders[compound].decode(code)
            if (
                not decoded.monoisotopic
                or decoded.adduct not in matlab_annotation_adducts(phase)
            ):
                continue
            if not np.isfinite(mass) or mass == 0:
                continue
            item = byname[name]
            neutral = masses[(name.casefold(), decoded.species)]
            if len(neutral) != 1:
                raise ValueError("Ambiguous chemical mass for a declared standard")
            neutral = float(next(iter(neutral)))
            rows.append(
                dict(
                    stream=phase + ("_Pos" if mass > 0 else "_Neg"),
                    name=decoded.species + " (IS)",
                    **{"class": decoded.lipid_class},
                    carbon=decoded.carbon,
                    db=decoded.double_bonds,
                    adduct=decoded.adduct,
                    ref_mz=abs(float(mass)),
                    ref_rt=np.nan,
                    is_is=True,
                    isotopologue="",
                    matlab_code=code,
                    matlab_compound=compound,
                    internal_standard_name=name,
                    internal_standard_quantity=item.quantity,
                    internal_standard_unit=item.unit,
                    internal_standard_neutral_mass=neutral,
                    standard_compound_id=f"{decoded.species}|{neutral:.5f}",
                )
            )
    columns = [
        "stream",
        "name",
        "class",
        "carbon",
        "db",
        "adduct",
        "ref_mz",
        "ref_rt",
        "is_is",
        "isotopologue",
        "matlab_code",
        "matlab_compound",
        "internal_standard_name",
        "internal_standard_quantity",
        "internal_standard_unit",
        "internal_standard_neutral_mass",
        "standard_compound_id",
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .drop_duplicates(["stream", "name", "adduct", "ref_mz"])
        .reset_index(drop=True)
    )
