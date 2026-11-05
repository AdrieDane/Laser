"""Project-owned internal-standard declarations and library resolution."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re

import pandas as pd

from resource_paths import REFERENCE_DATA_DIRECTORY
from standard_lipidomics.project_layout import xcms_project_root

PROJECT_IS_PANELS = ("Project internal standards", "Project internal standards + Agilent tune mix", "Agilent tune mix")
CATALOG = REFERENCE_DATA_DIRECTORY / "matlab_pipeline" / "DataBase" / "internal_standards_all.tsv"


@dataclass(frozen=True)
class InternalStandard:
    name: str
    concentration: float
    unit: str


def find_project_is_file(project: str | Path) -> Path | None:
    selected = Path(project).expanduser().resolve()
    if selected.is_file():
        selected = selected.parent
    root = xcms_project_root(selected)
    # A selected calibration keeps its frozen IS copy, including archived runs.
    candidates = [root / "metadata" / "internal_standards.csv", root / "internal_standards.csv"]
    if selected != root:
        candidates.insert(0, selected / "internal_standards.csv")
    return next((path for path in candidates if path.is_file()), None)


def read_internal_standards_csv(path: str | Path) -> tuple[InternalStandard, ...]:
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"name", "concentration", "unit"}.issubset(reader.fieldnames):
            raise ValueError(f"{path}: required columns are name, concentration, unit.")
        result = []
        seen = set()
        for number, row in enumerate(reader, 2):
            if not any(value for value in row.values()):
                continue
            name, unit = str(row.get("name") or "").strip(), str(row.get("unit") or "").strip()
            try:
                quantity = float(row.get("concentration", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid concentration on row {number}.") from exc
            if not name or not unit or not math.isfinite(quantity) or quantity <= 0:
                raise ValueError(f"{path}: row {number} needs a name, positive finite concentration, and unit.")
            if name.casefold() in seen:
                raise ValueError(f"{path}: duplicate internal standard {name}.")
            seen.add(name.casefold())
            result.append(InternalStandard(name, quantity, unit))
    if not result:
        raise ValueError(f"{path}: no internal standards were supplied.")
    return tuple(result)


def project_is_signature(project: str | Path) -> tuple[str, str]:
    path = find_project_is_file(project)
    return (str(path), hashlib.sha256(path.read_bytes()).hexdigest()) if path else ("", "")


def project_is_definitions(project: str | Path) -> tuple[Path | None, tuple[InternalStandard, ...]]:
    path = find_project_is_file(project)
    if path is None:
        return None, ()
    definitions = read_internal_standards_csv(path)
    catalog = pd.read_csv(CATALOG, sep="\t", encoding="latin-1").fillna("")
    known = set(catalog["name"].astype(str).str.casefold())
    unknown = [item.name for item in definitions if item.name.casefold() not in known]
    if unknown:
        raise ValueError(f"{path}: internal standards absent from the library: {', '.join(unknown)}.")
    return path, definitions


def _ion_key(value: str) -> str:
    return re.sub(r"[()]", "", value).replace("HCOO", "CHO2")


def project_is_reference(project: str | Path, reference: pd.DataFrame, *, stream: str) -> pd.DataFrame:
    """Resolve only declared IS names, using exact library masses and phase RTs."""
    from annotator.reference import parse_lipid_name
    from matlab_annotation.vendor.composition_initializer import _ADDUCTS
    from matlab_annotation.vendor.compound_model import PROTON_MASS

    path, definitions = project_is_definitions(project)
    columns = ["standard", "name", "adduct", "ref_mz", "ref_rt", "source",
               "concentration", "unit", "is_file"]
    if path is None:
        return pd.DataFrame(columns=columns)
    phase, polarity = stream.split("_")
    polarity = polarity.casefold()
    catalog = pd.read_csv(CATALOG, sep="\t", encoding="latin-1").fillna("")
    ions = {_ion_key(label): (addition, charge) for label, addition, charge in _ADDUCTS.values()}
    rows = []
    for item in definitions:
        records = catalog[catalog["name"].str.casefold().eq(item.name.casefold())]
        matched = False
        for record in records.to_dict(orient="records"):
            trait = str(record["trait"])
            identities = reference["name"].astype(str).str.replace(r"\s+\(IS\)$", "", regex=True)
            if phase == "NP":
                identities = identities.str.replace(r"^(?:1-acyl|2-acyl)\s+", "", regex=True)
            matches = reference[identities.eq(trait) & reference["polarity"].astype(str).eq(polarity)]
            if matches.empty:
                # The species can be absent from the chemical grid (short/odd
                # chains or labelled spikes). Reuse only ion channels already
                # defined for its class; mass and RT still come from the IS library.
                lipid_class = re.sub(r"^(?:1-acyl|2-acyl)\s+", "", trait).split("(", 1)[0]
                matches = reference[reference["class"].astype(str).eq(lipid_class)
                                    & reference["polarity"].astype(str).eq(polarity)]
            for candidate in matches.drop_duplicates("adduct").to_dict(orient="records"):
                ion = ions.get(_ion_key(str(candidate["adduct"])))
                if ion is None:
                    continue
                addition, charge = ion
                rt = pd.to_numeric(record.get(f"RT_{phase}"), errors="coerce")
                # Separate NP/RP catalogue rows can carry different traits.
                if len(records) > 1 and not math.isfinite(rt):
                    continue
                parsed = parse_lipid_name(trait)
                rows.append({**candidate, "standard": item.name, "name": f"{trait} (IS)",
                    "ref_mz": (float(record["mass"]) + addition) / abs(charge) + (1 if charge > 0 else -1) * PROTON_MASS,
                    "ref_rt": float(rt), "source": "project internal standard", "is_is": True,
                    "formula": "", "charge": charge,
                    "carbon": parsed.carbon, "db": parsed.db, "base": parsed.base,
                    "concentration": item.concentration, "unit": item.unit, "is_file": str(path)})
                matched = True
        if not matched:
            rows.append({"standard": item.name, "name": item.name, "adduct": "(reference unavailable)",
                         "ref_mz": float("nan"), "ref_rt": float("nan"), "source": "project internal standard",
                         "concentration": item.concentration, "unit": item.unit, "is_file": str(path)})
    return pd.DataFrame(rows).drop_duplicates(["standard", "adduct", "ref_mz", "ref_rt"])
