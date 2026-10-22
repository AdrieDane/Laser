"""Unused research: recover theoretical C:DB nodes without a successful RT fit.

Uses the native immutable chemical catalogue and code/mass operations. No
curated reference, raw feature, dataset identity or guessed formula is read.
These are theoretical targets only, never accepted full annotations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .automatic import matlab_annotation_adducts
from .chemical_targets import FORMAT
from .decode import CodeDecoder
from .standard_rt_library import ion_key
from .vendor.compound_model import phase_composition, theoretical_mz
from .xcms_project import load_bundled_compound_catalog


@dataclass(frozen=True)
class _Context:
    compounds: object


def catalogue_missing_targets(
    reference, reference_root, envelope, *, stream, declared_standards
):
    """Enumerate missing native monoisotopic nodes inside OTHER-group bounds.

    Unlike a mass-lattice extrapolation, this does not require two surviving
    reference nodes PER adduct. A failed per-class RT mapping cannot erase the
    class's chemical definition. Frozen bounds still require>=3 training groups.
    Only native C:DB axes/adducts are used; unsupported non-C:DB or missing
    chemical definitions stay unavailable. No O/P/positional class merging.
    Actual declared spike masses (not names alone) protect labelled standards.
    """
    if envelope.get("format") != FORMAT:
        raise ValueError("Unsupported grouped chemical envelope")
    if not {"name", "adduct"}.issubset(reference):
        raise ValueError("Existing reference must identify target names/adducts")
    if not {"adduct", "ref_mz", "internal_standard_quantity"}.issubset(
        declared_standards
    ):
        raise ValueError(
            "Explicit declared-spike catalogue required, including an empty declaration"
        )
    if stream not in {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}:
        raise ValueError("Unknown measurement stream")
    catalog = load_bundled_compound_catalog(reference_root)
    project = _Context(catalog)
    phase = stream[:2]
    positive = stream.endswith("Pos")
    existing = {
        (str(n), ion_key(a))
        for n, a in reference[["name", "adduct"]].itertuples(index=False, name=None)
    }
    std = declared_standards[
        pd.to_numeric(
            declared_standards.internal_standard_quantity, errors="coerce"
        ).gt(0)
    ].copy()
    if "stream" in std:
        std = std[std.stream.eq(stream)]
    std["_ion"] = std.adduct.map(ion_key)
    spikes = {
        str(a): g.ref_mz.to_numpy(float) for a, g in std.groupby("_ion", sort=False)
    }
    rows, audits = [], []
    for compound in range(1, catalog.count + 1):
        decoder = CodeDecoder(project, compound, phase)
        cls = canonical_lipid_class(decoder.lipid_class)
        bounds = envelope["models"].get(stream + "|" + cls)
        if (
            bounds is None
            or bounds.get("training_cohorts", 0) < 3
            or "x:x" not in decoder.template
        ):
            continue
        attr = catalog.composition_attr[compound - 1]
        # Some catalogue rows have no channel in this phase. Never synthesize it.
        if phase not in np.asarray(attr.get("column", []), dtype=str):
            continue
        native = phase_composition(attr, phase)
        if len(decoder.bases) != 5:
            continue
        isotope = [i for i, label in enumerate(decoder.axis(2)) if label == ""]
        if len(isotope) != 1:
            continue
        carbon = {int(c): i for i, c in enumerate(decoder.axis(3)) if str(c).isdigit()}
        db = {int(d): i for i, d in enumerate(decoder.axis(4)) if str(d).isdigit()}
        carbon = {
            c: i
            for c, i in carbon.items()
            if bounds["carbon_min"] <= c <= bounds["carbon_max"]
        }
        db = {d: i for d, i in db.items() if bounds["db_min"] <= d <= bounds["db_max"]}
        for ai, adduct in enumerate(decoder.axis(0)):
            if (
                adduct not in matlab_annotation_adducts(phase)
                or (native.charges[ai] > 0) != positive
            ):
                continue
            added, blocked = 0, 0
            for c, ci in carbon.items():
                for d, di in db.items():
                    if d > c // 2:
                        continue
                    code = decoder.code_for({0: ai, 2: isotope[0], 3: ci, 4: di})
                    decoded = decoder.decode(code)
                    key = (decoded.species, ion_key(adduct))
                    if key in existing:
                        continue
                    mz = abs(theoretical_mz(attr, phase, code, model=native))
                    if not np.isfinite(mz) or mz <= 0:
                        continue
                    protected = spikes.get(ion_key(adduct), np.array([], float))
                    if (
                        len(protected)
                        and (np.abs(protected - mz) / protected < 0.05e-6).any()
                    ):
                        blocked += 1
                        continue
                    rows.append(
                        dict(
                            name=decoded.species,
                            **{"class": decoder.lipid_class},
                            base="",
                            adduct=adduct,
                            carbon=c,
                            db=d,
                            ref_mz=mz,
                            ref_rt=np.nan,
                            matlab_code=code,
                            matlab_compound=compound,
                            is_is=False,
                            isotopologue="",
                            chemical_target_generated=True,
                            chemical_target_origin="native_catalogue_without_RT_dependency",
                        )
                    )
                    existing.add(key)
                    added += 1
            audits.append(
                dict(
                    stream=stream,
                    lipid_class=cls,
                    adduct=adduct,
                    added=added,
                    blocked_declared_spike_masses=blocked,
                    training_cohorts=int(bounds["training_cohorts"]),
                )
            )
    columns = [
        "name",
        "class",
        "base",
        "adduct",
        "carbon",
        "db",
        "ref_mz",
        "ref_rt",
        "matlab_code",
        "matlab_compound",
        "is_is",
        "isotopologue",
        "chemical_target_generated",
        "chemical_target_origin",
    ]
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(audits)
