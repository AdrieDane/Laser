"""Research-only native chemistry completion for underspecified class-ion grids.

Uses the existing catalogue/radix/mass decoder independently of RT-grid fitting.
No guessed formulas/adducts or query curation. Only class/mode envelopes supported
by at least three allowed training projects are eligible. The caller must match
query-excluded RT models and enforce the original strict candidate windows.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .automatic import matlab_annotation_adducts
from .chemical_targets import envelope_range_settings
from .decode import CodeDecoder
from .vendor.compound_model import phase_composition, theoretical_mz
from .xcms_project import load_bundled_compound_catalog


@dataclass(frozen=True)
class _Context:
    compounds: object


def compile_catalogue_completion(
    reference_root, reference, envelope, *, stream, declared_standards
):
    """Create only missing identities for class-ion groups with <2 seed C:DBs.

    Existing targets are neither changed nor replaced. A correctly seeded group
    remains the responsibility of the existing validated mass-lattice compiler.
    RTs remain unknown here; an excluded-project atlas must supply them before
    strict mass/RT matching. Native masses must agree with any existing matching
    reference identity, otherwise fail closed rather than silently replace it.
    """
    if stream not in {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}:
        raise ValueError("Explicit measurement mode required")
    required = {"name", "class", "adduct", "carbon", "db", "ref_mz"}
    if not required.issubset(reference):
        raise ValueError("Explicit existing chemical reference required")
    if not {"stream", "ref_mz", "adduct", "is_is"}.issubset(declared_standards):
        raise ValueError("Explicit declared-standard catalogue required")
    if len(declared_standards) and not declared_standards.is_is.eq(True).all():
        raise ValueError("Declared catalogue may contain only internal standards")
    ranges = envelope_range_settings(envelope, stream)
    catalog = load_bundled_compound_catalog(reference_root)
    context = _Context(catalog)
    mono = reference.loc[~reference.name.astype(str).str.endswith(" (IS)")].copy()
    if "isotopologue" in mono:
        mono = mono.loc[mono.isotopologue.fillna("").astype(str).eq("")]
    mono["_class"] = mono["class"].map(canonical_lipid_class)
    groups = {(c, a): g for (c, a), g in mono.groupby(["_class", "adduct"])}
    old_identities = set(mono[["name", "adduct"]].itertuples(index=False, name=None))
    spikes = declared_standards.loc[declared_standards.stream.eq(stream)]
    rows, audit = [], []
    for compound, attr in enumerate(catalog.composition_attr, start=1):
        formula = str(catalog.legacy_fields["chem"][compound - 1]).strip()
        r_group = str(catalog.legacy_fields["R_group"][compound - 1])
        if not formula or r_group not in {"FA", "FAfull"}:
            continue
        decoder = CodeDecoder(context, compound, stream[:2])
        cls = canonical_lipid_class(decoder.lipid_class)
        limits = envelope["models"].get(stream + "|" + cls)
        if not limits or limits["training_cohorts"] < 3:
            continue
        try:
            model = phase_composition(attr, stream[:2])
        except ValueError:
            continue
        for ai, ion in enumerate(model.text[0]):
            if ion not in matlab_annotation_adducts(stream[:2]):
                continue
            charge = model.charges[ai]
            if (charge > 0) != stream.endswith("Pos"):
                continue
            previous = groups.get((cls, ion), mono.iloc[:0])
            seeds = previous[["carbon", "db"]].drop_duplicates()
            if len(seeds) >= 2:
                continue
            known = {n: g.ref_mz.to_numpy(float) for n, g in previous.groupby("name")}
            protected_masses = (
                pd.to_numeric(
                    spikes.loc[spikes.adduct.eq(ion), "ref_mz"], errors="coerce"
                )
                .dropna()
                .to_numpy(float)
            )
            added, blocked = 0, 0
            for ci, ctext in enumerate(decoder.axis(3)):
                carbon = int(ctext)
                if not limits["carbon_min"] <= carbon <= limits["carbon_max"]:
                    continue
                for di, dtext in enumerate(decoder.axis(4)):
                    db = int(dtext)
                    if not limits["db_min"] <= db <= min(limits["db_max"], carbon // 2):
                        continue
                    code = decoder.code_for({0: ai, 3: ci, 4: di})
                    decoded = decoder.decode(code)
                    if not decoded.monoisotopic:
                        raise AssertionError(
                            "Catalogue completion may not emit isotope peaks"
                        )
                    name = decoded.species
                    if not ranges.decide(cls, carbon, db, species=name).allowed:
                        continue
                    mz = abs(theoretical_mz(attr, stream[:2], code, model=model))
                    if not np.isfinite(mz) or mz <= 0:
                        continue
                    if name in known and np.max(np.abs(known[name] - mz)) > 2e-5:
                        raise ValueError(
                            "Native catalogue and existing reference masses disagree"
                        )
                    if (name, ion) in old_identities:
                        continue
                    if len(protected_masses) and np.any(
                        np.abs(protected_masses - mz) <= max(1e-5, mz * 0.05e-6)
                    ):
                        blocked += 1
                        continue
                    rows.append(
                        dict(
                            name=name,
                            **{"class": decoded.lipid_class},
                            adduct=ion,
                            carbon=carbon,
                            db=db,
                            ref_mz=mz,
                            ref_rt=np.nan,
                            stream=stream,
                            isotopologue="",
                            is_is=False,
                            matlab_code=np.nan,
                            matlab_compound=compound,
                            chemical_target_generated=True,
                            chemical_target_origin="native_catalogue_underspecified_lattice",
                        )
                    )
                    added += 1
            audit.append(
                dict(
                    stream=stream,
                    lipid_class=cls,
                    adduct=ion,
                    original_seed_compositions=len(seeds),
                    added_targets=added,
                    declared_mass_aliases_blocked=blocked,
                    training_cohorts=limits["training_cohorts"],
                )
            )
    columns = [
        "name",
        "class",
        "adduct",
        "carbon",
        "db",
        "ref_mz",
        "ref_rt",
        "stream",
        "isotopologue",
        "is_is",
        "matlab_code",
        "matlab_compound",
        "chemical_target_generated",
        "chemical_target_origin",
    ]
    targets = pd.DataFrame(rows, columns=columns)
    if len(targets):
        if targets.duplicated(["name", "adduct"]).any():
            raise ValueError(
                "Multiple native definitions produce the same target identity"
            )
        targets = targets.reset_index(drop=True)
    return targets, ranges, pd.DataFrame(audit)
