"""Research-only chemical target completion, not enabled in the app.

Offline bounds must be trained without the evaluated cohort. At annotation
time only numerical class/mode bounds and a chemical catalog are needed; no
manual feature identities, curated workbooks or dataset names are consulted.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from annotator.composition_ranges import (
    ClassCompositionRange,
    CompositionRange,
    CompositionRangeSettings,
    DEFAULT_COMPOSITION_RANGE_SETTINGS,
    canonical_lipid_class,
    major_group_for_class,
)
from .class_candidates import _composition_allowed

FORMAT = "laser-composition-envelope-v1"
CH2 = 14.015650
H2 = 2.015650
COMPOSITION = re.compile(r"^(?P<head>[^()]+\((?:[dtm]|[OP]-|O\+P-)?)(\d+):(\d+)\)$")


def _default_bounds(cls):
    settings = DEFAULT_COMPOSITION_RANGE_SETTINGS
    rule = settings.class_lookup().get(cls)
    if rule is None:
        return None
    lo, hi, dlo, dhi = (
        rule.limits.carbon_min,
        rule.limits.carbon_max,
        rule.limits.db_min,
        rule.limits.db_max,
    )
    group = settings.group_lookup().get(rule.major_group)
    if group is not None and settings.use_group_ranges and rule.use_group_range:
        lo, hi = max(lo, group.carbon_min), min(hi, group.carbon_max)
        dlo, dhi = max(dlo, group.db_min), min(dhi, group.db_max)
    return lo, hi, dlo, dhi


def fit_composition_envelopes(
    observations, *, minimum_cohorts=3, carbon_padding=2, db_padding=1
):
    """Explicit offline option: widen defaults only from TRAINING compositions.

    The trained envelope never narrows a current default. Small fixed padding
    permits unseen neighboring sums; the classifier's smoothed priors remain
    soft evidence. Existing user-specified limits are not implicitly overridden:
    the caller must explicitly select the returned experimental settings.
    """
    required = {"cohort", "stream", "class", "carbon", "db"}
    if not required.issubset(observations):
        raise ValueError("Composition observations need cohort/mode/class/C/DB")
    if minimum_cohorts < 2 or carbon_padding < 0 or db_padding < 0:
        raise ValueError("Invalid composition support or padding")
    data = observations[list(sorted(required))].copy()
    data["class"] = data["class"].map(canonical_lipid_class)
    values = data[["carbon", "db"]].apply(pd.to_numeric, errors="coerce")
    if (
        not np.isfinite(values).all().all()
        or not (values.carbon > 0).all()
        or not (values.db >= 0).all()
    ):
        raise ValueError("Invalid chemical composition observations")
    if not values.eq(np.floor(values)).all().all():
        raise ValueError("Composition axes must be integers")
    data[["carbon", "db"]] = values.astype(int)
    output = {}
    for (stream, cls), group in data.groupby(["stream", "class"], sort=True):
        cohorts = group.cohort.nunique()
        if cohorts < minimum_cohorts:
            continue
        bounds = _default_bounds(cls)
        lo = max(1, int(group.carbon.min()) - carbon_padding)
        hi = int(group.carbon.max()) + carbon_padding
        dlo = max(0, int(group.db.min()) - db_padding)
        dhi = int(group.db.max()) + db_padding
        if bounds is not None:
            lo, hi, dlo, dhi = (
                min(lo, bounds[0]),
                max(hi, bounds[1]),
                min(dlo, bounds[2]),
                max(dhi, bounds[3]),
            )
        output[stream + "|" + cls] = dict(
            carbon_min=lo,
            carbon_max=hi,
            db_min=dlo,
            db_max=dhi,
            training_cohorts=int(cohorts),
            training_compositions=len(group[["carbon", "db"]].drop_duplicates()),
        )
    return dict(
        format=FORMAT,
        models=output,
        minimum_cohorts=minimum_cohorts,
        carbon_padding=carbon_padding,
        db_padding=db_padding,
    )


def envelope_range_settings(envelope, stream):
    if envelope.get("format") != FORMAT:
        raise ValueError("Unsupported composition envelope")
    rules = []
    for key, model in envelope["models"].items():
        phase, cls = key.split("|", 1)
        if phase != stream:
            continue
        limits = CompositionRange(
            **{k: model[k] for k in ("carbon_min", "carbon_max", "db_min", "db_max")}
        )
        rules.append(
            ClassCompositionRange(
                cls, major_group_for_class(cls), limits, use_group_range=False
            )
        )
    # Retain original defaults for unsupported classes, including group bounds.
    defaults = DEFAULT_COMPOSITION_RANGE_SETTINGS
    byclass = {r.lipid_class: r for r in rules}
    for cls, rule in defaults.class_lookup().items():
        if cls not in byclass:
            bounds = _default_bounds(cls)
            if bounds is not None:
                byclass[cls] = ClassCompositionRange(
                    cls,
                    rule.major_group,
                    CompositionRange(*bounds),
                    use_group_range=False,
                )
    return CompositionRangeSettings(
        enabled=True,
        use_group_ranges=False,
        group_ranges=(),
        class_ranges=tuple(byclass.values()),
        source="explicit grouped-training experimental envelope",
    )


def _charge(ion):
    match = re.search(r"\](?:\((\d+)[+-]\)|(\d+)[+-])$", str(ion))
    return int(next(v for v in match.groups() if v)) if match else 1


def compile_additional_targets(reference, envelope, *, stream, declared_standards=None):
    """Fill absent/previously filtered C:DB targets from verified chemical axes.

    A C:DB mass lattice is accepted only if all existing monoisotopic reference
    masses agree with CH2 / minus-H2 increments (charge-aware). Labels and
    structural subclasses stay distinct. Isoprenoid/non-C:DB names are not
    extrapolated. Mass-identical declared spikes remain standards, while a
    labelled spike cannot suppress its unlabelled endogenous counterpart.
    Returned rows are NEW proposals, never accepted annotations or legacy codes.
    Supply the separate declared-standard catalog: an inherited reference can
    omit ALL standard rows. Legacy callers without that catalog are explicitly
    flagged in the audit and cannot claim complete spike protection.
    """
    required = {"name", "class", "adduct", "carbon", "db", "ref_mz", "ref_rt"}
    if not required.issubset(reference):
        raise ValueError("Chemical reference missing required coordinates")
    settings = envelope_range_settings(envelope, stream)
    rows = []
    audit = []
    standard = reference.name.astype(str).str.endswith(" (IS)")
    spikes = reference[standard]
    protection_input = "embedded_reference" if len(spikes) else "unavailable"
    if declared_standards is not None:
        if not {"name", "adduct", "ref_mz"}.issubset(declared_standards):
            raise ValueError("Declared-standard protection needs names/adducts/masses")
        if not declared_standards.name.astype(str).str.endswith(" (IS)").all():
            raise ValueError("Declared-standard catalog must contain only standards")
        spikes = pd.concat([spikes, declared_standards], ignore_index=True)
        protection_input = "explicit_declared_catalog"
    endogenous = reference[~standard].copy()
    endogenous["canonical_class"] = endogenous["class"].map(canonical_lipid_class)
    for (cls, ion), group in endogenous.groupby(
        ["canonical_class", "adduct"], sort=True
    ):
        model = envelope["models"].get(stream + "|" + cls)
        if model is None:
            continue
        representative = group.iloc[0]
        template = COMPOSITION.fullmatch(str(representative["name"]))
        if template is None:
            continue
        charge = _charge(ion)
        intercept = (
            group.ref_mz - (group.carbon * CH2 - group.db * H2) / charge
        ).to_numpy(float)
        finite = intercept[np.isfinite(intercept)]
        if len(finite) < 2:
            continue
        neutral_offset = float(np.median(finite))
        if np.max(np.abs(finite - neutral_offset)) > 2e-5:
            audit.append(
                dict(
                    stream=stream,
                    lipid_class=cls,
                    adduct=ion,
                    status="inconsistent_mass_lattice",
                    added=0,
                )
            )
            continue
        existing = {
            r["name"]: r for r in group.drop_duplicates("name").to_dict("records")
        }
        declared = spikes[spikes.adduct.eq(ion)]
        spike_masses = {}
        for spike in declared.itertuples():
            spike_masses.setdefault(str(spike.name).removesuffix(" (IS)"), []).append(
                float(spike.ref_mz)
            )
        added = 0
        blocked = 0
        for carbon in range(model["carbon_min"], model["carbon_max"] + 1):
            for db in range(model["db_min"], min(model["db_max"], carbon // 2) + 1):
                name = f"{template.group('head')}{carbon}:{db})"
                previous = existing.get(name)
                if previous is not None and _composition_allowed(
                    DEFAULT_COMPOSITION_RANGE_SETTINGS, cls, carbon, db, name
                ):
                    continue
                mz = neutral_offset + (carbon * CH2 - db * H2) / charge
                if any(
                    abs(mz - s) <= max(1e-5, abs(mz) * 0.05e-6)
                    for s in spike_masses.get(name, [])
                ):
                    blocked += 1
                    continue
                row = (
                    dict(previous) if previous is not None else representative.to_dict()
                )
                row.update(
                    name=name,
                    carbon=carbon,
                    db=db,
                    ref_mz=mz,
                    ref_rt=previous["ref_rt"] if previous is not None else np.nan,
                    matlab_code=np.nan,
                    chemical_target_generated=True,
                    chemical_target_origin="group_trained_composition_envelope",
                )
                row.pop("canonical_class", None)
                for key in ("np_unresolved_source_name", "np_unresolved_source_class"):
                    if key in row:
                        row[key] = ""
                rows.append(row)
                added += 1
        audit.append(
            dict(
                stream=stream,
                lipid_class=cls,
                adduct=ion,
                status="compiled",
                added=added,
                mass_identical_standards_blocked=blocked,
                standard_protection_input=protection_input,
                charge=charge,
                maximum_lattice_mass_residual_Da=float(
                    np.max(np.abs(finite - neutral_offset))
                ),
            )
        )
    return (
        pd.DataFrame(
            rows,
            columns=list(
                dict.fromkeys(
                    [
                        *reference.columns,
                        "matlab_code",
                        "chemical_target_generated",
                        "chemical_target_origin",
                    ]
                )
            ),
        ),
        settings,
        pd.DataFrame(audit),
    )
