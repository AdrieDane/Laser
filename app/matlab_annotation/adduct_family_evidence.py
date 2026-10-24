"""Research-only, annotation-adduct-aware companion-ion measurements.

Native class/phase adduct chemistry supplies mass differences; original
same-stream profiles supply evidence. Review-only ions stay review evidence,
never new accepted identities. No annotations, labels, RT models or file writes.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from annotator.peak_evidence import sample_roles, _matrix, _median, _profile_correlation
from .decode import CodeDecoder
from .standard_rt_library import ion_key
from .vendor.compound_model import phase_composition, PROTON_MASS
from .xcms_project import load_bundled_compound_catalog

PREFIX = "q_adduct_family_"
COLUMNS = [
    "source_configured",
    "expected_other_ions",
    "nearby_other_ions",
    "profile_supported_ions",
    "ambiguous_other_ions",
    "best_correlation",
    "best_RT_seconds",
    "best_abs_ppm",
    "best_log_intensity_ratio",
    "best_profile_pairs",
    "source_abs_charge",
    "biological_profiles",
]


@dataclass(frozen=True)
class _Context:
    compounds: object


@lru_cache(maxsize=4)
def native_adduct_families(reference_root):
    catalog = load_bundled_compound_catalog(reference_root)
    context, families = _Context(catalog), {}
    for phase in ("RP", "NP"):
        for compound, attr in enumerate(catalog.composition_attr, start=1):
            cls = canonical_lipid_class(
                CodeDecoder(context, compound, phase).lipid_class
            )
            try:
                composition = phase_composition(attr, phase)
            except ValueError:
                continue
            values = families.setdefault((phase, cls), {})
            for ion, mass, charge in zip(
                composition.text[0],
                composition.mass[0],
                composition.charges,
                strict=True,
            ):
                key = ion_key(ion)
                chemical = (float(mass), int(charge))
                if key in values and values[key] != chemical:
                    raise ValueError("Native adduct definitions disagree")
                values[key] = chemical
    return families


def adduct_family_evidence(pool, features, reference_root, *, stream):
    """Twelve numeric measurements per candidate; same3ppm/2s partner search.

    Candidate m/z determines neutral mass using its ACTUAL adduct/charge.
    Companion m/z comes from the other native ions of the same class/phase.
    Each companion ion counts at most once; the same original feature may never
    serve as its own companion. >=4 biological (or explicit unknown-role)
    samples required for correlation; QC/blank-only data cannot supply it.
    No claimed EIC shape, cross-acquisition sample pairing or lipid identity.
    """
    if stream not in {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}:
        raise ValueError("Explicit same-stream acquisition mode required")
    required = {"stream", "class", "adduct", "ref_mz", "feat_feature_id"}
    if not required.issubset(pool) or not pool.index.is_unique:
        raise ValueError("Unique physical candidate rows with adduct/mass required")
    if len(pool) and not pool.stream.eq(stream).all():
        raise ValueError("Companion evidence cannot combine acquisition streams")
    if not {"feature_id", "mz", "rt"}.issubset(features):
        raise ValueError("Original raw feature IDs/mass/RT required")
    if features.feature_id.astype(str).duplicated().any():
        raise ValueError("Original feature IDs must be unique")
    if {"training_positive", "training_label_known", "same_manual_peak"} & set(pool):
        raise ValueError("Only label-free physical queries are permitted")
    output = pd.DataFrame(
        np.nan, index=pool.index, columns=[PREFIX + c for c in COLUMNS]
    )
    for c in COLUMNS[:5]:
        output[PREFIX + c] = 0.0
    if pool.empty:
        return output
    families = native_adduct_families(str(Path(reference_root).resolve()))
    ids = features.feature_id.astype(str).to_numpy()
    by_id = {fid: i for i, fid in enumerate(ids)}
    if not set(pool.feat_feature_id.astype(str)).issubset(by_id):
        raise ValueError("Query features are absent from this acquisition")
    mz = pd.to_numeric(features.mz, errors="coerce").to_numpy(float)
    rt = pd.to_numeric(features.rt, errors="coerce").to_numpy(float)
    valid = np.flatnonzero(np.isfinite(mz) & np.isfinite(rt) & (mz > 0))
    ordered = valid[np.argsort(mz[valid], kind="stable")]
    sorted_mass = mz[ordered]
    samples, roles, _ = sample_roles(features)
    bio = [c for c in samples if roles[c] == "biological"]
    columns = bio or [c for c in samples if roles[c] == "unknown"]
    profiles = _matrix(features, columns)
    medians = _median(profiles)
    output[PREFIX + "biological_profiles"] = float(bool(bio))
    memo = {}
    for idx, cls, ion, theoretical, fid in zip(
        pool.index,
        pool["class"].map(canonical_lipid_class),
        pool.adduct.map(ion_key),
        pd.to_numeric(pool.ref_mz, errors="coerce"),
        pool.feat_feature_id.astype(str),
        strict=True,
    ):
        key = (cls, ion, float(theoretical), fid)
        if key in memo:
            output.loc[idx] = memo[key]
            continue
        family = families.get((stream[:2], cls), {})
        if ion not in family or not np.isfinite(theoretical) or theoretical <= 0:
            memo[key] = output.loc[idx].to_numpy()
            continue
        offset, charge = family[ion]
        if not charge or (charge > 0) != stream.endswith("Pos"):
            raise ValueError("Candidate adduct and acquisition polarity disagree")
        neutral = charge * (np.sign(charge) * theoretical - PROTON_MASS) - offset
        source = by_id[fid]
        row = output.loc[idx].to_dict()
        row[PREFIX + "source_configured"] = 1.0
        row[PREFIX + "source_abs_charge"] = float(abs(charge))
        partners, best = [], None
        for partner_ion, (partner_offset, partner_charge) in family.items():
            if partner_ion == ion or (partner_charge > 0) != (charge > 0):
                continue
            row[PREFIX + "expected_other_ions"] += 1.0
            expected = abs((neutral + partner_offset) / partner_charge + PROTON_MASS)
            if not np.isfinite(expected) or expected <= 0:
                continue
            lo = np.searchsorted(sorted_mass, expected * (1 - 3e-6))
            hi = np.searchsorted(sorted_mass, expected * (1 + 3e-6), side="right")
            hits = ordered[lo:hi]
            hits = hits[(hits != source) & (np.abs(rt[hits] - rt[source]) * 60 <= 2)]
            hits = hits[np.abs((mz[hits] / expected - 1) * 1e6) < 3]
            if len(hits):
                row[PREFIX + "nearby_other_ions"] += 1.0
            if len(hits) > 1:
                row[PREFIX + "ambiguous_other_ions"] += 1.0
            ion_best = None
            for j in hits:
                corr, count = _profile_correlation(profiles[source], profiles[j], 4)
                if not np.isfinite(corr):
                    continue
                mass_error = abs((mz[j] / expected - 1) * 1e6)
                seconds = abs(rt[j] - rt[source]) * 60
                rank = (
                    corr,
                    -seconds,
                    -mass_error,
                    medians[j] if np.isfinite(medians[j]) else -1,
                )
                if ion_best is None or rank > ion_best[0]:
                    ion_best = (rank, j, count, mass_error, seconds)
            if ion_best is not None:
                partners.append(ion_best)
                if best is None or ion_best[0] > best[0]:
                    best = ion_best
        # One original partner feature may not inflate the support count if
        # multiple catalogue channels happen to share a mass.
        row[PREFIX + "profile_supported_ions"] = float(
            len({p[1] for p in partners if p[0][0] >= 0.9})
        )
        if best is not None:
            _, partner, count, mass_error, seconds = best
            row.update(
                {
                    PREFIX + "best_correlation": best[0][0],
                    PREFIX + "best_RT_seconds": seconds,
                    PREFIX + "best_abs_ppm": mass_error,
                    PREFIX + "best_log_intensity_ratio": np.log1p(medians[partner])
                    - np.log1p(medians[source]),
                    PREFIX + "best_profile_pairs": float(count),
                }
            )
        output.loc[idx] = pd.Series(row)
        memo[key] = output.loc[idx].to_numpy()
    return output.replace([np.inf, -np.inf], np.nan)
