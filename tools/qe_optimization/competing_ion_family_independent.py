"""Scalar independent expected-ion-set and comparative-evidence reconstruction."""

import math

import numpy as np
import pandas as pd


def independent_signature(row, families):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.standard_rt_library import ion_key
    from matlab_annotation.vendor.compound_model import PROTON_MASS

    cls, phase = canonical_lipid_class(row["class"]), row.stream[:2]
    family = families.get((phase, cls), {})
    if not family and phase == "NP" and cls.endswith("[O+P]"):
        left, right = (
            families.get((phase, cls[:-5] + suffix), {}) for suffix in ("[O]", "[P]")
        )
        if left and left == right:
            family = left
    source = family.get(ion_key(row.adduct))
    if source is None:
        return ()
    addition, z = source
    if z == 0 or (z > 0) != row.stream.endswith("Pos"):
        raise ValueError("Independent native source polarity mismatch")
    neutral = abs(z) * float(row.ref_mz) - z * PROTON_MASS - addition
    masses = []
    for delta, charge in family.values():
        if charge == 0 or (charge > 0) == (z > 0):
            continue
        masses.append(
            (neutral + delta) / abs(charge) + math.copysign(PROTON_MASS, charge)
        )
    return tuple(sorted(set(masses)))


def independent_measure(
    pool, evidence, families, physical, fields, measures, contrasts, prefix, raw_prefix
):
    from annotator.composition_ranges import canonical_lipid_class

    signatures = {i: independent_signature(row, families) for i, row in pool.iterrows()}
    entries = [(i, row) for i, row in pool.iterrows()]
    results, links = [], []
    by_feature = {}
    for i, row in entries:
        by_feature.setdefault((row.stream, str(row.feat_feature_id)), []).append(
            (i, row)
        )
    for i, row in entries:
        values = dict.fromkeys(fields, np.nan)
        own = signatures[i]
        values.update(
            source_configured=float(bool(own)),
            alternative_families=0.0,
            measured_alternative_families=0.0,
        )
        options, seen = [], set()
        if physical.loc[i] and own:
            for j, other in by_feature[(row.stream, str(row.feat_feature_id))]:
                if not physical.loc[j] or canonical_lipid_class(
                    other["class"]
                ) == canonical_lipid_class(row["class"]):
                    continue
                sig = signatures[j]
                if not sig:
                    continue
                shared = False
                for a in own:
                    for b in sig:
                        if abs((a - b) / b * 1e6) <= 0.05:
                            shared = True
                if shared:
                    continue
                key = tuple(float(f"{a:.8f}") for a in sig)
                if key in seen:
                    continue
                seen.add(key)
                links.append(dict(source_index=i, alternative_index=j))
                if evidence.loc[j, raw_prefix + "profile_measurement_available"] != 1:
                    continue
                raw = float(evidence.loc[j, raw_prefix + "best_raw_correlation"])
                if math.isfinite(raw):
                    normalized = float(
                        evidence.loc[j, raw_prefix + "best_normalized_correlation"]
                    )
                    ppm = float(evidence.loc[j, raw_prefix + "best_abs_ppm"])
                    options.append(
                        (
                            float(f"{raw:.12f}"),
                            float(f"{normalized:.12f}")
                            if math.isfinite(normalized)
                            else -2,
                            -float(f"{ppm:.8f}") if math.isfinite(ppm) else -math.inf,
                            j,
                        )
                    )
            values["alternative_families"] = float(len(seen))
            values["measured_alternative_families"] = float(len(options))
            # Stable tie resolution, not row-index ordering.
            if options:
                winner = sorted(options, key=lambda item: item[:3], reverse=True)[0][3]
                for field in measures:
                    values["alternative_" + field] = float(
                        evidence.loc[winner, raw_prefix + field]
                    )
                for field in contrasts:
                    a, b = (
                        float(evidence.loc[i, raw_prefix + field]),
                        float(evidence.loc[winner, raw_prefix + field]),
                    )
                    if math.isfinite(a) and math.isfinite(b):
                        values["own_minus_alternative_" + field] = a - b
        results.append(values)
    return pd.DataFrame(results, index=pool.index).add_prefix(prefix), pd.DataFrame(
        links, columns=["source_index", "alternative_index"]
    )
