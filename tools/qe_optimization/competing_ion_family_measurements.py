"""Comparative native-ion evidence from existing raw43 measurements only.

The opposite-polarity mass sets must be DISJOINT: shared ions do not distinguish
classes, and identical O/P or BMP/PG ion families never create a contrast.
No raw search, score model, identity decision, file access or acceptance gate.
"""

import numpy as np
import pandas as pd

PREFIX = "q_competing_ion_family_"
RAW_PREFIX = "q_raw_counterpart_"
MEASURES = (
    "best_raw_correlation",
    "best_normalized_correlation",
    "best_within_group_correlation",
    "best_leave_one_out_correlation",
    "best_normalized_leave_one_out_correlation",
    "best_log_ratio_mad",
    "best_abs_ppm",
    "best_raw_RT_delta_seconds",
    "best_profile_pairs",
)
CONTRASTS = MEASURES[:5]
FIELDS = (
    "source_configured",
    "alternative_families",
    "measured_alternative_families",
    *("alternative_" + m for m in MEASURES),
    *("own_minus_alternative_" + m for m in CONTRASTS),
)


def signature(row, families):
    from matlab_annotation.raw_counterpart_evidence import _family
    from matlab_annotation.raw_counterpart_rt import template_mass
    from matlab_annotation.standard_rt_library import ion_key
    from matlab_annotation.vendor.compound_model import PROTON_MASS

    family = _family(families, row.stream[:2], row["class"])
    backbone = template_mass(
        float(row.ref_mz),
        ion_key(row.adduct),
        family,
        positive=row.stream.endswith("Pos"),
    )
    if backbone is None:
        return ()
    return tuple(
        sorted(
            {
                abs((backbone + offset) / charge + PROTON_MASS)
                for offset, charge in family.values()
                if charge and (charge > 0) != row.stream.endswith("Pos")
            }
        )
    )


def overlap(a, b):
    return any(abs(x / y - 1) * 1e6 <= 0.05 for x in a for y in b)


def measure(pool, evidence, families, physical):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.raw_counterpart_rt import SUPERVISION

    if SUPERVISION & set(pool):
        raise ValueError("Competing-family descriptors refuse supervision")
    if (
        not pool.index.equals(evidence.index)
        or not pool.index.equals(physical.index)
        or not physical.isin([False, True]).all()
    ):
        raise ValueError("Aligned candidate/measurement/physical tables required")
    if not {
        RAW_PREFIX + f for f in (*MEASURES, "profile_measurement_available")
    }.issubset(evidence):
        raise ValueError("Complete original raw43 numeric evidence required")
    p = pool.assign(_cls=pool["class"].map(canonical_lipid_class))
    signatures = {idx: signature(row, families) for idx, row in p.iterrows()}
    result = pd.DataFrame(np.nan, index=p.index, columns=[PREFIX + f for f in FIELDS])
    result[PREFIX + "source_configured"] = [float(bool(signatures[i])) for i in p.index]
    result[PREFIX + "alternative_families"] = 0.0
    result[PREFIX + "measured_alternative_families"] = 0.0
    links = []
    for (_, _), part in p.groupby(["stream", "feat_feature_id"], sort=False):
        eligible = part.loc[physical.loc[part.index]]
        for idx, row in part.iterrows():
            own = signatures[idx]
            if not physical.loc[idx] or not own:
                continue
            candidates = []
            seen = set()
            for j, other in eligible.iterrows():
                other_sig = signatures[j]
                if other._cls == row._cls or not other_sig or overlap(own, other_sig):
                    continue
                # Quantization is ONLY a deduplication/ranking key. Numeric
                # measurements, masses and reported contrasts remain unrounded.
                key = tuple(round(x, 8) for x in other_sig)
                if key in seen:
                    continue
                seen.add(key)
                links.append(dict(source_index=idx, alternative_index=j))
                if evidence.loc[j, RAW_PREFIX + "profile_measurement_available"] != 1:
                    continue
                raw = float(evidence.loc[j, RAW_PREFIX + "best_raw_correlation"])
                if not np.isfinite(raw):
                    continue
                norm = float(
                    evidence.loc[j, RAW_PREFIX + "best_normalized_correlation"]
                )
                ppm = float(evidence.loc[j, RAW_PREFIX + "best_abs_ppm"])
                rank = (
                    round(raw, 12),
                    round(norm, 12) if np.isfinite(norm) else -2,
                    -round(ppm, 8) if np.isfinite(ppm) else -np.inf,
                )
                candidates.append((rank, j))
            result.loc[idx, PREFIX + "alternative_families"] = len(seen)
            result.loc[idx, PREFIX + "measured_alternative_families"] = len(candidates)
            if not candidates:
                continue
            best = max(candidates, key=lambda item: item[0])[1]
            for field in MEASURES:
                result.loc[idx, PREFIX + "alternative_" + field] = evidence.loc[
                    best, RAW_PREFIX + field
                ]
            for field in CONTRASTS:
                a, b = (float(evidence.loc[i, RAW_PREFIX + field]) for i in (idx, best))
                if np.isfinite(a) and np.isfinite(b):
                    result.loc[idx, PREFIX + "own_minus_alternative_" + field] = a - b
    return result, pd.DataFrame(links, columns=["source_index", "alternative_index"])
