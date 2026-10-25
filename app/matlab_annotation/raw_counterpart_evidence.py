"""Unused raw opposite-polarity measurements, NOT annotation confidence/FDR.

No files, model fits, curation, new candidate identities or default changes.
Unassigned original partner peaks may supply evidence; native review-only ions
are never emitted as full annotation channels by this helper.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from annotator.composition_ranges import canonical_lipid_class
from .adduct_family_evidence import native_adduct_families
from .paired_sample_evidence import _profiles, _comparison, _correlation
from .raw_counterpart_rt import (
    opposite,
    template_mass,
    mapped_counterpart_rt,
    SUPERVISION,
)
from .standard_rt_library import ion_key
from .vendor.compound_model import PROTON_MASS

PREFIX = "q_raw_counterpart_"
FIELDS = (
    "source_configured",
    "pairing_available",
    "expected_native_ions",
    "mass_hit_ions",
    "distinct_mass_hits",
    "paired_bio_specimens",
    "profile_measurement_available",
    "valid_profile_hits",
    "ambiguous_profile_hits",
    "best_raw_correlation",
    "best_normalized_correlation",
    "best_within_group_correlation",
    "best_spearman_correlation",
    "best_leave_one_out_correlation",
    "best_normalized_leave_one_out_correlation",
    "best_log_ratio_location",
    "best_normalized_log_ratio_location",
    "best_log_ratio_mad",
    "best_abs_ppm",
    "best_raw_RT_delta_seconds",
    "best_profile_pairs",
    "best_within_group_dof",
    "profile_margin",
    "RT_map_available",
    "RT_map_anchor_compounds",
    "RT_map_mad_seconds",
    "best_RT_residual_seconds",
    "RT_supported_profile_hits",
    "best_RT_supported_correlation",
    "best_RT_supported_normalized_correlation",
    "best_RT_supported_within_group_correlation",
    "best_RT_supported_abs_ppm",
)


def _family(families, phase, cls):
    cls = canonical_lipid_class(cls)
    value = families.get((phase, cls))
    if value is not None:
        return value
    # NP's existing unresolved O/P reporter is not a new chemical class. Use
    # its branch ion templates ONLY when both complete native definitions agree.
    if phase == "NP" and cls.endswith("[O+P]"):
        head = cls[:-5]
        a, b = families.get((phase, head + "[O]")), families.get((phase, head + "[P]"))
        if a and a == b:
            return a
    return {}


def _minimum_leave_one_out(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 8:
        return np.nan
    n = len(a) - 1
    sx, sy = a.sum() - a, b.sum() - b
    vx = (a * a).sum() - a * a - sx * sx / n
    vy = (b * b).sum() - b * b - sy * sy / n
    denominator = np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
    if not np.isfinite(denominator).all() or (denominator <= 1e-12).any():
        return np.nan
    covariance = (a * b).sum() - a * b - sx * sy / n
    return float(np.clip(covariance / denominator, -1, 1).min())


def _profile_result(a, b, ca, cb, labels):
    result = _comparison(a, b, ca, cb, labels)
    if result is None:
        return None
    valid = np.isfinite(a) & np.isfinite(b)
    x, y = a[valid], b[valid]
    nx, ny = (a - ca)[valid], (b - cb)[valid]
    norm = np.isfinite(nx) & np.isfinite(ny)
    result.update(
        spearman=_correlation(rankdata(x), rankdata(y)),
        leave_one_out=_minimum_leave_one_out(x, y),
        normalized_leave_one_out=_minimum_leave_one_out(nx, ny),
        log_ratio_location=float(np.median(y - x)),
        normalized_log_ratio_location=float(np.median(ny[norm] - nx[norm]))
        if norm.sum() >= 8
        else np.nan,
    )
    return result


def _validate_original_queries(pool, features_by_stream, stream, target_stream):
    required = {"stream", "class", "adduct", "ref_mz", "feat_feature_id", "feat_rt"}
    if (
        not required.issubset(pool)
        or not pool.index.is_unique
        or not pool.stream.eq(stream).all()
    ):
        raise ValueError(
            "Unique explicit same-stream physical candidate queries required"
        )
    if SUPERVISION & set(pool):
        raise ValueError("Raw counterpart measurements refuse curator supervision")
    if stream not in features_by_stream:
        raise ValueError("Original source feature table required")
    source = features_by_stream[stream]
    for name, features in features_by_stream.items():
        if name not in {stream, target_stream}:
            continue
        if (
            not {"feature_id", "mz", "rt", "intensity"}.issubset(features)
            or features.feature_id.astype(str).duplicated().any()
        ):
            raise ValueError("Unique original raw feature IDs/coordinates required")
    source_ids = source.feature_id.astype(str)
    if not set(pool.feat_feature_id.astype(str)).issubset(set(source_ids)):
        raise ValueError("Query original feature IDs absent from source acquisition")
    observed = source.set_index(source_ids, drop=False).loc[
        pool.feat_feature_id.astype(str)
    ]
    for query, raw in (("feat_rt", "rt"), ("feat_mz", "mz")):
        if query in pool:
            np.testing.assert_allclose(
                pool[query].to_numpy(float),
                observed[raw].to_numpy(float),
                rtol=8 * np.finfo(float).eps,
                atol=0,
                equal_nan=True,
            )


def raw_counterpart_evidence(
    pool, features_by_stream, reference_root, specimen_mapping, *, stream, rt_links=None
):
    """32 numeric fields, same-phase OTHERpolarity rawstrict3ppm searches.

    Profiles require explicit paired biological IDs,>=8positive pairs; within-
    group comparisons also require>=4residualDOF. No guessed RT relation: mapped
    RTsupport exists ONLYif independently supplied declared-spike links qualify.
    Unknown RT/profile evidence staysNaN, not fabricated no-partner proof.
    """
    target_stream = opposite(stream)
    _validate_original_queries(pool, features_by_stream, stream, target_stream)
    output = pd.DataFrame(
        np.nan, index=pool.index, columns=[PREFIX + c for c in FIELDS]
    )
    for c in (
        "source_configured",
        "pairing_available",
        "profile_measurement_available",
        "RT_map_available",
    ):
        output[PREFIX + c] = 0.0
    if pool.empty:
        return output
    source = features_by_stream[stream]
    families = native_adduct_families(str(Path(reference_root).resolve()))
    if rt_links is not None and (
        rt_links.get("source_stream") != stream
        or rt_links.get("target_stream") != target_stream
    ):
        raise ValueError(
            "RTlinks cannot cross RP/NP phases or reverse polarity silently"
        )
    target = features_by_stream.get(target_stream)
    if target is not None:
        masses = pd.to_numeric(target.mz, errors="coerce").to_numpy(float)
        times = pd.to_numeric(target.rt, errors="coerce").to_numpy(float) * 60
        intensity = pd.to_numeric(target.intensity, errors="coerce").to_numpy(float)
        valid = np.flatnonzero(
            np.isfinite(masses)
            & np.isfinite(times)
            & np.isfinite(intensity)
            & (masses > 0)
            & (times > 0)
            & (intensity > 0)
        )
        order = valid[np.argsort(masses[valid], kind="stable")]
        sorted_masses = masses[order]
        target_ids = target.feature_id.astype(str).to_numpy()
    plans = {}
    requested_source, requested_target = set(), set()
    for idx, r in pool.iterrows():
        cls, ion, fid = (
            canonical_lipid_class(r["class"]),
            ion_key(r.adduct),
            str(r.feat_feature_id),
        )
        key = (cls, ion, float(r.ref_mz), fid)
        if key not in plans:
            family = _family(families, stream[:2], cls)
            backbone = template_mass(
                float(r.ref_mz), ion, family, positive=stream.endswith("Pos")
            )
            expected = [
                abs((backbone + offset) / charge + PROTON_MASS)
                for offset, charge in family.values()
                if backbone is not None
                and charge
                and (charge > 0) == target_stream.endswith("Pos")
            ]
            hits = {}
            ion_hits = 0
            if target is not None:
                for mz in expected:
                    lo = np.searchsorted(sorted_masses, mz * (1 - 3e-6))
                    hi = np.searchsorted(sorted_masses, mz * (1 + 3e-6), side="right")
                    found = order[lo:hi]
                    found = found[np.abs((masses[found] / mz - 1) * 1e6) < 3]
                    if len(found):
                        ion_hits += 1
                    for i in found:
                        ppm = abs((masses[i] / mz - 1) * 1e6)
                        if int(i) not in hits or ppm < hits[int(i)]:
                            hits[int(i)] = float(ppm)
            plans[key] = dict(
                indices=[],
                configured=backbone is not None,
                expected=len(expected),
                hits=hits,
                ion_hits=ion_hits,
                source_seconds=float(r.feat_rt) * 60,
            )
            requested_source.add(fid)
            if target is not None:
                requested_target.update(target_ids[list(hits)])
        plans[key]["indices"].append(idx)
    available = (
        target is not None
        and bool(specimen_mapping.get(stream))
        and bool(specimen_mapping.get(target_stream))
    )
    if available:
        source_profiles, source_centers, source_labels = _profiles(
            source, requested_source, specimen_mapping[stream]
        )
        target_profiles, target_centers, target_labels = _profiles(
            target, requested_target, specimen_mapping[target_stream]
        )
        shared = sorted(set(source_labels) & set(target_labels))
        if any(source_labels[s] != target_labels[s] for s in shared):
            raise ValueError("Explicit paired biological specimen groups disagree")
        sx, sy = (
            source_centers.reindex(shared).to_numpy(float),
            target_centers.reindex(shared).to_numpy(float),
        )
        labels = [source_labels[s] for s in shared]
    else:
        shared = []
    for (cls, _ion, _reference, fid), plan in plans.items():
        row = {PREFIX + c: np.nan for c in FIELDS}
        row.update(
            {
                PREFIX + "source_configured": float(plan["configured"]),
                PREFIX + "pairing_available": float(available),
                PREFIX + "expected_native_ions": float(plan["expected"]),
                PREFIX + "profile_measurement_available": 0.0,
                PREFIX + "RT_map_available": 0.0,
            }
        )
        mapping = (
            mapped_counterpart_rt(rt_links, plan["source_seconds"], cls)
            if rt_links is not None
            else None
        )
        if mapping is not None:
            expected_rt, map_info = mapping
            row.update(
                {
                    PREFIX + "RT_map_available": 1.0,
                    PREFIX + "RT_map_anchor_compounds": float(
                        map_info["anchor_compounds"]
                    ),
                    PREFIX + "RT_map_mad_seconds": float(map_info["mad_seconds"]),
                }
            )
        if target is not None:
            row[PREFIX + "mass_hit_ions"] = float(plan["ion_hits"])
            row[PREFIX + "distinct_mass_hits"] = float(len(plan["hits"]))
        if available:
            row[PREFIX + "paired_bio_specimens"] = float(len(shared))
        comparisons = []
        if available and len(shared) >= 8 and fid in source_profiles.index:
            a = source_profiles.loc[fid, shared].to_numpy(float)
            for i, ppm in plan["hits"].items():
                other_id = target_ids[i]
                if other_id not in target_profiles.index:
                    continue
                result = _profile_result(
                    a,
                    target_profiles.loc[other_id, shared].to_numpy(float),
                    sx,
                    sy,
                    labels,
                )
                if result is None:
                    continue
                result.update(
                    partner_index=i,
                    ppm=ppm,
                    raw_rt_delta=float(times[i] - plan["source_seconds"]),
                    rt_residual=float(times[i] - expected_rt)
                    if mapping is not None
                    else np.nan,
                )
                comparisons.append(result)
        if comparisons:
            comparisons.sort(
                key=lambda r: (
                    r["raw"],
                    r["normalized"] if np.isfinite(r["normalized"]) else -2.0,
                    -r["ppm"],
                ),
                reverse=True,
            )
            best = comparisons[0]
            row.update(
                {
                    PREFIX + "profile_measurement_available": 1.0,
                    PREFIX + "valid_profile_hits": float(len(comparisons)),
                    PREFIX + "ambiguous_profile_hits": float(
                        max(0, sum(r["raw"] >= 0.8 for r in comparisons) - 1)
                    ),
                }
            )
            for field, value in (
                ("best_raw_correlation", best["raw"]),
                ("best_normalized_correlation", best["normalized"]),
                ("best_within_group_correlation", best["within"]),
                ("best_spearman_correlation", best["spearman"]),
                ("best_leave_one_out_correlation", best["leave_one_out"]),
                (
                    "best_normalized_leave_one_out_correlation",
                    best["normalized_leave_one_out"],
                ),
                ("best_log_ratio_location", best["log_ratio_location"]),
                (
                    "best_normalized_log_ratio_location",
                    best["normalized_log_ratio_location"],
                ),
                ("best_log_ratio_mad", best["mad"]),
                ("best_abs_ppm", best["ppm"]),
                ("best_raw_RT_delta_seconds", best["raw_rt_delta"]),
                ("best_profile_pairs", best["n"]),
                ("best_within_group_dof", best["dof"]),
            ):
                row[PREFIX + field] = float(value)
            if len(comparisons) > 1:
                row[PREFIX + "profile_margin"] = float(
                    best["raw"] - comparisons[1]["raw"]
                )
            if mapping is not None:
                row[PREFIX + "best_RT_residual_seconds"] = best["rt_residual"]
                supported = [r for r in comparisons if abs(r["rt_residual"]) <= 10]
                row[PREFIX + "RT_supported_profile_hits"] = float(len(supported))
                if supported:
                    rtbest = supported[0]
                    for field, value in (
                        ("best_RT_supported_correlation", rtbest["raw"]),
                        (
                            "best_RT_supported_normalized_correlation",
                            rtbest["normalized"],
                        ),
                        (
                            "best_RT_supported_within_group_correlation",
                            rtbest["within"],
                        ),
                        ("best_RT_supported_abs_ppm", rtbest["ppm"]),
                    ):
                        row[PREFIX + field] = float(value)
        output.loc[plan["indices"], :] = np.tile(
            [row[c] for c in output.columns], (len(plan["indices"]), 1)
        )
    return output.replace([np.inf, -np.inf], np.nan)
