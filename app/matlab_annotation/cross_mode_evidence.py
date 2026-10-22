"""Independent automatic evidence from other measurement streams.

This helper never reads curation or guesses specimen pairing from filenames or
column order. Profile comparisons use only matching validated biological GROUP
labels, not claimed individual-sample pairs. Positional/ether classes stay
distinct. Counts and residuals are evidence, never proof of identification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from annotator.peak_evidence import sample_roles, _profile_correlation


def _key(frame):
    return pd.Series(
        list(
            zip(
                frame["class"].map(canonical_lipid_class),
                pd.to_numeric(frame.carbon, errors="coerce"),
                pd.to_numeric(frame.db, errors="coerce"),
            )
        ),
        index=frame.index,
    )


def _strong(automatic):
    if automatic.empty:
        return automatic.copy()
    x = automatic.copy()
    std = x.get("is_is", pd.Series(False, index=x.index)).fillna(False).astype(bool)
    std |= x.name.astype(str).str.endswith(" (IS)") | x.name.astype(str).str.startswith(
        "IS: "
    )
    confidence = pd.to_numeric(
        x.get("post_regression_confidence", pd.Series(np.nan, index=x.index)),
        errors="coerce",
    )
    ppm = pd.to_numeric(x.ppm_error, errors="coerce")
    x = x[~std & confidence.ge(0.9) & ppm.abs().lt(1.5)].copy()
    x["key"] = _key(x)
    x["seconds"] = pd.to_numeric(x.feat_rt, errors="coerce") * 60
    x = x[np.isfinite(x[["carbon", "db", "seconds"]]).all(axis=1)]
    # Multiple adducts do not multiply support. Disagreeing peaks cannot supply
    # an unambiguous other-stream RT or abundance profile.
    spread = x.groupby(["stream", "key"]).seconds.transform("max") - x.groupby(
        ["stream", "key"]
    ).seconds.transform("min")
    x = x[spread.le(3)].sort_values("feat_intensity", ascending=False, kind="stable")
    return x.drop_duplicates(["stream", "key"])


def _group_profiles(features, requested):
    empty = pd.DataFrame(index=pd.Index(sorted(requested), name="feature_id"))
    if features.attrs.get("sample_group_mapping_status") != "validated":
        return empty
    columns, roles, _ = sample_roles(features)
    mapping = features.attrs.get("sample_group_map", {})
    groups = {}
    for col in columns:
        if roles[col] == "biological" and mapping.get(col):
            groups.setdefault(str(mapping[col]).strip().casefold(), []).append(col)
    subset = features[features.feature_id.astype(str).isin(requested)].copy()
    if subset.feature_id.astype(str).duplicated().any():
        raise ValueError("Original feature IDs must be unique within each stream")
    output = pd.DataFrame(index=subset.feature_id.astype(str))
    for group, cols in groups.items():
        values = subset[cols].apply(pd.to_numeric, errors="coerce")
        values = values.where(np.isfinite(values) & values.ge(0))
        output[group] = values.mean(axis=1).to_numpy()
    return output


def cross_mode_evidence(pool, features_by_stream, automatic):
    """Describe OTHER-stream support using frozen automatic annotations only.

    Same-phase RT offsets require >=8 OTHER shared automatic compositions with
    median absolute spread <=3s and 90th percentile <=6s. Own composition never
    calibrates its RT offset. No RT comparisons across NP/RP phases are made.
    Profiles require >=4 matching validated biological groups; otherwise unknown.
    """
    required = {"stream", "class", "carbon", "db", "feat_feature_id", "feat_rt"}
    if not required.issubset(pool):
        raise ValueError(
            "Cross-mode queries require stream/composition/original feature coordinates"
        )
    out = pd.DataFrame(index=pool.index)
    out["q_cross_mode_supporting_streams"] = 0
    out["q_cross_mode_same_phase_support"] = 0
    for c in [
        "q_cross_mode_best_group_profile_correlation",
        "q_cross_mode_profile_groups",
        "q_cross_mode_RT_residual_seconds",
        "q_cross_mode_RT_anchor_count",
    ]:
        out[c] = np.nan
    strong = _strong(automatic)
    if strong.empty or pool.empty:
        return out
    keys = _key(pool)
    streams = list(features_by_stream)
    profiles, lookup, paired = {}, {}, {}
    for stream in streams:
        part = strong[strong.stream.eq(stream)]
        lookup[stream] = {r.key: r for r in part.itertuples()}
        wanted = set(pool.loc[pool.stream.eq(stream), "feat_feature_id"].astype(str))
        wanted.update(part.feat_feature_id.astype(str))
        profiles[stream] = _group_profiles(features_by_stream[stream], wanted)
    for target in streams:
        for source in streams:
            if target == source or target[:2] != source[:2]:
                continue
            shared = set(lookup[target]) & set(lookup[source])
            paired[(target, source)] = {
                k: lookup[target][k].seconds - lookup[source][k].seconds for k in shared
            }
    # Each original feature/composition query is calculated once, irrespective
    # of repeated target adduct rows in the candidate pool.
    queries = pool.assign(_key=keys, _fid=pool.feat_feature_id.astype(str))
    for (stream, key, fid), indices in queries.groupby(
        ["stream", "_key", "_fid"], sort=False
    ).groups.items():
        support = same_phase = 0
        correlations, residuals = [], []
        target_profile = profiles[stream]
        for other in streams:
            if other == stream or key not in lookup[other]:
                continue
            anchor = lookup[other][key]
            support += 1
            if other[:2] == stream[:2]:
                same_phase += 1
                offsets = np.asarray(
                    [v for k, v in paired[(stream, other)].items() if k != key]
                )
                if len(offsets) >= 8:
                    center = float(np.median(offsets))
                    error = np.abs(offsets - center)
                    if (
                        abs(center) <= 30
                        and np.median(error) <= 3
                        and np.quantile(error, 0.9) <= 6
                    ):
                        observed = float(pool.loc[indices[0], "feat_rt"]) * 60
                        residuals.append(
                            (observed - anchor.seconds - center, len(offsets))
                        )
            other_profile = profiles[other]
            common = sorted(set(target_profile.columns) & set(other_profile.columns))
            other_id = str(anchor.feat_feature_id)
            if (
                len(common) >= 4
                and fid in target_profile.index
                and other_id in other_profile.index
            ):
                corr, n = _profile_correlation(
                    target_profile.loc[fid, common].to_numpy(float),
                    other_profile.loc[other_id, common].to_numpy(float),
                    4,
                )
                if np.isfinite(corr):
                    correlations.append((corr, n))
        out.loc[indices, "q_cross_mode_supporting_streams"] = support
        out.loc[indices, "q_cross_mode_same_phase_support"] = same_phase
        if correlations:
            corr, n = max(correlations)
            out.loc[indices, "q_cross_mode_best_group_profile_correlation"] = corr
            out.loc[indices, "q_cross_mode_profile_groups"] = n
        if residuals:
            residual, n = min(residuals, key=lambda p: abs(p[0]))
            out.loc[indices, "q_cross_mode_RT_residual_seconds"] = residual
            out.loc[indices, "q_cross_mode_RT_anchor_count"] = n
    return out
