"""Research-only abundance evidence using EXPLICIT cross-mode specimen links.

No annotation decisions, filename-offset guesses, or curated labels here. The
caller validates acquisition metadata with specimen_pairs first. Technical
replicates count once and blank/QC/standard samples never supply correlations.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from annotator.peak_evidence import sample_roles
from matlab_annotation.cross_mode_evidence import _key, _strong
from matlab_annotation.specimen_pairs import _group


FIELDS = (
    "q_paired_supporting_streams",
    "q_paired_best_log_correlation",
    "q_paired_median_log_correlation",
    "q_paired_best_normalized_correlation",
    "q_paired_best_within_group_correlation",
    "q_paired_best_log_ratio_mad",
    "q_paired_profile_specimens",
    "q_paired_within_group_dof",
)


def _profiles(features, requested, mapping):
    columns, roles, _ = sample_roles(features)
    groups = features.attrs.get("sample_group_map", {})
    specimens, labels = {}, {}
    for col in columns:
        if col not in mapping or roles[col] != "biological":
            continue
        sid = str(mapping[col])
        label = _group(groups.get(col, ""))
        if not label:
            raise ValueError("Paired biological specimen needs a validated group")
        if sid in labels and labels[sid] != label:
            raise ValueError("Paired specimen has conflicting groups")
        labels[sid] = label
        specimens.setdefault(sid, []).append(col)
    subset = features[features.feature_id.astype(str).isin(requested)]
    if subset.feature_id.astype(str).duplicated().any():
        raise ValueError("Original feature IDs must be unique per stream")
    profiles = pd.DataFrame(index=subset.feature_id.astype(str))
    centers = {}
    for sid, cols in specimens.items():
        values = subset[cols].apply(pd.to_numeric, errors="coerce")
        values = values.where(np.isfinite(values) & values.gt(0))
        # Average replicate intensities BEFORE logarithm: each physical specimen
        # supplies one observation, irrespective of acquisition multiplicity.
        profiles[sid] = np.log(values.mean(axis=1).to_numpy())
        raw = features[cols].apply(pd.to_numeric, errors="coerce")
        raw = raw.where(np.isfinite(raw) & raw.gt(0)).mean(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            centers[sid] = float(np.nanmedian(np.log(raw.to_numpy())))
    return profiles, pd.Series(centers, dtype=float), labels


def _correlation(a, b):
    if np.std(a) < 0.05 or np.std(b) < 0.05:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _comparison(a, b, ca, cb, labels):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 8:
        return None
    a, b = a[valid], b[valid]
    ca, cb = ca[valid], cb[valid]
    labels = np.asarray(labels)[valid]
    raw = _correlation(a, b)
    if not np.isfinite(raw):
        return None
    centered = np.isfinite(ca) & np.isfinite(cb)
    norm = (
        _correlation((a - ca)[centered], (b - cb)[centered])
        if centered.sum() >= 8
        else np.nan
    )
    ar, br, dof = [], [], 0
    for label in sorted(set(labels)):
        group = labels == label
        if group.sum() < 2:
            continue
        ar.extend(a[group] - a[group].mean())
        br.extend(b[group] - b[group].mean())
        dof += int(group.sum()) - 1
    within = _correlation(ar, br) if len(ar) >= 8 and dof >= 4 else np.nan
    ratio = a - b
    return dict(raw=raw, normalized=norm, within=within, dof=dof,
                mad=float(np.median(np.abs(ratio - np.median(ratio)))), n=len(a))


def paired_sample_evidence(pool, features_by_stream, automatic, specimen_mapping):
    """Eight OTHER-stream descriptors; unknown stays NaN, not invented zero.

    Exact canonical class/C/DB automatic anchors (baseline confidence>=.9,
    absppm<1.5, disagreeing adduct RTs withheld). No self-stream support or RT
    transfers. At least8 positive paired biological specimens with variable
    log profiles; within-group comparisons also need >=4 residual degrees of
    freedom. Normalization uses each sample's raw global log-intensity median,
    not curated annotations or target-dependent normalization.
    """
    required = {"stream", "class", "carbon", "db", "feat_feature_id"}
    if not required.issubset(pool):
        raise ValueError("Paired evidence requires original feature/composition queries")
    out = pd.DataFrame(np.nan, index=pool.index, columns=FIELDS)
    strong = _strong(automatic)
    if pool.empty or strong.empty or not specimen_mapping:
        return out
    profiles, centers, labels, lookup = {}, {}, {}, {}
    for stream, features in features_by_stream.items():
        part = strong[strong.stream.eq(stream)]
        lookup[stream] = {r.key: r for r in part.itertuples()}
        wanted = set(pool.loc[pool.stream.eq(stream), "feat_feature_id"].astype(str))
        wanted.update(part.feat_feature_id.astype(str))
        profiles[stream], centers[stream], labels[stream] = _profiles(
            features, wanted, specimen_mapping.get(stream, {})
        )
    queries = pool.assign(_key=_key(pool), _fid=pool.feat_feature_id.astype(str))
    for (stream, key, fid), indices in queries.groupby(
        ["stream", "_key", "_fid"], sort=False
    ).groups.items():
        if not labels[stream] or fid not in profiles[stream].index:
            continue
        comparisons = []
        for other in profiles:
            if other == stream or key not in lookup[other]:
                continue
            anchor = str(lookup[other][key].feat_feature_id)
            common = sorted(set(labels[stream]) & set(labels[other]))
            if any(labels[stream][sid] != labels[other][sid] for sid in common):
                raise ValueError("Cross-mode specimen groups disagree")
            if len(common) < 8 or anchor not in profiles[other].index:
                continue
            result = _comparison(
                profiles[stream].loc[fid, common].to_numpy(float),
                profiles[other].loc[anchor, common].to_numpy(float),
                centers[stream].reindex(common).to_numpy(float),
                centers[other].reindex(common).to_numpy(float),
                [labels[stream][sid] for sid in common],
            )
            if result is not None:
                comparisons.append(result)
        out.loc[indices, FIELDS[0]] = len(comparisons)
        if not comparisons:
            continue
        best = max(comparisons, key=lambda x: x["raw"])
        out.loc[indices, FIELDS[1]] = best["raw"]
        out.loc[indices, FIELDS[2]] = np.median([c["raw"] for c in comparisons])
        for field, statistic in [(FIELDS[3], "normalized"), (FIELDS[4], "within")]:
            available = [c[statistic] for c in comparisons if np.isfinite(c[statistic])]
            if available:
                out.loc[indices, field] = max(available)
        out.loc[indices, FIELDS[5]] = min(c["mad"] for c in comparisons)
        out.loc[indices, FIELDS[6]] = best["n"]
        within = [c for c in comparisons if np.isfinite(c["within"])]
        if within:
            out.loc[indices, FIELDS[7]] = max(within, key=lambda c: c["within"])["dof"]
    return out
