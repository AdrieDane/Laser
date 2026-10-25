"""Unused research: resolve COMPLETE isomer clusters, not analytical confidence.

Only numerical score proposals; no annotations/curation/files/default changes.
Maximum old score becomes a feature-detection PROXY, never an identity/FDR
probability. Original pointwise scores must remain separately available.
"""

from itertools import combinations

import numpy as np
import pandas as pd

from .evidence_training import validate_feature_scope
from .isomer_label_model import predict_isomer_label_preferences
from .mode_evidence_experts import MODES

PAIR_COLUMNS = ["cluster_id", "left_index", "right_index", "preference"]


def _physical_clusters(pool):
    """Stable label-independent versions; non-clique mass chains withheld."""
    keys = ["stream", "feat_feature_id", "adduct"]
    if (
        not {*keys, "name", "class", "ref_mz", "is_is"}.issubset(pool)
        or not pool.index.is_unique
    ):
        raise ValueError("Unique explicit original physical label hypotheses required")
    if not pool.is_is.isin([True, False]).all() or not pool.stream.isin(MODES).all():
        raise ValueError("Explicit standard flags and known measurement modes required")
    numeric = pd.to_numeric(pool.ref_mz, errors="coerce")
    if not np.isfinite(numeric).all() or numeric.le(0).any():
        raise ValueError("Finite positive physical theoretical masses required")
    physical = pool.assign(ref_mz=numeric)
    versions = physical.groupby([*keys, "name"])
    inconsistent = (
        versions.ref_mz.transform("max") - versions.ref_mz.transform("min")
    ).gt(1e-8) | versions.is_is.transform("nunique").gt(1)
    unique = physical.assign(_ambiguous_version=inconsistent).drop_duplicates(
        [*keys, "name"], keep="first"
    )
    unique = unique.loc[unique.groupby(keys).name.transform("size").ge(2)]
    clusters = []
    for _, group in unique.groupby(keys, sort=False):
        mass = pd.to_numeric(group.ref_mz, errors="coerce")
        group = group.loc[np.isfinite(mass) & mass.gt(0)].sort_values(
            "ref_mz", kind="stable"
        )
        if len(group) < 2:
            continue
        components = (group.ref_mz.diff().gt(1e-8)).cumsum()
        for _, part in group.groupby(components, sort=False):
            if (
                len(part) < 2
                or part._ambiguous_version.any()
                or float(part.ref_mz.max()) - float(part.ref_mz.min()) > 1e-8
            ):
                continue
            clusters.append(part.index.tolist())
    return clusters


def isomer_label_preferences(
    data, model, *, protected_features_by_stream, protected_masses_by_stream
):
    """Generate blind preferences with complete-cluster chemical/spike guards."""
    validate_feature_scope(
        data,
        model["training_reference_permissions"],
        model["excluded_groups"],
        query_is_training=False,
    )
    if set(protected_features_by_stream) != set(MODES) or set(
        protected_masses_by_stream
    ) != set(MODES):
        raise ValueError("Explicit spike protections for ALL four modes required")
    pool, evidence = data["pool"], data["evidence"]
    masses = {
        s: np.asarray(list(v), float) for s, v in protected_masses_by_stream.items()
    }
    if any(not np.isfinite(v).all() or np.any(v <= 0) for v in masses.values()):
        raise ValueError("Declared spike masses must be finite and positive")
    protected = {s: set(map(str, v)) for s, v in protected_features_by_stream.items()}
    rows = []
    for cluster_id, indices in enumerate(_physical_clusters(pool)):
        group = pool.loc[indices]
        first = group.iloc[0]
        names = group.name.astype(str)
        if (
            group.is_is.any()
            or names.str.startswith("IS: ").any()
            or names.str.endswith(" (IS)").any()
        ):
            continue
        if str(first.feat_feature_id) in protected[first.stream]:
            continue
        measured = group[["feat_mz", "feat_rt", "feat_intensity"]].to_numpy(float)
        if (
            not np.isfinite(measured).all()
            or np.any(measured[:, 0] <= 0)
            or np.any(measured[:, 1:] < 0)
        ):
            continue
        if not np.allclose(measured, measured[0], rtol=8 * np.finfo(float).eps, atol=0):
            raise ValueError(
                "Isomer labels disagree on one original feature's measurements"
            )
        if (
            np.abs(
                (measured[:, 0] - group.ref_mz.to_numpy(float))
                / group.ref_mz.to_numpy(float)
                * 1e6
            )
            >= 3
        ).any():
            continue
        spike_masses = masses[first.stream]
        if (
            len(spike_masses)
            and np.min(np.abs(float(first.feat_mz) - spike_masses) / spike_masses * 1e6)
            < 5.01
        ):
            continue
        ev = evidence.loc[indices]
        if (
            "q_declared_standard_spike_feature" in ev
            and pd.to_numeric(ev.q_declared_standard_spike_feature, errors="coerce")
            .ge(0.5)
            .any()
        ):
            continue
        rt_columns = [
            c
            for c in (
                "absolute_RT_error_seconds",
                "q_absolute_atlas_RT_error_seconds",
                "q_absolute_fitted_RT_error_seconds",
            )
            if c in ev
        ]
        rt = ev[rt_columns].to_numpy(float)
        if (
            not len(rt_columns)
            or not (np.isfinite(rt) & (np.abs(rt) <= 10)).any(axis=1).all()
        ):
            continue
        rows.extend(
            dict(cluster_id=cluster_id, left_index=a, right_index=b, preference=np.nan)
            for a, b in combinations(indices, 2)
        )
    result = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    if len(result):
        result["preference"] = predict_isomer_label_preferences(
            data, model, result.left_index.tolist(), result.right_index.tolist()
        )
    return result


def resolve_isomer_scores(pool, scores, preferences, thresholds):
    """Complete supported unique pairwise winner only; 1 explicitly disables.

    Return adjusted detection proxies plus private decision audit. Both original
    pool and pointwise scores remain unchanged. Non-isomer competitors stay intact.
    """
    if not scores.index.equals(pool.index) or not set(PAIR_COLUMNS).issubset(
        preferences
    ):
        raise ValueError(
            "Aligned base scores and explicit blind pair preferences required"
        )
    if set(thresholds) != set(MODES) or any(
        not np.isfinite(t) or not 0.55 <= t <= 1 for t in thresholds.values()
    ):
        raise ValueError(
            "Explicit [.55,1] conditional thresholds for ALL modes required"
        )
    finite = pd.to_numeric(scores, errors="coerce").dropna()
    if not np.isfinite(finite).all() or ((finite < 0) | (finite > 1)).any():
        raise ValueError("Invalid original pointwise scores")
    values = preferences.preference.dropna()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid conditional isomer preferences")
    output, audit = scores.copy(), []
    clusters = _physical_clusters(pool)
    for cid, pairs in preferences.groupby("cluster_id", sort=False):
        if not isinstance(cid, (int, np.integer)) or not 0 <= cid < len(clusters):
            raise ValueError("Unknown complete physical isomer cluster")
        members = clusters[cid]
        expected = {frozenset(p) for p in combinations(members, 2)}
        actual = [
            frozenset(p)
            for p in pairs[["left_index", "right_index"]].itertuples(
                index=False, name=None
            )
        ]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(
                "Every pair of the COMPLETE physical cluster required exactly once"
            )
        first = pool.loc[members[0]]
        names = pool.loc[members, "name"].astype(str)
        if (
            pool.loc[members, "is_is"].any()
            or names.str.startswith("IS: ").any()
            or names.str.endswith(" (IS)").any()
        ):
            raise ValueError("Cannot resolve standard-label hypotheses")
        threshold = thresholds[first.stream]
        row = dict(
            cluster_id=int(cid),
            stream=first.stream,
            feat_feature_id=str(first.feat_feature_id),
            member_count=len(members),
            winner_index=np.nan,
            conditional_min_preference=np.nan,
            feature_detection_proxy=np.nan,
            threshold=threshold,
            decision="unresolved",
        )
        if threshold == 1:
            row["decision"] = "isomer_resolution_disabled"
        elif pairs.preference.isna().any():
            row["decision"] = "unknown_or_unsupported_pair_in_complete_cluster"
        else:
            support = {i: [] for i in members}
            for pair in pairs.itertuples():
                support[pair.left_index].append(float(pair.preference))
                support[pair.right_index].append(1 - float(pair.preference))
            winners = [i for i, p in support.items() if min(p) >= threshold]
            if len(winners) == 1:
                winner = winners[0]
                # Include ALL original reference versions of these same physical
                # labels, not only the label-independent representatives above.
                versions = (
                    pool.stream.eq(first.stream)
                    & pool.feat_feature_id.astype(str).eq(str(first.feat_feature_id))
                    & pool.adduct.eq(first.adduct)
                    & pool.name.isin(names)
                )
                indices = pool.index[versions]
                old_scores = scores.loc[indices]
                if old_scores.isna().any():
                    row["decision"] = "unscored_original_label_version"
                else:
                    proxy = float(old_scores.max())
                    output.loc[indices] = np.nan
                    winning_versions = indices[
                        pool.loc[indices, "name"].eq(pool.loc[winner, "name"])
                    ]
                    output.loc[winning_versions] = proxy
                    row.update(
                        winner_index=winner,
                        conditional_min_preference=min(support[winner]),
                        feature_detection_proxy=proxy,
                        decision="resolved_isomer_label_detection_proxy",
                    )
            else:
                row["decision"] = "no_unique_supported_pairwise_winner"
        audit.append(row)
    return output, pd.DataFrame(audit)
