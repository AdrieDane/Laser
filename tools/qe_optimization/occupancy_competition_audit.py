"""Read-only competition masks; no annotation assignment or cutoff learning."""

import numpy as np
import pandas as pd

FIELDS = (
    "physical",
    "identity_free",
    "feature_free",
    "identity_best",
    "feature_top",
    "feature_clear",
    "residual_top",
    "residual_clear",
    "blocked_by_called_other_peak",
    "blocker_score_advantage",
    "relaxed_opportunity",
)


def normalized(pool, previous):
    from standard_lipidomics.validation import _normalise, _species
    from matlab_annotation.standard_rt_library import ion_key

    p = pool.copy()
    p["_species"] = p.name.map(_species)
    p["_ion"] = p.adduct.map(ion_key)
    p["_fid"] = p.feat_feature_id.astype(str)
    p["_identity"] = list(zip(p.stream, p._species, p._ion))
    p["_feature"] = list(zip(p.stream, p._fid))
    a = _normalise(previous.assign(adduct=previous.adduct.map(ion_key)), manual=False)
    calls = {}
    for r in a.itertuples():
        calls.setdefault((r.stream, r.species, r.adduct), set()).add(str(r.feature_id))
    return p, calls


def measure(pool, scores, previous, physical):
    from matlab_annotation.raw_counterpart_rt import SUPERVISION

    if SUPERVISION & set(pool) or SUPERVISION & set(previous):
        raise ValueError("Read-only graph refuses supervision")
    if not scores.index.equals(pool.index) or not physical.index.equals(pool.index):
        raise ValueError("Aligned candidate scores/physical masks required")
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError("Finite native scores required")
    p, calls = normalized(pool, previous)
    occupied = {(key[0], fid) for key, fids in calls.items() for fid in fids}
    ordered = (
        p.assign(_score=scores)
        .sort_values(["_score", "feat_intensity"], ascending=False, kind="stable")
        .drop_duplicates(["_identity", "_fid"])
    )
    identity_best = set(ordered.drop_duplicates("_identity").index)
    top = ordered.drop_duplicates("_feature")
    top_by_feature = {r._feature: idx for idx, r in top.iterrows()}
    old_scores = {(r._identity, r._fid): float(r._score) for _, r in ordered.iterrows()}
    next_score = ordered.groupby("_feature")._score.shift(-1)
    clear = set(
        ordered.loc[next_score.isna() | ordered._score.sub(next_score).ge(0.05)].index
    )
    residual = ordered.loc[~ordered._identity.isin(set(calls))]
    residual_top = set(residual.drop_duplicates("_feature").index)
    residual_next = residual.groupby("_feature")._score.shift(-1)
    residual_clear = set(
        residual.loc[
            residual_next.isna() | residual._score.sub(residual_next).ge(0.05)
        ].index
    )
    result = []
    for idx, r in p.iterrows():
        winner = ordered.loc[top_by_feature[r._feature]]
        fids = calls.get(winner._identity, set())
        displaced = (
            winner._identity != r._identity and len(fids) == 1 and r._fid not in fids
        )
        advantage = np.nan
        if displaced:
            accepted_fid = next(iter(fids))
            accepted_score = old_scores.get((winner._identity, accepted_fid), np.nan)
            advantage = accepted_score - float(winner._score)
        row = dict(
            physical=bool(physical.loc[idx]),
            identity_free=r._identity not in calls,
            feature_free=r._feature not in occupied,
            identity_best=idx in identity_best,
            feature_top=idx in top.index,
            feature_clear=idx in clear,
            residual_top=idx in residual_top,
            residual_clear=idx in residual_clear,
            blocked_by_called_other_peak=displaced,
            blocker_score_advantage=advantage,
        )
        row["relaxed_opportunity"] = (
            all(
                row[k]
                for k in (
                    "physical",
                    "identity_free",
                    "feature_free",
                    "identity_best",
                    "residual_top",
                    "residual_clear",
                    "blocked_by_called_other_peak",
                )
            )
            and np.isfinite(advantage)
            and advantage >= 0.05
            and scores.loc[idx] >= 0.05
        )
        result.append(row)
    return pd.DataFrame(result, index=pool.index)[list(FIELDS)]


def independent_measure(pool, scores, previous, physical):
    """Scalar list/group enumeration, no vectorized ranking/mask helper reuse."""
    # Shared canonical equality conventions are not measurements or scores.
    p, calls = normalized(pool, previous)
    records = [(idx, row) for idx, row in p.iterrows()]
    order = sorted(
        range(len(records)),
        key=lambda j: (
            -float(scores.loc[records[j][0]]),
            -float(records[j][1].feat_intensity),
            j,
        ),
    )
    chosen, seen = [], set()
    for j in order:
        idx, row = records[j]
        key = (row._identity, row._fid)
        if key not in seen:
            chosen.append((idx, row))
            seen.add(key)
    by_feature, by_identity = {}, {}
    for idx, r in chosen:
        by_feature.setdefault(r._feature, []).append((idx, r))
        by_identity.setdefault(r._identity, []).append((idx, r))
    occupied = {(key[0], fid) for key, fids in calls.items() for fid in fids}
    result = []
    for idx, r in records:
        alternatives = by_feature[r._feature]
        first_idx, first = alternatives[0]
        residual = [(j, v) for j, v in alternatives if v._identity not in calls]
        # Clear is descriptive even for non-top rows, matching the producer's
        # consecutive-score definition, not a claimed accepted alternative.
        own_position = next(
            (j for j, (i, _) in enumerate(alternatives) if i == idx), None
        )
        own_clear = own_position is not None and (
            own_position == len(alternatives) - 1
            or scores.loc[idx] - scores.loc[alternatives[own_position + 1][0]] >= 0.05
        )
        residual_position = next(
            (j for j, (i, _) in enumerate(residual) if i == idx), None
        )
        residual_is_clear = residual_position is not None and (
            residual_position == len(residual) - 1
            or scores.loc[idx] - scores.loc[residual[residual_position + 1][0]] >= 0.05
        )
        fids = calls.get(first._identity, set())
        displaced = (
            first._identity != r._identity and len(fids) == 1 and r._fid not in fids
        )
        advantage = np.nan
        if displaced:
            fid = next(iter(fids))
            found = [
                j
                for j, candidate in by_identity[first._identity]
                if candidate._fid == fid
            ]
            if found:
                advantage = float(scores.loc[found[0]] - scores.loc[first_idx])
        row = dict(
            physical=bool(physical.loc[idx]),
            identity_free=r._identity not in calls,
            feature_free=r._feature not in occupied,
            identity_best=by_identity[r._identity][0][0] == idx,
            feature_top=first_idx == idx,
            feature_clear=own_clear,
            residual_top=bool(residual) and residual[0][0] == idx,
            residual_clear=residual_is_clear,
            blocked_by_called_other_peak=displaced,
            blocker_score_advantage=advantage,
        )
        row["relaxed_opportunity"] = bool(
            row["physical"]
            and row["identity_free"]
            and row["feature_free"]
            and row["identity_best"]
            and row["residual_top"]
            and row["residual_clear"]
            and displaced
            and np.isfinite(advantage)
            and advantage >= 0.05
            and scores.loc[idx] >= 0.05
        )
        result.append(row)
    return pd.DataFrame(result, index=pool.index)[list(FIELDS)]
