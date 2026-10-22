"""Research-only bounded discovery and conservative additive peak selection.

Chemical/RT targets and OTHER-mode automatic anchors are explicit inputs. No
curated labels, dataset names, guessed formulas or specimen pairing are read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .automatic import matlab_annotation_adducts
from .cross_mode_evidence import _key, _strong
from .standard_rt_library import ion_key


def paired_discovery_candidates(
    targets, features, automatic, old_candidates, *, stream,
    declared_standard_masses=(),
):
    """Native targets, >=3 RT cohorts/MAD<=10s, raw<3ppm/<=30s.

    Require a strong exact canonical class/C/DB automatic anchor in ANOTHER
    stream, never use NP/RP RT transfer. Retain ALL physical alternatives for an
    identity with at least one candidate absent from007. Old candidates remain
    competitors, not forced out before ranking. Actual declared-spike masses
    withhold the WHOLE target, not only an inconvenient competing feature.
    Paired biological profiles and learned acceptance are checked separately.
    """
    if stream not in {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}:
        raise ValueError("Unknown stream")
    required = {"name", "class", "carbon", "db", "adduct", "ref_mz",
                "atlas_rt_seconds", "atlas_training_cohorts", "atlas_training_mad_seconds"}
    if not required.issubset(targets):
        raise ValueError("Independent chemical/RT targets required")
    if {"training_positive", "same_manual_peak", "manual_feature_id"} & set(targets):
        raise ValueError("Curated labels cannot enter physical discovery")
    if features.feature_id.astype(str).duplicated().any():
        raise ValueError("Original feature IDs must be unique")
    strong = _strong(automatic)
    supported = set(_key(strong[strong.stream.ne(stream)])) if len(strong) else set()
    old = {(str(n), ion_key(a), str(f)) for n, a, f in old_candidates[
        ["name", "adduct", "feat_feature_id"]].itertuples(index=False, name=None)}
    mz = pd.to_numeric(features.mz, errors="coerce").to_numpy(float)
    rt = pd.to_numeric(features.rt, errors="coerce").to_numpy(float) * 60
    intensity = pd.to_numeric(features.intensity, errors="coerce").to_numpy(float)
    ids = features.feature_id.astype(str).to_numpy()
    valid = np.flatnonzero(np.isfinite(mz) & (mz > 0) & np.isfinite(rt)
                           & np.isfinite(intensity) & (intensity > 0))
    order = valid[np.argsort(mz[valid], kind="stable")]
    spikes = np.asarray(list(declared_standard_masses), float)
    spikes = spikes[np.isfinite(spikes) & (spikes > 0)]
    target_keys = _key(targets)
    rows = []
    for idx, target in targets.drop_duplicates(["name", "adduct"]).iterrows():
        if target_keys.loc[idx] not in supported:
            continue
        if str(target["name"]).endswith(" (IS)") or str(target["name"]).startswith("IS: "):
            continue
        ion = str(target.adduct)
        if ion not in matlab_annotation_adducts(stream[:2]) or ion_key(ion).endswith("+") != stream.endswith("Pos"):
            continue
        mass, expected, cohorts, mad = map(float, (
            target.ref_mz, target.atlas_rt_seconds, target.atlas_training_cohorts,
            target.atlas_training_mad_seconds))
        if not np.isfinite([mass, expected, cohorts, mad]).all() or mass <= 0 or expected <= 0 or cohorts < 3 or mad > 10:
            continue
        if len(spikes) and (np.abs(spikes - mass) / spikes < 3e-6).any():
            continue
        lo, hi = np.searchsorted(mz[order], [mass * (1 - 3e-6), mass * (1 + 3e-6)], side="left")
        hits = order[lo:hi]
        hits = hits[(np.abs(mz[hits] / mass - 1) * 1e6 < 3)
                    & (np.abs(rt[hits] - expected) <= 30)]
        part = []
        for pos in hits:
            part.append(dict(
                target, stream=stream, feat_feature_id=ids[pos], feat_mz=mz[pos],
                feat_rt=rt[pos] / 60, feat_intensity=intensity[pos], ref_rt=expected / 60,
                rt_expected=expected / 60, rt_error=(rt[pos] - expected) / 60,
                ppm_error=(mz[pos] / mass - 1) * 1e6, is_is=False, isotopologue="",
                automatic_accepted=False, automatic_candidate=True, atlas_available=True,
                auto_annotation_status="bounded_paired_discovery_candidate",
                discovery_novel_to007=(str(target["name"]), ion_key(ion), ids[pos]) not in old,
            ))
        if any(r["discovery_novel_to007"] for r in part):
            rows.extend(part)
    return pd.DataFrame(rows)


def select_paired_additions(pool, scores, annotations, *, threshold, margin=0.1):
    """Immutable old calls; clear best identity AND best feature, no cascades.

    Every raw alternative competes BEFORE novelty, occupancy or profile gates.
    New feature must be absent from007, score>=training cutoff, >=8 paired
    biological specimens and variable log-profile correlation>=.8. An occupied
    winner withholds completion; a weaker free peak cannot borrow its label.
    Every existing identity, feature and standard/manual call remains unchanged.
    """
    if not .5 <= threshold <= 1 or not 0 <= margin <= 1:
        raise ValueError("Invalid additive discovery operating point")
    if not pool.index.equals(scores.index) or not pool.index.is_unique:
        raise ValueError("Discovery scores must align with unique query rows")
    if pool.empty:
        return pool.assign(class_selection_score=scores).copy()
    required = {"discovery_novel_to007", "q_paired_best_log_correlation",
                "q_paired_profile_specimens", "feat_feature_id", "name", "adduct"}
    if not required.issubset(pool):
        raise ValueError("Explicit novelty/paired-profile evidence required")
    ranked = pool.assign(class_selection_score=pd.to_numeric(scores, errors="coerce"))
    ranked = ranked.sort_values(["class_selection_score", "feat_intensity"], ascending=False, kind="stable")
    ranked = ranked.drop_duplicates(["name", "adduct", "feat_feature_id"])
    identity = ranked.name.astype(str) + "|" + ranked.adduct.map(ion_key)
    feature = ranked.feat_feature_id.astype(str)
    identity_rank = ranked.groupby(identity, sort=False).cumcount()
    feature_rank = ranked.groupby(feature, sort=False).cumcount()
    next_identity = ranked.groupby(identity, sort=False).class_selection_score.shift(-1)
    next_feature = ranked.groupby(feature, sort=False).class_selection_score.shift(-1)
    identity_clear = next_identity.isna() | (ranked.class_selection_score - next_identity).ge(margin)
    feature_clear = next_feature.isna() | (ranked.class_selection_score - next_feature).ge(margin)
    selected = ranked[identity_rank.eq(0) & feature_rank.eq(0) & identity_clear
                      & feature_clear & ranked.class_selection_score.ge(threshold)].copy()
    existing = {(str(n), ion_key(a)) for n, a in annotations[["name", "adduct"]].itertuples(index=False, name=None)}
    occupied = set(annotations.feat_feature_id.astype(str))
    keep = [bool(r.discovery_novel_to007)
            and (str(r.name), ion_key(r.adduct)) not in existing
            and str(r.feat_feature_id) not in occupied
            and np.isfinite(r.q_paired_best_log_correlation)
            and r.q_paired_best_log_correlation >= .8
            and r.q_paired_profile_specimens >= 8
            for r in selected.itertuples()]
    selected = selected.loc[keep].copy()
    selected["automatic_accepted"] = True
    selected["auto_annotation_status"] = "paired_profile_supported_new_discovery"
    selected["annotation_evidence"] = "native chemistry; OTHER-group RT; explicit paired biological profiles; grouped training-only additive score/cutoff"
    return selected
