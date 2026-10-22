"""Conservative NP completion from independent raw mass-series RT hypotheses.

Unused research policy. Existing annotations are NEVER removed or re-peaked.
New identities require a unique raw hit before occupancy filtering, common
OTHER-group composition support, and no competing feature/standard claim.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .raw_grid_candidates import match_raw_grid_candidates


def complete_np_series(
    annotations,
    features,
    targets,
    priors,
    *,
    stream,
    declared_standard_masses=(),
    min_prevalence=0.25,
    min_training_groups=3,
):
    if not 0 < min_prevalence <= 1 or min_training_groups < 3:
        raise ValueError("NP series completion requires explicit other-group support")
    if not stream.startswith("NP"):
        return annotations.copy(), pd.DataFrame(), pd.DataFrame()
    # Deliberately no occupied/standard feature filtering until AFTER counting
    # raw hits. Such filtering must not erase a genuine physical ambiguity.
    raw = match_raw_grid_candidates(
        targets, features, pd.DataFrame(), stream=stream, ppm=1.5, rt_seconds=5.0
    )
    if raw.empty:
        return annotations.copy(), raw, pd.DataFrame()
    raw = raw.drop_duplicates(["name", "adduct", "feat_feature_id"]).reset_index(
        drop=True
    )
    raw["automatic_accepted"] = False
    spikes = np.asarray(list(declared_standard_masses), dtype=float)
    spikes = spikes[np.isfinite(spikes) & (spikes > 0)]
    existing = set(annotations[["name", "adduct"]].itertuples(index=False, name=None))
    occupied = set(annotations.feat_feature_id.astype(str))
    proposed, audit = [], []
    for (name, adduct), part in raw.groupby(["name", "adduct"], sort=False):
        r = part.iloc[0]
        action = None
        if (name, adduct) in existing:
            action = "retain_existing_identity"
        elif part.feat_feature_id.astype(str).nunique() != 1:
            action = "withhold_multiple_raw_hits"
        else:
            cls = canonical_lipid_class(r["class"])
            prior = priors.get(stream + "|" + cls, {})
            groups = int(prior.get("cohorts", 0))
            support = int(prior.get("joint", {}).get(f"{int(r.carbon)}:{int(r.db)}", 0))
            fid = str(r.feat_feature_id)
            if groups < min_training_groups or support < max(
                min_training_groups, min_prevalence * groups
            ):
                action = "withhold_insufficient_training_composition_support"
            elif fid in occupied:
                action = "withhold_occupied_original_feature"
            elif len(spikes) and (
                (abs(spikes - float(r.ref_mz)) / spikes < 3e-6).any()
                or (abs(spikes - float(r.feat_mz)) / spikes < 3e-6).any()
            ):
                action = "withhold_declared_standard_mass"
            else:
                proposed.append(r.copy())
                continue
        audit.append(
            dict(
                action=action,
                name=name,
                adduct=adduct,
                raw_hits=int(part.feat_feature_id.nunique()),
            )
        )
    claims = Counter(str(r.feat_feature_id) for r in proposed)
    additions = []
    for r in proposed:
        fid = str(r.feat_feature_id)
        if claims[fid] > 1:
            audit.append(
                dict(
                    action="withhold_competing_identity_claims",
                    name=r["name"],
                    adduct=r.adduct,
                    feature_id=fid,
                )
            )
            continue
        r["automatic_accepted"] = True
        r["auto_annotation_status"] = "independent_NP_series_completion"
        r["class_selection_score"] = np.nan
        r["annotation_evidence"] = (
            "OTHER-composition RT grid; unique1.5ppm/5s raw hit; OTHER-group common composition"
        )
        additions.append(r.to_dict())
        raw.loc[
            raw.name.eq(r["name"])
            & raw.adduct.eq(r.adduct)
            & raw.feat_feature_id.astype(str).eq(fid),
            "automatic_accepted",
        ] = True
        audit.append(
            dict(
                action="add_unique_independent_NP_series_hit",
                name=r["name"],
                adduct=r.adduct,
                feature_id=fid,
            )
        )
    result = (
        pd.concat([annotations, pd.DataFrame(additions)], ignore_index=True)
        if additions
        else annotations.copy()
    )
    pd.testing.assert_frame_equal(
        result.iloc[: len(annotations)][annotations.columns].reset_index(drop=True),
        annotations.reset_index(drop=True),
        check_dtype=False,
    )
    assert len(result) >= len(annotations)
    assert not result.feat_feature_id.astype(str).duplicated().any()
    return result, raw, pd.DataFrame(audit)
