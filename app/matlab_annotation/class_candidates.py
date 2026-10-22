"""Strict original-feature candidates for an offline-trained class RT atlas.

No curation is read here. This is an unused development component, not an
enabled annotation policy. Existing user composition limits remain binding.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from annotator.composition_ranges import (
    CompositionRangeSettings, DEFAULT_COMPOSITION_RANGE_SETTINGS, canonical_lipid_class,
)
from .automatic import matlab_annotation_adducts
from .retention_atlas import predict_aligned_retention_atlas


@lru_cache(maxsize=100_000)
def _composition_allowed(settings, lipid_class, carbon, db, species):
    # Repeated adducts/cohorts/folds share the same immutable range decision.
    return settings.decide(lipid_class, carbon, db, species=species).allowed


def prepare_class_candidates(
    annotations: pd.DataFrame, reference: pd.DataFrame, features: pd.DataFrame,
    *, stream: str, atlas: dict,
    ranges: CompositionRangeSettings = DEFAULT_COMPOSITION_RANGE_SETTINGS,
    ppm: float = 3.0, rt_seconds: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """Join chemical targets to ORIGINAL features inside strict mass/RT bounds.

    The atlas replaces a target's RT reference, not the observed coordinates or
    feature ID. Unsupported RT predictions fall back to the preceding fitted
    reference and remain explicitly marked. Standards are reserved by the
    caller, never regenerated as endogenous candidate identities.
    """
    if not 0 < ppm <= 3 or not 0 < rt_seconds <= 10:
        raise ValueError("Class candidates cannot exceed 3 ppm / 10 RT seconds")
    required = {"name", "class", "adduct", "carbon", "db", "ref_mz", "ref_rt"}
    if not required.issubset(reference):
        raise ValueError("Chemical reference is missing candidate coordinates")
    if not {"feature_id", "mz", "rt", "intensity"}.issubset(features):
        raise ValueError("Original features require IDs, m/z, RT and intensity")
    if features.feature_id.astype(str).duplicated().any():
        raise ValueError("Original feature IDs must be unique within a stream")
    targets = reference[
        reference.adduct.isin(matlab_annotation_adducts(stream[:2]))
        & ~reference.name.astype(str).str.endswith(" (IS)")
    ].drop_duplicates(["name", "adduct"]).copy()
    allowed = [_composition_allowed(ranges, r["class"], r["carbon"], r["db"], r["name"])
               for r in targets.to_dict("records")]
    targets = targets.loc[allowed].reset_index(drop=True)
    rt_predictions = predict_aligned_retention_atlas(targets, annotations, atlas, stream=stream)
    targets = targets.join(rt_predictions)
    targets["baseline_ref_rt"] = targets.ref_rt
    targets["atlas_available"] = targets.atlas_rt_seconds.notna()
    targets["ref_rt"] = targets.atlas_rt_seconds.div(60).fillna(targets.ref_rt)
    masses = pd.to_numeric(features.mz, errors="coerce").to_numpy(float)
    rts = pd.to_numeric(features.rt, errors="coerce").to_numpy(float)
    intensities = pd.to_numeric(features.intensity, errors="coerce").to_numpy(float)
    ids = features.feature_id.astype(str).to_numpy()
    valid = np.flatnonzero(np.isfinite(masses) & np.isfinite(rts) & (masses > 0))
    order = valid[np.argsort(masses[valid], kind="stable")]
    sorted_masses = masses[order]
    standards = annotations.get("is_is", pd.Series(False, index=annotations.index)).fillna(False).astype(bool)
    protected = set(annotations.loc[standards, "feat_feature_id"].astype(str)) if len(annotations) else set()
    rows = []
    for target in targets.to_dict("records"):
        mz, rt = float(target["ref_mz"]), float(target["ref_rt"])
        if not np.isfinite(mz) or not np.isfinite(rt) or mz <= 0 or rt <= 0:
            continue
        lower = np.searchsorted(sorted_masses, mz * (1 - ppm * 1e-6))
        upper = np.searchsorted(sorted_masses, mz * (1 + ppm * 1e-6), side="right")
        for position in order[lower:upper]:
            error = (masses[position] - mz) / mz * 1e6
            if abs(error) >= ppm or abs(rts[position] - rt)*60 > rt_seconds or ids[position] in protected:
                continue
            row = dict(target)
            row.update(stream=stream, feat_feature_id=ids[position], feat_mz=masses[position],
                       feat_rt=rts[position], feat_intensity=intensities[position],
                       rt_expected=rt, rt_error=rts[position]-rt, ppm_error=error,
                       isotopologue="", is_is=False, automatic_accepted=False,
                       automatic_candidate=True, auto_annotation_status="class_rt_atlas_candidate")
            rows.append(row)
    pool = pd.DataFrame(rows)
    audit = dict(stream=stream, targets=len(targets), atlas_targets=int(targets.atlas_available.sum()),
                 original_features=len(features), candidate_pairs=len(pool),
                 protected_standard_features=len(protected), ppm=ppm, rt_seconds=rt_seconds)
    return pool, audit


def candidate_evidence(pool: pd.DataFrame) -> pd.DataFrame:
    """Numerical/categorical evidence only; no names, IDs or cohort features.

    Chemical class, adduct and total composition are legitimate predictors.
    Raw feature/target identifiers are used only to calculate within-run
    competition statistics; they are never returned as model input columns.
    """
    if pool.empty:
        return pd.DataFrame(index=pool.index)
    evidence = pd.DataFrame(index=pool.index)
    for column in ("stream", "class", "adduct"):
        evidence[column] = pool[column].astype(str)
    evidence["class"] = evidence["class"].map(canonical_lipid_class)
    evidence["stream_class"] = evidence.stream + "|" + evidence["class"]
    mass = pd.to_numeric(pool.ppm_error, errors="coerce")
    rt = pd.to_numeric(pool.rt_error, errors="coerce")*60
    intensity = np.log1p(pd.to_numeric(pool.feat_intensity, errors="coerce").clip(lower=0))
    evidence["mass_error_ppm"] = mass
    evidence["absolute_mass_error_ppm"] = mass.abs()
    evidence["squared_mass_error_ppm"] = mass**2
    evidence["RT_error_seconds"] = rt
    evidence["absolute_RT_error_seconds"] = rt.abs()
    evidence["squared_RT_error_seconds"] = rt**2
    evidence["log_intensity"] = intensity
    identity = pool["name"].astype(str) + "|" + pool.adduct.astype(str)
    evidence["identity_relative_log_intensity"] = intensity - intensity.groupby(identity).transform("max")
    cls = pool["class"].astype(str)
    evidence["class_centered_log_intensity"] = intensity - intensity.groupby(cls).transform("median")
    evidence["identity_feature_count"] = identity.groupby(identity).transform("size")
    feature = pool.feat_feature_id.astype(str)
    evidence["feature_target_count"] = feature.groupby(feature).transform("size")
    for column in ("carbon", "db"):
        evidence[column] = pd.to_numeric(pool[column], errors="coerce")
    evidence["carbon_odd"] = evidence.carbon % 2
    evidence["db_per_carbon"] = evidence.db/evidence.carbon.clip(lower=1)
    for column in ("atlas_training_cohorts", "atlas_training_mad_seconds",
                   "atlas_local_support_count", "atlas_local_shift_seconds", "atlas_available"):
        evidence[column] = pd.to_numeric(pool[column], errors="coerce")
    # Additional optional quality evidence must be measured upstream from raw
    # intensities, not reconstructed from curator decisions or EIC-free widths.
    for column in sorted(c for c in pool if c.startswith("q_")):
        evidence[column] = pd.to_numeric(pool[column], errors="coerce")
    return evidence.replace([np.inf, -np.inf], np.nan)


def fit_class_composition_priors(observations: pd.DataFrame) -> dict:
    """Offline TRAINING-only composition prevalence, never a hard whitelist."""
    required = {"cohort", "stream", "class", "carbon", "db"}
    if not required.issubset(observations):
        raise ValueError("Composition-prior observations are incomplete")
    data = observations[sorted(required)].copy()
    data["class"] = data["class"].map(canonical_lipid_class)
    data = data.drop_duplicates(["cohort", "stream", "class", "carbon", "db"])
    output = {}
    for (stream, cls), group in data.groupby(["stream", "class"], sort=True):
        joint = group.groupby(["carbon", "db"]).cohort.nunique()
        output[stream+"|"+cls] = {
            "cohorts": int(group.cohort.nunique()),
            "joint": {f"{int(c)}:{int(d)}": int(n) for (c, d), n in joint.items()},
            "carbon": {str(int(c)): int(n) for c, n in group.groupby("carbon").cohort.nunique().items()},
            "db": {str(int(d)): int(n) for d, n in group.groupby("db").cohort.nunique().items()},
        }
    return output


def attach_composition_prior_evidence(evidence: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """Use smoothed support rather than rejecting an unseen long-chain lipid."""
    output = evidence.copy()
    rows = []
    for row in evidence.to_dict("records"):
        prior = priors.get(row["stream_class"], {})
        n = prior.get("cohorts", 0)
        c, d = str(int(row["carbon"])), str(int(row["db"]))
        rows.append({
            "composition_training_cohorts": n,
            "composition_joint_frequency": (prior.get("joint", {}).get(c+":"+d, 0)+.5)/(n+1),
            "composition_carbon_frequency": (prior.get("carbon", {}).get(c, 0)+.5)/(n+1),
            "composition_db_frequency": (prior.get("db", {}).get(d, 0)+.5)/(n+1),
        })
    return pd.concat([output, pd.DataFrame(rows, index=output.index)], axis=1)


def score_linear_evidence(evidence: pd.DataFrame, model: dict) -> pd.Series:
    """Portable DictVectorizer-style logistic scorer; no sklearn at runtime."""
    if model.get("format") != "laser-class-evidence-logistic-v1":
        raise ValueError("Unsupported class-evidence model")
    linear = np.full(len(evidence), float(model["intercept"]))
    weights = model["coefficients"]
    for column in evidence:
        values = evidence[column]
        if pd.api.types.is_numeric_dtype(values):
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
            missing = ~np.isfinite(numeric)
            replacement = float(model.get("imputation", {}).get(column, 0.0))
            numeric = np.where(missing, replacement, numeric)
            linear += float(weights.get(column, 0.0))*numeric
            linear += float(weights.get(column+"_missing", 0.0))*missing
        else:
            linear += np.asarray([float(weights.get(column+"="+str(v), 0.0)) for v in values])
    scores = 1/(1+np.exp(-np.clip(linear, -60, 60)))
    return pd.Series(scores, index=evidence.index)


def select_class_candidates(pool: pd.DataFrame, scores: pd.Series, *, threshold: float = .65,
                            ambiguity_margin: float = .05) -> pd.DataFrame:
    """Require one clear label per feature and one original feature per identity.

    A model score estimates agreement with the training curation process, not
    an analytical identification probability or false-discovery rate.
    """
    if not .5 <= threshold <= 1 or not 0 <= ambiguity_margin <= 1:
        raise ValueError("Invalid class-selection threshold or ambiguity margin")
    if not scores.index.equals(pool.index):
        raise ValueError("Candidate score indices must match the candidate table")
    ranked = pool.assign(class_selection_score=scores)
    ranked = ranked.sort_values(["class_selection_score", "feat_intensity"], ascending=False, kind="stable")
    # Repeated reference rows cannot create a spurious ambiguity or extra votes.
    ranked = ranked.drop_duplicates(["feat_feature_id", "name", "adduct"])
    rank = ranked.groupby("feat_feature_id").cumcount()
    next_score = ranked.groupby("feat_feature_id").class_selection_score.shift(-1)
    clear = next_score.isna() | (ranked.class_selection_score-next_score).ge(ambiguity_margin)
    selected = ranked[rank.eq(0) & clear & ranked.class_selection_score.ge(threshold)].copy()
    selected = selected.drop_duplicates(["name", "adduct"])
    selected["automatic_accepted"] = True
    selected["auto_annotation_status"] = "class_rt_atlas_selected"
    return selected
