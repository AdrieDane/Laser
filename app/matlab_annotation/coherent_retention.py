"""Unused research adapter: blind, composition-cross-fitted numerical RT shifts.

No candidate creation, scoring, annotations, file access or production defaults.
The caller supplies fingerprint-verified references and spike measurements from
the EXACT requested exclusion mask. This module verifies scope metadata too.
"""

import hashlib

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .automatic import matlab_annotation_adducts
from .class_candidates import _composition_allowed
from .coherent_rt_shift import FIELDS, estimate_coherent_rt_shift
from .retention_atlas import predict_aligned_retention_atlas


def composition_block(carbon, db):
    """Five deterministic chemical-coordinate blocks, NOT biological CV folds."""
    return (np.asarray(carbon, dtype=int) // 2 + 3 * np.asarray(db, dtype=int)) % 5


def retention_family(lipid_class, stream):
    """Share only explicit RP isomer families; NP retains its own exact classes."""
    cls = canonical_lipid_class(lipid_class)
    if not stream.startswith("RP_"):
        return cls
    for group in (
        {"PC[O]", "PC[P]"},
        {"PE[O]", "PE[P]"},
        {"LPC[O]", "LPC[P]"},
        {"LPE[O]", "LPE[P]"},
        {"DG[O]", "DG[P]"},
        {"1-acyl LPC", "2-acyl LPC"},
        {"1-acyl LPE", "2-acyl LPE"},
    ):
        if cls in group:
            return "+".join(sorted(group))
    return cls


def _protected_mass(values, declared_masses):
    masses = np.sort(np.asarray(declared_masses, float))
    if not len(masses):
        return np.zeros(len(values), dtype=bool)
    lower = np.searchsorted(masses, values / (1 + 5.01e-6), side="left")
    upper = np.searchsorted(masses, values / (1 - 5.01e-6), side="right")
    return lower != upper


def coherent_retention_predictions(
    targets,
    features,
    automatic,
    declared,
    measurements,
    bundle,
    *,
    stream,
    query_group,
    expected_excluded_groups,
    ranges,
):
    """Return numeric predictions and an audit, NEVER original-peak identities.

    Mass hits from the top intensity decile and even-carbon anchor compositions
    vote for a shared shift. These restrictions apply ONLY to anchors, not the
    prediction/annotation universe. Native ions, <1.5ppm, ±90s, >=3 training
    projects and <=5s training MAD; actual spike masses/FIDs excluded. For EACH
    target, its entire C/DB block across ALL branches in the family is excluded
    from raw-query anchor construction. Thus its own peak cannot align itself.
    RP isomer branches receive the SAME offset for a given chemical block;
    their trained/base relative RT differences are not collapsed or reversed.
    Unsupported/ambiguous/out-of-anchor-domain targets retain baseline RT.
    """
    if stream not in {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}:
        raise ValueError("Explicit acquisition stream required")
    if bundle.get("format") != "laser-nested-reference-bundle-v1":
        raise ValueError("Explicit nested reference bundle required")
    training = set(bundle["training_groups"])
    excluded = set(bundle["excluded_groups"])
    if (
        training & excluded
        or query_group in training
        or query_group not in excluded
        or excluded != set(expected_excluded_groups)
    ):
        raise ValueError("Exact query/outer/inner reference exclusions required")
    atlas = bundle["atlas"]
    if atlas["training_scope_sha256"] != hashlib.sha256(
        "\n".join(sorted(training)).encode()
    ).hexdigest() or atlas["training_cohorts"] != len(training):
        raise ValueError("Atlas training fingerprint disagrees with reference scope")
    if (
        not {"name", "class", "adduct", "carbon", "db", "ref_mz"}.issubset(targets)
        or not targets.index.is_unique
    ):
        raise ValueError("Indexed chemical targets with mass and C/DB required")
    if targets.duplicated(["name", "adduct"]).any():
        raise ValueError("Unique target/ion pairs required")
    if not {"feature_id", "mz", "rt", "intensity"}.issubset(features):
        raise ValueError("Original physical features required")
    if (
        features.feature_id.isna().any()
        or features.feature_id.astype(str).duplicated().any()
    ):
        raise ValueError("Unique nonmissing original feature IDs required")
    if not {"ref_mz", "stream"}.issubset(declared):
        raise ValueError("Explicit stream-specific declared mass catalog required")
    for frame in (targets, features, automatic, declared, measurements):
        if {
            "training_positive",
            "training_label_known",
            "same_manual_peak",
            "manual_feature_id",
            "manual_annotation",
            "manual_confirmed",
            "manually_confirmed",
        } & set(frame):
            raise ValueError("Only blind physical inputs, never query supervision")
        if "stream" in frame and len(frame) and not frame.stream.eq(stream).all():
            raise ValueError("Cannot combine acquisition streams")
    masses = pd.to_numeric(declared.ref_mz, errors="raise").to_numpy(float)
    if not np.isfinite(masses).all() or (masses <= 0).any():
        raise ValueError("Finite positive declared masses required")
    protected = set()
    for frame in (automatic, measurements):
        if "feat_feature_id" in frame:
            std = (
                frame.get("is_is", pd.Series(False, index=frame.index))
                .fillna(False)
                .astype(bool)
            )
            # All independently measured features are protected, even if a
            # downstream quality flag later withholds that standard as evidence.
            if frame is measurements:
                std = pd.Series(True, index=frame.index)
            protected.update(frame.loc[std, "feat_feature_id"].dropna().astype(str))
    clean = automatic.copy()
    if len(clean):
        keep = ~_protected_mass(
            pd.to_numeric(clean.feat_mz, errors="coerce").to_numpy(float), masses
        )
        keep &= ~clean.feat_feature_id.astype(str).isin(protected).to_numpy()
        clean = clean.loc[keep].copy()
    base = predict_aligned_retention_atlas(targets, clean, atlas, stream=stream)
    out = pd.DataFrame(index=targets.index)
    out["baseline_rt_seconds"] = base.atlas_rt_seconds
    out["coherent_rt_seconds"] = base.atlas_rt_seconds
    out["coherent_shift_seconds"] = np.nan
    out["coherent_available"] = False
    data = targets.copy()
    data["class"] = data["class"].map(canonical_lipid_class)
    data["_family"] = data["class"].map(lambda c: retention_family(c, stream))
    data["_block"] = -1
    numeric = data[["carbon", "db", "ref_mz"]].apply(pd.to_numeric, errors="raise")
    finite = np.isfinite(numeric).all(axis=1)
    finite &= numeric.carbon.gt(0) & numeric.db.ge(0) & numeric.ref_mz.gt(0)
    finite &= numeric.carbon.eq(numeric.carbon.round()) & numeric.db.eq(
        numeric.db.round()
    )
    data.loc[finite, "_block"] = composition_block(
        numeric.loc[finite, "carbon"], numeric.loc[finite, "db"]
    )
    eligible = finite & base.atlas_in_domain & base.atlas_training_cohorts.ge(3)
    eligible &= base.atlas_training_mad_seconds.le(5) & base.atlas_rt_seconds.notna()
    eligible &= ~data.name.astype(str).str.endswith(" (IS)")
    eligible &= (
        data.get("isotopologue", pd.Series("", index=data.index))
        .fillna("")
        .astype(str)
        .eq("")
    )
    eligible &= ~_protected_mass(numeric.ref_mz.to_numpy(float), masses)
    ions = set(matlab_annotation_adducts(stream[:2]))
    eligible &= data.adduct.isin(ions) & data.adduct.astype(str).str.rstrip(
        ")"
    ).str.endswith("+" if stream.endswith("Pos") else "-")
    eligible &= pd.Series(
        [
            _composition_allowed(ranges, r["class"], r["carbon"], r["db"], r["name"])
            if finite.loc[i]
            else False
            for i, r in data.iterrows()
        ],
        index=data.index,
    )
    raw = features[["feature_id", "mz", "rt", "intensity"]].copy()
    for c in ("mz", "rt", "intensity"):
        raw[c] = pd.to_numeric(raw[c], errors="raise")
    valid = np.isfinite(raw[["mz", "rt", "intensity"]]).all(axis=1)
    valid &= raw.mz.gt(0) & raw.rt.gt(0) & raw.intensity.gt(0)
    valid &= ~raw.feature_id.astype(str).isin(protected)
    valid &= ~_protected_mass(raw.mz.to_numpy(float), masses)
    raw = raw.loc[valid].copy()
    if len(raw):
        raw = raw[raw.intensity.ge(raw.intensity.quantile(0.9))].sort_values(
            "mz", kind="stable"
        )
    observed_mass = raw.mz.to_numpy(float)
    rows = []
    # Strict mass-search interval first; reference names/feature IDs only link
    # physical measurements and deduplicate votes, never predict an offset.
    for i, target in data.loc[eligible & numeric.carbon.mod(2).eq(0)].iterrows():
        mass = float(target.ref_mz)
        lo, hi = np.searchsorted(
            observed_mass, [mass * (1 - 1.5e-6), mass * (1 + 1.5e-6)]
        )
        for feature in raw.iloc[lo:hi].itertuples():
            residual = float(feature.rt * 60 - base.loc[i, "atlas_rt_seconds"])
            ppm = float((feature.mz - mass) / mass * 1e6)
            if abs(ppm) >= 1.5 or abs(residual) > 90:
                continue
            rows.append(
                dict(
                    feature_id=str(feature.feature_id),
                    composition_id=f"{target['class']}|{int(target.carbon)}|{int(target.db)}",
                    series_id=target["class"],
                    carbon=int(target.carbon),
                    db=int(target.db),
                    mass_error_ppm=ppm,
                    rt_residual_seconds=residual,
                    intensity=float(feature.intensity),
                    _family=target._family,
                    _block=int(target._block),
                )
            )
    hits = pd.DataFrame(rows, columns=[*sorted(FIELDS), "_family", "_block"])
    # Different native ions of a composition can share a peak only if their
    # theoretical masses agree. Keep its best mass edge, not an arbitrary alias.
    if len(hits):
        hits["_absolute_ppm"] = hits.mass_error_ppm.abs()
        hits = (
            hits.sort_values("_absolute_ppm", kind="stable")
            .drop_duplicates(["_family", "feature_id", "composition_id"])
            .drop(columns="_absolute_ppm")
        )
    audits = []
    for family, indices in (
        data.loc[eligible].groupby("_family", sort=True).groups.items()
    ):
        family_hits = hits[hits._family.eq(family)]
        for block in sorted(data.loc[indices, "_block"].unique()):
            anchor = family_hits[family_hits._block.ne(block)]
            result = estimate_coherent_rt_shift(anchor[sorted(FIELDS)])
            audits.append(
                dict(
                    stream=stream,
                    family=family,
                    excluded_composition_block=int(block),
                    physical_mass_hits=len(anchor),
                    **result,
                )
            )
            if not result["supported"]:
                continue
            apply = data.index.isin(indices) & data._block.eq(block)
            apply &= data.carbon.between(
                result["carbon_min"] - 2, result["carbon_max"] + 2
            )
            apply &= data.db.between(max(0, result["db_min"] - 1), result["db_max"] + 1)
            apply &= base.atlas_rt_seconds.add(result["shift_seconds"]).gt(0)
            out.loc[apply, "coherent_shift_seconds"] = result["shift_seconds"]
            out.loc[apply, "coherent_available"] = True
            out.loc[apply, "coherent_rt_seconds"] += result["shift_seconds"]
    return out, pd.DataFrame(audits)
