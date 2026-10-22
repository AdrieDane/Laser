"""Unused research: one-pass completion from strong automatic RT anchors.

No newly completed lipid calibrates another lipid in this pass. All other-group
models are explicit inputs, never read from the evaluated curated workbook.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .automatic import matlab_annotation_adducts
from .retention_atlas import (
    _prepare_automatic_anchors,
    _alignment_from_prepared_anchors,
    predict_retention_atlas,
)
from .standard_rt_library import ion_key


def fit_class_channels(observations):
    required = {"cohort", "stream", "class", "adduct"}
    if not required.issubset(observations):
        raise ValueError("Channel training needs explicit project groups")
    data = observations[list(sorted(required))].copy()
    data["class"] = data["class"].map(canonical_lipid_class)
    data["adduct"] = data.adduct.map(ion_key)
    models = {}
    for (stream, cls), part in data.groupby(["stream", "class"], sort=True):
        models[stream + "|" + cls] = dict(
            cohorts=int(part.cohort.nunique()),
            adducts={
                str(k): int(v)
                for k, v in part.groupby("adduct").cohort.nunique().items()
            },
        )
    return dict(format="laser-class-channel-support-v1", models=models)


def predict_strong_anchor_rt(targets, automatic, atlas, *, stream):
    """Six OTHER compositions, robust residual agreement, no target self-vote.

    Strong anchors have frozen curator-agreement score>=.95 AND<1.5ppm. Their
    scores are not analytical identification probabilities. Other-class warp
    uses the same bounded robust atlas alignment. Each query needs six distinct
    other compositions, residual MAD<=2s/p90<=5s, shift<=60s and nearby C/DB.
    """
    prediction = predict_retention_atlas(targets.assign(stream=stream), atlas)
    prediction["anchor_refined_rt_seconds"] = np.nan
    prediction["refined_anchor_count"] = 0
    prediction["refined_anchor_mad_seconds"] = np.nan
    prediction["refined_anchor_p90_seconds"] = np.nan
    prediction["refined_anchor_shift_seconds"] = np.nan
    required = {
        "class_selection_score",
        "class",
        "carbon",
        "db",
        "feat_rt",
        "ppm_error",
    }
    if not required.issubset(automatic):
        return prediction
    source = automatic.copy()
    source["post_regression_confidence"] = pd.to_numeric(
        source.class_selection_score, errors="coerce"
    )
    source = source[source.post_regression_confidence.ge(0.95)]
    source = source.sort_values(
        "post_regression_confidence", ascending=False, kind="stable"
    )
    # Accepted candidates already carry old atlas metadata; never reuse it as
    # the independent atlas prediction or cause colliding joins.
    source = source.drop(
        columns=[
            "atlas_rt_seconds",
            "atlas_training_mad_seconds",
            "atlas_training_cohorts",
            "atlas_in_domain",
        ],
        errors="ignore",
    )
    anchors = _prepare_automatic_anchors(source, atlas, stream)
    canonical = targets["class"].map(canonical_lipid_class)
    for cls, indices in canonical.groupby(canonical, sort=False).groups.items():
        local = anchors[anchors._canonical_class.eq(cls)]
        if len(local) < 6:
            continue
        warp = _alignment_from_prepared_anchors(
            anchors[anchors._canonical_class.ne(cls)]
        )
        residual = local.feat_rt.to_numpy(float) * 60 - (
            local.atlas_rt_seconds.to_numpy(float) * warp["slope"]
            + warp["intercept_seconds"]
        )
        for index in indices:
            c, d = float(targets.loc[index, "carbon"]), float(targets.loc[index, "db"])
            other = ~(local.carbon.eq(c) & local.db.eq(d))
            values = residual[other.to_numpy()]
            support = local[other]
            if len(values) < 6 or not np.isfinite(values).all():
                continue
            shift = float(np.median(values))
            deviation = np.abs(values - shift)
            mad, p90 = float(np.median(deviation)), float(np.quantile(deviation, 0.9))
            if abs(shift) > 60 or mad > 2 or p90 > 5:
                continue
            if not support.carbon.min() - 2 <= c <= support.carbon.max() + 2:
                continue
            if not max(0, support.db.min() - 1) <= d <= support.db.max() + 1:
                continue
            expected = (
                float(prediction.loc[index, "atlas_rt_seconds"]) * warp["slope"]
                + warp["intercept_seconds"]
                + shift
            )
            if not np.isfinite(expected) or expected <= 0:
                continue
            prediction.loc[
                index,
                [
                    "anchor_refined_rt_seconds",
                    "refined_anchor_count",
                    "refined_anchor_mad_seconds",
                    "refined_anchor_p90_seconds",
                    "refined_anchor_shift_seconds",
                ],
            ] = [expected, len(values), mad, p90, shift]
    return prediction


def complete_strong_anchor_series(
    annotations,
    features,
    targets,
    priors,
    channels,
    *,
    stream,
    declared_standard_masses=(),
):
    """Unique raw1.5ppm/5s hit BEFORE occupancy, no old call removed/re-peaked."""
    if channels.get("format") != "laser-class-channel-support-v1":
        raise ValueError("Unsupported class/adduct support model")
    required = {
        "name",
        "class",
        "adduct",
        "carbon",
        "db",
        "ref_mz",
        "ref_rt",
        "anchor_refined_rt_seconds",
        "refined_anchor_count",
    }
    if not required.issubset(targets):
        raise ValueError("Completion requires explicit independent anchored RT")
    if features.feature_id.astype(str).duplicated().any():
        raise ValueError("Original features must be unique")
    mz = features.mz.to_numpy(float)
    rt = features.rt.to_numpy(float) * 60
    intensity = features.intensity.to_numpy(float)
    ids = features.feature_id.astype(str).to_numpy()
    valid = (
        np.isfinite(mz)
        & (mz > 0)
        & np.isfinite(rt)
        & np.isfinite(intensity)
        & (intensity > 0)
    )
    order = np.flatnonzero(valid)
    order = order[np.argsort(mz[order], kind="stable")]
    existing = set(annotations[["name", "adduct"]].itertuples(index=False, name=None))
    occupied = set(annotations.feat_feature_id.astype(str))
    spikes = np.asarray(list(declared_standard_masses), float)
    spikes = spikes[np.isfinite(spikes) & (spikes > 0)]
    raw, proposed, audit = [], [], []
    targets = targets.drop_duplicates(["name", "adduct"])
    for r in targets.to_dict("records"):
        name, adduct = r["name"], r["adduct"]
        if (
            (name, adduct) in existing
            or str(name).endswith(" (IS)")
            or str(name).startswith("IS: ")
        ):
            continue
        if adduct not in matlab_annotation_adducts(stream[:2]):
            continue
        expected = float(r["anchor_refined_rt_seconds"])
        if not np.isfinite(expected) or r["refined_anchor_count"] < 6:
            continue
        cls = canonical_lipid_class(r["class"])
        prior = priors.get(stream + "|" + cls, {})
        groups = int(prior.get("cohorts", 0))
        support = int(
            prior.get("joint", {}).get(f"{int(r['carbon'])}:{int(r['db'])}", 0)
        )
        channel = channels["models"].get(stream + "|" + cls, {})
        cg = int(channel.get("cohorts", 0))
        cs = int(channel.get("adducts", {}).get(ion_key(adduct), 0))
        if (
            support < max(3, 0.25 * groups)
            or groups < 3
            or cs < max(3, 0.5 * cg)
            or cg < 3
        ):
            continue
        mass = float(r["ref_mz"])
        if not np.isfinite(mass) or mass <= 0:
            continue
        lo, hi = np.searchsorted(mz[order], [mass * (1 - 1.5e-6), mass * (1 + 1.5e-6)])
        hits = order[lo:hi]
        hits = hits[
            (np.abs(mz[hits] / mass - 1) * 1e6 < 1.5)
            & (np.abs(rt[hits] - expected) <= 5)
        ]
        for pos in hits:
            raw.append(
                dict(
                    r,
                    stream=stream,
                    feat_feature_id=ids[pos],
                    feat_mz=mz[pos],
                    feat_rt=rt[pos] / 60,
                    feat_intensity=intensity[pos],
                    ref_rt=expected / 60,
                    baseline_ref_rt=r["ref_rt"],
                    ppm_error=(mz[pos] / mass - 1) * 1e6,
                    rt_error=(rt[pos] - expected) / 60,
                    rt_expected=expected / 60,
                    is_is=False,
                    isotopologue="",
                    automatic_accepted=False,
                    automatic_candidate=True,
                    class_selection_score=np.nan,
                    auto_annotation_status="strong_anchor_RT_candidate",
                )
            )
        if len(hits) != 1:
            if len(hits):
                audit.append(
                    dict(action="withhold_multiple_raw_hits", name=name, adduct=adduct)
                )
            continue
        pos = hits[0]
        if ids[pos] in occupied:
            continue
        if len(spikes) and (
            (np.abs(spikes - mass) / spikes < 3e-6).any()
            or (np.abs(spikes - mz[pos]) / spikes < 3e-6).any()
        ):
            continue
        proposed.append(raw[-1].copy())
    claims = Counter(r["feat_feature_id"] for r in proposed)
    additions = []
    for r in proposed:
        if claims[r["feat_feature_id"]] > 1:
            audit.append(
                dict(
                    action="withhold_competing_identity_claims",
                    name=r["name"],
                    adduct=r["adduct"],
                )
            )
            continue
        r["automatic_accepted"] = True
        r["auto_annotation_status"] = "strong_automatic_anchor_RT_completion"
        r["annotation_evidence"] = (
            "six OTHER high-score compositions; unique1.5ppm/5s original hit; OTHER-group composition/adduct support"
        )
        additions.append(r)
        audit.append(
            dict(
                action="add_unique_strong_anchor_hit",
                name=r["name"],
                adduct=r["adduct"],
                feature_id=r["feat_feature_id"],
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
    assert not result.feat_feature_id.astype(str).duplicated().any()
    raw = pd.DataFrame(raw)
    if len(raw):
        selected = {(r["name"], r["adduct"], r["feat_feature_id"]) for r in additions}
        raw["automatic_accepted"] = [
            k in selected
            for k in raw[["name", "adduct", "feat_feature_id"]].itertuples(
                index=False, name=None
            )
        ]
    return result, raw, pd.DataFrame(audit)
