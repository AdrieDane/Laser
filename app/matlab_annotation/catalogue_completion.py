"""Research-only completion for independently restored catalogue targets.

Old calls are immutable. Targets must already be absent from the old chemical
reference, with RT from OTHER project groups and blind automatic run alignment.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .automatic import matlab_annotation_adducts
from .standard_rt_library import ion_key

_ALIGNED = {"automatic_anchor_alignment", "identity_warp_no_reliable_improvement"}


def complete_catalogue_targets(
    annotations,
    features,
    targets,
    priors,
    channels,
    *,
    stream,
    declared_standard_masses=(),
):
    """Fixed unique1.5ppm/5s rule, not an identification-probability estimate.

    Require>=3 other-group RT cohorts/trainingMAD<=5s, reliable independent run
    alignment, composition in>=25% and>=3 other groups, class/adduct in>=50% and
    >=3 other groups. Count all raw hits BEFORE occupancy/standard filtering.
    ALL competing new identity claims on one feature are withheld. No refit,
    iterations, score threshold changes, old removals or old peak replacements.
    """
    if stream not in {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}:
        raise ValueError("Unknown stream")
    if channels.get("format") != "laser-class-channel-support-v1":
        raise ValueError("Unsupported held-out class/channel support")
    needed = {
        "name",
        "class",
        "adduct",
        "carbon",
        "db",
        "ref_mz",
        "atlas_rt_seconds",
        "atlas_training_cohorts",
        "atlas_training_mad_seconds",
        "atlas_global_status",
        "chemical_target_origin",
    }
    if not needed.issubset(targets):
        raise ValueError(
            "Catalogue completion needs explicit independent chemical/RT provenance"
        )
    if {
        "training_positive",
        "same_manual_peak",
        "manual_feature_id",
        "training_label_known",
    } & set(targets):
        raise ValueError("Curated labels must not enter catalogue completion")
    if features.feature_id.astype(str).duplicated().any():
        raise ValueError("Original feature IDs must be unique")
    mz = pd.to_numeric(features.mz, errors="coerce").to_numpy(float)
    rt = pd.to_numeric(features.rt, errors="coerce").to_numpy(float) * 60
    intensity = pd.to_numeric(features.intensity, errors="coerce").to_numpy(float)
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
    existing = {
        (str(n), ion_key(a))
        for n, a in annotations[["name", "adduct"]].itertuples(index=False, name=None)
    }
    occupied = set(annotations.feat_feature_id.astype(str))
    spikes = np.asarray(list(declared_standard_masses), float)
    spikes = spikes[np.isfinite(spikes) & (spikes > 0)]
    raw, proposed, audit = [], [], []
    for r in targets.drop_duplicates(["name", "adduct"]).to_dict("records"):
        name, adduct = str(r["name"]), str(r["adduct"])
        if (
            (name, ion_key(adduct)) in existing
            or name.endswith(" (IS)")
            or name.startswith("IS: ")
        ):
            continue
        if adduct not in matlab_annotation_adducts(stream[:2]) or ion_key(
            adduct
        ).endswith("+") != stream.endswith("Pos"):
            continue
        if r["chemical_target_origin"] != "native_catalogue_without_RT_dependency":
            raise ValueError(
                "Only independently restored native catalogue targets are admissible"
            )
        mass, expected = float(r["ref_mz"]), float(r["atlas_rt_seconds"])
        mad = float(r["atlas_training_mad_seconds"])
        if not np.isfinite([mass, expected, mad]).all() or mass <= 0 or expected <= 0:
            continue
        if (
            r["atlas_training_cohorts"] < 3
            or mad > 5
            or r["atlas_global_status"] not in _ALIGNED
        ):
            continue
        cls = canonical_lipid_class(r["class"])
        p = priors.get(stream + "|" + cls, {})
        groups = int(p.get("cohorts", 0))
        support = int(p.get("joint", {}).get(f"{int(r['carbon'])}:{int(r['db'])}", 0))
        channel = channels["models"].get(stream + "|" + cls, {})
        cg = int(channel.get("cohorts", 0))
        cs = int(channel.get("adducts", {}).get(ion_key(adduct), 0))
        if (
            groups < 3
            or support < max(3, 0.25 * groups)
            or cg < 3
            or cs < max(3, 0.5 * cg)
        ):
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
                    baseline_ref_rt=r.get("ref_rt", np.nan),
                    rt_expected=expected / 60,
                    rt_error=(rt[pos] - expected) / 60,
                    ppm_error=(mz[pos] / mass - 1) * 1e6,
                    is_is=False,
                    isotopologue="",
                    automatic_accepted=False,
                    automatic_candidate=True,
                    class_selection_score=np.nan,
                    auto_annotation_status="restored_catalogue_candidate",
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
        r["auto_annotation_status"] = "restored_native_catalogue_completion"
        r["annotation_evidence"] = (
            "native chemical catalogue; OTHER-group RT/composition/channel support; unique1.5ppm/5s original peak"
        )
        additions.append(r)
        audit.append(
            dict(
                action="add_unique_restored_catalogue_hit",
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
