"""Research-only additive mass-drift discovery; raw coordinates stay original.

Fixed physical gates, no curated files, learned acceptance or dataset-name
predictors. Callers must supply query-excluded RT/standard references and keep
all previous accepted calls immutable. Window eligibility is not identification.
"""

import numpy as np
import pandas as pd

from .automatic import matlab_annotation_adducts
from .cross_mode_evidence import _key, _strong
from .declared_standard_context import _qualified_measurements
from .standard_mass_context import _summary
from .standard_rt_library import ion_key


def independently_measured_mass_anchors(measured):
    """Each declared actual labelled neutral mass votes once in calibration."""
    columns = ["stream", "neutral_key", "feature_id", "reference_mz", "ppm"]
    if {"training_positive", "training_label_known", "same_manual_peak"} & set(
        measured
    ):
        raise ValueError("No supervision may enter independent mass anchors")
    if measured.empty:
        return pd.DataFrame(columns=columns)
    if "internal_standard_neutral_mass" not in measured:
        raise ValueError("Actual declared neutral masses required")
    good = _qualified_measurements(measured)
    neutral = pd.to_numeric(good.internal_standard_neutral_mass, errors="coerce")
    good = good[np.isfinite(neutral) & neutral.gt(0)].copy()
    return (
        pd.DataFrame(
            dict(
                stream=good.stream,
                neutral_key=neutral.loc[good.index].round(6).astype(str),
                feature_id=good.feat_feature_id.astype(str),
                reference_mz=good.ref_mz,
                ppm=good._ppm,
            )
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )


def calibrated_mass_candidates(
    targets, features, anchors, old_candidates, *, stream, declared_standard_masses=()
):
    """All corrected<3ppm/atlas<=10s alternatives before any acceptance gate.

    RT requires>=3other-project cohorts/MAD<=10s. Calibration requires>=6distinct
    neutral masses,MAD<=.6ppm,p90<=1.5ppm,abs median<=2ppm (existing mass policy).
    Every target excludes all calibrator neutral groups with a channel within
    3ppm of its mass. Whole targets within5.01ppm of any declared spike are withheld
    (raw search3+2ppm plus multiplicative rounding), preventing spike self-anchors.
    New calls may ONLY be previously absent pairs with rawabsppm>=3. Other raw
    alternatives remain competitors even when not eligible for adding a call.
    """
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
        "atlas_in_domain",
    }
    if not needed.issubset(targets) or not {
        "feature_id",
        "mz",
        "rt",
        "intensity",
    }.issubset(features):
        raise ValueError(
            "Explicit original features and independent chemical/RT targets required"
        )
    if {"training_positive", "training_label_known", "same_manual_peak"} & set(targets):
        raise ValueError("No supervision may enter discovery")
    if not {"stream", "neutral_key", "feature_id", "reference_mz", "ppm"}.issubset(
        anchors
    ):
        raise ValueError("Declared actual-mass anchors required")
    if not features.feature_id.astype(str).is_unique:
        raise ValueError("Original feature IDs must be unique")
    if stream not in {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}:
        raise ValueError("Unknown stream")
    local = anchors[anchors.stream.eq(stream)]
    old = {
        (str(n), ion_key(a), str(f))
        for n, a, f in old_candidates[["name", "adduct", "feat_feature_id"]].itertuples(
            index=False, name=None
        )
    }
    mz = pd.to_numeric(features.mz, errors="coerce").to_numpy(float)
    rt = pd.to_numeric(features.rt, errors="coerce").to_numpy(float) * 60
    intensity = pd.to_numeric(features.intensity, errors="coerce").to_numpy(float)
    ids = features.feature_id.astype(str).to_numpy()
    valid = np.flatnonzero(
        np.isfinite(mz)
        & (mz > 0)
        & np.isfinite(rt)
        & (rt > 0)
        & np.isfinite(intensity)
        & (intensity > 0)
    )
    order = valid[np.argsort(mz[valid], kind="stable")]
    spikes = np.asarray(list(declared_standard_masses), float)
    spikes = spikes[np.isfinite(spikes) & (spikes > 0)]
    cache, rows = {}, []
    for target in targets.drop_duplicates(["name", "adduct"]).to_dict("records"):
        if pd.notna(target.get("isotopologue")) and str(target["isotopologue"]).strip():
            continue
        if str(target["name"]).endswith(" (IS)") or str(target["name"]).startswith(
            "IS: "
        ):
            continue
        ion = str(target["adduct"])
        if ion not in matlab_annotation_adducts(stream[:2]) or ion_key(ion).endswith(
            "+"
        ) != stream.endswith("Pos"):
            continue
        mass, expected, cohorts, mad = map(
            float,
            (
                target["ref_mz"],
                target["atlas_rt_seconds"],
                target["atlas_training_cohorts"],
                target["atlas_training_mad_seconds"],
            ),
        )
        if (
            not np.isfinite([mass, expected, cohorts, mad]).all()
            or mass <= 0
            or expected <= 0
            or cohorts < 3
            or mad > 10
            or not isinstance(target["atlas_in_domain"], (bool, np.bool_))
            or not target["atlas_in_domain"]
        ):
            continue
        if len(spikes) and (np.abs(spikes / mass - 1) * 1e6 <= 5.01).any():
            continue
        excluded = frozenset(
            local.loc[(local.reference_mz / mass - 1).abs().lt(3e-6), "neutral_key"]
        )
        if excluded not in cache:
            cache[excluded] = _summary(local[~local.neutral_key.isin(excluded)])
        calibration = cache[excluded]
        if not calibration["enabled"]:
            continue
        center = calibration["center"]
        shifted = mass * (1 + center * 1e-6)
        lo, hi = np.searchsorted(
            mz[order], [shifted * (1 - 3e-6), shifted * (1 + 3e-6)]
        )
        hits = order[lo:hi]
        error = (mz[hits] / shifted - 1) * 1e6
        hits = hits[(np.abs(error) < 3) & (np.abs(rt[hits] - expected) <= 10)]
        for pos in hits:
            raw = (mz[pos] / mass - 1) * 1e6
            corrected = (mz[pos] / shifted - 1) * 1e6
            rows.append(
                dict(
                    target,
                    stream=stream,
                    feat_feature_id=ids[pos],
                    feat_mz=mz[pos],
                    feat_rt=rt[pos] / 60,
                    feat_intensity=intensity[pos],
                    ref_rt=expected / 60,
                    rt_expected=expected / 60,
                    rt_error=(rt[pos] - expected) / 60,
                    ppm_error=raw,
                    calibrated_mass_error_ppm=corrected,
                    mass_calibration_center_ppm=center,
                    mass_calibration_anchor_count=calibration["count"],
                    mass_calibration_MAD_ppm=calibration["mad"],
                    mass_calibration_p90_ppm=calibration["p90"],
                    discovery_novel_mass_pair=(
                        str(target["name"]),
                        ion_key(ion),
                        ids[pos],
                    )
                    not in old
                    and abs(raw) >= 3,
                    is_is=False,
                    isotopologue="",
                    automatic_candidate=True,
                    automatic_accepted=False,
                    auto_annotation_status="standard_calibrated_mass_candidate",
                )
            )
    return pd.DataFrame(rows)


def select_calibrated_additions(pool, automatic, previous, *, stream):
    """Fixed conservative completion; no old call can be removed or replaced.

    Full10s/3ppm candidate pool determines uniqueness BEFORE fine gates or
    occupancy. Require one physical feature for identity and one identity for
    feature, RT<=3s,MAD<=5s and strong exact class/CDB anchor in ANOTHER stream.
    An occupied winner or existing identity withholds completion; no promotion.
    """
    if pool.empty:
        return pool.copy()
    if not pool.stream.eq(stream).all():
        raise ValueError("One stream at a time for immutable additive selection")
    unique = pool.drop_duplicates(["name", "adduct", "feat_feature_id"]).copy()
    identity = unique.name.astype(str) + "|" + unique.adduct.map(ion_key)
    identity_count = unique.groupby(identity).feat_feature_id.transform("nunique")
    feature_count = unique.groupby(unique.feat_feature_id.astype(str)).name.transform(
        "size"
    )
    strong = _strong(automatic)
    supported = set(_key(strong[strong.stream.ne(stream)])) if len(strong) else set()
    evidence = _key(unique).isin(supported)
    existing = {
        (str(n), ion_key(a))
        for n, a in previous[["name", "adduct"]].itertuples(index=False, name=None)
    }
    occupied = set(previous.feat_feature_id.astype(str))
    free = pd.Series(
        [
            (str(n), ion_key(a)) not in existing and str(f) not in occupied
            for n, a, f in unique[["name", "adduct", "feat_feature_id"]].itertuples(
                index=False, name=None
            )
        ],
        index=unique.index,
    )
    selected = unique[
        identity_count.eq(1)
        & feature_count.eq(1)
        & unique.discovery_novel_mass_pair
        & unique.rt_error.abs().le(3 / 60)
        & unique.atlas_training_mad_seconds.le(5)
        & evidence
        & free
    ].copy()
    selected["automatic_accepted"] = True
    selected["auto_annotation_status"] = "standard_calibrated_unique_mass_RT_other_mode"
    selected["annotation_evidence"] = (
        "Declared independent standards; leave-target-out mass center; corrected<3ppm; RT<=3s; OTHER-group RT>=3cohorts/MAD<=5s; unique before occupancy; strong OTHER-mode class/CDB support"
    )
    return selected
