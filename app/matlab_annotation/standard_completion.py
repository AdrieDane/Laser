"""Research-only, conservative handling of explicitly declared standards.

No curated workbook or dataset name is read. Chemical declarations, frozen
OTHER-group models and original features are explicit caller-supplied inputs.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .standard_rt_library import (
    ion_key,
    nominal_standard_name,
    standard_report_name,
    predict_standard_retention,
)
from .standard_rt_transfer import (
    predict_standard_class_transfer,
    combine_standard_rt_sources,
)


def fit_standard_channel_prior(observations):
    required = {"cohort", "stream", "name", "adduct"}
    if not required.issubset(observations):
        raise ValueError(
            "Standard-channel priors require grouped standard observations"
        )
    data = observations[list(sorted(required))].copy()
    data["name"] = data.name.map(nominal_standard_name)
    data["adduct"] = data.adduct.map(ion_key)
    models = {}
    for (stream, name), group in data.groupby(["stream", "name"], sort=True):
        models[stream + "|" + name] = dict(
            cohorts=int(group.cohort.nunique()),
            adducts={
                str(a): int(n)
                for a, n in group.groupby("adduct").cohort.nunique().items()
            },
        )
    return dict(format="laser-standard-channel-prior-v1", models=models)


def _standards(frame):
    flags = (
        frame.get("is_is", pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(bool)
    )
    return (
        flags
        | frame.name.astype(str).str.endswith(" (IS)")
        | frame.name.astype(str).str.startswith("IS: ")
    )


def _manual(frame):
    result = pd.Series(False, index=frame.index)
    for c in ["manual_confirmed", "manually_confirmed", "manual_annotation"]:
        if c in frame:
            result |= frame[c].fillna(False).astype(bool)
    if "status" in frame:
        result |= frame.status.astype(str).str.contains(
            "manual", case=False, regex=False
        )
    return result


def _matching_targets(row, targets, *, allow_reporting_name):
    names = targets.name.map(standard_report_name)
    wanted = standard_report_name(row["name"])
    if not str(row["name"]).endswith(" (IS)"):
        wanted += " (IS)"
    same = names.eq(wanted)
    if allow_reporting_name:
        same |= targets.standard_report_name.map(standard_report_name).eq(wanted)
    ion = targets.adduct.map(ion_key).eq(ion_key(row["adduct"]))
    mass = float(row.get("ref_mz", np.nan))
    close = (
        (targets.ref_mz - mass)
        .abs()
        .le(np.maximum(1e-5, targets.ref_mz.abs() * 0.05e-6))
    )
    return targets[same & ion & close]


def _clean_existing(annotations, targets, channels, stream):
    kept = annotations.copy()
    standards = _standards(kept)
    manual = _manual(kept)
    remove = []
    audit = []
    for i, row in kept.iterrows():
        if manual.loc[i]:
            continue
        if standards.loc[i]:
            prior = channels["models"].get(
                stream + "|" + nominal_standard_name(row["name"])
            )
            if (
                prior is not None
                and prior["cohorts"] >= 3
                and prior["adducts"].get(ion_key(row["adduct"]), 0) == 0
            ):
                remove.append(i)
                audit.append(
                    dict(
                        action="demote_unvalidated_standard_channel",
                        feature_id=str(row.feat_feature_id),
                        name=row["name"],
                    )
                )
                continue
            matches = _matching_targets(row, targets, allow_reporting_name=True)
            if len(matches) != 1:
                continue
            target = matches.iloc[0]
            label = str(target.standard_report_name)
            old = standard_report_name(row["name"])
            if not label or label == old:
                continue
            if re.match(r"^[12]-acyl", old):
                continue  # Never overwrite an explicit other sn position.
            expected = float(target.standard_rt_seconds)
            mass = (
                (float(row.feat_mz) - float(target.ref_mz)) / float(target.ref_mz) * 1e6
            )
            if (
                not np.isfinite(expected)
                or abs(float(row.feat_rt) * 60 - expected) > 10
                or abs(mass) >= 3
            ):
                continue
            kept.loc[i, "name"] = label
            kept.loc[i, "class"] = target.standard_report_class or row["class"]
            kept.loc[i, "auto_annotation_status"] = (
                "declared_standard_reporting_consensus"
            )
            audit.append(
                dict(
                    action="resolve_standard_reporting_label",
                    feature_id=str(row.feat_feature_id),
                    name=label,
                )
            )
        else:
            matches = _matching_targets(row, targets, allow_reporting_name=True)
            if len(matches):
                remove.append(i)
                audit.append(
                    dict(
                        action="demote_mass_identical_spike_alias",
                        feature_id=str(row.feat_feature_id),
                        name=row["name"],
                    )
                )
    demoted = kept.loc[remove].copy()
    if len(demoted):
        demoted["automatic_accepted"] = False
        demoted["automatic_candidate"] = True
        demoted["auto_annotation_status"] = "declared_standard_companion_or_alias"
    return kept.drop(index=remove).copy(), demoted, audit


def _discover_unique(targets, features, kept):
    masses = pd.to_numeric(features.mz, errors="coerce").to_numpy(float)
    times = pd.to_numeric(features.rt, errors="coerce").to_numpy(float) * 60
    intensity = pd.to_numeric(features.intensity, errors="coerce").to_numpy(float)
    ids = features.feature_id.astype(str).to_numpy()
    valid = np.flatnonzero(
        np.isfinite(masses)
        & np.isfinite(times)
        & np.isfinite(intensity)
        & (masses > 0)
        & (intensity > 0)
    )
    order = valid[np.argsort(masses[valid], kind="stable")]
    sorted_mass = masses[order]
    occupied = set(kept.feat_feature_id.astype(str))
    existing = kept[_standards(kept)].copy()
    if len(existing):
        existing["_nominal"] = existing.name.map(nominal_standard_name)
        existing["_ion"] = existing.adduct.map(ion_key)
    proposals = []
    candidates = []
    audit = []
    for target in targets.to_dict("records"):
        name = target["standard_report_name"] or target["name"]
        rt = float(target["standard_rt_seconds"])
        mz = float(target["ref_mz"])
        if len(existing):
            same = existing._nominal.eq(
                nominal_standard_name(target["name"])
            ) & existing._ion.eq(ion_key(target["adduct"]))
            if "ref_mz" in existing:
                same &= (
                    pd.to_numeric(existing.ref_mz, errors="coerce")
                    .sub(mz)
                    .abs()
                    .le(max(1e-5, mz * 0.05e-6))
                )
            if same.any():
                continue  # Preserve old peak choice rather than force an ambiguous replacement.
        if not target["standard_adduct_supported"] or not np.isfinite(rt):
            audit.append(dict(action="standard_reference_unavailable", name=name))
            continue
        lo = np.searchsorted(sorted_mass, mz * (1 - 3e-6))
        hi = np.searchsorted(sorted_mass, mz * (1 + 3e-6), side="right")
        hits = order[lo:hi]
        hits = hits[
            (np.abs((masses[hits] - mz) / mz * 1e6) < 3)
            & (np.abs(times[hits] - rt) <= 10)
        ]
        rows = []
        for j in hits:
            row = {
                **target,
                "name": name,
                "class": target["standard_report_class"] or target["class"],
                "feat_feature_id": str(ids[j]),
                "feat_mz": masses[j],
                "feat_rt": times[j] / 60,
                "feat_intensity": intensity[j],
                "rt_expected": rt / 60,
                "ref_rt": rt / 60,
                "rt_error": (times[j] - rt) / 60,
                "ppm_error": (masses[j] - mz) / mz * 1e6,
                "is_is": True,
                "isotopologue": "",
                "automatic_candidate": True,
                "automatic_accepted": False,
                "auto_annotation_status": "declared_standard_mass_RT_candidate",
            }
            rows.append(row)
            candidates.append(row)
        # Ambiguity is resolved BEFORE occupied-feature exclusion.
        if len(rows) != 1:
            audit.append(
                dict(
                    action="standard_ambiguous_or_absent",
                    name=name,
                    candidates=len(rows),
                )
            )
            continue
        row = rows[0]
        if row["feat_feature_id"] in occupied:
            audit.append(
                dict(
                    action="standard_feature_occupied",
                    name=name,
                    feature_id=row["feat_feature_id"],
                )
            )
            continue
        proposals.append(row)
    proposals = pd.DataFrame(proposals)
    if len(proposals):
        conflict = proposals.feat_feature_id.duplicated(keep=False)
        for row in proposals[conflict].to_dict("records"):
            audit.append(
                dict(
                    action="conflicting_declared_standard_identities",
                    name=row["name"],
                    feature_id=row["feat_feature_id"],
                )
            )
        proposals = proposals[~conflict].copy()
        proposals["automatic_accepted"] = True
        proposals["auto_annotation_status"] = "declared_standard_unique_mass_RT"
        for row in proposals.to_dict("records"):
            audit.append(
                dict(
                    action="complete_declared_standard",
                    name=row["name"],
                    feature_id=row["feat_feature_id"],
                )
            )
    return proposals, pd.DataFrame(candidates), audit


def complete_declared_standards(
    annotations,
    features,
    declared,
    standard_library,
    atlas,
    transfer,
    channels,
    *,
    stream,
    alignment_annotations,
):
    """Conservative completion with a FROZEN independent alignment stage.

    Never bootstrap new standards into their own alignment. Manual rows and
    occupied non-alias features are protected. Additional channels without any
    support in >=3 training groups remain companion candidates. Existing peak
    choices are not forcibly replaced by this first completion experiment.
    """
    if channels.get("format") != "laser-standard-channel-prior-v1":
        raise ValueError("Unsupported declared-standard channel prior")
    if features.feature_id.astype(str).duplicated().any():
        raise ValueError("Original features must have unique IDs")
    primary = predict_standard_retention(
        declared, alignment_annotations, standard_library, stream=stream
    )
    other = predict_standard_class_transfer(
        declared, alignment_annotations, atlas, transfer, stream=stream
    )
    targets = declared.join(combine_standard_rt_sources(primary, other))
    kept, demoted, audit = _clean_existing(annotations, targets, channels, stream)
    added, candidates, decisions = _discover_unique(targets, features, kept)
    selected = pd.concat([kept, added], ignore_index=True)
    assert not selected.feat_feature_id.astype(str).duplicated().any()
    return (
        selected,
        pd.concat([candidates, demoted], ignore_index=True),
        pd.DataFrame([*audit, *decisions]),
    )
