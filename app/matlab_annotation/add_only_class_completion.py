"""Unused research: conservative additions; no files, models or default changes."""

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from standard_lipidomics.validation import _species
from .automatic import matlab_annotation_adducts
from .coherent_retention import _protected_mass
from .guarded_class_operating_point import clear_identity_winners
from .standard_rt_library import ion_key

MODES = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")


def free_completion_winners(
    pool, scores, previous, *, stream, protected_features, declared_masses
):
    """Freeze identity winner before physical/ambiguity/occupancy filters."""
    if stream not in MODES or not pool.stream.eq(stream).all():
        raise ValueError("One explicit supported mode required")
    if len(previous) and not previous.stream.eq(stream).all():
        raise ValueError("Previous occupancy belongs to another mode")
    p = pool.copy()
    # Normalization is for equality only, not alias merging in output text.
    p["_completion_name"] = p.name.map(_species)
    p["_completion_ion"] = p.adduct.map(ion_key)
    strongest = (
        p.assign(class_selection_score=scores)
        .sort_values(
            ["class_selection_score", "feat_intensity"], ascending=False, kind="stable"
        )
        .drop_duplicates(["_completion_name", "_completion_ion"])
    )
    fixed_keys = set(strongest.index)
    selected = clear_identity_winners(p, scores, ambiguity_margin=0.05)
    selected = selected.loc[selected.index.isin(fixed_keys)].copy()
    selected["lipid_class"] = selected["class"].map(canonical_lipid_class)
    if selected.empty:
        return selected.drop(columns=["_completion_name", "_completion_ion"])
    occupied = set(previous.feat_feature_id.astype(str)) | set(
        map(str, protected_features)
    )
    identities = set(zip(previous.name.map(_species), previous.adduct.map(ion_key)))
    keep = ~selected.feat_feature_id.astype(str).isin(occupied)
    keep &= pd.Series(
        [
            k not in identities
            for k in zip(selected._completion_name, selected._completion_ion)
        ],
        index=selected.index,
    )
    # Require physical boolean standard flags; never bool("False").
    if not selected.is_is.isin([False, True]).all():
        raise ValueError("Physical pool requires explicit boolean standard flags")
    keep &= ~selected.is_is.astype(bool)
    keep &= ~selected.name.astype(str).str.endswith(" (IS)")
    keep &= ~selected.name.astype(str).str.startswith("IS: ")
    allowed = {ion_key(a) for a in matlab_annotation_adducts(stream[:2])}
    keep &= selected.adduct.map(ion_key).isin(allowed)
    keep &= selected.adduct.map(ion_key).str.endswith(
        "+" if stream.endswith("Pos") else "-"
    )
    keep &= (
        selected.get("isotopologue", pd.Series("", index=selected.index))
        .fillna("")
        .astype(str)
        .eq("")
    )
    numeric = selected[["ref_mz", "feat_mz", "feat_rt", "feat_intensity"]].to_numpy(
        float
    )
    keep &= np.isfinite(numeric).all(axis=1) & (numeric > 0).all(axis=1)
    keep &= ((selected.feat_mz / selected.ref_mz - 1) * 1e6).abs().lt(3)
    masses = np.asarray(list(declared_masses), float)
    if not np.isfinite(masses).all() or (masses <= 0).any():
        raise ValueError("Finite positive declared spike masses required")
    keep &= ~_protected_mass(selected.feat_mz.to_numpy(float), masses)
    keep &= ~_protected_mass(selected.ref_mz.to_numpy(float), masses)
    deviations = [
        (
            selected.feat_rt * 60 - pd.to_numeric(selected[c], errors="coerce") * scale
        ).abs()
        for c, scale in (("ref_rt", 60), ("rt_expected", 60), ("atlas_rt_seconds", 1))
        if c in selected
    ]
    if not deviations:
        raise ValueError("Authorized numeric RT reference required")
    keep &= pd.concat(deviations, axis=1).min(axis=1).le(10 + 1e-9)
    result = (
        selected.loc[keep].drop(columns=["_completion_name", "_completion_ion"]).copy()
    )
    return result


def select_completion(proposals, thresholds):
    """Unlisted class/modes are disabled, including scores exactly one."""
    cuts = pd.Series(
        [
            float(thresholds.get(s, {}).get(c, 1.0))
            for s, c in proposals[["stream", "lipid_class"]].itertuples(
                index=False, name=None
            )
        ],
        index=proposals.index,
        dtype=float,
    )
    if not np.isfinite(cuts).all() or not cuts.between(0, 1).all():
        raise ValueError("Explicit valid class cutoffs required")
    selected = proposals.loc[
        cuts.lt(1) & proposals.class_selection_score.ge(cuts)
    ].copy()
    selected["class_completion_threshold"] = cuts.loc[selected.index]
    selected["automatic_accepted"] = True
    selected["automatic_candidate"] = False
    selected["auto_annotation_status"] = "conservative_add_only_class_completion"
    return selected


def class_completion_curves(
    proposals, manual, baseline_counts, *, training_groups, excluded_groups
):
    """Only authorized project-held-out TRAINING predictions may carry labels."""
    allowed, excluded = set(training_groups), set(excluded_groups)
    if not allowed or not excluded or allowed & excluded:
        raise ValueError("Disjoint explicit training/excluded groups required")
    if (set(proposals.project) | set(manual.project)) - allowed:
        raise ValueError("Excluded project entered calibration")
    if manual.internal_standard.fillna(False).any():
        raise ValueError("Only endogenous manual calibration rows allowed")
    for c in (
        "training_positive",
        "training_label_known",
        "calibration_identity_in_manual",
    ):
        if not proposals[c].isin([True, False]).all():
            raise ValueError("Explicit calibration booleans required")
    if (proposals.training_positive & ~proposals.training_label_known).any():
        raise ValueError("Positive labels must be known")
    if (
        not np.isfinite(proposals.class_selection_score).all()
        or not proposals.class_selection_score.between(0, 1).all()
    ):
        raise ValueError("Finite agreement ranks in0..1 required")
    p, m = proposals.copy(), manual.copy()
    p["lipid_class"] = p.lipid_class.map(canonical_lipid_class)
    m["lipid_class"] = m.lipid_class.map(canonical_lipid_class)
    keys = sorted(
        set(map(tuple, m[["stream", "lipid_class"]].to_numpy()))
        | set(map(tuple, p[["stream", "lipid_class"]].to_numpy()))
    )
    rows = []
    for stream, cls in keys:
        q = p.loc[p.stream.eq(stream) & p.lipid_class.eq(cls)]
        ref = m.loc[m.stream.eq(stream) & m.lipid_class.eq(cls)]
        n, base = len(ref), int(baseline_counts.get((stream, cls), 0))
        if not 0 <= base <= n:
            raise ValueError("Invalid reconstructed baseline count")
        eligible = (
            n >= 100 and ref.project.nunique() >= 3 and base < 0.9 * n and cls != "X4"
        )
        pos = q.loc[q.training_positive]
        levels = {1.0}
        if eligible and len(pos):
            values = np.sort(pos.class_selection_score.to_numpy(float))[::-1]
            levels.update(
                float(values[int(np.ceil(f * len(values))) - 1])
                for f in np.arange(0.05, 1.001, 0.05)
            )
        for cut in sorted(levels, reverse=True):
            picked = q.loc[q.class_selection_score.ge(cut)] if cut < 1 else q.iloc[:0]
            correct = int(picked.training_positive.sum())
            extras = int((~picked.calibration_identity_in_manual).sum())
            wrong = int(
                (
                    picked.calibration_identity_in_manual
                    & picked.training_label_known
                    & ~picked.training_positive
                ).sum()
            )
            if base + correct > n:
                raise ValueError(
                    "Occupied correct identities must not be proposed twice"
                )
            counts = picked.groupby("project").training_positive.agg(["sum", "size"])
            remaining = [
                (correct - r["sum"]) / (len(picked) - r["size"])
                for _, r in counts.iterrows()
                if len(picked) > r["size"]
            ]
            leave_one_out = min(remaining, default=0.0)
            substantial = counts.loc[counts["size"].ge(3)]
            worst = (
                float((substantial["sum"] / substantial["size"]).min())
                if len(substantial)
                else 0.0
            )
            feasible = cut == 1.0 or bool(
                eligible
                and correct >= 10
                and pos.loc[pos.class_selection_score.ge(cut), "project"].nunique() >= 3
                and wrong == 0
                and extras <= 0.2 * correct
                and extras <= 0.02 * n
                and correct >= 0.8 * len(picked)
                and leave_one_out >= 0.8
                and worst >= 0.5
            )
            rows.append(
                dict(
                    stream=stream,
                    lipid_class=cls,
                    threshold=cut,
                    manual=n,
                    manual_projects=int(ref.project.nunique()),
                    core_correct=base,
                    eligible=eligible,
                    added_correct=correct,
                    extras=extras,
                    different_peak=wrong,
                    added_calls=len(picked),
                    positive_projects=int(
                        picked.loc[picked.training_positive, "project"].nunique()
                    ),
                    minimum_leave_one_project_agreement=leave_one_out,
                    worst_substantial_project_agreement=worst,
                    feasible=feasible,
                    capped_gain=min(correct, max(0, int(np.ceil(0.9 * n)) - base)),
                )
            )
    return pd.DataFrame(rows)


def choose_completion_thresholds(table):
    thresholds = {s: {} for s in MODES}
    chosen = []
    for (stream, cls), group in table.groupby(["stream", "lipid_class"], sort=True):
        feasible = group.loc[group.feasible]
        if len(feasible.loc[feasible.threshold.eq(1)]) != 1:
            raise ValueError("Each class requires one explicit disabled fallback")
        best = feasible.sort_values(
            ["capped_gain", "extras", "added_calls", "threshold"],
            ascending=[False, True, True, False],
            kind="stable",
        ).iloc[0]
        thresholds[stream][cls] = float(best.threshold)
        chosen.append(best.name)
    return thresholds, table.loc[chosen].copy()
