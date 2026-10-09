"""Shared reversed-phase lipid-isomer policy.

RP separates positional lyso isomers and ether/plasmalogen siblings.  Candidate
engines may enumerate those identities independently, but acceptance must be a
joint decision: one feature cannot simultaneously be an O- and P- identity,
and LPC/LPE 1-acyl/2-acyl peaks must follow their chromatographic order.

The positional decision is shared with the standard XCMS workflow.  The small
adapter here gives the 4D workflow the same policy while preserving all losing
candidates for review and audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from matlab_annotation.isomer_resolver import (
    PositionalIsomerSettings,
    resolve_positional_lyso_isomers,
)


_POSITIONAL_LYSO = re.compile(
    r"^(?P<position>[12])-acyl\s*(?P<family>LPC|LPE)\(",
    re.IGNORECASE,
)


def positional_lyso_class(name: object) -> str | None:
    """Return the canonical RP class encoded by a positional lyso name."""
    match = _POSITIONAL_LYSO.match(str(name).strip())
    if not match:
        return None
    return f"{int(match.group('position'))}-acyl{match.group('family').upper()}"


def rp_grid_class(
    classes: pd.Series,
    bases: pd.Series,
    names: pd.Series | None = None,
) -> pd.Series:
    """Return RP-resolved class labels used by grids, reports and the viewer."""
    if names is None:
        names = pd.Series("", index=classes.index, dtype=object)
    values: list[str] = []
    for lipid_class, base, name in zip(classes, bases, names, strict=True):
        positional = positional_lyso_class(name)
        if positional:
            values.append(positional)
        elif isinstance(base, str) and base.strip():
            values.append(f"{lipid_class}-{base.strip()}")
        else:
            values.append(str(lipid_class))
    return pd.Series(values, index=classes.index, dtype=object)


def _row_keys(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["feat_feature_id"].astype(str) + "\x1f"
        + frame["name"].astype(str) + "\x1f"
        + frame["adduct"].astype(str)
    )


def _ether_pair_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {
        "feat_feature_id", "class", "base", "carbon", "db", "adduct",
    }.issubset(frame.columns):
        return frame.iloc[0:0].copy()
    base = frame["base"].fillna("").astype(str).str.upper()
    meta = frame.loc[base.isin(("O", "P"))].copy()
    if meta.empty:
        return meta
    meta["_rp_base"] = base.loc[meta.index]
    carbon = pd.to_numeric(meta["carbon"], errors="coerce")
    double_bonds = pd.to_numeric(meta["db"], errors="coerce")
    # O-(C:DB+1) and P-(C:DB) are elemental-composition isobars.
    equivalent_db = double_bonds - meta["_rp_base"].eq("O").astype(int)
    isotopologue = meta.get(
        "isotopologue", pd.Series("", index=meta.index, dtype=object)
    ).fillna("").astype(str)
    meta["_rp_isomer_group"] = list(zip(
        meta["feat_feature_id"].astype(str),
        meta["class"].astype(str),
        carbon,
        equivalent_db,
        meta["adduct"].astype(str),
        isotopologue,
    ))
    return meta


def resolve_ether_plasmalogen_isomers(
    annotations: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select one RP O/P identity per feature and preserve alternatives.

    Only true elemental-composition siblings compete. A solitary O or P match
    is retained. If both pass the annotation windows, the candidate with the
    best combined mass/RT/CCS score wins; the other remains in ``all_pairs`` as
    an explicit review candidate.
    """
    accepted = annotations.copy()
    decorated = candidates.copy()
    for frame in (accepted, decorated):
        if "ether_plasmalogen_selected" not in frame:
            frame["ether_plasmalogen_selected"] = False
        if "ether_plasmalogen_status" not in frame:
            frame["ether_plasmalogen_status"] = "not_applicable"

    accepted_meta = _ether_pair_metadata(accepted)
    candidate_meta = _ether_pair_metadata(decorated)
    if accepted_meta.empty:
        return accepted, decorated, pd.DataFrame()

    remove_indices: list[object] = []
    audit: list[dict[str, object]] = []
    accepted_keys = set(_row_keys(accepted_meta))
    for group_key, group in accepted_meta.groupby(
        "_rp_isomer_group", sort=False, dropna=False,
    ):
        bases = set(group["_rp_base"])
        score = pd.to_numeric(
            group.get("score", pd.Series(np.inf, index=group.index)),
            errors="coerce",
        ).fillna(np.inf)
        ordered = group.assign(_rp_score=score).sort_values(
            ["_rp_score", "name"], kind="stable",
        )
        winner = ordered.index[0]
        losers = [index for index in group.index if index != winner]
        remove_indices.extend(losers)
        status = "resolved_by_evidence_score" if bases == {"O", "P"} else (
            "single_subclass_supported"
        )
        accepted.loc[winner, "ether_plasmalogen_selected"] = True
        accepted.loc[group.index, "ether_plasmalogen_status"] = status
        runner_score = (
            float(ordered.iloc[1]["_rp_score"]) if len(ordered) > 1 else np.nan
        )
        audit.append({
            "feature_id": str(group.iloc[0]["feat_feature_id"]),
            "lipid_class": str(group.iloc[0]["class"]),
            "carbon": group_key[2],
            "equivalent_double_bonds": group_key[3],
            "adduct": str(group.iloc[0]["adduct"]),
            "candidate_subclasses": "; ".join(sorted(bases)),
            "selected_identity": str(accepted.loc[winner, "name"]),
            "selected_subclass": str(ordered.loc[winner, "_rp_base"]),
            "selected_score": float(ordered.iloc[0]["_rp_score"]),
            "runner_up_score": runner_score,
            "score_margin": runner_score - float(ordered.iloc[0]["_rp_score"]),
            "status": status,
        })

    if remove_indices:
        accepted = accepted.drop(index=remove_indices)

    if not candidate_meta.empty:
        for _, group in candidate_meta.groupby(
            "_rp_isomer_group", sort=False, dropna=False,
        ):
            passing = group[_row_keys(group).isin(accepted_keys)]
            if passing.empty:
                decorated.loc[group.index, "ether_plasmalogen_status"] = (
                    "outside_annotation_windows"
                )
                continue
            passing_scores = pd.to_numeric(
                passing.get("score", pd.Series(np.inf, index=passing.index)),
                errors="coerce",
            ).fillna(np.inf)
            selected = passing_scores.sort_values(kind="stable").index[0]
            selected_name = str(passing.loc[selected, "name"])
            decorated.loc[group.index, "ether_plasmalogen_status"] = (
                "resolved_by_evidence_score"
                if set(passing["_rp_base"]) == {"O", "P"}
                else "single_subclass_supported"
            )
            decorated.loc[
                group.index[group["name"].astype(str).eq(selected_name)],
                "ether_plasmalogen_selected",
            ] = True

    return (
        accepted.reset_index(drop=True),
        decorated.reset_index(drop=True),
        pd.DataFrame(audit),
    )


@dataclass(frozen=True)
class RpIsomerResolution:
    annotations: pd.DataFrame
    all_pairs: pd.DataFrame
    positional_audit: pd.DataFrame
    ether_plasmalogen_audit: pd.DataFrame


def resolve_rp_isomers(
    annotations: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    reference: pd.DataFrame,
    ppm_tolerance: float,
) -> RpIsomerResolution:
    """Apply all shared RP isomer decisions to a candidate-engine result."""
    resolved, decorated, positional_audit = resolve_positional_lyso_isomers(
        annotations,
        candidates,
        reference=reference,
        settings=PositionalIsomerSettings(ppm_tolerance=ppm_tolerance),
    )
    resolved, decorated, ether_audit = resolve_ether_plasmalogen_isomers(
        resolved, decorated,
    )
    for frame in (resolved, decorated):
        if {"class", "base", "name"}.issubset(frame.columns):
            frame["grid_class"] = rp_grid_class(
                frame["class"], frame["base"], frame["name"],
            )
    return RpIsomerResolution(
        annotations=resolved,
        all_pairs=decorated,
        positional_audit=positional_audit,
        ether_plasmalogen_audit=ether_audit,
    )
