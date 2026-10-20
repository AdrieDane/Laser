"""Final, additive check of explicitly declared PE internal standards.

Known short-chain spikes need not follow an endogenous class grid. This pass
requires one clear monoisotopic declared-standard peak inside the original
mass/RT limits; it does not guess standards from endogenous compositions.
It runs after all phase-local models, so it cannot influence their fitting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .result_assembly import assemble_result


def _keys(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series("", index=frame.index, dtype=object)
    return (
        frame[["feat_feature_id", "name", "adduct"]]
        .fillna("")
        .astype(str)
        .agg("\x1f".join, axis=1)
    )


def complete_pe_internal_standards(
    result: dict,
    *,
    enabled: bool = True,
    ppm_tolerance: float = 3.0,
    rt_seconds: float = 10.0,
) -> dict:
    if not 0 < ppm_tolerance <= 3 or not 0 < rt_seconds <= 10:
        raise ValueError("PE standards must stay within 3 ppm / 10 seconds")
    if not enabled or result.get("pe_standard_completion_applied"):
        return result.copy()
    updated = result.copy()
    streams = {s: p.copy() for s, p in result.get("streams", {}).items()}
    audits = []
    added_count = 0
    for stream, part in streams.items():
        annotations = part.get("annotations", pd.DataFrame())
        pairs = part.get("all_pairs", pd.DataFrame())
        required = {
            "class",
            "name",
            "adduct",
            "feat_feature_id",
            "is_is",
            "internal_standard_name",
            "ppm_error",
            "rt_error",
        }
        if pairs.empty or not required.issubset(pairs):
            continue
        adduct = "[M-H]-" if stream.endswith("Neg") else "[M+H]+"
        mask = pairs["class"].eq("PE") & pairs.adduct.eq(adduct)
        mask &= pairs.is_is.fillna(False).astype(bool)
        mask &= pairs.internal_standard_name.fillna("").astype(str).ne("")
        mask &= pairs.ppm_error.abs().lt(ppm_tolerance)
        mask &= pairs.rt_error.abs().mul(60).le(rt_seconds)
        mask &= (
            pairs.get("isotopologue", pd.Series("", index=pairs.index))
            .fillna("")
            .eq("")
        )
        pool = pairs[mask].copy()
        occupied = (
            set(annotations.feat_feature_id.astype(str))
            if not annotations.empty
            else set()
        )
        names = (
            set(zip(annotations["name"], annotations.adduct))
            if not annotations.empty
            else set()
        )
        proposals = []
        for (name, ion, _), group in pool.groupby(
            ["name", "adduct", "internal_standard_name"], sort=True
        ):
            # Resolve ambiguity before occupied-feature exclusion: a conflicting
            # accepted feature cannot make an ambiguous standard look unique.
            group = group.drop_duplicates("feat_feature_id")
            if len(group) != 1 or (name, ion) in names:
                continue
            row = group.iloc[[0]].copy()
            if str(row.feat_feature_id.iloc[0]) in occupied:
                continue
            row["automatic_accepted"] = True
            row["automatic_candidate"] = True
            row["auto_annotation_status"] = "pe_declared_standard_supported"
            row["status"] = "pe_declared_standard_supported"
            row["rank_for_feature"] = 1
            row["standard_stream"], row["chromatography"] = stream, stream.split("_")[0]
            row["pe_standard_completion_supported"] = True
            row["grid_coherence_exemption"] = (
                "declared_PE_standard_unique_mass_RT_match"
            )
            proposals.append(row)
            occupied.add(str(row.feat_feature_id.iloc[0]))
            names.add((name, ion))
        added = (
            pd.concat(proposals, ignore_index=True, sort=False)
            if proposals
            else pool.iloc[:0].copy()
        )
        if not added.empty:
            part["annotations"] = pd.concat(
                [annotations, added], ignore_index=True, sort=False
            )
            chosen = set(_keys(added))
            part["all_pairs"] = pd.concat(
                [pairs[~_keys(pairs).isin(chosen)], added],
                ignore_index=True,
                sort=False,
            )
            added_count += len(added)
        audits.append(
            dict(
                phase=stream.split("_")[0],
                stream=stream,
                method="pe_declared_internal_standard",
                status="completed",
                added_pe=len(added),
                restored_pe_internal_standards=len(added),
                criterion="explicitly declared; unique monoisotopic peak; <3 ppm; <=10 seconds; unused feature",
            )
        )
    updated["streams"] = streams
    if added_count:
        assembled = assemble_result(streams, [], Path("."), Path("."))
        for name in (
            "all_candidates",
            "automatic_annotations",
            "identified_lipids",
            "equivalent_lipids",
        ):
            updated[name] = assembled[name]
        standards = updated.get("matlab_internal_standard_audit", pd.DataFrame()).copy()
        if not standards.empty:
            for index, row in standards.iterrows():
                accepted = sum(
                    int(
                        p["annotations"]
                        .get("internal_standard_name", pd.Series(dtype=str))
                        .eq(row.internal_standard_name)
                        .sum()
                    )
                    for s, p in streams.items()
                    if s.startswith(str(row.phase) + "_")
                )
                standards.loc[index, "accepted_matches"] = accepted
                if accepted:
                    standards.loc[index, "status"] = "accepted"
            updated["matlab_internal_standard_audit"] = standards
    existing = updated.get("matlab_pe_completion_audit", pd.DataFrame())
    updated["matlab_pe_completion_audit"] = pd.concat(
        [existing, pd.DataFrame(audits)], ignore_index=True, sort=False
    )
    updated["pe_standard_completion_applied"] = True
    return updated
