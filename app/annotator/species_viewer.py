"""Programmatic grouping and MS/MS summaries for the lipid-species viewer.

The review GUI usually shows one best feature per lipid species on a class
grid.  This module deliberately inverts that relationship: one lipid species is
selected and every feature–adduct candidate carrying that exact species ID is
retained.  Strict class-rule failures can be supplied separately so the viewer
can show why they were excluded without changing the annotation result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = ("feat_feature_id", "name", "adduct")


def collect_lipid_species_candidates(
    annotations: pd.DataFrame,
    rule_failures: pd.DataFrame | None = None,
    grid_failures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one row per feature–species–adduct candidate.

    ``annotations`` contains candidates retained by the current annotation
    rules. ``rule_failures`` may contain candidates removed by strict MS/MS
    filtering. ``grid_failures`` may contain candidates removed by experimental
    local grid-coherence filtering. Retained rows win when the same identity
    occurs in more than one input, which is the expected case when either
    optional filter is disabled.
    """
    kept = annotations.copy()
    kept["rule_excluded"] = False
    kept["grid_excluded"] = False
    frames = [kept]
    if rule_failures is not None and not rule_failures.empty:
        excluded = rule_failures.copy()
        excluded["rule_excluded"] = True
        excluded["grid_excluded"] = False
        frames.append(excluded)
    if grid_failures is not None and not grid_failures.empty:
        excluded = grid_failures.copy()
        excluded["rule_excluded"] = False
        excluded["grid_excluded"] = True
        frames.append(excluded)
    if not frames or all(frame.empty for frame in frames):
        empty = annotations.copy()
        empty["rule_excluded"] = pd.Series(dtype=bool)
        empty["grid_excluded"] = pd.Series(dtype=bool)
        return empty

    candidates = pd.concat(frames, ignore_index=True, sort=False)
    missing = [column for column in IDENTITY_COLUMNS if column not in candidates]
    if missing:
        raise ValueError(
            "Species candidates require columns: " + ", ".join(missing)
        )
    candidates["feat_feature_id"] = candidates["feat_feature_id"].astype(str)
    candidates["name"] = candidates["name"].astype(str)
    candidates["adduct"] = candidates["adduct"].astype(str)
    candidates["key"] = (
        candidates["feat_feature_id"]
        + "|" + candidates["name"]
        + "|" + candidates["adduct"]
    )
    candidates = candidates.drop_duplicates(
        list(IDENTITY_COLUMNS), keep="first"
    )
    candidates["_feature_number"] = pd.to_numeric(
        candidates["feat_feature_id"], errors="coerce"
    )
    candidates = candidates.sort_values(
        [
            "name", "rule_excluded", "grid_excluded", "_feature_number",
            "feat_feature_id", "adduct",
        ],
        na_position="last",
    ).drop(columns="_feature_number")
    return candidates.reset_index(drop=True)


def lipid_species_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    """Summarise candidate and evidence counts for every exact species ID."""
    columns = [
        "species_id", "lipid_class", "candidate_pairs", "feature_count",
        "msms_spectra", "rule_supported", "rule_absent", "rule_excluded",
        "grid_excluded",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    work = candidates.copy()
    for column, default in (
        ("class", ""),
        ("msms_has_spectrum", False),
        ("msms_status", "not_evaluated"),
        ("rule_excluded", False),
        ("grid_excluded", False),
    ):
        if column not in work:
            work[column] = default
    work["_has_msms"] = work["msms_has_spectrum"].fillna(False).astype(bool)
    work["_supported"] = work["msms_status"].eq("supported")
    work["_absent"] = work["msms_status"].eq("not_detected")
    work["_excluded"] = work["rule_excluded"].fillna(False).astype(bool)
    work["_grid_excluded"] = work["grid_excluded"].fillna(False).astype(bool)

    feature_ids = work["feat_feature_id"].astype(str)
    for target, mask in (
        ("_msms_feature", work["_has_msms"]),
        ("_supported_feature", work["_supported"]),
        ("_absent_feature", work["_absent"]),
        ("_excluded_feature", work["_excluded"]),
        ("_grid_excluded_feature", work["_grid_excluded"]),
    ):
        work[target] = feature_ids.where(mask)
    class_column = "grid_class" if "grid_class" in work else "class"
    grouped = work.groupby("name", sort=False, dropna=False)
    result = grouped.agg(
        lipid_class=(
            class_column,
            lambda values: " / ".join(sorted({
                value for value in values.dropna().astype(str) if value
            })),
        ),
        candidate_pairs=("feat_feature_id", "size"),
        feature_count=("feat_feature_id", "nunique"),
        msms_spectra=("_msms_feature", "nunique"),
        rule_supported=("_supported_feature", "nunique"),
        rule_absent=("_absent_feature", "nunique"),
        rule_excluded=("_excluded_feature", "nunique"),
        grid_excluded=("_grid_excluded_feature", "nunique"),
    ).reset_index().rename(columns={"name": "species_id"})
    result["species_id"] = result["species_id"].astype(str)
    return (
        result.reindex(columns=columns)
        .sort_values(
            ["feature_count", "candidate_pairs", "species_id"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def class_evidence_text(row: pd.Series) -> str:
    """Return a concise product-ion/neutral-loss evidence description."""
    rule_id = str(row.get("msms_rule_id", "") or "").strip()
    status = str(row.get("msms_status", "not_evaluated") or "not_evaluated")
    evidence_name = str(
        row.get("msms_rule_evidence_name", "") or "class-specific evidence"
    )
    evidence_type = str(
        row.get("msms_rule_evidence_type", "") or "product_ion"
    )
    if not rule_id:
        return "No class-specific MS/MS rule configured"
    if status == "no_msms":
        return f"{evidence_name}: no MS/MS spectrum"
    if status == "not_detected":
        target = pd.to_numeric(
            pd.Series([row.get("msms_rule_target_mz")]), errors="coerce"
        ).iloc[0]
        target_text = (
            f" {target:.4f} Da"
            if np.isfinite(target) and evidence_type == "neutral_loss"
            else f" m/z {target:.4f}"
            if np.isfinite(target)
            else ""
        )
        kind = "neutral loss" if evidence_type == "neutral_loss" else "fragment"
        return f"{evidence_name}: {kind}{target_text} not detected"
    if status != "supported":
        return f"{evidence_name}: not evaluated"

    observed_fragment = pd.to_numeric(
        pd.Series([row.get("msms_rule_observed_mz")]), errors="coerce"
    ).iloc[0]
    observed_value = pd.to_numeric(
        pd.Series([row.get("msms_rule_observed_value")]), errors="coerce"
    ).iloc[0]
    relative = pd.to_numeric(
        pd.Series([row.get("msms_rule_relative_intensity")]), errors="coerce"
    ).iloc[0]
    if evidence_type == "neutral_loss":
        detail = (
            f"neutral loss {observed_value:.4f} Da"
            if np.isfinite(observed_value)
            else "neutral loss detected"
        )
        if np.isfinite(observed_fragment):
            detail += f" via fragment m/z {observed_fragment:.4f}"
    elif evidence_type == "fatty_acyl":
        detected = str(row.get("fatty_acyl_detected", "") or "")
        detail = f"detected ({detected})" if detected else "detected"
    else:
        detail = (
            f"fragment m/z {observed_fragment:.4f} detected"
            if np.isfinite(observed_fragment)
            else "fragment detected"
        )
    if np.isfinite(relative):
        detail += f" · {relative:.1f}% relative intensity"
    return f"{evidence_name}: {detail}"
