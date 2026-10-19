"""RP-only, ordered-isomer anchors with independent local RT correction.

The reference grid is the calibrated RTDB prior, not the staged mapper's final
regression. Strong, unambiguous isomer pairs establish bounded local offsets
from that prior. A composition never contributes to its own prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from annotator.grid_coherence import score_grid_coherence

from .confidence import (
    PostRegressionCompletionSettings,
    _candidate_features,
    _load_model,
    _preserve_count,
    _protected_rows,
    _row_keys,
    _score_features,
    _select_confident,
    _strict_candidate_mask,
)
from .fitted_grid_completion import FittedGridCompletionSettings


# Additional channels retain the preceding engine's decision; these are the
# method's validated primary channels for introducing new RP isomer calls.
PRIMARY_CHANNELS = {
    "PC-O": "[M+H]+",
    "PC-P": "[M+H]+",
    "PE-O": "[M-H]-",
    "PE-P": "[M-H]-",
    "1-acylLPC": "[M+H]+",
    "2-acylLPC": "[M+H]+",
    "1-acylLPE": "[M-H]-",
    "2-acylLPE": "[M-H]-",
}


@dataclass(frozen=True)
class RPIsomerAnchorSettings:
    enabled: bool = True
    anchor_rt_seconds: float = 30.0
    final_rt_seconds: float = 10.0
    gap_error_seconds: float = 8.0
    minimum_separation_seconds: float = 3.0
    pair_score_margin: float = 0.15
    singleton_rt_margin_seconds: float = 3.0
    minimum_support_groups: int = 4
    maximum_anchor_groups: int = 20
    maximum_composition_distance: float = 8.0
    selection_margin: float = 0.10
    grid_threshold: float = 0.45
    model_path: Path = FittedGridCompletionSettings().model_path


def _validate(settings: RPIsomerAnchorSettings) -> None:
    positive = (
        settings.anchor_rt_seconds,
        settings.final_rt_seconds,
        settings.gap_error_seconds,
        settings.minimum_separation_seconds,
        settings.singleton_rt_margin_seconds,
        settings.maximum_composition_distance,
    )
    if any(not np.isfinite(x) or x <= 0 for x in positive):
        raise ValueError("RP isomer tolerances must be finite and positive")
    if settings.final_rt_seconds > 10 or settings.anchor_rt_seconds > 30:
        raise ValueError("RP isomer refinement cannot widen the 30s/10s envelopes")
    if settings.minimum_support_groups < 2:
        raise ValueError("RP isomer prediction requires independent support groups")
    if settings.maximum_anchor_groups < settings.minimum_support_groups + 1:
        raise ValueError("RP isomer anchor capacity must permit leave-one-group-out")
    if not 0 <= settings.grid_threshold <= 1:
        raise ValueError("RP isomer grid threshold must be in [0, 1]")
    if any(
        not np.isfinite(x) or x < 0
        for x in (
            settings.pair_score_margin,
            settings.selection_margin,
        )
    ):
        raise ValueError("RP isomer ambiguity margins must be finite and nonnegative")


def _reference_groups(reference: pd.DataFrame) -> pd.DataFrame:
    if reference.empty or not {
        "class",
        "name",
        "adduct",
        "ref_rt",
        "ref_mz",
        "carbon",
        "db",
    }.issubset(reference):
        return reference.iloc[:0].copy()
    r = reference[
        reference["class"].isin(PRIMARY_CHANNELS)
        & reference["adduct"].eq(reference["class"].map(PRIMARY_CHANNELS))
        & pd.to_numeric(reference["ref_rt"], errors="coerce").notna()
        & ~reference["name"].astype(str).str.endswith(" (IS)")
    ].reset_index(drop=True)
    if r.empty:
        return r
    r["_rp_family"] = (
        r["class"]
        .str.replace(r"^[12]-acyl", "", regex=True)
        .str.replace(r"-[OP]$", "_ether", regex=True)
    )
    # O-C:D and P-C:(D-1) share a formula, not O-C:D and P-C:D.
    r["_rp_total_db"] = r["db"] + r["class"].str.endswith("-P").astype(int)
    r["_rp_group"] = list(
        zip(
            r["_rp_family"],
            r["carbon"],
            r["_rp_total_db"],
            r["adduct"],
            strict=True,
        )
    )
    r["_rp_member"] = np.where(
        r["class"].str.endswith("-O") | r["class"].str.startswith("2-acyl"),
        "early",
        "late",
    )
    # Resolve the order from this method's reference, not from a universal
    # literature offset. Lyso order is additionally checked explicitly below.
    usable = []
    for _, group in r.groupby("_rp_group", sort=False):
        group = group.drop_duplicates(["name", "adduct"])
        if len(group) != 2:
            continue
        mz = pd.to_numeric(group["ref_mz"], errors="coerce").to_numpy(float)
        if not np.isfinite(mz).all() or not np.isclose(mz[0], mz[1], rtol=0, atol=1e-6):
            continue
        if str(group["_rp_family"].iloc[0]).endswith("_ether"):
            ordered = group.sort_values("ref_rt", kind="stable")
            r.loc[ordered.index, "_rp_member"] = ["early", "late"]
        else:
            early = group.loc[group["_rp_member"].eq("early"), "ref_rt"]
            late = group.loc[group["_rp_member"].eq("late"), "ref_rt"]
            if early.empty or late.empty or float(early.iloc[0]) >= float(late.iloc[0]):
                continue
        usable.extend(group.index)
    return r.loc[usable].copy()


def _join_features(
    reference: pd.DataFrame,
    features: pd.DataFrame,
    window_seconds: float,
) -> pd.DataFrame:
    """Mass-window join retaining the original XCMS IDs and intensities."""
    if reference.empty or features.empty:
        return pd.DataFrame()
    masses = pd.to_numeric(features["mz"], errors="coerce").to_numpy(float)
    rts = pd.to_numeric(features["rt"], errors="coerce").to_numpy(float)
    valid = np.flatnonzero(np.isfinite(masses) & np.isfinite(rts) & (masses > 0))
    order = valid[np.argsort(masses[valid], kind="stable")]
    sorted_mass = masses[order]
    rows = []
    for target in reference.to_dict("records"):
        mz = float(target["ref_mz"])
        rt = float(target["ref_rt"])
        lo = np.searchsorted(sorted_mass, mz * (1 - 3e-6))
        hi = np.searchsorted(sorted_mass, mz * (1 + 3e-6), side="right")
        for position in order[lo:hi]:
            ppm = (masses[position] - mz) / mz * 1e6
            if abs(ppm) >= 3 or abs(rts[position] - rt) * 60 > window_seconds:
                continue
            feature = features.iloc[position]
            row = dict(target)
            row.update({f"feat_{c}": value for c, value in feature.items()})
            row.update(
                {
                    "feat_feature_id": str(feature["feature_id"]),
                    "rt_expected": rt,
                    "rt_error": rts[position] - rt,
                    "ppm_error": ppm,
                    "isotopologue": "",
                    "is_is": False,
                    "display_name": str(target["name"]),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _choose_pair(
    group: pd.DataFrame,
    window: float,
    settings: RPIsomerAnchorSettings,
) -> list[int]:
    early = group[group["_rp_member"].eq("early")]
    late = group[group["_rp_member"].eq("late")]
    options = []
    maximum = np.log1p(max(0.0, float(group["feat_intensity"].max())))
    for i, e in early.iterrows():
        for j, late_row in late.iterrows():
            if str(e["feat_feature_id"]) == str(late_row["feat_feature_id"]):
                continue
            gap = (float(late_row["feat_rt"]) - float(e["feat_rt"])) * 60
            expected = (float(late_row["rt_expected"]) - float(e["rt_expected"])) * 60
            if min(gap, expected) < settings.minimum_separation_seconds:
                continue
            gap_error = abs(gap - expected)
            if gap_error > settings.gap_error_seconds:
                continue
            residual = (
                abs(float(e["rt_error"])) + abs(float(late_row["rt_error"]))
            ) * 60
            score = (
                residual / (2 * window) + 0.5 * gap_error / settings.gap_error_seconds
            )
            score += (
                abs(float(e["ppm_error"])) + abs(float(late_row["ppm_error"]))
            ) / 12
            log_signal = np.log1p(max(0.0, float(e["feat_intensity"])))
            log_signal += np.log1p(max(0.0, float(late_row["feat_intensity"])))
            score -= 0.08 * (log_signal - 2 * maximum)
            options.append((score, i, j))
    options.sort()
    if not options or (
        len(options) > 1 and options[1][0] - options[0][0] < settings.pair_score_margin
    ):
        return []
    return [options[0][1], options[0][2]]


def _anchor_pairs(
    candidates: pd.DataFrame,
    settings: RPIsomerAnchorSettings,
) -> pd.DataFrame:
    pairs = []
    for _, group in candidates.groupby("_rp_group", sort=False):
        indices = _choose_pair(group, settings.anchor_rt_seconds, settings)
        if indices:
            pair = group.loc[indices].copy()
            pair["rp_anchor_pair_intensity"] = pair["feat_intensity"].min()
            pairs.append(pair)
    if not pairs:
        return candidates.iloc[:0].copy()
    seeds = pd.concat(pairs)
    retained = []
    for _, group in seeds.groupby(["_rp_family", "adduct"], sort=False):
        keys = (
            group.sort_values(
                "rp_anchor_pair_intensity", ascending=False, kind="stable"
            )["_rp_group"]
            .drop_duplicates()
            .head(settings.maximum_anchor_groups)
        )
        group = group[group["_rp_group"].isin(keys)].copy()
        for lipid_class in group["class"].unique():
            source = group[group["class"].eq(lipid_class)]
            residual = source["rt_error"] * 60
            center = residual.median()
            mad = (residual - center).abs().median()
            bad = source.loc[
                (residual - center).abs().gt(max(5.0, 3 * 1.4826 * mad)),
                "_rp_group",
            ]
            group = group[~group["_rp_group"].isin(bad)]
        retained.append(group)
    return pd.concat(retained) if retained else seeds.iloc[:0].copy()


def _correct_reference(
    reference: pd.DataFrame,
    seeds: pd.DataFrame,
    settings: RPIsomerAnchorSettings,
) -> pd.DataFrame:
    corrected = reference.copy()
    corrected["rp_isomer_prior_rt"] = corrected["ref_rt"]
    corrected["rp_isomer_anchor_count"] = 0
    corrected["rp_isomer_shift_seconds"] = np.nan
    if seeds.empty:
        return corrected
    for key, group in reference.groupby("_rp_group", sort=False):
        family, carbon, db, adduct = key
        support = seeds[
            seeds["_rp_family"].eq(family)
            & seeds["adduct"].eq(adduct)
            & ~seeds["_rp_group"].isin([key])
        ]
        support = support[
            (support["carbon"] - carbon).abs()
            + 2 * (support["_rp_total_db"] - db).abs()
            <= settings.maximum_composition_distance
        ]
        count = support["_rp_group"].nunique()
        if count < settings.minimum_support_groups:
            continue
        if support["carbon"].max() - support["carbon"].min() < 4 and (
            support["_rp_total_db"].max() - support["_rp_total_db"].min() < 2
        ):
            continue
        for _member, nodes in group.groupby("_rp_member", sort=False):
            residual = (
                support.loc[support["class"].eq(nodes["class"].iloc[0]), "rt_error"]
                * 60
            )
            shift = float(residual.median())
            if not np.isfinite(shift) or abs(shift) > settings.anchor_rt_seconds:
                continue
            corrected.loc[nodes.index, "rp_isomer_shift_seconds"] = shift
            corrected.loc[nodes.index, "rp_isomer_anchor_count"] = count
            corrected.loc[nodes.index, "ref_rt"] = nodes["ref_rt"] + shift / 60
    for _, group in corrected.groupby("_rp_group", sort=False):
        early = group.loc[group["_rp_member"].eq("early"), "ref_rt"]
        late = group.loc[group["_rp_member"].eq("late"), "ref_rt"]
        if (
            early.empty
            or late.empty
            or (float(late.iloc[0]) - float(early.iloc[0])) * 60
            < settings.minimum_separation_seconds
        ):
            corrected.loc[group.index, "rp_isomer_anchor_count"] = 0
    return corrected


def _resolve_supported(
    candidates: pd.DataFrame,
    settings: RPIsomerAnchorSettings,
) -> pd.DataFrame:
    chosen = []
    for _, group in candidates.groupby("_rp_group", sort=False):
        indices = _choose_pair(group, settings.final_rt_seconds, settings)
        if indices:
            chosen.extend(indices)
            continue
        # A single O/P form or primary 1-acyl may be present. Never invent the
        # other form, and never create a singleton 2-acyl assignment.
        ranked = group.assign(_error=group["rt_error"].abs() * 60).sort_values(
            "_error", kind="stable"
        )
        if len(ranked) > 1 and (
            ranked.iloc[1]["_error"] - ranked.iloc[0]["_error"]
            < settings.singleton_rt_margin_seconds
        ):
            continue
        best = ranked.iloc[0]
        if not str(best["class"]).startswith("2-acyl"):
            chosen.append(best.name)
    selected = candidates.loc[chosen].copy()
    if selected.empty:
        return selected
    selected["rp_isomer_selected"] = True
    selected["positional_isomer_selected"] = selected["class"].str.contains("acyl")
    selected["rt_evidence_mode"] = "rp_ordered_anchor_series"
    selected = score_grid_coherence(
        selected,
        3.0,
        settings.final_rt_seconds / 60,
        1.0,
        use_ccs=False,
        threshold=settings.grid_threshold,
    )
    selected = selected[selected["grid_coherence_pass"]].copy()
    # This is a deterministic ranking utility, NOT a calibrated probability.
    # Keep it separate from the existing cohort-model confidence column.
    selected["rp_isomer_evidence_score"] = (
        0.95
        - 0.10 * selected["ppm_error"].abs() / 3.0
        - 0.10 * selected["rt_error"].abs() * 60 / settings.final_rt_seconds
        - 0.10 * (1 - selected["grid_coherence_score"].fillna(0.0))
    )
    return selected


def propose_rp_isomer_assignments(
    reference: pd.DataFrame,
    features: pd.DataFrame,
    settings: RPIsomerAnchorSettings = RPIsomerAnchorSettings(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find ordered anchors, correct RT locally, then resolve weaker features."""
    _validate(settings)
    grouped = _reference_groups(reference)
    broad = _join_features(grouped, features, settings.anchor_rt_seconds)
    if broad.empty:
        return broad, pd.DataFrame(), grouped
    seeds = _anchor_pairs(broad, settings)
    corrected = _correct_reference(grouped, seeds, settings)
    supported = corrected[
        corrected["rp_isomer_anchor_count"].ge(settings.minimum_support_groups)
    ]
    pool = _join_features(supported, features, settings.final_rt_seconds)
    if pool.empty:
        return pool, seeds, corrected
    return _resolve_supported(pool, settings), seeds, corrected


def _without_orphan_new_two_acyl(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected
    one = selected[selected["class"].isin(["1-acylLPC", "1-acylLPE"])]
    one_keys = set(
        zip(
            one["class"].str.removeprefix("1-acyl"),
            one["carbon"],
            one["db"],
            one["adduct"],
            strict=True,
        )
    )
    new = (
        selected.get("rp_isomer_selected", pd.Series(False, index=selected.index))
        .fillna(False)
        .astype(bool)
    )
    retained = (
        selected.get(
            "post_regression_count_retained", pd.Series(False, index=selected.index)
        )
        .fillna(False)
        .astype(bool)
    )
    orphan = [
        str(row["class"]).startswith("2-acyl")
        and (str(row["class"])[6:], row["carbon"], row["db"], row["adduct"])
        not in one_keys
        for _, row in selected.iterrows()
    ]
    return selected.loc[~(np.asarray(orphan) & new & ~retained)].copy()


def refine_rp_isomer_annotations(
    annotations: pd.DataFrame,
    all_pairs: pd.DataFrame,
    reference: pd.DataFrame,
    features: pd.DataFrame,
    *,
    stream: str,
    settings: RPIsomerAnchorSettings = RPIsomerAnchorSettings(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep the baseline count, standards and NP; audit every RP promotion."""
    audit = {
        "stream": stream,
        "enabled": settings.enabled,
        "baseline_annotations": len(annotations),
    }

    def unchanged(status: str, detail: str = ""):
        return (
            annotations,
            all_pairs,
            pd.DataFrame(
                [
                    {
                        **audit,
                        "status": status,
                        "detail": detail,
                        "final_annotations": len(annotations),
                    }
                ]
            ),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    if not str(stream).startswith("RP_"):
        return unchanged("not_applicable")
    if not settings.enabled or annotations.empty:
        return unchanged("disabled" if not settings.enabled else "no_baseline")
    if reference.empty or features.empty:
        return unchanged("no_isomer_reference_or_features")
    _validate(settings)
    try:
        model = _load_model(settings.model_path)
    except (OSError, ValueError) as error:
        return unchanged("model_unavailable", str(error))
    if "post_regression_confidence" not in annotations:
        return unchanged("baseline_confidence_unavailable")
    protected = _protected_rows(annotations)
    reserved = set(protected["feat_feature_id"].astype(str))
    available_features = features[~features["feature_id"].astype(str).isin(reserved)]
    new, seeds, corrected = propose_rp_isomer_assignments(
        reference, available_features, settings
    )
    if new.empty:
        return unchanged("no_supported_assignments")
    new = new[~new["feat_feature_id"].astype(str).isin(reserved)].copy()
    if new.empty:
        return unchanged("protected_features_only")
    oldpool = all_pairs[
        _strict_candidate_mask(all_pairs, stream, model["candidate_limits"])
    ]
    oldpool = oldpool[~_row_keys(oldpool).isin(set(_row_keys(new)))]
    scoring = pd.concat([oldpool, new], ignore_index=True, sort=False)
    scores = _score_features(_candidate_features(scoring, stream, model), model)
    new["post_regression_confidence"] = scores.iloc[-len(new) :].to_numpy()
    new["post_regression_model"] = str(model["format"])
    new["post_regression_eligible"] = True
    combined = pd.concat(
        [new, annotations], ignore_index=True, sort=False
    ).drop_duplicates(
        ["feat_feature_id", "name", "adduct", "matlab_code"], keep="first"
    )
    protected_keys = set(_row_keys(protected))
    combined = combined[~_row_keys(combined).isin(protected_keys)].copy()
    combined = combined[~combined["feat_feature_id"].astype(str).isin(reserved)]
    baseline_keys = set(_row_keys(annotations)) - protected_keys
    baseline = combined[_row_keys(combined).isin(baseline_keys)]
    if len(baseline) != len(annotations) - len(protected):
        return unchanged("baseline_pair_unavailable")
    candidate_features = _candidate_features(combined, stream, model)
    original_confidence = combined["post_regression_confidence"].copy()
    # Reuse the conflict/count machinery with a temporary ranking column;
    # restore the actual cohort confidence before returning any table.
    combined["rp_isomer_selection_score"] = combined.get(
        "rp_isomer_evidence_score", pd.Series(np.nan, index=combined.index)
    ).fillna(original_confidence)
    combined["post_regression_confidence"] = combined["rp_isomer_selection_score"]
    selected, _, _ = _select_confident(
        combined,
        candidate_features,
        combined["rp_isomer_selection_score"],
        PostRegressionCompletionSettings(ambiguity_margin=settings.selection_margin),
    )
    selected = _without_orphan_new_two_acyl(selected)
    selected, retained = _preserve_count(
        selected, baseline, combined, candidate_features
    )
    safe = _without_orphan_new_two_acyl(selected)
    if len(safe) != len(selected):
        selected, retained = _preserve_count(
            safe, baseline, combined, candidate_features
        )
    selected["post_regression_confidence"] = original_confidence.loc[selected.index]
    supported = (
        selected.get("rp_isomer_selected", pd.Series(False, index=selected.index))
        .fillna(False)
        .astype(bool)
        & ~selected["post_regression_count_retained"]
    )
    selected.loc[supported, "auto_annotation_status"] = "rp_isomer_anchor_supported"
    selected.loc[supported, "status"] = "rp_isomer_anchor_supported"
    selected["automatic_accepted"] = True
    selected["automatic_candidate"] = True
    accepted = pd.concat([selected, protected], ignore_index=True, sort=False)
    if (
        len(accepted) != len(annotations)
        or accepted["feat_feature_id"].astype(str).duplicated().any()
    ):
        return unchanged("count_or_feature_conflict")
    accepted["rank_for_feature"] = accepted.groupby("feat_feature_id").cumcount()
    candidates = (
        pd.concat([new, all_pairs], ignore_index=True, sort=False)
        .drop_duplicates(
            ["feat_feature_id", "name", "adduct", "matlab_code"], keep="first"
        )
        .reset_index(drop=True)
    )
    status = dict(
        zip(_row_keys(accepted), accepted["auto_annotation_status"], strict=True)
    )
    keys = _row_keys(candidates)
    candidates["automatic_accepted"] = keys.isin(status)
    candidates["automatic_candidate"] = candidates["automatic_accepted"]
    candidates["auto_annotation_status"] = keys.map(status).fillna(
        "review_only_blue_candidate"
    )
    retained_mask = (
        accepted.get(
            "post_regression_count_retained", pd.Series(False, index=accepted.index)
        )
        .fillna(False)
        .astype(bool)
    )
    candidates["post_regression_count_retained"] = keys.isin(
        set(_row_keys(accepted.loc[retained_mask]))
    )
    candidates["rp_isomer_selected"] = (
        candidates.get("rp_isomer_selected", pd.Series(False, index=candidates.index))
        .fillna(False)
        .astype(bool)
    )
    summary = pd.DataFrame(
        [
            {
                **audit,
                "status": "completed",
                "anchor_pairs": len(seeds) // 2,
                "supported_assignments": len(new),
                "selected_supported_assignments": int(supported.sum()),
                "baseline_calls_retained": retained,
                "final_annotations": len(accepted),
                "minimum_independent_support": settings.minimum_support_groups,
            }
        ]
    )
    seeds = seeds.copy()
    corrected = corrected.copy()
    seeds["stream"] = stream
    corrected["stream"] = stream
    return accepted, candidates, summary, seeds, corrected


__all__ = [
    "RPIsomerAnchorSettings",
    "propose_rp_isomer_assignments",
    "refine_rp_isomer_annotations",
]
