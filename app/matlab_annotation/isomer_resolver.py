"""Conservative resolution of positional lyso-lipid isomers.

The RP MATLAB catalogue contains separate 1-acyl and 2-acyl LPC/LPE targets.
They are precursor-mass isomers, so resolving every XCMS feature independently
can assign both chromatographic peaks to the same positional target.  This
module treats the two target identities as one assignment problem. LPC is
resolved asymmetrically: the abundant 1-acyl feature is selected first and an
earlier 2-acyl feature is accepted only when it is supported as its optional
partner. LPE retains the conservative paired-peak rule.

The resolver is deliberately outside ``matlab_annotation/vendor``.  It is an
XCMS-only acceptance policy layered on the unchanged MATLAB mapper output.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd


_POSITIONAL_LYSO = re.compile(
    r"^(?P<position>[12])-acyl\s*(?P<family>LPC|LPE)"
    r"\((?P<carbon>\d+):(?P<db>\d+)\)(?:\s+\(IS\))?$",
    re.IGNORECASE,
)


def _monoisotopic(values: pd.Series) -> pd.Series:
    """Accept both XCMS' blank label and 4D's numeric zero convention."""
    text = values.fillna("").astype(str).str.strip().str.casefold()
    numeric = pd.to_numeric(values, errors="coerce")
    return text.isin(("", "m", "m+0", "monoisotopic")) | numeric.eq(0)


@dataclass(frozen=True)
class PositionalIsomerSettings:
    """Guardrails for accepting positional LPC/LPE feature assignments."""

    maximum_rt_error_seconds: float = 10.0
    maximum_gap_error_seconds: float = 8.0
    minimum_peak_separation_seconds: float = 3.0
    minimum_assignment_margin: float = 0.20
    ppm_tolerance: float = 3.0
    maximum_local_rt_adjustment_seconds: float = 10.0
    minimum_local_support_groups: int = 2
    maximum_local_composition_distance: int = 4
    allow_lpe_one_acyl_singleton: bool = True


def _identity(value: object):
    match = _POSITIONAL_LYSO.match(str(value).strip())
    if not match:
        return None
    return {
        "position": int(match.group("position")),
        "family": match.group("family").upper(),
        "carbon": int(match.group("carbon")),
        "db": int(match.group("db")),
    }


def _decorate(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    defaults = {
        "positional_isomer_status": "not_applicable",
        "positional_isomer_pair_id": "",
        "positional_isomer_position": "",
        "positional_isomer_selected": False,
        "positional_isomer_rt_expected": np.nan,
        "positional_isomer_rt_error_seconds": np.nan,
        "positional_isomer_observed_gap_seconds": np.nan,
        "positional_isomer_expected_gap_seconds": np.nan,
        "positional_isomer_gap_error_seconds": np.nan,
        "positional_isomer_assignment_score": np.nan,
        "positional_isomer_assignment_margin": np.nan,
        "positional_isomer_local_rt_adjustment_seconds": 0.0,
        "positional_isomer_local_support_groups": 0,
        "positional_isomer_seed_identity_valid": False,
    }
    for column, default in defaults.items():
        if column not in out:
            out[column] = default
    if {"name", "feat_rt", "rt_expected"}.issubset(out.columns):
        positional = out["name"].map(_identity).notna()
        if "is_is" in out:
            positional &= ~out["is_is"].fillna(False).astype(bool)
        if "isotopologue" in out:
            positional &= _monoisotopic(out["isotopologue"])
        expected = pd.to_numeric(out["rt_expected"], errors="coerce")
        observed = pd.to_numeric(out["feat_rt"], errors="coerce")
        out.loc[positional, "positional_isomer_rt_expected"] = expected.loc[
            positional
        ]
        out.loc[positional, "positional_isomer_rt_error_seconds"] = (
            observed.loc[positional] - expected.loc[positional]
        ) * 60.0
    return out


def _candidate_metadata(candidates: pd.DataFrame) -> pd.DataFrame:
    parsed = candidates["name"].map(_identity)
    positional = parsed.notna()
    if "is_is" in candidates:
        positional &= ~candidates["is_is"].fillna(False).astype(bool)
    if "isotopologue" in candidates:
        positional &= _monoisotopic(candidates["isotopologue"])
    meta = candidates.loc[positional].copy()
    if meta.empty:
        return meta
    values = pd.DataFrame(parsed.loc[positional].tolist(), index=meta.index)
    for column in values:
        meta[f"_isomer_{column}"] = values[column]
    meta["_isomer_adduct"] = (
        meta["adduct"].astype(str) if "adduct" in meta else ""
    )
    if "isotopologue" in meta:
        isotopologue = meta["isotopologue"].fillna("").astype(str)
        meta["_isomer_isotopologue"] = isotopologue.where(
            ~_monoisotopic(meta["isotopologue"]), ""
        )
    else:
        meta["_isomer_isotopologue"] = ""
    meta["_isomer_group"] = list(zip(
        meta["_isomer_family"],
        meta["_isomer_carbon"],
        meta["_isomer_db"],
        meta["_isomer_adduct"],
        meta["_isomer_isotopologue"],
    ))
    seed_species = meta.get(
        "positional_isomer_seed_species", pd.Series("", index=meta.index)
    ).fillna("").astype(str)
    valid_seed = []
    for index, seed in seed_species.items():
        identity = _identity(seed)
        valid_seed.append(bool(
            identity is not None
            and identity["family"] == str(meta.at[index, "_isomer_family"])
            and int(identity["carbon"]) == int(meta.at[index, "_isomer_carbon"])
            and int(identity["db"]) == int(meta.at[index, "_isomer_db"])
        ))
    meta["_isomer_seed_identity_valid"] = valid_seed
    return meta


def _reference_metadata(reference: pd.DataFrame) -> pd.DataFrame:
    if reference.empty or "name" not in reference:
        return pd.DataFrame()
    parsed = reference["name"].map(_identity)
    positional = parsed.notna()
    meta = reference.loc[positional].copy()
    if meta.empty:
        return meta
    values = pd.DataFrame(parsed.loc[positional].tolist(), index=meta.index)
    for column in values:
        meta[f"_isomer_{column}"] = values[column]
    meta["_isomer_adduct"] = (
        meta["adduct"].astype(str) if "adduct" in meta else ""
    )
    meta["_isomer_isotopologue"] = ""
    meta["_isomer_group"] = list(zip(
        meta["_isomer_family"],
        meta["_isomer_carbon"],
        meta["_isomer_db"],
        meta["_isomer_adduct"],
        meta["_isomer_isotopologue"],
    ))
    return meta


def _expand_positional_counterparts(
    candidates: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Restore both positional targets for every observed lyso feature.

    MATLAB sibling resolution may carry only one positional proposal into the
    staged pool.  The reference grid still contains both target codes and the
    RTDB separation.  Apply the median final-model shift to that reference pair
    and materialise the missing feature/position rows before joint assignment.
    """
    out = candidates.copy()
    if "positional_isomer_candidate_source" not in out:
        out["positional_isomer_candidate_source"] = "matlab_mapper"
    if "positional_isomer_seed_anchor" not in out:
        out["positional_isomer_seed_anchor"] = False
    if "positional_isomer_seed_species" not in out:
        out["positional_isomer_seed_species"] = ""
    meta = _candidate_metadata(out)
    refs = _reference_metadata(reference)
    if meta.empty or refs.empty:
        return out

    synthetic = []
    for key, ref_group in refs.groupby("_isomer_group", sort=False):
        if set(ref_group["_isomer_position"].astype(int)) != {1, 2}:
            continue
        observed = meta[
            meta["_isomer_group"].map(lambda value: value == key)
        ].copy()
        if observed.empty:
            continue
        reference_by_position = {
            int(row["_isomer_position"]): row
            for _, row in ref_group.sort_index().iterrows()
        }
        # A retained RTDB seed identifies a concrete XCMS peak before the
        # positional resolver runs.  Use that feature to position the shared
        # 1-/2-acyl reference pair.  Without this constraint, a coherent series
        # one peak too late can shift both identities and replace a correct
        # second-peak 1-acyl anchor with a third peak.
        shifts = []
        seed_rows = observed[
            observed["positional_isomer_seed_species"]
            .fillna("").astype(str).ne("")
            & observed["_isomer_seed_identity_valid"].fillna(False).astype(bool)
        ].drop_duplicates("feat_feature_id", keep="first")
        for _, row in seed_rows.iterrows():
            seed_identity = _identity(row["positional_isomer_seed_species"])
            ref_row = (
                reference_by_position.get(int(seed_identity["position"]))
                if seed_identity is not None else None
            )
            if ref_row is None:
                continue
            shift = float(row["feat_rt"]) - float(ref_row["ref_rt"])
            if np.isfinite(shift):
                shifts.append(shift)
        if not shifts:
            for _, row in observed.iterrows():
                ref_row = reference_by_position.get(
                    int(row["_isomer_position"])
                )
                if ref_row is None:
                    continue
                shift = float(row["rt_expected"]) - float(ref_row["ref_rt"])
                if np.isfinite(shift):
                    shifts.append(shift)
        if not shifts:
            continue
        shift = float(np.median(shifts))
        target_rt = {
            position: float(row["ref_rt"]) + shift
            for position, row in reference_by_position.items()
        }
        observed["_ppm_abs"] = pd.to_numeric(
            observed["ppm_error"], errors="coerce"
        ).abs()
        feature_rows = observed.sort_values(
            ["feat_feature_id", "_ppm_abs"], kind="stable"
        ).drop_duplicates("feat_feature_id", keep="first")
        for _, base in feature_rows.iterrows():
            feature_id = str(base["feat_feature_id"])
            for position, ref_row in reference_by_position.items():
                existing = observed[
                    observed["feat_feature_id"].astype(str).eq(feature_id)
                    & observed["_isomer_position"].eq(position)
                ].index
                expected = target_rt[position]
                if len(existing):
                    out.loc[existing, "rt_expected"] = expected
                    out.loc[existing, "ref_rt"] = expected
                    out.loc[existing, "rt_error"] = (
                        pd.to_numeric(out.loc[existing, "feat_rt"], errors="coerce")
                        - expected
                    )
                    continue
                row = base.copy()
                for column in (
                    "name", "class", "base", "adduct", "carbon", "db",
                    "ref_mz", "matlab_code", "matlab_compound",
                ):
                    if column in ref_row:
                        row[column] = ref_row[column]
                row["display_name"] = f"{row['name']} {row['adduct']}".strip()
                row["rt_expected"] = expected
                row["ref_rt"] = expected
                row["rt_error"] = float(row["feat_rt"]) - expected
                row["ppm_error"] = (
                    (float(row["feat_mz"]) - float(row["ref_mz"]))
                    / float(row["ref_mz"]) * 1e6
                )
                row["automatic_accepted"] = False
                row["automatic_candidate"] = True
                row["positional_isomer_seed_anchor"] = (
                    str(row.get("positional_isomer_seed_species", ""))
                    == str(row["name"])
                )
                row["auto_annotation_status"] = (
                    "positional_isomer_counterpart_candidate"
                )
                row["positional_isomer_candidate_source"] = "rt_grid_counterpart"
                synthetic.append(row)
    if synthetic:
        out = pd.concat(
            [out, pd.DataFrame(synthetic)], ignore_index=True, sort=False
        )
    return out


def _best_identity_rows(
    group: pd.DataFrame,
    position: int,
    settings: PositionalIsomerSettings,
) -> pd.DataFrame:
    selected = group[group["_isomer_position"].eq(position)].copy()
    if selected.empty:
        return selected
    selected["_rt_error_seconds"] = (
        pd.to_numeric(selected["feat_rt"], errors="coerce")
        - pd.to_numeric(selected["rt_expected"], errors="coerce")
    ) * 60.0
    selected["_ppm_abs"] = pd.to_numeric(
        selected["ppm_error"], errors="coerce"
    ).abs()
    selected["_intensity"] = pd.to_numeric(
        selected.get("feat_intensity", 0.0), errors="coerce"
    ).fillna(0.0)
    selected = selected[
        np.isfinite(selected["_rt_error_seconds"])
        & np.isfinite(selected["_ppm_abs"])
    ]
    if selected.empty:
        return selected
    seed_anchor = selected.get(
        "positional_isomer_seed_anchor",
        pd.Series(False, index=selected.index),
    ).fillna(False).astype(bool) & selected.get(
        "_isomer_seed_identity_valid",
        pd.Series(False, index=selected.index),
    ).fillna(False).astype(bool)
    if seed_anchor.any():
        # A retained automatic seed identifies the concrete peak for this
        # positional identity.  It is a constraint, not merely a tie-breaker:
        # allowing other same-mass rows back into the assignment can shift the
        # whole pair one chromatographic peak too late.
        selected = selected.loc[seed_anchor].copy()
    selected["_row_cost"] = (
        (
            selected["_rt_error_seconds"].abs()
            / settings.maximum_rt_error_seconds
        ) ** 2
        + (selected["_ppm_abs"] / settings.ppm_tolerance) ** 2
    ) ** 0.5
    selected["_seed_priority"] = np.where(
        selected.get(
            "positional_isomer_seed_anchor",
            pd.Series(False, index=selected.index),
        ).fillna(False).astype(bool)
        & selected.get(
            "_isomer_seed_identity_valid",
            pd.Series(False, index=selected.index),
        ).fillna(False).astype(bool),
        0,
        1,
    )
    return selected.sort_values(
        ["_seed_priority", "_row_cost", "_intensity"],
        ascending=[True, True, False], kind="stable",
    ).drop_duplicates("feat_feature_id", keep="first")


def _pair_id(key: tuple) -> str:
    family, carbon, db, adduct, isotopologue = key
    suffix = f":{isotopologue}" if isotopologue else ""
    return f"{family}({carbon}:{db}):{adduct}{suffix}"


def _eligible_rows(
    rows: pd.DataFrame, settings: PositionalIsomerSettings,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    return rows[
        rows["_rt_error_seconds"].abs().le(
            settings.maximum_rt_error_seconds
        )
        & rows["_ppm_abs"].lt(settings.ppm_tolerance)
    ].copy()


def _resolve_lpc_group(
    group: pd.DataFrame,
    settings: PositionalIsomerSettings,
) -> dict:
    """Select abundant 1-acyl first, then conditionally resolve earlier 2-acyl."""
    one_all = _best_identity_rows(group, 1, settings)
    two_all = _best_identity_rows(group, 2, settings)
    features = set(group["feat_feature_id"].astype(str))
    base = {
        "candidate_features": len(features),
        "one_acyl_candidates": len(one_all),
        "two_acyl_candidates": len(two_all),
        "selected_one_index": None,
        "selected_two_index": None,
        "observed_gap_seconds": np.nan,
        "expected_gap_seconds": np.nan,
        "gap_error_seconds": np.nan,
        "assignment_score": np.nan,
        "assignment_margin": np.nan,
    }
    one = _eligible_rows(one_all, settings)
    if one.empty:
        return {**base, "status": "unresolved_missing_one_acyl"}

    seeded = one.get(
        "positional_isomer_seed_anchor", pd.Series(False, index=one.index)
    ).fillna(False).astype(bool)
    if not seeded.any():
        # Within the mass/RT guardrails, abundance is the primary LPC prior.
        # RT/mass fit breaks equal-intensity alternatives.
        one = one.sort_values(
            ["_intensity", "_row_cost"],
            ascending=[False, True], kind="stable",
        )
    primary_index = one.index[0]
    primary = one.loc[primary_index]
    primary_result = {
        **base,
        "selected_one_index": primary_index,
        "assignment_score": float(primary["_row_cost"]),
    }

    two = _eligible_rows(two_all, settings)
    if two.empty or len(features) < 2:
        return {
            **primary_result,
            "status": "one_acyl_primary_only_no_two_acyl",
        }

    assignments = []
    saw_reference_order = False
    saw_earlier_peak = False
    for two_index, two_row in two.iterrows():
        if str(primary["feat_feature_id"]) == str(two_row["feat_feature_id"]):
            continue
        expected_gap = (
            float(primary["rt_expected"]) - float(two_row["rt_expected"])
        ) * 60.0
        if not np.isfinite(expected_gap) or expected_gap <= 0:
            continue
        saw_reference_order = True
        observed_gap = (
            float(primary["feat_rt"]) - float(two_row["feat_rt"])
        ) * 60.0
        if observed_gap < settings.minimum_peak_separation_seconds:
            continue
        saw_earlier_peak = True
        gap_error = abs(observed_gap - expected_gap)
        if gap_error > settings.maximum_gap_error_seconds:
            continue
        score = (
            float(primary["_row_cost"])
            + float(two_row["_row_cost"])
            + gap_error / settings.maximum_gap_error_seconds
        )
        assignments.append({
            "one_index": primary_index,
            "two_index": two_index,
            "observed_gap_seconds": observed_gap,
            "expected_gap_seconds": expected_gap,
            "gap_error_seconds": gap_error,
            "assignment_score": score,
        })

    if not assignments:
        reason = (
            "reference_order" if not saw_reference_order
            else "no_earlier_peak" if not saw_earlier_peak
            else "gap_mismatch"
        )
        return {
            **primary_result,
            "status": f"one_acyl_primary_only_{reason}",
        }

    assignments.sort(key=lambda item: item["assignment_score"])
    best = assignments[0]
    margin = (
        assignments[1]["assignment_score"] - best["assignment_score"]
        if len(assignments) > 1 else float("inf")
    )
    if margin < settings.minimum_assignment_margin:
        return {
            **primary_result,
            "observed_gap_seconds": best["observed_gap_seconds"],
            "expected_gap_seconds": best["expected_gap_seconds"],
            "gap_error_seconds": best["gap_error_seconds"],
            "assignment_margin": margin,
            "status": "one_acyl_primary_only_ambiguous_two_acyl",
        }
    return {
        **base,
        "selected_one_index": primary_index,
        "selected_two_index": best["two_index"],
        "observed_gap_seconds": best["observed_gap_seconds"],
        "expected_gap_seconds": best["expected_gap_seconds"],
        "gap_error_seconds": best["gap_error_seconds"],
        "assignment_score": best["assignment_score"],
        "assignment_margin": margin,
        "status": "paired_rt_order_supported",
    }


def _resolve_lpe_group(
    group: pd.DataFrame,
    settings: PositionalIsomerSettings,
) -> dict:
    one = _best_identity_rows(group, 1, settings)
    two = _best_identity_rows(group, 2, settings)
    features = set(group["feat_feature_id"].astype(str))
    base = {
        "candidate_features": len(features),
        "one_acyl_candidates": len(one),
        "two_acyl_candidates": len(two),
        "selected_one_index": None,
        "selected_two_index": None,
        "observed_gap_seconds": np.nan,
        "expected_gap_seconds": np.nan,
        "gap_error_seconds": np.nan,
        "assignment_score": np.nan,
        "assignment_margin": np.nan,
    }
    one_eligible = _eligible_rows(one, settings)
    two_eligible = _eligible_rows(two, settings)
    if len(features) < 2 or two_eligible.empty:
        # LPE abundance is not a positional discriminator.  A unique 1-acyl
        # peak can nevertheless be retained when it has an exact seed or a
        # locally supported RT correction.  The reciprocal rule is forbidden:
        # 2-acyl never survives without its 1-acyl primary.
        if settings.allow_lpe_one_acyl_singleton and not one_eligible.empty:
            seeded = one_eligible.get(
                "positional_isomer_seed_anchor",
                pd.Series(False, index=one_eligible.index),
            ).fillna(False).astype(bool) & one_eligible.get(
                "_isomer_seed_identity_valid",
                pd.Series(False, index=one_eligible.index),
            ).fillna(False).astype(bool)
            locally_supported = bool(
                pd.to_numeric(
                    group.get(
                        "_positional_local_support_groups",
                        pd.Series(0, index=group.index),
                    ), errors="coerce",
                ).fillna(0).max()
                >= settings.minimum_local_support_groups
            )
            candidates = one_eligible.loc[seeded] if seeded.any() else (
                one_eligible if locally_supported else one_eligible.iloc[0:0]
            )
            if len(candidates) == 1:
                primary = candidates.sort_values(
                    ["_row_cost", "_intensity"],
                    ascending=[True, False], kind="stable",
                ).iloc[0]
                return {
                    **base,
                    "selected_one_index": primary.name,
                    "assignment_score": float(primary["_row_cost"]),
                    "status": "one_acyl_primary_only_lpe_supported",
                }
        return {
            **base,
            "status": (
                "unresolved_single_peak"
                if len(features) < 2 else "unresolved_missing_position"
            ),
        }
    if one_eligible.empty:
        return {**base, "status": "unresolved_missing_one_acyl"}

    assignments = []
    saw_reference_order = False
    saw_observed_order = False
    saw_rt_fit = False
    for one_index, one_row in one_eligible.iterrows():
        for two_index, two_row in two_eligible.iterrows():
            if str(one_row["feat_feature_id"]) == str(two_row["feat_feature_id"]):
                continue
            expected_gap = (
                float(one_row["rt_expected"]) - float(two_row["rt_expected"])
            ) * 60.0
            if not np.isfinite(expected_gap) or expected_gap <= 0:
                continue
            saw_reference_order = True
            observed_gap = (
                float(one_row["feat_rt"]) - float(two_row["feat_rt"])
            ) * 60.0
            if observed_gap < settings.minimum_peak_separation_seconds:
                continue
            saw_observed_order = True
            one_error = abs(float(one_row["_rt_error_seconds"]))
            two_error = abs(float(two_row["_rt_error_seconds"]))
            gap_error = abs(observed_gap - expected_gap)
            if (
                one_error > settings.maximum_rt_error_seconds
                or two_error > settings.maximum_rt_error_seconds
                or gap_error > settings.maximum_gap_error_seconds
                or float(one_row["_ppm_abs"]) >= settings.ppm_tolerance
                or float(two_row["_ppm_abs"]) >= settings.ppm_tolerance
            ):
                continue
            saw_rt_fit = True
            score = (
                float(one_row["_row_cost"])
                + float(two_row["_row_cost"])
                + gap_error / settings.maximum_gap_error_seconds
            )
            assignments.append({
                "one_index": one_index,
                "two_index": two_index,
                "observed_gap_seconds": observed_gap,
                "expected_gap_seconds": expected_gap,
                "gap_error_seconds": gap_error,
                "assignment_score": score,
            })
    if not assignments:
        if not saw_reference_order:
            status = "unresolved_reference_order"
        elif not saw_observed_order:
            status = "unresolved_coelution_or_order"
        elif not saw_rt_fit:
            status = "unresolved_rt_fit"
        else:
            status = "unresolved_no_assignment"
        return {**base, "status": status}

    assignments.sort(key=lambda item: item["assignment_score"])
    best = assignments[0]
    margin = (
        assignments[1]["assignment_score"] - best["assignment_score"]
        if len(assignments) > 1 else float("inf")
    )
    if margin < settings.minimum_assignment_margin:
        return {
            **base,
            **{key: value for key, value in best.items() if not key.endswith("index")},
            "assignment_margin": margin,
            "status": "unresolved_assignment_margin",
        }
    return {
        **base,
        "selected_one_index": best["one_index"],
        "selected_two_index": best["two_index"],
        "observed_gap_seconds": best["observed_gap_seconds"],
        "expected_gap_seconds": best["expected_gap_seconds"],
        "gap_error_seconds": best["gap_error_seconds"],
        "assignment_score": best["assignment_score"],
        "assignment_margin": margin,
        "status": "paired_rt_order_supported",
    }


def _resolve_group(
    group: pd.DataFrame,
    settings: PositionalIsomerSettings,
) -> dict:
    if str(group["_isomer_family"].iloc[0]).upper() == "LPC":
        return _resolve_lpc_group(group, settings)
    return _resolve_lpe_group(group, settings)


def resolve_positional_lyso_isomers(
    annotations: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    reference: pd.DataFrame | None = None,
    settings: PositionalIsomerSettings = PositionalIsomerSettings(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return accepted annotations, decorated candidates and isomer audit.

    Every positional LPC/LPE group represented in the candidate pool is taken
    out of ordinary mapper acceptance. LPC restores the strongest credible
    1-acyl row first and restores a 2-acyl row only when it is a unique,
    RT-consistent earlier partner. LPE restores exactly two rows only when a
    unique early-2-acyl/late-1-acyl assignment exists. Other position-specific
    candidates remain visible in ``all_pairs`` for review.
    """
    positive = (
        settings.maximum_rt_error_seconds,
        settings.maximum_gap_error_seconds,
        settings.minimum_peak_separation_seconds,
        settings.ppm_tolerance,
        settings.maximum_local_rt_adjustment_seconds,
        settings.maximum_local_composition_distance,
    )
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("positional-isomer tolerances must be finite and positive")
    if (
        not np.isfinite(settings.minimum_assignment_margin)
        or settings.minimum_assignment_margin < 0
    ):
        raise ValueError("positional-isomer score margin must be finite and >=0")
    if settings.minimum_local_support_groups < 1:
        raise ValueError("positional-isomer local support must be at least one group")

    accepted = _decorate(annotations)
    expanded = _expand_positional_counterparts(
        candidates, reference if reference is not None else pd.DataFrame()
    )
    decorated = _decorate(expanded)
    if decorated.empty or "name" not in decorated:
        return accepted, decorated, pd.DataFrame()

    meta = _candidate_metadata(decorated)
    if meta.empty:
        return accepted, decorated, pd.DataFrame()
    decorated.loc[
        meta.index, "positional_isomer_seed_identity_valid"
    ] = meta["_isomer_seed_identity_valid"].astype(bool)

    accepted_meta = _candidate_metadata(accepted) if not accepted.empty else accepted
    group_keys = set(meta["_isomer_group"])
    if not accepted_meta.empty:
        remove = accepted_meta["_isomer_group"].isin(group_keys)
        accepted = accepted.drop(index=accepted_meta.index[remove])

    grouped = list(meta.groupby("_isomer_group", sort=False))
    strict_results = {
        key: _resolve_group(group, settings) for key, group in grouped
    }
    # Learn only a bounded, local residual correction from already credible
    # groups.  This is deliberately not a wider global RT window: support must
    # come from neighbouring compositions in the same family/adduct series.
    support_points: list[dict] = []
    for key, group in grouped:
        result = strict_results[key]
        one_index = result.get("selected_one_index")
        if one_index is None:
            continue
        status = str(result.get("status", ""))
        selected = group.loc[one_index]
        trustworthy = status == "paired_rt_order_supported" or bool(
            selected.get("positional_isomer_seed_anchor", False)
            and selected.get("_isomer_seed_identity_valid", False)
        )
        if not trustworthy:
            continue
        family, carbon, db, adduct, isotopologue = key
        residual = (
            float(selected["feat_rt"]) - float(selected["rt_expected"])
        ) * 60.0
        if np.isfinite(residual):
            support_points.append({
                "key": key, "family": family, "carbon": int(carbon),
                "db": int(db), "adduct": adduct,
                "isotopologue": isotopologue, "residual": residual,
            })

    def result_rank(result: dict) -> int:
        status = str(result.get("status", ""))
        if status == "paired_rt_order_supported":
            return 3
        if status.startswith("one_acyl_primary_only_"):
            return 2
        return 0

    promoted = []
    audit_rows = []
    for key, group in grouped:
        result = strict_results[key]
        effective_group = group
        family, carbon, db, adduct, isotopologue = key
        neighbours = [
            item for item in support_points
            if item["key"] != key
            and item["family"] == family
            and item["adduct"] == adduct
            and item["isotopologue"] == isotopologue
            and (
                abs(int(item["carbon"]) - int(carbon))
                + 2 * abs(int(item["db"]) - int(db))
            ) <= settings.maximum_local_composition_distance
        ]
        local_support = len(neighbours)
        local_adjustment = 0.0
        if local_support >= settings.minimum_local_support_groups:
            local_adjustment = float(np.clip(
                np.median([item["residual"] for item in neighbours]),
                -settings.maximum_local_rt_adjustment_seconds,
                settings.maximum_local_rt_adjustment_seconds,
            ))
            adjusted = group.copy()
            adjusted["rt_expected"] = (
                pd.to_numeric(adjusted["rt_expected"], errors="coerce")
                + local_adjustment / 60.0
            )
            if "ref_rt" in adjusted:
                adjusted["ref_rt"] = (
                    pd.to_numeric(adjusted["ref_rt"], errors="coerce")
                    + local_adjustment / 60.0
                )
            adjusted["rt_error"] = (
                pd.to_numeric(adjusted["feat_rt"], errors="coerce")
                - pd.to_numeric(adjusted["rt_expected"], errors="coerce")
            )
            adjusted["_positional_local_support_groups"] = local_support
            adjusted_result = _resolve_group(adjusted, settings)
            current_score = float(result.get("assignment_score", np.inf))
            adjusted_score = float(
                adjusted_result.get("assignment_score", np.inf)
            )
            if (
                result_rank(adjusted_result) > result_rank(result)
                or (
                    result_rank(adjusted_result) == result_rank(result)
                    and adjusted_score < current_score
                )
            ):
                result = adjusted_result
                effective_group = adjusted
        group_indices = group.index
        decorated.loc[
            group_indices, "positional_isomer_local_rt_adjustment_seconds"
        ] = local_adjustment if effective_group is not group else 0.0
        decorated.loc[
            group_indices, "positional_isomer_local_support_groups"
        ] = local_support
        if effective_group is not group:
            for column in ("rt_expected", "ref_rt", "rt_error"):
                if column in effective_group:
                    decorated.loc[group_indices, column] = effective_group[column]
            decorated.loc[
                group_indices, "positional_isomer_rt_expected"
            ] = effective_group["rt_expected"]
            decorated.loc[
                group_indices, "positional_isomer_rt_error_seconds"
            ] = (
                pd.to_numeric(effective_group["feat_rt"], errors="coerce")
                - pd.to_numeric(effective_group["rt_expected"], errors="coerce")
            ) * 60.0
        pair_id = _pair_id(key)
        decorated.loc[group_indices, "positional_isomer_status"] = result["status"]
        decorated.loc[group_indices, "positional_isomer_pair_id"] = pair_id
        decorated.loc[group_indices, "positional_isomer_position"] = (
            group["_isomer_position"].astype(str) + "-acyl"
        )
        for column, key_name in (
            ("positional_isomer_observed_gap_seconds", "observed_gap_seconds"),
            ("positional_isomer_expected_gap_seconds", "expected_gap_seconds"),
            ("positional_isomer_gap_error_seconds", "gap_error_seconds"),
            ("positional_isomer_assignment_score", "assignment_score"),
            ("positional_isomer_assignment_margin", "assignment_margin"),
        ):
            decorated.loc[group_indices, column] = result[key_name]

        selected_indices = [
            index for index in (
                result["selected_two_index"], result["selected_one_index"]
            ) if index is not None
        ]
        if selected_indices:
            decorated.loc[selected_indices, "positional_isomer_selected"] = True
            chosen = decorated.loc[selected_indices].copy()
            chosen["automatic_accepted"] = True
            chosen["automatic_candidate"] = True
            chosen["auto_annotation_status"] = (
                "matlab_positional_isomer_paired"
                if result["selected_two_index"] is not None
                else "matlab_one_acyl_primary"
            )
            chosen["rank_for_feature"] = 0
            promoted.append(chosen)

        selected_one = result.get("selected_one_index")
        selected_two = result.get("selected_two_index")
        selected_one_row = (
            effective_group.loc[selected_one] if selected_one is not None else None
        )
        selected_two_row = (
            effective_group.loc[selected_two] if selected_two is not None else None
        )
        audit_rows.append({
            "pair_id": pair_id,
            "family": family,
            "carbon": int(carbon),
            "double_bonds": int(db),
            "adduct": adduct,
            "isotopologue": isotopologue,
            "status": result["status"],
            "candidate_features": result["candidate_features"],
            "one_acyl_candidates": result["one_acyl_candidates"],
            "two_acyl_candidates": result["two_acyl_candidates"],
            "observed_gap_seconds": result["observed_gap_seconds"],
            "expected_gap_seconds": result["expected_gap_seconds"],
            "gap_error_seconds": result["gap_error_seconds"],
            "assignment_score": result["assignment_score"],
            "assignment_margin": result["assignment_margin"],
            "local_rt_adjustment_seconds": (
                local_adjustment if effective_group is not group else 0.0
            ),
            "local_support_groups": local_support,
            "selected_one_feature_id": (
                str(selected_one_row["feat_feature_id"])
                if selected_one_row is not None else ""
            ),
            "selected_two_feature_id": (
                str(selected_two_row["feat_feature_id"])
                if selected_two_row is not None else ""
            ),
            "selected_one_seed_species": (
                str(selected_one_row.get("positional_isomer_seed_species", ""))
                if selected_one_row is not None else ""
            ),
            "selected_one_seed_identity_valid": (
                bool(selected_one_row.get("_isomer_seed_identity_valid", False))
                if selected_one_row is not None else False
            ),
            "selected_before_grid": len(selected_indices),
        })

    if promoted:
        accepted = pd.concat([accepted, *promoted], ignore_index=True, sort=False)
    accepted = accepted.reset_index(drop=True)
    decorated = decorated.reset_index(drop=True)
    return accepted, decorated, pd.DataFrame(audit_rows)
