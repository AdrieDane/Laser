"""Optional final PE correction with label-free quality and pairwise peak ranking.

Never reads manual annotations. Only automatic ordinary PE can be replaced;
manual confirmations, standards and other lipid classes are reserved. NP-positive
is intentionally unchanged. Additions require independent admission evidence,
not merely a high ranking score; RP-positive also needs all three other streams.
Disabled by default.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from annotator.composition_ranges import (
    CompositionRangeSettings,
    DEFAULT_COMPOSITION_RANGE_SETTINGS,
)
from annotator.peak_evidence import peak_evidence
from resource_paths import REFERENCE_DATA_DIRECTORY
from .pe_rebuild import (
    PERebuildSettings,
    _pe_keys,
    pe_rebuild_evidence,
    prepare_pe_rebuild_candidates,
    score_pe_rebuild,
)
from .result_assembly import assemble_result
from .pe_consensus import add_rp_consensus_evidence, rp_consensus_checks


PEERS = {"RP_Pos": "RP_Neg", "RP_Neg": "RP_Pos", "NP_Neg": "NP_Pos"}
MODEL_FILENAMES = {s: f"experimental_pe_selector_{s.lower()}.json" for s in PEERS}


@dataclass(frozen=True)
class ExperimentalPESelectionSettings:
    enabled: bool = False
    minimum_pair_score: float = 0.90
    minimum_rank_score: float = 0.65
    ppm_tolerance: float = 3.0
    rt_seconds: float = 10.0
    search_seconds: float = 40.0
    model_directory: Path = REFERENCE_DATA_DIRECTORY


def _validate(settings: ExperimentalPESelectionSettings) -> None:
    if (
        not 0.5 <= settings.minimum_pair_score <= 1
        or not 0.5 <= settings.minimum_rank_score <= 1
    ):
        raise ValueError("Experimental PE preference thresholds must be in [0.5, 1]")
    if not 0 < settings.ppm_tolerance <= 3 or not 0 < settings.rt_seconds <= 10:
        raise ValueError("Experimental PE cannot exceed 3 ppm / 10 local RT seconds")
    if not 0 < settings.search_seconds <= 60:
        raise ValueError("Experimental PE discovery must remain within 60 seconds")


def experimental_pe_evidence(pool: pd.DataFrame) -> pd.DataFrame:
    """Numerical observations only; no feature IDs, labels or dataset names."""
    out = pe_rebuild_evidence(pool)
    for c in sorted(c for c in pool if c.startswith("q_")):
        values = pd.to_numeric(pool[c], errors="coerce")
        out[c] = values
        out[c + "_known"] = values.notna().astype(float)
    # Ordinary PE has five headgroup/glycerol carbons in addition to the acyl
    # sum. This is a rough natural 13C envelope check, not structural proof.
    expected = np.log((pool.carbon + 5) * 0.0107 / 0.9893)
    out["q_isotope_upper_envelope_error"] = (
        pool.q_isotope_upper_log_ratio - expected
    ).abs()
    out["q_isotope_lower_envelope_error"] = (
        -pool.q_isotope_lower_log_ratio - expected
    ).abs()
    out["within_median_intensity"] = pool.q_log_median - pool.q_log_median.groupby(
        pool["name"]
    ).transform("max")
    return out.replace([np.inf, -np.inf], np.nan).fillna(-50.0)


def prepare_experimental_pe_candidates(
    annotations: pd.DataFrame,
    reference: pd.DataFrame,
    features: pd.DataFrame,
    counterpart_annotations: pd.DataFrame,
    counterpart_features: pd.DataFrame,
    *,
    stream: str,
    settings: ExperimentalPESelectionSettings,
    ranges: CompositionRangeSettings = DEFAULT_COMPOSITION_RANGE_SETTINGS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _validate(settings)
    if stream not in PEERS:
        return pd.DataFrame(), pd.DataFrame(), {}
    pool, anchors, context = prepare_pe_rebuild_candidates(
        annotations,
        reference,
        features,
        counterpart_annotations,
        counterpart_features,
        settings=PERebuildSettings(
            ppm_tolerance=settings.ppm_tolerance, search_seconds=settings.search_seconds
        ),
        ranges=ranges,
        target_adduct="[M-H]-" if stream.endswith("Neg") else "[M+H]+",
    )
    if pool.empty:
        return pool, anchors, context
    quality = peak_evidence(
        features, pool.feat_feature_id, polarity=stream.split("_")[1]
    )
    pool = pool.merge(
        quality,
        left_on="feat_feature_id",
        right_on="feature_id",
        how="left",
        validate="many_to_one",
    )
    return pool, anchors, context


def pair_preference(a: np.ndarray, b: np.ndarray, model: dict) -> np.ndarray:
    """Enforce antisymmetry; a preference score is not calibrated accuracy."""
    columns = model["features"]
    forward = score_pe_rebuild(pd.DataFrame(a - b, columns=columns), model)
    reverse = score_pe_rebuild(pd.DataFrame(b - a, columns=columns), model)
    return 0.5 + 0.5 * (forward - reverse)


def rank_experimental_pe(
    pool: pd.DataFrame, model: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evidence = experimental_pe_evidence(pool)
    required = model["features"]
    x = evidence[required]
    ranked = pool.copy()
    ranked["pe_experimental_rank_score"] = 0.5
    batches, slices, offset = [], [], 0
    for _, group in ranked.groupby("name", sort=True):
        if len(group) < 2:
            continue
        values = x.loc[group.index].to_numpy()
        a = np.repeat(values, len(group), axis=0)
        b = np.tile(values, (len(group), 1))
        batches.append((a, b))
        slices.append((group.index, offset, len(group)))
        offset += len(group) ** 2
    if not batches:
        return ranked, x
    all_preferences = pair_preference(
        np.concatenate([a for a, _ in batches]),
        np.concatenate([b for _, b in batches]),
        model,
    )
    for indices, offset, size in slices:
        pref = all_preferences[offset : offset + size**2].reshape(size, size)
        ranked.loc[indices, "pe_experimental_rank_score"] = (
            pref.sum(axis=1) - np.diag(pref)
        ) / (size - 1)
    return ranked, x


def _incumbent_preferences(
    ranked: pd.DataFrame, evidence: pd.DataFrame, annotations: pd.DataFrame, model: dict
) -> pd.Series:
    """Score all current-versus-proposed duels in one portable-model batch."""
    current = annotations.drop_duplicates("name", keep=False)
    fids = dict(zip(current["name"], current.feat_feature_id.astype(str)))
    lookup = {
        (str(r["name"]), str(r["feat_feature_id"])): i
        for i, r in ranked[["name", "feat_feature_id"]].iterrows()
    }
    target, prior = [], []
    for index, name in ranked["name"].items():
        old = lookup.get((str(name), fids.get(name, "")))
        if old is not None:
            target.append(index)
            prior.append(old)
    output = pd.Series(np.nan, index=ranked.index)
    if target:
        output.loc[target] = pair_preference(
            evidence.loc[target].to_numpy(), evidence.loc[prior].to_numpy(), model
        )
    return output


def _true_flags(frame: pd.DataFrame, column: str) -> pd.Series:
    # Treat strings explicitly: bool("False") would protect every CSV row.
    return (
        frame.get(column, pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(str)
        .str.casefold()
        .isin({"true", "1", "1.0"})
    )


def _protected(frame: pd.DataFrame) -> pd.Series:
    protected = _true_flags(frame, "is_is") | _true_flags(frame, "manual_confirmed")
    protected |= _true_flags(frame, "manual_override")
    protected |= frame.get("annotation_source", pd.Series("", index=frame.index)).eq(
        "manual"
    )
    protected |= (
        frame.get("internal_standard_name", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .ne("")
    )
    protected |= (
        frame.get("name", pd.Series("", index=frame.index))
        .astype(str)
        .str.endswith(" (IS)")
    )
    return protected


def _admission_evidence(row) -> tuple[bool, bool, bool, bool]:
    expected = np.log((row.carbon + 5) * 0.0107 / 0.9893)
    upper = bool(
        np.isfinite(row.q_isotope_upper_correlation)
        and row.q_isotope_upper_correlation >= 0.8
        and abs(row.q_isotope_upper_log_ratio - expected) <= 0.7
    )
    lower = bool(
        np.isfinite(row.q_isotope_lower_correlation)
        and row.q_isotope_lower_correlation >= 0.9
        and abs(-row.q_isotope_lower_log_ratio - expected) <= 0.5
    )
    counterpart = bool(
        pd.notna(row.pe_negative_rt)
        and abs(row.feat_rt - row.pe_negative_rt) * 60 <= 10
    )
    blank = bool(
        pd.isna(row.q_log_sample_blank_ratio)
        or row.q_log_sample_blank_ratio >= np.log(3)
    )
    return upper, lower, counterpart, blank


def select_experimental_pe(
    annotations: pd.DataFrame,
    all_pairs: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    stream: str,
    model: dict,
    settings: ExperimentalPESelectionSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Resolve a frozen pool; this function never fits anything from curation."""
    _validate(settings)
    if not settings.enabled or stream not in PEERS:
        return (
            annotations.copy(),
            all_pairs.copy(),
            pd.DataFrame(),
            pd.DataFrame(),
            dict(
                stream=stream,
                status="disabled" if not settings.enabled else "not_applicable",
                added_PE=0,
                replaced_PE=0,
            ),
        )
    ranked, evidence = rank_experimental_pe(pool, model)
    ranked["pe_experimental_incumbent_preference_score"] = _incumbent_preferences(
        ranked,
        evidence,
        annotations,
        model,
    )
    reserved = set(
        annotations.loc[
            annotations["class"].ne("PE") | _protected(annotations), "feat_feature_id"
        ].astype(str)
    )
    reserved |= set(all_pairs.loc[_protected(all_pairs), "feat_feature_id"].astype(str))
    locked = set(annotations.loc[_protected(annotations), "name"].astype(str))
    locked |= set(all_pairs.loc[_protected(all_pairs), "name"].astype(str))
    occupied = dict(
        zip(annotations.feat_feature_id.astype(str), annotations["name"].astype(str))
    )
    ranked["pe_experimental_mass_pass"] = ranked.pe_corrected_mass_error_ppm.abs().lt(
        settings.ppm_tolerance
    )
    ranked["pe_experimental_rt_pass"] = ranked.pe_local_rt_error_seconds.abs().le(
        settings.rt_seconds
    )
    ranked["pe_experimental_local_support_pass"] = ranked.pe_support_count.ge(5)
    ranked["pe_experimental_profile_pass"] = ranked.q_profile_samples.ge(4)
    ranked["pe_experimental_identity_pass"] = (
        ranked["class"].eq("PE")
        & ranked.adduct.eq("[M-H]-" if stream.endswith("Neg") else "[M+H]+")
        & ~_protected(ranked)
        & ranked.get("isotopologue", pd.Series("", index=ranked.index))
        .fillna("")
        .eq("")
    )
    ranked["pe_experimental_feature_available"] = [
        str(f) not in reserved
        and (str(f) not in occupied or occupied[str(f)] == str(n))
        for f, n in zip(ranked.feat_feature_id, ranked["name"])
    ]
    if stream == "RP_Pos":
        consensus = rp_consensus_checks(
            ranked,
            ppm_tolerance=settings.ppm_tolerance,
            rt_seconds=settings.rt_seconds,
            minimum_rank_score=settings.minimum_rank_score,
        )
        ranked = pd.concat([ranked, consensus], axis=1)
        ranked["pe_rp_consensus_pass"] = consensus.all(axis=1)
    ranked["pe_experimental_decision"] = "not_selected"
    ranked["pe_experimental_final_selected"] = False
    chosen_indices, changes = [], []
    gates = [
        "pe_experimental_mass_pass",
        "pe_experimental_rt_pass",
        "pe_experimental_local_support_pass",
        "pe_experimental_profile_pass",
        "pe_experimental_identity_pass",
        "pe_experimental_feature_available",
    ]
    for name, group in ranked.groupby("name", sort=True):
        current = annotations[annotations["name"].eq(name)]
        reason = "no_candidate_passing_independent_gates"
        allowed = group[group[gates].all(axis=1)]
        if name in locked:
            reason = "protected_manual_or_standard_identity"
        elif len(current) > 1:
            reason = "multiple_existing_choices_preserved"
        elif not allowed.empty:
            best = allowed.sort_values(
                ["pe_experimental_rank_score", "q_log_median", "feat_feature_id"],
                ascending=[False, False, True],
                kind="stable",
            ).iloc[0]
            isotope, isotope_child, counterpart, blank = _admission_evidence(best)
            if not current.empty:
                fid = str(current.feat_feature_id.iloc[0])
                previous = group[group.feat_feature_id.astype(str).eq(fid)]
                reason = "incumbent_outside_bounded_discovery"
                if best.feat_feature_id == fid:
                    reason = "retained_current_peak"
                elif not previous.empty:
                    prior = previous.iloc[0]
                    pair = float(best.pe_experimental_incumbent_preference_score)
                    mass_gain = abs(prior.pe_corrected_mass_error_ppm) - abs(
                        best.pe_corrected_mass_error_ppm
                    )
                    rt_gain = abs(prior.pe_local_rt_error_seconds) - abs(
                        best.pe_local_rt_error_seconds
                    )
                    reason = "withheld_weak_pair_preference"
                    if (
                        pair >= settings.minimum_pair_score
                        and best.pe_experimental_rank_score
                        >= settings.minimum_rank_score
                    ):
                        reason = "withheld_mass_RT_dominance_guard"
                        # RP-positive PE is the less discriminating ion mode.
                        # Require independent improvement in BOTH fits there;
                        # a good pooled score must not justify trading a correct
                        # dataset-level peak for a better-looking alternative.
                        improves = (
                            (mass_gain >= 0.3 and rt_gain >= 2)
                            if stream == "RP_Pos"
                            else (mass_gain >= 0.3 or rt_gain >= 2)
                        )
                        if mass_gain >= -0.2 and rt_gain >= -1 and improves:
                            reason = "withheld_supported_isotope_child"
                            if not isotope_child:
                                reason = "replace_automatic_PE"
                                chosen_indices.append(best.name)
                                changes.append(
                                    dict(
                                        stream=stream,
                                        name=name,
                                        adduct=best.adduct,
                                        old_feature_id=fid,
                                        new_feature_id=str(best.feat_feature_id),
                                        change="replacement",
                                        pair_preference_score=pair,
                                        mass_gain_ppm=mass_gain,
                                        local_rt_gain_seconds=rt_gain,
                                        isotope_support=isotope,
                                        opposite_mode_support=counterpart,
                                    )
                                )
            elif stream == "RP_Pos":
                reason = "withheld_RP_positive_consensus_admission"
                # Do not search for an alternative passing admission after the
                # preferred peak failed: that can promote the wrong quantifier.
                if best.pe_rp_consensus_pass and not isotope_child:
                    reason = "add_consensus_supported_RP_positive_PE"
                    chosen_indices.append(best.name)
                    changes.append(
                        dict(
                            stream=stream,
                            name=name,
                            adduct=best.adduct,
                            old_feature_id="",
                            new_feature_id=str(best.feat_feature_id),
                            change="addition",
                            pair_preference_score=np.nan,
                            mass_gain_ppm=np.nan,
                            local_rt_gain_seconds=np.nan,
                            isotope_support=isotope,
                            opposite_mode_support=counterpart,
                            NP_positive_feature_id=best.get(
                                "pe_rp_np_positive_feature_id", ""
                            ),
                            NP_negative_feature_id=best.get(
                                "pe_rp_np_negative_feature_id", ""
                            ),
                            admission_rule="RP_positive_four_stream_consensus",
                        )
                    )
            elif stream != "NP_Neg":
                reason = "additions_outside_NP_negative_disabled"
            else:
                reason = "withheld_NP_negative_admission"
                admission = (
                    best.pe_corrected_mass_error_ppm.__abs__()
                    < min(2.5, settings.ppm_tolerance)
                    and best.pe_local_rt_error_seconds.__abs__()
                    <= min(5, settings.rt_seconds)
                    and best.q_detection_fraction >= 0.5
                    and blank
                    and not isotope_child
                    and (isotope or counterpart)
                )
                unambiguous = (
                    len(allowed) == 1
                    or best.pe_experimental_rank_score >= settings.minimum_rank_score
                )
                if admission and unambiguous:
                    reason = "add_supported_NP_negative_PE"
                    chosen_indices.append(best.name)
                    changes.append(
                        dict(
                            stream=stream,
                            name=name,
                            adduct=best.adduct,
                            old_feature_id="",
                            new_feature_id=str(best.feat_feature_id),
                            change="addition",
                            pair_preference_score=np.nan,
                            mass_gain_ppm=np.nan,
                            local_rt_gain_seconds=np.nan,
                            isotope_support=isotope,
                            opposite_mode_support=counterpart,
                        )
                    )
        ranked.loc[group.index, "pe_experimental_decision"] = reason
    chosen = ranked.loc[chosen_indices].copy()
    # The same raw feature cannot be consumed by two proposed identities.
    if chosen.feat_feature_id.astype(str).duplicated().any():
        raise ValueError("Experimental PE proposals collide on an original feature")
    replaced_names = {r["name"] for r in changes if r["change"] == "replacement"}
    chosen["automatic_accepted"], chosen["automatic_candidate"] = True, True
    chosen["pe_experimental_final_selected"] = True
    chosen["pe_experimental_evidence_pass"] = True
    chosen["pe_final_selected"] = True
    chosen["pe_decision"] = "accepted_experimental"
    chosen["auto_annotation_status"] = "pe_experimental_supported"
    chosen["status"] = "pe_experimental_supported"
    chosen["rt_evidence_mode"] = "experimental_PE_local_RT_quality_and_pair_preference"
    chosen["rank_for_feature"] = 1
    chosen["standard_stream"], chosen["chromatography"] = stream, stream.split("_")[0]
    # Keep historical grid scores and flags. This separate, explicit rule can
    # supersede an endogenous grid rejection without rewriting the old audit.
    chosen["grid_coherence_exemption"] = "experimental_PE_independent_evidence"
    selected = pd.concat(
        [annotations[~annotations["name"].isin(replaced_names)], chosen],
        ignore_index=True,
        sort=False,
    )
    if selected.feat_feature_id.astype(str).duplicated().any():
        raise ValueError("Experimental PE produced conflicting primary features")
    updated_pairs = all_pairs.copy()
    old_keys = set(_pe_keys(annotations[annotations["name"].isin(replaced_names)]))
    replaced = _pe_keys(updated_pairs).isin(old_keys)
    for flag in ("automatic_accepted", "automatic_candidate", "pe_final_selected"):
        if flag in updated_pairs:
            updated_pairs.loc[replaced, flag] = False
    updated_pairs.loc[replaced, "pe_experimental_decision"] = "replaced_automatic_PE"
    chosen_keys = set(_pe_keys(chosen))
    updated_pairs = pd.concat(
        [updated_pairs[~_pe_keys(updated_pairs).isin(chosen_keys)], chosen],
        ignore_index=True,
        sort=False,
    )
    final_keys = set(_pe_keys(selected))
    ranked["pe_experimental_final_selected"] = _pe_keys(ranked).isin(final_keys)
    changes_frame = pd.DataFrame(
        changes,
        columns=[
            "stream",
            "name",
            "adduct",
            "old_feature_id",
            "new_feature_id",
            "change",
            "pair_preference_score",
            "mass_gain_ppm",
            "local_rt_gain_seconds",
            "isotope_support",
            "opposite_mode_support",
            "NP_positive_feature_id",
            "NP_negative_feature_id",
            "admission_rule",
        ],
    )
    audit = dict(
        stream=stream,
        status="completed",
        method="experimental_PE_quality_pair_selector",
        baseline_annotations=len(annotations),
        final_annotations=len(selected),
        candidate_rows=len(ranked),
        added_PE=sum(r["change"] == "addition" for r in changes),
        replaced_PE=len(replaced_names),
        minimum_pair_score=settings.minimum_pair_score,
        minimum_rank_score=settings.minimum_rank_score,
        mass_rule="strict corrected error below configured tolerance, <=3 ppm",
        RT_rule="target-excluded local prediction, <=10 seconds; new identities <=5 seconds",
        RP_positive_admission="mass<2.5ppm; localRT<=5s; rank>=0.8; isotope correlation>=0.9; biological detection>=80%; measured blank contrast>=3; accepted RP-negative within5s and both frozen NP primary identities",
    )
    return selected, updated_pairs, ranked, changes_frame, audit


def refine_experimental_pe_stream(
    annotations: pd.DataFrame,
    all_pairs: pd.DataFrame,
    reference: pd.DataFrame,
    features: pd.DataFrame,
    counterpart_annotations: pd.DataFrame,
    counterpart_features: pd.DataFrame,
    *,
    stream: str,
    settings: ExperimentalPESelectionSettings,
    ranges: CompositionRangeSettings = DEFAULT_COMPOSITION_RANGE_SETTINGS,
    confirmation_annotations: dict[str, pd.DataFrame] | None = None,
) -> tuple:
    _validate(settings)
    audit = dict(
        stream=stream,
        status="disabled" if not settings.enabled else "not_applicable",
        method="experimental_PE_quality_pair_selector",
        added_PE=0,
        replaced_PE=0,
    )
    unchanged = (annotations.copy(), all_pairs.copy(), pd.DataFrame(), pd.DataFrame())
    if not settings.enabled or stream not in PEERS:
        return (*unchanged, pd.DataFrame([audit]))
    try:
        path = Path(settings.model_directory) / MODEL_FILENAMES[stream]
        source = path.read_bytes()
        payload = json.loads(source)
        if (
            payload.get("format") != "laser-experimental-pe-selector-v1"
            or payload.get("stream") != stream
        ):
            raise ValueError("Experimental PE model format/stream mismatch")
        pool, anchors, context = prepare_experimental_pe_candidates(
            annotations,
            reference,
            features,
            counterpart_annotations,
            counterpart_features,
            stream=stream,
            settings=settings,
            ranges=ranges,
        )
        if pool.empty:
            audit.update(status="no_bounded_PE_candidates", **context)
            return (*unchanged, pd.DataFrame([audit]))
        if stream == "RP_Pos":
            pool = add_rp_consensus_evidence(pool, confirmation_annotations or {})
        selected, pairs, decisions, changes, audit = select_experimental_pe(
            annotations,
            all_pairs,
            pool,
            stream=stream,
            model=payload["pair_model"],
            settings=settings,
        )
        audit.update(
            context,
            model_sha256=hashlib.sha256(source).hexdigest(),
            model_training_datasets=";".join(payload.get("training_datasets", [])),
            independent_anchor_rows=len(anchors),
        )
        return selected, pairs, decisions, changes, pd.DataFrame([audit])
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        np.linalg.LinAlgError,
    ) as error:
        audit.update(status="experiment_unavailable", detail=str(error))
        return (*unchanged, pd.DataFrame([audit]))


def apply_experimental_pe_selector(
    result: dict,
    *,
    settings: ExperimentalPESelectionSettings = ExperimentalPESelectionSettings(),
    ranges: CompositionRangeSettings = DEFAULT_COMPOSITION_RANGE_SETTINGS,
) -> dict:
    """Apply once, after all default phase models and the standard safeguard."""
    _validate(settings)
    if not settings.enabled or result.get("experimental_pe_selector_applied"):
        return result.copy()
    updated = result.copy()
    streams = {s: p.copy() for s, p in result.get("streams", {}).items()}
    preceding = {s: p["annotations"].copy() for s, p in streams.items()}
    audits, decisions, changes = [], [], []
    changed = False
    for stream, peer in PEERS.items():
        if stream not in streams:
            continue
        p = streams[stream]
        p["annotations"], p["all_pairs"], detail, replacements, audit = (
            refine_experimental_pe_stream(
                p["annotations"],
                p["all_pairs"],
                p["reference"],
                p["features"],
                preceding.get(peer, pd.DataFrame()),
                streams.get(peer, {}).get("features", pd.DataFrame()),
                stream=stream,
                settings=settings,
                ranges=ranges,
                confirmation_annotations=preceding,
            )
        )
        detail = detail.assign(stream=stream)
        p["experimental_pe_decisions"], p["experimental_pe_changes"] = (
            detail,
            replacements,
        )
        audits.append(audit)
        decisions.append(detail)
        changes.append(replacements)
        changed |= not replacements.empty
    updated["streams"] = streams
    if changed:
        assembled = assemble_result(streams, [], Path("."), Path("."))
        for name in (
            "all_candidates",
            "automatic_annotations",
            "identified_lipids",
            "equivalent_lipids",
        ):
            updated[name] = assembled[name]
    for name, tables in (
        ("matlab_experimental_pe_audit", audits),
        ("matlab_experimental_pe_decisions", decisions),
        ("matlab_experimental_pe_changes", changes),
    ):
        updated[name] = (
            pd.concat(tables, ignore_index=True, sort=False)
            if tables
            else pd.DataFrame()
        )
    updated["experimental_pe_selector_applied"] = True
    return updated
