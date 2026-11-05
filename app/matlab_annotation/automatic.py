"""Automatic, conservative anchors for the MATLAB staged annotation loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scientific_defaults import SCIENTIFIC_DEFAULTS

from annotator.composition_ranges import (
    DEFAULT_COMPOSITION_RANGE_SETTINGS,
    CompositionRangeSettings,
)
from annotator.fa_model import expected_compositions

from .composition_filter import filter_compound_set_targets
from .decode import CodeDecoder
from .vendor.annotation_engine import AnnotationEngine
from .vendor.annotation_runner import _target_database_rt_seconds
from .vendor.model_dispatch import (
    UnsupportedPhaseDefinition,
    build_calibrated_annotation_model,
)
from .vendor.naming import compound_display_template
from .vendor.target_model import build_compound_set_target_table
from .xcms_project import with_memberships


# These are the RP annotation channels retained by the MATLAB application in
# its raw primary/equivalent exports. Sodium, potassium, chloride and formate
# signals are useful review evidence but are not promoted to lipid identities.
MATLAB_RP_ANNOTATION_ADDUCTS = frozenset({
    "[M+H]+", "[M+NH4]+", "[M-H]-", "[M-2H](2-)",
})

# NP uses the adduct channels explicitly configured for NP in the MATLAB
# catalogue. Unlike RP, sodium/chloride/formate and dehydrated ions are part of
# the normal-phase measurement policy rather than review-only companions.
MATLAB_NP_ANNOTATION_ADDUCTS = frozenset({
    "[M+H]+", "[M+H-H2O]+", "[M+NH4]+", "[M+Na]+",
    "[M-H]-", "[M-2H](2-)", "[M-4H](4-)", "[M+Cl]-", "[M+CHO2]-",
})


def matlab_annotation_adducts(phase: str) -> frozenset[str]:
    """Return the MATLAB export channels retained for one chromatography."""
    phase = str(phase).upper()
    if phase == "RP":
        return MATLAB_RP_ANNOTATION_ADDUCTS
    if phase == "NP":
        return MATLAB_NP_ANNOTATION_ADDUCTS
    raise ValueError(f"automatic MATLAB annotation requires NP or RP, not {phase!r}")

BIOLOGICAL_ABUNDANCE_PRIOR = "biological_abundance_prior"
AUTOMATIC_REGRESSION_INLIERS = "automatic_regression_inliers"
UNIFIED_EVIDENCE_ANCHORS = "unified_evidence_anchors"
AUTOMATIC_ANCHOR_SELECTION_MODES = (
    UNIFIED_EVIDENCE_ANCHORS,
    BIOLOGICAL_ABUNDANCE_PRIOR,
    AUTOMATIC_REGRESSION_INLIERS,
)

POST_REGRESSION_POSITIONAL_RESOLUTION = "post_regression"
ITERATIVE_POSITIONAL_REFIT = "iterative_refit"
POSITIONAL_ISOMER_WORKFLOW_MODES = (
    POST_REGRESSION_POSITIONAL_RESOLUTION,
    ITERATIVE_POSITIONAL_REFIT,
)


def _canonical_species_name(value: object) -> str:
    """Normalise harmless whitespace differences in lipid species labels.

    The bundled MATLAB catalogue renders positional lyso species as
    ``1-acylLPC`` whereas the chemical reference used for trusted XCMS seeds
    renders ``1-acyl LPC``.  They are the same identity for consensus
    matching; the original spelling is retained in result tables.
    """

    return "".join(str(value).split())


def _rtdb_only_lyso_groups(
    project, engine, group_compounds, *, phase: str,
) -> set[int]:
    """Return positional 1-/2-acyl lyso groups eligible for RTDB seeding.

    The ordinary LPC/LPE pair is an MS1 positional-isomer set: both members
    have the same precursor formula and therefore the same theoretical m/z.
    The XCMS-only workflow can still seed this group from the calibrated RTDB
    evidence, but it cannot honestly require an independent chemical-reference
    row for each positional label.  Keep this exception narrowly scoped to the
    two known lyso families; all other groups retain reference-consensus
    protection.
    """

    out: set[int] = set()
    if phase != "RP":
        return out
    composition = np.asarray(
        project.compounds.legacy_fields["composition"], dtype=object
    )
    for group_compound in group_compounds:
        try:
            active = engine.siblings(phase, int(group_compound))
        except Exception:
            continue
        labels = set()
        for compound in active:
            parts = composition[int(compound) - 1]
            labels.add(_canonical_species_name(
                f"{parts[0]}{project.compounds.classes[int(compound) - 1]}{parts[1]}"
            ))
        if labels in ({"1-acylLPC", "2-acylLPC"}, {"1-acylLPE", "2-acylLPE"}):
            out.add(int(group_compound))
    return out


@dataclass(frozen=True)
class AutomaticAnchorSettings:
    """Guardrails for promoting XCMS/RTDB matches to regression anchors."""

    ppm_tolerance: float = 3.0
    rt_tolerance_seconds: float = 30.0
    maximum_score: float = 1.0
    minimum_score_margin: float = 0.20
    minimum_anchors: int = 5
    maximum_anchors: int = 20
    anchor_selection_mode: str = UNIFIED_EVIDENCE_ANCHORS
    expected_abundance_top_k: int = 15
    minimum_carbon_span: int = 4
    minimum_double_bond_span: int = 2
    grid_coherence_rt_tolerance_seconds: float = 10.0
    grid_coherence_version: str = "v1"
    use_grid_coherence: bool = True
    grid_coherence_threshold: float = (
        SCIENTIFIC_DEFAULTS.grid_coherence_threshold
    )
    positional_isomer_rt_tolerance_seconds: float = 10.0
    positional_isomer_gap_tolerance_seconds: float = 8.0
    positional_isomer_minimum_separation_seconds: float = 3.0
    positional_isomer_score_margin: float = 0.20
    positional_isomer_local_rt_adjustment_seconds: float = 10.0
    positional_isomer_local_support_groups: int = 2
    positional_isomer_local_composition_distance: int = 4
    positional_isomer_workflow_mode: str = (
        POST_REGRESSION_POSITIONAL_RESOLUTION
    )
    use_post_regression_completion: bool = True
    post_regression_confidence_threshold: float = 0.50
    post_regression_ambiguity_margin: float = 0.10
    use_fitted_grid_completion: bool = True
    fitted_grid_feature_ambiguity_margin: float = 0.10
    use_rp_isomer_anchor_refinement: bool = True
    use_pe_completion: bool = True
    pe_completion_confidence_threshold: float = 0.90
    use_pe_rebuild: bool = True
    pe_rebuild_score_threshold: float = 0.10
    pe_rebuild_feature_margin: float = 0.03
    pe_rebuild_search_seconds: float = 40.0
    use_experimental_pe_selector: bool = False
    excluded_anchor_species: tuple[str, ...] = (
        "1-acylLPC(20:5)",
        "1-acylLPC(22:1)",
        "1-acylLPC(22:5)",
    )


def _canonical_positional_anchor_species(value: object) -> str:
    """Compare positional exclusions against the primary 1-acyl identity."""
    species = str(value)
    if species.startswith("2-acylLPC("):
        return "1-acylLPC(" + species.removeprefix("2-acylLPC(")
    if species.startswith("2-acylLPE("):
        return "1-acylLPE(" + species.removeprefix("2-acylLPE(")
    return species


def _is_expected_abundance_anchor(decoded, *, top_k: int) -> bool:
    """True for protected standards or biologically expected compositions."""
    if bool(decoded.internal_standard):
        return True
    if decoded.carbon is None or decoded.double_bonds is None:
        return False
    expected = expected_compositions(
        decoded.lipid_class, top_k=int(top_k)
    )
    return (int(decoded.carbon), int(decoded.double_bonds)) in expected


def _anchor_selection_allows(
    decoded,
    *,
    mode: str,
    top_k: int,
) -> bool:
    """Return whether a trusted candidate may enter RT regression."""
    if mode == AUTOMATIC_REGRESSION_INLIERS:
        return True
    if mode == UNIFIED_EVIDENCE_ANCHORS:
        # One class-aware model, not competing annotation engines. LPC gains
        # from composition-wide trusted inliers because its abundant 1-acyl
        # peak is subsequently guarded by explicit early/late resolution.
        # LPE has no reliable abundance ordering, and the other classes were
        # more stable in the example cohorts with the biological panel, so
        # they retain that conservative eligibility rule.
        species = _canonical_species_name(decoded.species)
        if species.startswith(("1-acylLPC(", "2-acylLPC(")):
            return True
        return _is_expected_abundance_anchor(decoded, top_k=top_k)
    if mode == BIOLOGICAL_ABUNDANCE_PRIOR:
        return _is_expected_abundance_anchor(decoded, top_k=top_k)
    raise ValueError(
        "unknown automatic anchor-selection mode: "
        f"{mode!r}; expected one of {AUTOMATIC_ANCHOR_SELECTION_MODES}"
    )


def _candidate_matches(
    project,
    reference,
    engine: AnnotationEngine,
    calibration,
    group_compound: int,
    settings: AutomaticAnchorSettings,
    composition_ranges: CompositionRangeSettings,
    *,
    phase: str,
) -> tuple[pd.DataFrame, str]:
    """Return stage-1/2 RTDB matches for one sibling group."""
    try:
        active = engine.siblings(phase, group_compound)
        model = build_calibrated_annotation_model(
            project, calibration, phase=phase, compounds=active
        )
        targets = build_compound_set_target_table(
            project, reference, phase, active
        )
        targets, _ = filter_compound_set_targets(
            project,
            targets,
            active,
            phase=phase,
            settings=composition_ranges,
        )
        database_rt = _target_database_rt_seconds(
            project, reference, phase, active, targets, model.rt_preprocessing
        )
    except (UnsupportedPhaseDefinition, ValueError, np.linalg.LinAlgError) as error:
        return pd.DataFrame(), f"{type(error).__name__}: {error}"

    target_mask = (
        # Positive codes are endogenous lipids; MATLAB reserves negative
        # composition codes for internal standards.  Both are real targets.
        (targets.codes != 0)
        & (targets.stages <= 2)
        & np.isfinite(targets.theoretical_mz)
        & np.isfinite(database_rt)
    )
    target_indices = np.flatnonzero(target_mask)
    decoders = {
        compound: CodeDecoder(project, compound, phase) for compound in active
    }
    target_indices = np.asarray([
        target_index for target_index in target_indices
        if decoders[
            int(active[int(targets.sibling_positions[target_index]) - 1])
        ].decode(int(targets.codes[target_index])).adduct
        in matlab_annotation_adducts(phase)
    ], dtype=np.int64)
    if not len(target_indices):
        return pd.DataFrame(), "no stage-1/2 targets with RTDB predictions"

    data = project.phases[phase]
    observed_mz = np.asarray(data.mz_grouped, dtype=float)
    observed_rt = np.asarray(data.rt_grouped, dtype=float)
    records: list[dict] = []
    for target_index in target_indices:
        target_mz = float(targets.theoretical_mz[target_index])
        tolerance = abs(target_mz) * settings.ppm_tolerance * 1e-6
        lower = int(np.searchsorted(observed_mz, target_mz - tolerance))
        upper = int(np.searchsorted(
            observed_mz, target_mz + tolerance, side="right"
        ))
        if upper <= lower:
            continue
        rows: np.ndarray = np.arange(lower, upper, dtype=np.int64)
        rt_error = observed_rt[rows] - float(database_rt[target_index])
        rows = rows[np.abs(rt_error) <= settings.rt_tolerance_seconds]
        if not len(rows):
            continue
        rt_error = observed_rt[rows] - float(database_rt[target_index])
        ppm_error = (
            (observed_mz[rows] - target_mz) / target_mz * 1e6
        )
        score = np.sqrt(
            (ppm_error / settings.ppm_tolerance) ** 2
            + (rt_error / settings.rt_tolerance_seconds) ** 2
        )
        sibling = int(targets.sibling_positions[target_index])
        compound = int(active[sibling - 1])
        code = int(targets.codes[target_index])
        decoded = decoders[compound].decode(code)
        for row, ppm, delta_rt, value in zip(
            rows, ppm_error, rt_error, score, strict=True
        ):
            records.append({
                "row": int(row),
                "group_compound": int(group_compound),
                "compound": compound,
                "sibling": sibling,
                "code": code,
                "species": decoded.species,
                "adduct": decoded.adduct,
                "stage": int(targets.stages[target_index]),
                "target_mz": target_mz,
                "database_rt_seconds": float(database_rt[target_index]),
                "ppm_error": float(ppm),
                "rt_error_seconds": float(delta_rt),
                "score": float(value),
            })
    return pd.DataFrame(records), ""


def derive_automatic_memberships(
    project,
    reference,
    group_compounds: list[int] | tuple[int, ...],
    *,
    phase: str,
    settings: AutomaticAnchorSettings = AutomaticAnchorSettings(),
    composition_ranges: CompositionRangeSettings = (
        DEFAULT_COMPOSITION_RANGE_SETTINGS
    ),
    trusted_anchors: set[tuple[int, str, str]] | None = None,
    progress=None,
):
    """Return a project seeded only by globally unambiguous XCMS/RTDB matches.

    Candidate anchors are first generated independently for every selectable
    lipid definition, then resolved globally per observed feature.  This is
    essential for common isobaric pairs: one peak cannot seed both PC/PE or
    PG/BMP merely because each class was evaluated in a separate sweep.
    """
    phase = str(phase).upper()
    if phase not in project.phases:
        raise ValueError(f"project does not contain requested phase {phase!r}")
    if not 0 < settings.ppm_tolerance <= 3.0:
        raise ValueError("automatic MATLAB anchor tolerance must be >0 and <=3 ppm")
    if settings.anchor_selection_mode not in AUTOMATIC_ANCHOR_SELECTION_MODES:
        raise ValueError(
            "unknown automatic anchor-selection mode: "
            f"{settings.anchor_selection_mode!r}; expected one of "
            f"{AUTOMATIC_ANCHOR_SELECTION_MODES}"
        )
    if (
        not isinstance(settings.expected_abundance_top_k, int)
        or settings.expected_abundance_top_k <= 0
    ):
        raise ValueError("expected-abundance anchor count must be a positive integer")
    engine = AnnotationEngine(project=project, reference=reference)
    calibration = engine.calibration(phase)
    frames: list[pd.DataFrame] = []
    base_audit: list[dict] = []
    total = max(len(group_compounds), 1)
    for index, group_compound in enumerate(group_compounds, start=1):
        if progress:
            progress(
                0.05 + 0.35 * index / total,
                f"{phase}: deriving automatic RT anchors ({index}/{len(group_compounds)})",
            )
        matches, detail = _candidate_matches(
            project, reference, engine, calibration, int(group_compound),
            settings, composition_ranges, phase=phase,
        )
        if not matches.empty:
            frames.append(matches)
        base_audit.append({
            "phase": phase,
            "group_compound": int(group_compound),
            "definition": compound_display_template(project, int(group_compound) - 1),
            "anchor_selection_mode": settings.anchor_selection_mode,
            "raw_anchor_candidates": int(len(matches)),
            "detail": detail,
        })

    candidates = (
        pd.concat(frames, ignore_index=True)
        if frames else pd.DataFrame(columns=[
            "row", "group_compound", "compound", "sibling", "code", "stage",
            "species", "adduct",
            "target_mz", "database_rt_seconds", "ppm_error",
            "rt_error_seconds", "score",
        ])
    )
    if candidates.empty:
        audit = pd.DataFrame(base_audit)
        audit["globally_unique_candidates"] = 0
        audit["anchor_mode_candidates"] = 0
        audit["anchors_mode_filtered"] = 0
        audit["abundance_prior_candidates"] = 0
        audit["anchors_abundance_filtered"] = 0
        audit["anchors_retained"] = 0
        audit["anchor_status"] = "no_candidates"
        return project, audit, candidates

    if trusted_anchors is not None:
        # A declared spike is trusted from the project panel plus its dedicated
        # database mass/RT.  It must not be rejected merely because the generic
        # reference also contains an endogenous isobar with the same sum
        # composition.  Endogenous targets still require independent consensus.
        canonical_trusted = {
            (int(feature), _canonical_species_name(name), str(adduct))
            for feature, name, adduct in trusted_anchors
        }
        # Positional LPC/LPE members are deliberately not independently
        # represented in the chemical reference.  Their calibrated RTDB
        # targets are valid seed evidence even when the legacy consensus table
        # contains only one positional spelling (or none for 2-acyl).
        rtdb_only_groups = _rtdb_only_lyso_groups(
            project, engine, group_compounds, phase=phase
        )
        trusted = np.asarray([
            int(code) < 0
            or int(group_compound) in rtdb_only_groups
            or (
                int(row), _canonical_species_name(species), str(adduct)
            ) in canonical_trusted
            for row, species, adduct, code, group_compound in candidates[
                ["row", "species", "adduct", "code", "group_compound"]
            ].itertuples(index=False, name=None)
        ], dtype=bool)
        candidates = candidates.loc[trusted].reset_index(drop=True)
        if candidates.empty:
            audit = pd.DataFrame(base_audit)
            audit["globally_unique_candidates"] = 0
            audit["anchor_mode_candidates"] = 0
            audit["anchors_mode_filtered"] = 0
            audit["abundance_prior_candidates"] = 0
            audit["anchors_abundance_filtered"] = 0
            audit["anchors_retained"] = 0
            audit["anchor_status"] = "no_reference_consensus"
            return project, audit, candidates

    # Duplicate rows for the same theoretical identity can arise at catalogue
    # sibling boundaries.  They are not independent alternatives.
    candidates = candidates.sort_values("score", kind="stable").drop_duplicates(
        ["row", "compound", "code"], keep="first"
    )
    anchor_decoders: dict[int, CodeDecoder] = {}
    abundance_allowed = []
    selection_allowed = []
    for compound, code in candidates[["compound", "code"]].itertuples(
        index=False, name=None
    ):
        compound = int(compound)
        decoder = anchor_decoders.get(compound)
        if decoder is None:
            decoder = CodeDecoder(project, compound, phase)
            anchor_decoders[compound] = decoder
        decoded = decoder.decode(int(code))
        abundance_allowed.append(_is_expected_abundance_anchor(
            decoded,
            top_k=settings.expected_abundance_top_k,
        ))
        selection_allowed.append(_anchor_selection_allows(
            decoded,
            mode=settings.anchor_selection_mode,
            top_k=settings.expected_abundance_top_k,
        ))
    candidates["abundance_prior_allowed"] = abundance_allowed
    candidates["anchor_selection_allowed"] = selection_allowed
    # These exclusions belong to the RP positional-LPC workflow. NP represents
    # LPC as one combined phase identity and must not inherit positional rules.
    excluded_species = (
        {
            _canonical_positional_anchor_species(species)
            for species in settings.excluded_anchor_species
        }
        if phase == "RP" else set()
    )
    candidates["anchor_species_excluded"] = candidates["species"].map(
        _canonical_positional_anchor_species
    ).isin(excluded_species)
    # Apply the selected eligibility mode before resolving global feature
    # conflicts. Otherwise an ineligible isobar could beat an eligible anchor
    # by a small RTDB score difference and make the trustworthy candidate
    # disappear.
    eligible = candidates[
        candidates["anchor_selection_allowed"]
        & ~candidates["anchor_species_excluded"]
    ].copy()
    resolved: list[pd.Series] = []
    for _, alternatives in eligible.groupby("row", sort=False):
        ordered = alternatives.sort_values("score", kind="stable")
        best = ordered.iloc[0]
        if float(best["score"]) > settings.maximum_score:
            continue
        if len(ordered) > 1:
            margin = float(ordered.iloc[1]["score"] - best["score"])
            if margin < settings.minimum_score_margin:
                continue
        resolved.append(best)
    unique = pd.DataFrame(resolved, columns=candidates.columns)

    retained_frames: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    base_by_group = {item["group_compound"]: item for item in base_audit}
    for group_compound in group_compounds:
        group_candidates = candidates[
            candidates["group_compound"].eq(group_compound)
        ]
        mode_filtered_count = int(
            (~group_candidates["anchor_selection_allowed"]).sum()
        )
        abundance_filtered_count = (
            mode_filtered_count
            if settings.anchor_selection_mode == BIOLOGICAL_ABUNDANCE_PRIOR
            else 0
        )
        excluded_count = int(
            (
                group_candidates["anchor_selection_allowed"]
                & group_candidates["anchor_species_excluded"]
            ).sum()
        )
        group = unique[unique["group_compound"].eq(group_compound)].copy()
        globally_unique_count = int(len(group))
        group["anchor_intensity"] = np.asarray(
            project.phases[phase].representative_intensity, dtype=float
        )[group["row"].astype(np.int64).to_numpy()]
        if settings.anchor_selection_mode == UNIFIED_EVIDENCE_ANCHORS:
            group["_unified_anchor_priority"] = np.select(
                [group["code"].lt(0), group["abundance_prior_allowed"]],
                [0, 1], default=2,
            )
            sort_columns = [
                "_unified_anchor_priority", "score", "anchor_intensity"
            ]
            sort_ascending = [True, True, False]
        else:
            sort_columns = ["score", "anchor_intensity"]
            sort_ascending = [True, False]
        group = group.sort_values(
            sort_columns,
            ascending=sort_ascending,
            kind="stable",
        ).drop_duplicates(
            ["compound", "code"], keep="first"
        )
        mode_candidate_count = int(len(group))
        abundance_prior_count = int(group["abundance_prior_allowed"].sum())
        # Declared spikes are protected controls, not ordinary grid points.
        # Keep every matched internal standard first and use endogenous anchors
        # only for the remaining per-group slots.
        standards = group[group["code"].lt(0)]
        endogenous = group[group["code"].gt(0)]
        remaining = max(settings.maximum_anchors - len(standards), 0)
        group = pd.concat(
            [standards, endogenous.head(remaining)], ignore_index=True
        )
        carbon_values: list[int] = []
        db_values: list[int] = []
        for compound, code in zip(group.get("compound", []), group.get("code", [])):
            decoded = anchor_decoders[int(compound)].decode(int(code))
            if decoded.carbon is not None:
                carbon_values.append(int(decoded.carbon))
            if decoded.double_bonds is not None:
                db_values.append(int(decoded.double_bonds))
        carbon_span = max(carbon_values) - min(carbon_values) if carbon_values else 0
        db_span = max(db_values) - min(db_values) if db_values else 0
        sufficient = bool(
            len(group) >= settings.minimum_anchors
            and (
                carbon_span >= settings.minimum_carbon_span
                or db_span >= settings.minimum_double_bond_span
            )
        )
        if sufficient:
            retained_frames.append(group)
            status = "anchored"
        elif group.empty:
            status = "no_unambiguous_anchors"
        else:
            status = "insufficient_grid_span"
        base = base_by_group[int(group_compound)]
        audit_rows.append({
            **base,
            "globally_unique_candidates": globally_unique_count,
            "anchor_mode_candidates": mode_candidate_count,
            "anchors_mode_filtered": mode_filtered_count,
            "abundance_prior_candidates": abundance_prior_count,
            "anchors_abundance_filtered": abundance_filtered_count,
            "carbon_span": int(carbon_span),
            "double_bond_span": int(db_span),
            "anchors_retained": int(len(group)) if sufficient else 0,
            "anchors_excluded": excluded_count,
            "median_anchor_intensity": (
                float(group["anchor_intensity"].median())
                if sufficient else float("nan")
            ),
            "anchor_status": status,
        })

    retained = (
        pd.concat(retained_frames, ignore_index=True)
        if retained_frames else pd.DataFrame(columns=candidates.columns)
    )
    membership_lists: list[list[list[object]]] = [
        [] for _ in range(project.compounds.count)
    ]
    data = project.phases[phase]
    for row in retained.itertuples(index=False):
        feature_row = int(row.row)
        membership_lists[int(row.compound) - 1].append([
            phase, int(row.code), float(data.mz_grouped[feature_row]),
            float(data.rt_grouped[feature_row]),
        ])
    seeded = with_memberships(
        project, tuple(membership_lists)
    )
    return seeded, pd.DataFrame(audit_rows), retained


def derive_automatic_rp_memberships(
    project,
    reference,
    group_compounds: list[int] | tuple[int, ...],
    *,
    settings: AutomaticAnchorSettings = AutomaticAnchorSettings(),
    trusted_anchors: set[tuple[int, str, str]] | None = None,
    progress=None,
):
    """Backward-compatible RP wrapper around phase-aware anchor selection."""
    return derive_automatic_memberships(
        project,
        reference,
        group_compounds,
        phase="RP",
        settings=settings,
        trusted_anchors=trusted_anchors,
        progress=progress,
    )
