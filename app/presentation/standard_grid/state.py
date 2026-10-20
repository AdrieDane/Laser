"""Pure grid-state construction for the standard-lipidomics viewer.

Keeping this code independent from Streamlit makes the scientific display state
testable without starting the GUI.  The browser remains responsible only for
session state and presentation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import annotator.grid_model as grid_model
from scientific_defaults import SCIENTIFIC_DEFAULTS
from viewer_common import (
    apply_anchor_correction,
    apply_anchor_correction_to_line,
    fit_anchor_correction,
)


def _class_grid_rt(reference: pd.DataFrame) -> pd.DataFrame:
    grid = reference[["carbon", "db", "ref_mz", "grid_rt"]].copy()
    grid = grid.rename(columns={"grid_rt": "rt"})
    for column in grid.columns:
        grid[column] = pd.to_numeric(grid[column], errors="coerce")
    return grid


def select_plotted_species(candidates: pd.DataFrame) -> pd.DataFrame:
    """Select one displayed feature per lipid identity."""
    accepted = candidates.get(
        "automatic_candidate", pd.Series(False, index=candidates.index)
    ).fillna(False).astype(bool)
    plotted = (
        candidates.assign(_run_primary=np.where(accepted, 0, 1)).sort_values([
            "manual_primary", "_run_primary", "automatic_anchor_primary",
            "absresid", "absppm",
        ])
        .drop_duplicates(subset=["name"], keep="first")
        .drop(columns="_run_primary")
        .copy()
    )
    plotted["node_rt"] = plotted["rt_pred"]
    plotted["node_mz"] = plotted["ref_mz"]
    return plotted


def review_criteria_mask(
    candidates: pd.DataFrame,
    *,
    ppm_tolerance: float,
    rt_tolerance: float,
    use_rt: bool,
    use_grid_coherence: bool = False,
    grid_coherence_threshold: float = (
        SCIENTIFIC_DEFAULTS.grid_coherence_threshold
    ),
    rt_residual_column: str = "grid_resid",
) -> pd.Series:
    """Return the shared run/live-view inclusion mask."""
    mass_error = candidates["ppm_error"].copy()
    corrected = pd.to_numeric(
        candidates.get("pe_corrected_mass_error_ppm", pd.Series(np.nan, index=candidates.index)),
        errors="coerce",
    )
    rebuilt = _rebuilt_pe_display_mask(candidates) & corrected.notna()
    # This is the actual PE acceptance mass model, not a silent widening of
    # the raw window. Original ppm_error remains available in the details.
    mass_error.loc[rebuilt] = corrected.loc[rebuilt]
    keep = mass_error.abs() <= float(ppm_tolerance)
    if use_rt:
        keep &= candidates[rt_residual_column].abs() <= float(rt_tolerance)
    if use_grid_coherence:
        assessed = candidates.get(
            "grid_coherence_assessed",
            pd.Series(False, index=candidates.index),
        ).fillna(False).astype(bool)
        score = pd.to_numeric(
            candidates.get(
                "grid_coherence_score",
                pd.Series(np.nan, index=candidates.index),
            ),
            errors="coerce",
        )
        declared_pe = candidates.get(
            "auto_annotation_status", pd.Series("", index=candidates.index)
        ).eq("pe_declared_standard_supported")
        declared_pe &= candidates.get(
            "automatic_candidate", pd.Series(False, index=candidates.index)
        ).fillna(False).astype(bool)
        keep &= ~assessed | score.ge(float(grid_coherence_threshold)) | declared_pe
    return keep.fillna(False)


def _rebuilt_pe_display_mask(candidates: pd.DataFrame) -> pd.Series:
    status = candidates.get("auto_annotation_status", pd.Series("", index=candidates.index))
    accepted = candidates.get("automatic_candidate", pd.Series(False, index=candidates.index))
    return status.eq("pe_raw_evidence_supported") & accepted.fillna(False).astype(bool)


def apply_review_criteria(
    candidates: pd.DataFrame,
    *,
    ppm_tolerance: float,
    rt_tolerance: float,
    use_rt: bool,
    use_grid_coherence: bool = False,
    grid_coherence_threshold: float = (
        SCIENTIFIC_DEFAULTS.grid_coherence_threshold
    ),
) -> pd.DataFrame:
    """Apply view-only criteria without changing run annotations."""
    if candidates.empty:
        return candidates
    keep = review_criteria_mask(
        candidates,
        ppm_tolerance=ppm_tolerance,
        rt_tolerance=rt_tolerance,
        use_rt=use_rt,
        use_grid_coherence=use_grid_coherence,
        grid_coherence_threshold=grid_coherence_threshold,
    )
    return candidates[keep].copy()


def _prepare_candidates(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    lipid_class: str,
    adduct: str,
    confirmed: set[str],
    primary_overrides: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    ref_ca = reference[
        reference["grid_class"].eq(lipid_class)
        & reference["adduct"].astype(str).eq(adduct)
    ].copy()
    cand = candidates[
        candidates["grid_class"].eq(lipid_class)
        & candidates["adduct"].astype(str).eq(adduct)
    ].copy()
    mesh = (
        grid_model.build_mesh(_class_grid_rt(ref_ca), rt_col="rt")
        if not ref_ca.empty else None
    )
    cand["rt_pred_db"] = cand.get("ref_rt", pd.Series(dtype=float))
    if not ref_ca.empty and not cand.empty:
        shifted = (
            ref_ca.drop_duplicates(["name"], keep="first")
            .set_index("name")["grid_rt"]
        )
        cand["rt_pred_db"] = cand["name"].astype(str).map(shifted)
    if mesh is not None and not cand.empty:
        cand["rt_pred_db"] = grid_model.predict_rt(
            mesh["model"], cand["carbon"].to_numpy(), cand["db"].to_numpy()
        )
    cand["anchor"] = cand.get(
        "automatic_rt_anchor", pd.Series(False, index=cand.index)
    ).fillna(False).astype(bool)
    cand["confirmed"] = cand["name"].astype(str).isin(confirmed)
    cand["grid_resid_db"] = cand["feat_rt"] - cand["rt_pred_db"]
    cand["run_grid_resid"] = cand["grid_resid_db"]
    cand["absresid_db"] = cand["grid_resid_db"].abs()
    cand["absppm"] = cand["ppm_error"].abs()
    cand["manual_primary"] = [
        0 if primary_overrides.get(str(name)) == str(feature_id) else 1
        for name, feature_id in zip(cand.get("name", []), cand.get("feat_feature_id", []))
    ]
    cand["automatic_anchor_primary"] = np.where(cand["anchor"], 0, 1)
    return ref_ca, cand, mesh


def _apply_anchor_model(cand: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    anchor_seed = (
        cand.sort_values([
            "manual_primary", "automatic_anchor_primary",
            "absresid_db", "absppm",
        ]).drop_duplicates(subset=["name"], keep="first")
    )
    correction = fit_anchor_correction(anchor_seed[anchor_seed["confirmed"]])
    cand["rt_pred"] = apply_anchor_correction(
        correction,
        cand["rt_pred_db"].to_numpy(),
        cand["carbon"].to_numpy(),
        cand["db"].to_numpy(),
    )
    cand["grid_resid"] = cand["feat_rt"] - cand["rt_pred"]
    positional_expected = pd.to_numeric(
        cand.get("positional_isomer_rt_expected", pd.Series(np.nan, index=cand.index)),
        errors="coerce",
    )
    positional_error_seconds = pd.to_numeric(
        cand.get(
            "positional_isomer_rt_error_seconds",
            pd.Series(np.nan, index=cand.index),
        ),
        errors="coerce",
    )
    positional = positional_expected.notna() & positional_error_seconds.notna()
    cand.loc[positional, "rt_pred"] = positional_expected.loc[positional]
    cand.loc[positional, "grid_resid"] = positional_error_seconds.loc[positional] / 60.0
    cand.loc[positional, "run_grid_resid"] = positional_error_seconds.loc[positional] / 60.0
    pe_error = pd.to_numeric(
        cand.get("pe_local_rt_error_seconds", pd.Series(np.nan, index=cand.index)),
        errors="coerce",
    )
    pe = _rebuilt_pe_display_mask(cand) & pe_error.notna()
    cand.loc[pe, "rt_pred"] = cand.loc[pe, "feat_rt"] - pe_error.loc[pe] / 60.0
    cand.loc[pe, "grid_resid"] = pe_error.loc[pe] / 60.0
    cand.loc[pe, "run_grid_resid"] = pe_error.loc[pe] / 60.0
    cand["absresid"] = cand["grid_resid"].abs()
    cand["rt_error"] = cand["grid_resid"]
    return cand, correction


def _visible_mesh(
    ref_ca: pd.DataFrame,
    cand: pd.DataFrame,
    correction: dict,
) -> tuple[dict | None, dict]:
    if ref_ca.empty:
        return None, {}
    drawn = cand.dropna(subset=["carbon", "db"])
    mesh = grid_model.build_mesh(
        _class_grid_rt(ref_ca),
        rt_col="rt",
        draw_c=(drawn["carbon"].min(), drawn["carbon"].max()) if not drawn.empty else None,
        draw_d=(drawn["db"].min(), drawn["db"].max()) if not drawn.empty else None,
    )
    if mesh is None:
        return None, {}
    mesh_lines = {}
    for family, coord_key in (("db_lines", "db_line_coords"), ("c_lines", "c_line_coords")):
        xx, yy = mesh[family]
        cc, dd = mesh[coord_key]
        mesh_lines[family] = (
            apply_anchor_correction_to_line(correction, xx, cc, dd), yy
        )
    return mesh, mesh_lines


def build_class_grid_state(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    lipid_class: str,
    adduct: str,
    confirmed: set[str],
    primary_overrides: dict[str, str] | None = None,
    review: dict | None = None,
    run_criteria: dict | None = None,
) -> dict:
    """Resolve reference mesh, anchors, filters, and displayed primaries."""
    ref_ca, cand, _ = _prepare_candidates(
        reference,
        candidates,
        lipid_class,
        adduct,
        confirmed,
        primary_overrides or {},
    )
    cand, correction = _apply_anchor_model(cand)
    if run_criteria is None:
        cand["run_candidate"] = True
    else:
        cand["run_candidate"] = review_criteria_mask(
            cand, rt_residual_column="run_grid_resid", **run_criteria
        )
        cand["run_candidate"] |= cand.get(
            "automatic_candidate", pd.Series(False, index=cand.index)
        ).fillna(False).astype(bool)
    all_candidates = cand.copy()
    if review is not None:
        cand = apply_review_criteria(cand, **review)
    plot_df = select_plotted_species(cand)
    mesh, mesh_lines = _visible_mesh(ref_ca, cand, correction)
    return {
        "reference": ref_ca,
        "all_candidates": all_candidates,
        "candidates": cand,
        "plot_df": plot_df,
        "mesh": mesh,
        "mesh_lines": mesh_lines,
        "correction": correction,
    }
