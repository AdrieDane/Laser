"""MATLAB-engine audit presenters for the standard workflow."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def _download(label: str, table: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label, table.to_csv(index=False).encode(), file_name=filename,
        mime="text/csv", width="stretch",
    )


def _render_isomer_audits(result: dict) -> None:
    isomers = result.get("matlab_positional_isomer_audit", pd.DataFrame())
    if isinstance(isomers, pd.DataFrame) and not isomers.empty:
        paired = int(isomers["status"].eq("paired_rt_order_supported").sum())
        primary_only = int(isomers["status"].astype(str).str.startswith(
            "one_acyl_primary_only_").sum())
        adjustments = pd.to_numeric(
            isomers.get("local_rt_adjustment_seconds",
                        pd.Series(0.0, index=isomers.index)), errors="coerce",
        ).fillna(0.0)
        with st.expander("Positional LPC/LPE isomer resolution", expanded=False):
            st.caption(
                f"{paired} groups were accepted as paired peaks and {primary_only} "
                "groups as a 1-acyl primary without an accepted 2-acyl partner. "
                "LPC selects the strongest credible 1-acyl peak inside the RT "
                "guardrail; 2-acyl is optional and must be a unique, RT-consistent "
                "earlier peak. LPE uses no abundance preference and may retain a "
                "uniquely supported 1-acyl singleton, while 2-acyl still requires "
                f"1-acyl. {int(adjustments.abs().gt(0).sum())} groups used a bounded "
                "neighbouring-grid RT correction. Seed identity and post-grid "
                "retention are listed in the audit."
            )
            st.dataframe(isomers, hide_index=True, width="stretch")
            _download("Download positional-isomer audit CSV", isomers,
                      "LASER_matlab_positional_isomer_audit.csv")
    refit = result.get("matlab_positional_refit_audit", pd.DataFrame())
    if isinstance(refit, pd.DataFrame) and not refit.empty:
        used = int(refit.get("status", pd.Series(dtype=object)).astype(str)
                   .eq("refit_membership").sum())
        if used:
            with st.expander("Experimental positional RT refit memberships",
                             expanded=False):
                st.caption(f"{used} provisional LPC/LPE identities were fed back "
                           "into the second MATLAB RT regression. These are refit "
                           "inputs, not automatically accepted final annotations.")
                st.dataframe(refit, hide_index=True, width="stretch")
                _download("Download positional-refit audit CSV", refit,
                          "LASER_matlab_positional_refit_audit.csv")


def _render_grid_and_standards(result: dict) -> None:
    completion = result.get("matlab_post_regression_audit", pd.DataFrame())
    if isinstance(completion, pd.DataFrame) and not completion.empty:
        promoted = int(pd.to_numeric(
            completion.get("promoted_annotations"), errors="coerce"
        ).fillna(0).sum())
        final = int(pd.to_numeric(
            completion.get("final_annotations"), errors="coerce"
        ).fillna(0).sum())
        with st.expander("Post-regression confidence completion", expanded=False):
            st.caption(
                f"{promoted} fitted-grid candidates were promoted after curated-"
                f"cohort confidence calibration; {final} final annotations remain. "
                "Class-specific carbon/double-bond plausibility helps rank calls "
                "within the existing strict candidate limits. The baseline count "
                "is preserved per stream; any baseline calls retained to fill "
                "that count are listed in the audit."
            )
            st.dataframe(completion, hide_index=True, width="stretch")
            _download(
                "Download post-regression audit CSV",
                completion,
                "LASER_matlab_post_regression_audit.csv",
            )
    mass = result.get("matlab_mass_calibration_audit", pd.DataFrame())
    if isinstance(mass, pd.DataFrame) and not mass.empty:
        applied = int(mass["status"].eq("applied").sum())
        with st.expander("Per-stream m/z calibration", expanded=False):
            st.caption(
                f"{applied} of {len(mass)} available XCMS streams used an "
                "independently reviewed precursor-mass model. Raw m/z remains "
                "available beside calibrated m/z in the feature-level results."
            )
            st.dataframe(mass, hide_index=True, width="stretch")
            _download(
                "Download m/z-calibration audit CSV",
                mass,
                "LASER_matlab_mass_calibration_audit.csv",
            )
        anchors = result.get("matlab_mass_calibration_anchors", pd.DataFrame())
        if isinstance(anchors, pd.DataFrame) and not anchors.empty:
            with st.expander("Confirmed m/z-calibration standards", expanded=False):
                st.dataframe(anchors, hide_index=True, width="stretch")
                _download(
                    "Download m/z-calibration anchors CSV",
                    anchors,
                    "LASER_matlab_mass_calibration_anchors.csv",
                )
    coherence = result.get("matlab_grid_coherence_audit", pd.DataFrame())
    if isinstance(coherence, pd.DataFrame) and not coherence.empty:
        row = coherence.iloc[0]
        enabled = bool(row.get("used_for_annotation", False))
        label = "annotation filter enabled" if enabled else "diagnostic only"
        with st.expander(f"Grid coherence ({label})", expanded=False):
            suffix = (f"used to exclude {int(row.get('filtered_annotations', 0))} "
                      "assessed outlier annotation(s)." if enabled
                      else "not used for annotation or exclusion.")
            st.caption(
                f"Assessed {int(row.get('assessed', 0))} of "
                f"{int(row.get('annotations', 0))} annotations; "
                f"{int(row.get('diagnostic_outliers', 0))} score below "
                f"{float(row.get('threshold', 0.35)):.2f}. This parameter is {suffix}"
            )
            st.dataframe(coherence, hide_index=True, width="stretch")
            _download("Download grid-coherence audit CSV", coherence,
                      "LASER_matlab_grid_coherence_audit.csv")
    standards = result.get("matlab_internal_standard_audit", pd.DataFrame())
    if isinstance(standards, pd.DataFrame) and not standards.empty:
        recovered = int(standards["status"].eq("accepted").sum())
        with st.expander("Internal-standard recovery", expanded=False):
            st.caption(f"{recovered} of {len(standards)} declared standards were "
                       "accepted. Missing standards remain explicit in this audit.")
            st.dataframe(standards, hide_index=True, width="stretch")
            _download("Download internal-standard audit CSV", standards,
                      "LASER_matlab_internal_standard_audit.csv")


def _render_anchor_and_definition_audits(result: dict) -> None:
    seeds = result.get("matlab_seed_audit", pd.DataFrame())
    if isinstance(seeds, pd.DataFrame) and not seeds.empty:
        anchored = int(seeds["anchor_status"].eq("anchored").sum())
        total = int(seeds["anchors_retained"].sum())
        with st.expander("Automatic MATLAB anchor audit", expanded=False):
            st.caption(f"{total} unambiguous XCMS/RTDB anchors across {anchored} "
                       "lipid grids. Grids without enough independent carbon/double-"
                       "bond coverage are left unannotated.")
            st.dataframe(seeds.sort_values(["anchors_retained", "definition"],
                                           ascending=[False, True]),
                         hide_index=True, width="stretch")
            _download("Download automatic-anchor audit CSV", seeds,
                      "LASER_matlab_automatic_anchor_audit.csv")
    audit = result.get("matlab_compound_audit", pd.DataFrame())
    if not isinstance(audit, pd.DataFrame) or audit.empty:
        return
    failed = audit[audit["status"].eq("failed")]
    if not failed.empty:
        st.warning(f"{len(failed)} lipid definition(s) could not be mapped because "
                   "the MATLAB port raised an error on them. The rest of the sweep "
                   "is unaffected; the definition audit below names them.")
    with st.expander("MATLAB lipid definition audit", expanded=False):
        st.caption("Every lipid definition the sweep considered. `skipped` means "
                   "the definition has no charged adduct in that phase or no finite "
                   "project calibration. `failed` means the port raised an error.")
        counts = audit.groupby(["phase", "status"]).size().rename(
            "definitions").reset_index()
        st.dataframe(counts, hide_index=True, width="stretch")
        st.dataframe(audit.sort_values(["phase", "annotations"],
                                      ascending=[True, False]),
                     hide_index=True, width="stretch")
        _download("Download MATLAB definition audit CSV", audit,
                  "LASER_matlab_definition_audit.csv")


def render_matlab_audit(result: dict) -> None:
    """Present each audited MATLAB-engine decision in collapsed sections."""
    _render_isomer_audits(result)
    _render_grid_and_standards(result)
    _render_anchor_and_definition_audits(result)
