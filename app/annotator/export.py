"""Stable tabular exports for programmatic downstream analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from output_io import excel_safe_json_array

from .features import AREA_COLUMN_PREFIX, HEIGHT_COLUMN_PREFIX
from .exclusions import annotation_exclusion_mask
from .msms import msms_rule_result
from .annotated_feature_report import rank_identity_feature_matches


EXPORT_COLUMNS = [
    "feature_id", "feature_identity", "lipid_class", "lipid_subclass",
    "rp_resolved_lipid_class",
    "positional_isomer_selected", "positional_isomer_status",
    "positional_isomer_pair_id", "ether_plasmalogen_selected",
    "ether_plasmalogen_status",
    "adduct", "annotation_status", "automatic_annotation",
    "primary_for_identity", "identity_match_rank",
    "exclusion_reason", "msms_rule_excluded", "grid_coherence_excluded",
    "mzmine_shape_score_excluded",
    "boundary_apex_excluded", "boundary_apex_ratio_assessed",
    "maximum_boundary_apex_ratio",
    "quantification_measure",
    "multiple_identities_for_feature", "identity_count_for_feature",
    "all_matching_identities", "mz_raw", "mz", "reference_mz", "ppm_error",
    "multiple_features_for_identity", "feature_count_for_identity",
    "all_feature_ids_for_identity",
    "mass_calibration_applied", "mass_calibration_standard_features",
    "mass_calibration_model", "mass_calibration_coefficients",
    "mass_calibration_slope", "mass_calibration_intercept",
    "mass_calibration_rmse_before_ppm",
    "mass_calibration_rmse_after_ppm",
    "rt_min", "reference_rt_min", "rt_error_min", "mobility",
    "mobility_unit", "ccs_raw_a2", "ccs_calibrated_a2", "reference_ccs_a2",
    "ccs_error_percent", "annotation_score",
    "grid_coherence_score", "grid_coherence_neighbors",
    "grid_coherence_dimensions", "grid_coherence_assessed",
    "grid_coherence_outlier", "grid_coherence_pass",
    "class_rule_applicable", "class_rule_result",
    "class_rule_pass", "class_rule_status", "class_rule_id",
    "class_rule_description",
    "class_rule_evidence_type", "class_rule_evidence_name",
    "class_rule_target_mz", "class_rule_observed_fragment_mz",
    "class_rule_observed_value", "class_rule_error_da",
    "class_rule_relative_intensity_percent", "has_msms_spectrum",
    "fatty_acyl_rule_id", "fatty_acyl_status",
    "fatty_acyl_min_relative_intensity_percent",
    "detected_fatty_acyl_groups",
    "detected_fatty_acyl_count", "fatty_acyl_evidence_json",
    "fatty_acyl_fragment_intensities",
    "fatty_acyl_to_molecular_species_json",
    "molecular_species_candidates", "molecular_species_candidate_count",
    "molecular_species_ambiguous",
    "molecular_species_ranking_json", "most_likely_molecular_species",
    "most_likely_molecular_species_score",
    "most_likely_molecular_species_ambiguous",
    "annotation_evidence_tier", "annotation_confidence",
    "quantification_quality", "preferred_quantifier", "processing_qc_flags",
    "biological_detected_fraction", "biological_gapfilled_fraction",
    "qc_cv_height_percent", "qc_cv_area_percent", "sample_blank_ratio_height",
    "qc_rsd_internal_standard", "qc_rsd_internal_standard_feature_id",
    "qc_rsd_used_fallback", "qc_rsd_height_percent", "qc_rsd_area_percent",
    "qc_rsd_height_n", "qc_rsd_area_n",
    "is_normalization_status", "is_normalization_selection",
    "is_normalization_valid_height_n", "is_normalization_valid_area_n",
    "rt_shape_class", "rt_shape_class_confidence",
    "mzmine_rt_shape_score", "mzmine_mobility_shape_score",
    "mzmine_shape_score_assessed", "minimum_mzmine_shape_score",
    "mzmine_rt_shape_score_outlier", "mzmine_mobility_shape_score_outlier",
    "mzmine_shape_score_outlier", "mzmine_shape_score_excluded",
    "rt_double_peak_fraction", "median_fwhm_min", "fwhm_cv_percent",
    "median_tailing_factor", "median_asymmetry_factor",
    "median_boundary_apex_ratio", "median_height_area_ratio",
    "area_height_spearman", "median_peak_duration_min",
    "median_duration_fwhm_ratio", "max_consensus_rt_error_min",
    "median_mobility_fwhm", "mobility_fwhm_cv_percent",
    "double_mobility_peak_sample_fraction", "median_ms2_apex_distance_min",
]


def _sample_output_columns(
    sample_columns: list[str] | tuple[str, ...],
) -> list[str]:
    prefixes = (
        "intensity::", "height::", "area::",
        "is_corrected_height::", "is_corrected_area::",
    )
    return [f"{prefix}{sample}" for prefix in prefixes for sample in sample_columns]


def _sample_measurements(
    data: pd.DataFrame,
    sample_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Return raw and IS-corrected measurement channels for stable export."""
    quantification_measure = data.get(
        "feat_quantification_measure",
        pd.Series("", index=data.index, dtype="object"),
    ).fillna("").astype(str)
    columns: dict[str, pd.Series] = {}
    for sample in sample_columns:
        selected = pd.to_numeric(
            data.get(f"feat_{sample}", pd.Series(np.nan, index=data.index)),
            errors="coerce",
        )
        height = pd.to_numeric(
            data.get(
                f"feat_{HEIGHT_COLUMN_PREFIX}{sample}",
                pd.Series(np.nan, index=data.index),
            ),
            errors="coerce",
        )
        area = pd.to_numeric(
            data.get(
                f"feat_{AREA_COLUMN_PREFIX}{sample}",
                pd.Series(np.nan, index=data.index),
            ),
            errors="coerce",
        )
        # Legacy projects may only have the selected signal. Recover it only
        # when its provenance identifies the measure; never fabricate both.
        height = height.where(
            height.notna(), selected.where(quantification_measure.eq("peak_height"))
        )
        area = area.where(
            area.notna(),
            selected.where(quantification_measure.eq("peak_area_fallback")),
        )
        columns[f"intensity::{sample}"] = selected
        columns[f"height::{sample}"] = height
        columns[f"area::{sample}"] = area
        for measure in ("height", "area"):
            output_column = f"is_corrected_{measure}::{sample}"
            columns[output_column] = pd.to_numeric(
                data.get(output_column, pd.Series(np.nan, index=data.index)),
                errors="coerce",
            )
    return pd.DataFrame(columns, index=data.index)


def build_annotated_features_export(
    annotations: pd.DataFrame,
    sample_columns: list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    """Return one row per feature–identity match, including ambiguity and intensities.

    No lipid-species deduplication is performed. If one feature matches several
    database identities, every match remains a separate row and each row carries
    the total match count and complete identity list for that feature.
    """
    sample_output_columns = _sample_output_columns(sample_columns)
    if annotations.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS + sample_output_columns)

    data = rank_identity_feature_matches(annotations)
    excluded = annotation_exclusion_mask(data)
    # Exclusion tables retain an identity rank for audit, but an excluded row
    # must never be advertised as the selected annotation feature.
    data.loc[excluded, "primary_for_identity"] = False
    identities = (
        data["name"].astype(str) + " " + data["adduct"].astype(str)
    )
    data["_identity_with_adduct"] = identities
    feature_groups = data.groupby(data["feat_feature_id"].astype(str), sort=False)
    identity_count = feature_groups["_identity_with_adduct"].transform("nunique")
    all_identities = feature_groups["_identity_with_adduct"].transform(
        lambda values: "; ".join(dict.fromkeys(map(str, values)))
    )
    identity_groups = data.groupby("_identity_with_adduct", sort=False)
    feature_count_for_identity = identity_groups["feat_feature_id"].transform(
        lambda values: values.astype(str).nunique()
    )
    all_feature_ids_for_identity = identity_groups["feat_feature_id"].transform(
        lambda values: "; ".join(dict.fromkeys(values.astype(str)))
    )

    subclass = data.get("grid_class", data["class"]).astype(str)
    rule_id = data.get("msms_rule_id", pd.Series("", index=data.index)).fillna("")
    has_msms = data.get(
        "msms_has_spectrum", pd.Series(False, index=data.index)
    ).fillna(False).astype(bool)
    has_ccs = (
        pd.to_numeric(
            data.get("feat_ccs", pd.Series(np.nan, index=data.index)), errors="coerce"
        ).notna()
        & pd.to_numeric(
            data.get("ref_ccs", pd.Series(np.nan, index=data.index)), errors="coerce"
        ).notna()
    )
    class_supported = data.get(
        "msms_status", pd.Series("not_evaluated", index=data.index)
    ).astype(str).eq("supported")
    fatty_acyl_supported = data.get(
        "fatty_acyl_status", pd.Series("not_applicable", index=data.index)
    ).astype(str).eq("supported")
    evidence_tier = np.select(
        [
            class_supported & has_ccs,
            class_supported,
            has_msms & has_ccs,
            has_msms,
            has_ccs,
        ],
        ["MSMS_CLASS+CCS", "MSMS_CLASS", "MSMS+CCS", "MSMS", "CCS"],
        default="MASS_RT",
    )
    annotation_confidence = np.select(
        [
            (class_supported | fatty_acyl_supported) & has_ccs,
            class_supported | fatty_acyl_supported | (has_msms & has_ccs),
            has_msms | has_ccs,
        ],
        ["HIGH", "SUPPORTED", "MODERATE"],
        default="PROVISIONAL",
    )
    output = pd.DataFrame({
        "feature_id": data["feat_feature_id"].astype(str),
        "feature_identity": data["name"].astype(str),
        "lipid_class": data["class"].astype(str),
        "lipid_subclass": subclass,
        "rp_resolved_lipid_class": subclass,
        "positional_isomer_selected": data.get(
            "positional_isomer_selected", pd.Series(False, index=data.index),
        ).fillna(False).astype(bool),
        "positional_isomer_status": data.get(
            "positional_isomer_status",
            pd.Series("not_applicable", index=data.index),
        ),
        "positional_isomer_pair_id": data.get(
            "positional_isomer_pair_id", pd.Series("", index=data.index),
        ),
        "ether_plasmalogen_selected": data.get(
            "ether_plasmalogen_selected", pd.Series(False, index=data.index),
        ).fillna(False).astype(bool),
        "ether_plasmalogen_status": data.get(
            "ether_plasmalogen_status",
            pd.Series("not_applicable", index=data.index),
        ),
        "adduct": data["adduct"].astype(str),
        "annotation_status": data.get("status", pd.Series("identified", index=data.index)),
        "automatic_annotation": data.get(
            "automatic_candidate", pd.Series(True, index=data.index)
        ).astype(bool),
        "primary_for_identity": data["primary_for_identity"].astype(bool),
        "identity_match_rank": data["identity_match_rank"].astype("Int64"),
        "exclusion_reason": data.get(
            "exclusion_reason", pd.Series("", index=data.index)
        ),
        "msms_rule_excluded": data.get(
            "rule_excluded", pd.Series(False, index=data.index)
        ).fillna(False).astype(bool),
        "grid_coherence_excluded": data.get(
            "grid_excluded", pd.Series(False, index=data.index)
        ).fillna(False).astype(bool),
        "mzmine_shape_score_excluded": data.get(
            "mzmine_shape_score_excluded", pd.Series(False, index=data.index)
        ).fillna(False).astype(bool),
        "boundary_apex_excluded": data.get(
            "boundary_apex_excluded", pd.Series(False, index=data.index)
        ).fillna(False).astype(bool),
        "boundary_apex_ratio_assessed": data.get(
            "boundary_apex_ratio_assessed", pd.Series(False, index=data.index)
        ).fillna(False).astype(bool),
        "maximum_boundary_apex_ratio": data.get(
            "maximum_boundary_apex_ratio"
        ),
        "quantification_measure": data.get(
            "feat_quantification_measure",
            pd.Series("unspecified_intensity", index=data.index),
        ),
        "multiple_identities_for_feature": identity_count.gt(1),
        "identity_count_for_feature": identity_count.astype(int),
        "all_matching_identities": all_identities,
        "multiple_features_for_identity": feature_count_for_identity.gt(1),
        "feature_count_for_identity": feature_count_for_identity.astype(int),
        "all_feature_ids_for_identity": all_feature_ids_for_identity,
        "mz_raw": data.get("feat_mz_raw", data.get("feat_mz")),
        "mz": data.get("feat_mz"),
        "reference_mz": data.get("ref_mz"),
        "ppm_error": data.get("ppm_error"),
        "mass_calibration_applied": data.get(
            "mass_calibration_standard_features",
            pd.Series("", index=data.index),
        ).fillna("").astype(str).str.len().gt(0),
        "mass_calibration_standard_features": data.get(
            "mass_calibration_standard_features"
        ),
        "mass_calibration_model": data.get("mass_calibration_model"),
        "mass_calibration_coefficients": data.get(
            "mass_calibration_coefficients"
        ),
        "mass_calibration_slope": data.get("mass_calibration_slope"),
        "mass_calibration_intercept": data.get(
            "mass_calibration_intercept"
        ),
        "mass_calibration_rmse_before_ppm": data.get(
            "mass_calibration_rmse_before_ppm"
        ),
        "mass_calibration_rmse_after_ppm": data.get(
            "mass_calibration_rmse_after_ppm"
        ),
        "rt_min": data.get("feat_rt"),
        "reference_rt_min": data.get("ref_rt"),
        "rt_error_min": data.get("rt_error"),
        "mobility": data.get("feat_mobility"),
        "mobility_unit": data.get("feat_mobility_unit"),
        "ccs_raw_a2": data.get("feat_ccs_raw"),
        "ccs_calibrated_a2": data.get("feat_ccs"),
        "reference_ccs_a2": data.get("ref_ccs"),
        "ccs_error_percent": data.get("ccs_error_percent"),
        "annotation_score": data.get("score"),
        "grid_coherence_score": data.get("grid_coherence_score"),
        "grid_coherence_neighbors": data.get("grid_coherence_neighbors"),
        "grid_coherence_dimensions": data.get("grid_coherence_dimensions"),
        "grid_coherence_assessed": data.get(
            "grid_coherence_assessed", pd.Series(False, index=data.index)
        ),
        "grid_coherence_outlier": data.get(
            "grid_coherence_outlier", pd.Series(False, index=data.index)
        ),
        "grid_coherence_pass": data.get(
            "grid_coherence_pass", pd.Series(True, index=data.index)
        ),
        "class_rule_applicable": rule_id.astype(str).str.len().gt(0),
        "class_rule_result": data.apply(msms_rule_result, axis=1),
        "class_rule_pass": data.get("msms_rule_pass", pd.Series(pd.NA, index=data.index)),
        "class_rule_status": data.get("msms_status", pd.Series("not_evaluated", index=data.index)),
        "class_rule_id": rule_id,
        "class_rule_description": data.get("msms_rule_description"),
        "class_rule_evidence_type": data.get("msms_rule_evidence_type"),
        "class_rule_evidence_name": data.get("msms_rule_evidence_name"),
        "class_rule_target_mz": data.get("msms_rule_target_mz"),
        "class_rule_observed_fragment_mz": data.get("msms_rule_observed_mz"),
        "class_rule_observed_value": data.get("msms_rule_observed_value"),
        "class_rule_error_da": data.get("msms_rule_error_da"),
        "class_rule_relative_intensity_percent": data.get("msms_rule_relative_intensity"),
        "has_msms_spectrum": data.get("msms_has_spectrum", False),
        "fatty_acyl_rule_id": data.get("fatty_acyl_rule_id"),
        "fatty_acyl_status": data.get("fatty_acyl_status"),
        "fatty_acyl_min_relative_intensity_percent": data.get(
            "fatty_acyl_min_relative_intensity_percent"
        ),
        "detected_fatty_acyl_groups": data.get("fatty_acyl_detected"),
        "detected_fatty_acyl_count": data.get("fatty_acyl_detected_count"),
        "fatty_acyl_evidence_json": data.get("fatty_acyl_evidence_json"),
        "fatty_acyl_fragment_intensities": data.get(
            "fatty_acyl_fragment_intensities"
        ),
        "fatty_acyl_to_molecular_species_json": data.get(
            "fatty_acyl_species_mapping_json"
        ),
        "molecular_species_candidates": data.get("molecular_species_candidates"),
        "molecular_species_candidate_count": data.get(
            "molecular_species_candidate_count"
        ),
        "molecular_species_ambiguous": data.get("molecular_species_ambiguous"),
        "molecular_species_ranking_json": data.get(
            "molecular_species_ranking_json"
        ),
        "most_likely_molecular_species": data.get(
            "most_likely_molecular_species"
        ),
        "most_likely_molecular_species_score": data.get(
            "most_likely_molecular_species_score"
        ),
        "most_likely_molecular_species_ambiguous": data.get(
            "most_likely_molecular_species_ambiguous"
        ),
        "annotation_evidence_tier": evidence_tier,
        "annotation_confidence": annotation_confidence,
        "quantification_quality": data.get("quantification_quality"),
        "preferred_quantifier": data.get("preferred_quantifier"),
        "processing_qc_flags": data.get("processing_qc_flags"),
        "biological_detected_fraction": data.get("biological_detected_fraction"),
        "biological_gapfilled_fraction": data.get("biological_gapfilled_fraction"),
        "qc_cv_height_percent": data.get("qc_cv_height_percent"),
        "qc_cv_area_percent": data.get("qc_cv_area_percent"),
        "sample_blank_ratio_height": data.get("sample_blank_ratio_height"),
        "qc_rsd_internal_standard": data.get("qc_rsd_internal_standard"),
        "qc_rsd_internal_standard_feature_id": data.get(
            "qc_rsd_internal_standard_feature_id"
        ),
        "qc_rsd_used_fallback": data.get("qc_rsd_used_fallback"),
        "qc_rsd_height_percent": data.get("qc_rsd_height_percent"),
        "qc_rsd_area_percent": data.get("qc_rsd_area_percent"),
        "qc_rsd_height_n": data.get("qc_rsd_height_n"),
        "qc_rsd_area_n": data.get("qc_rsd_area_n"),
        "is_normalization_status": data.get("is_normalization_status"),
        "is_normalization_selection": data.get("is_normalization_selection"),
        "is_normalization_valid_height_n": data.get(
            "is_normalization_valid_height_n"
        ),
        "is_normalization_valid_area_n": data.get(
            "is_normalization_valid_area_n"
        ),
        "rt_shape_class": data.get("rt_shape_class"),
        "rt_shape_class_confidence": data.get("rt_shape_class_confidence"),
        "mzmine_rt_shape_score": data.get("mzmine_rt_shape_score"),
        "mzmine_mobility_shape_score": data.get("mzmine_mobility_shape_score"),
        "mzmine_shape_score_assessed": data.get("mzmine_shape_score_assessed"),
        "minimum_mzmine_shape_score": data.get("minimum_mzmine_shape_score"),
        "mzmine_rt_shape_score_outlier": data.get("mzmine_rt_shape_score_outlier"),
        "mzmine_mobility_shape_score_outlier": data.get("mzmine_mobility_shape_score_outlier"),
        "mzmine_shape_score_outlier": data.get("mzmine_shape_score_outlier"),
        "rt_double_peak_fraction": data.get("rt_double_peak_fraction"),
        "median_fwhm_min": data.get("median_fwhm_min"),
        "fwhm_cv_percent": data.get("fwhm_cv_percent"),
        "median_tailing_factor": data.get("median_tailing_factor"),
        "median_asymmetry_factor": data.get("median_asymmetry_factor"),
        "median_boundary_apex_ratio": data.get("median_boundary_apex_ratio"),
        "median_height_area_ratio": data.get("median_height_area_ratio"),
        "area_height_spearman": data.get("area_height_spearman"),
        "median_peak_duration_min": data.get("median_peak_duration_min"),
        "median_duration_fwhm_ratio": data.get(
            "median_duration_fwhm_ratio"
        ),
        "max_consensus_rt_error_min": data.get("max_consensus_rt_error_min"),
        "median_mobility_fwhm": data.get("median_mobility_fwhm"),
        "mobility_fwhm_cv_percent": data.get("mobility_fwhm_cv_percent"),
        "double_mobility_peak_sample_fraction": data.get(
            "double_mobility_peak_sample_fraction"
        ),
        "median_ms2_apex_distance_min": data.get(
            "median_ms2_apex_distance_min"
        ),
    })
    sample_measurements = _sample_measurements(data, sample_columns)
    if not sample_measurements.empty:
        output = pd.concat([output, sample_measurements], axis=1)
    # Excel can render CSV rows as broken across several worksheet rows when a
    # TG molecular-species ranking exceeds its 32,767-character cell limit.
    # Keep the highest-ranked valid JSON prefix below that limit. The complete
    # candidate names remain available in molecular_species_candidates.
    output["molecular_species_ranking_json"] = output[
        "molecular_species_ranking_json"
    ].map(excel_safe_json_array)
    return output.sort_values(
        ["feature_id", "feature_identity", "adduct"], kind="stable"
    ).reset_index(drop=True)
