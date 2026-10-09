"""One-row-per-match annotated-feature evidence report."""

import base64
import gzip
import re
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from annotator.annotated_feature_report import (
    _class_bar_chart,
    assemble_feature_evidence_candidates,
    build_lipid_class_map_data,
    rank_identity_feature_matches,
    render_annotation_summary_charts_report,
    render_annotated_feature_report,
    render_injection_order_report,
    render_kendrick_map_report,
    render_rt_mz_map_report,
    render_tic_report,
    write_annotation_summary_charts_report,
    write_annotated_feature_report,
)
from annotator.trace_gallery import SampleTrace
from annotator.feature_traces import SummedFeatureChromatogram


def _annotations():
    return pd.DataFrame({
        "feat_feature_id": ["11", "12", "21"],
        "name": ["PC(34:1)", "PC(34:1)", "PE(36:2)"],
        "class": ["PC", "PC", "PE"],
        "adduct": ["[M+H]+", "[M+H]+", "[M+H]+"],
        "status": ["accept", "accept", "accept"],
        "automatic_candidate": [True, True, True],
        "score": [0.18, 0.73, 0.24],
        "feat_mz": [760.5851, 760.5853, 744.5538],
        "feat_rt": [10.0, 10.3, 9.4],
        "rt_expected": [10.02, 10.02, 9.45],
        "rt_error": [-0.02, 0.28, -0.05],
        "ppm_error": [0.4, 0.8, -1.1],
        "grid_coherence_score": [0.81, 0.51, 0.73],
        "ccs_error_percent": [0.5, 1.4, -0.8],
        "feat_mobility": [1.24, 1.25, 1.31],
        "msms_status": ["supported", "no_msms", "supported"],
        "msms_has_spectrum": [True, False, True],
        "msms_rule_id": ["PC_184", "PC_184", "PE_196"],
        "msms_rule_pass": [True, pd.NA, True],
        "msms_rule_evidence_type": ["product_ion"] * 3,
        "msms_rule_evidence_name": [
            "phosphocholine", "phosphocholine", "PE headgroup",
        ],
        "msms_rule_observed_mz": [184.0733, np.nan, 196.0],
        "msms_rule_relative_intensity": [80.0, np.nan, 50.0],
    })


def _sample_trace(name="S1.d"):
    rt = np.linspace(9.5, 10.5, 18)
    mobility = np.linspace(1.18, 1.30, 14)
    return SampleTrace(
        sample_name=name,
        is_representative=True,
        rt_values=rt,
        rt_intensities=np.exp(-((rt - 10.0) ** 2) / .02) * 1e5,
        mobility_values=mobility,
        mobility_intensities=np.exp(-((mobility - 1.24) ** 2) / .0004) * 1e5,
    )


class AnnotatedFeatureReportTests(unittest.TestCase):

    def test_lipid_class_map_data_is_retained_unique_and_uses_ch2_kendrick_scale(self):
        frame = _annotations().copy()
        duplicate = frame.iloc[[0]].copy()
        rejected = frame.iloc[[2]].copy()
        rejected["feat_feature_id"] = "99"
        rejected["status"] = "reject"
        frame = pd.concat([frame, duplicate, rejected], ignore_index=True)

        points = build_lipid_class_map_data(frame)

        self.assertEqual(len(points), 2)
        self.assertEqual(set(points["feat_feature_id"]), {"11", "21"})
        point = points[points["feat_feature_id"].eq("11")].iloc[0]
        expected_kendrick = 760.5851 * 14.0 / 14.01565
        self.assertAlmostEqual(point["kendrick_mass"], expected_kendrick)
        self.assertEqual(point["nominal_kendrick_mass"], round(expected_kendrick))
        self.assertAlmostEqual(
            point["kendrick_mass_defect"],
            round(expected_kendrick) - expected_kendrick,
        )

    def test_lipid_maps_use_the_feature_laser_selected_for_each_identification(self):
        frame = _annotations().head(2).copy()
        # Feature 11 has stronger generic evidence, but LASER explicitly chose
        # feature 12. The report must reproduce that choice rather than re-rank.
        frame["primary_for_identity"] = [False, True]

        points = build_lipid_class_map_data(frame)

        self.assertEqual(points["feat_feature_id"].tolist(), ["12"])

    def test_lipid_maps_do_not_invent_a_cross_adduct_winner(self):
        frame = _annotations().iloc[[0]].copy()
        second_adduct = frame.copy()
        second_adduct["feat_feature_id"] = "13"
        second_adduct["adduct"] = "[M+Na]+"
        frame = pd.concat([frame, second_adduct], ignore_index=True)
        frame["primary_for_identity"] = True

        points = build_lipid_class_map_data(frame)

        self.assertEqual(set(points["feat_feature_id"]), {"11", "13"})

    def test_rt_mz_report_has_one_panel_per_class_and_hover_evidence(self):
        page = render_rt_mz_map_report(_annotations())

        self.assertEqual(page.count("class='lipid-map-card'"), 2)
        self.assertIn("retention time (min)", page)
        self.assertIn(">m/z<", page)
        self.assertIn("PC(34:1) [M+H]+ · feature #11", page)
        self.assertIn("linked MS/MS spectrum", page)
        self.assertIn("final retained feature LASER selected", page)
        self.assertNotIn("http://", page)

    def test_kendrick_report_states_formula_and_uses_standard_axes(self):
        page = render_kendrick_map_report(_annotations())

        self.assertEqual(page.count("class='lipid-map-card'"), 2)
        self.assertIn("m/z × 14 / 14.01565", page)
        self.assertIn("nominal Kendrick mass (CH₂)", page)
        self.assertIn("Kendrick mass defect (CH₂)", page)
        self.assertIn("ion/adduct-specific", page)
        self.assertIn("final retained feature LASER selected", page)

    def test_unique_species_chart_omits_all_explicitly_excluded_rows(self):
        frame = rank_identity_feature_matches(_annotations())
        frame["boundary_apex_excluded"] = [False, True, True]

        chart = _class_bar_chart(frame)

        self.assertIn("PC: 1 unique lipid IDs", chart)
        self.assertNotIn("PE: 1 unique lipid IDs", chart)

    def test_excluded_rivals_are_kept_only_for_a_surviving_identity(self):
        retained = _annotations().iloc[[0, 2]].copy()
        failures = pd.DataFrame({
            **{column: [value, value] for column, value in retained.iloc[0].items()},
            "feat_feature_id": ["12", "99"],
            "name": ["PC(34:1)", "TG(60:5)"],
            "class": ["PC", "TG"],
        })

        combined = assemble_feature_evidence_candidates(
            retained, msms_failures=failures
        )

        self.assertEqual(set(combined["feat_feature_id"]), {"11", "12", "21"})
        excluded = combined[combined["feat_feature_id"].eq("12")].iloc[0]
        self.assertTrue(excluded["rule_excluded"])
        self.assertFalse(excluded["automatic_candidate"])

    def test_an_excluded_rival_cannot_become_primary(self):
        frame = _annotations().head(2).copy()
        frame["rule_excluded"] = [False, True]
        frame["msms_status"] = ["no_msms", "supported"]
        frame["rt_error"] = [0.2, 0.01]

        ranked = rank_identity_feature_matches(frame)

        self.assertEqual(
            ranked.loc[ranked["primary_for_identity"], "feat_feature_id"].item(),
            "11",
        )

    def test_every_competing_feature_is_kept_and_primary_is_best_score(self):
        ranked = rank_identity_feature_matches(_annotations())
        pc = ranked[ranked["name"].eq("PC(34:1)")]

        self.assertEqual(set(pc["feat_feature_id"]), {"11", "12"})
        self.assertEqual(
            pc.loc[pc["primary_for_identity"], "feat_feature_id"].tolist(), ["11"]
        )
        self.assertEqual(set(pc["identity_match_rank"]), {1, 2})

    def test_an_explicit_primary_override_beats_a_better_automatic_match(self):
        frame = _annotations().head(2).copy()
        frame["status"] = ["accept", "reject"]
        frame["score"] = [0.8, 0.1]
        frame["rt_error"] = [0.3, 0.01]
        frame["primary_feature_override"] = [True, False]

        ranked = rank_identity_feature_matches(frame)

        self.assertEqual(
            ranked.loc[ranked["primary_for_identity"], "feat_feature_id"].item(),
            "11",
        )

    def test_confirming_msms_beats_a_feature_closer_to_the_rt_grid(self):
        frame = _annotations().head(2).copy()
        frame["msms_status"] = ["supported", "no_msms"]
        frame["rt_error"] = [0.20, 0.01]
        frame["ppm_error"] = [2.0, 0.1]

        ranked = rank_identity_feature_matches(frame)

        self.assertEqual(
            ranked.loc[ranked["primary_for_identity"], "feat_feature_id"].item(),
            "11",
        )

    def test_report_contains_one_row_per_match_all_plots_and_metrics(self):
        traces = {feature: [_sample_trace()] for feature in ("11", "12", "21")}
        spectra = {
            "11": {"precursor_mz": 760.5851, "peaks": [[184.0733, 1000], [86.1, 80]]},
            "21": {"precursor_mz": 744.5538, "peaks": [[196.0, 500]]},
        }

        page = render_annotated_feature_report(
            _annotations(), traces=traces, spectra=spectra
        )

        self.assertEqual(page.count("class='feature-row "), 3)
        self.assertEqual(
            page.count("class='badge primary'>SELECTED FOR LIPID ID<"), 2
        )
        self.assertIn("NOT SELECTED · ACCEPTED ALTERNATIVE", page)
        self.assertIn("mass error", page)
        self.assertIn("ΔRT", page)
        self.assertIn("grid coherence", page)
        self.assertIn("ΔCCS", page)
        self.assertIn("Ion-mobility trace", page)
        self.assertIn("MS/MS spectrum", page)
        self.assertIn("no linked MS/MS spectrum", page)
        self.assertIn("MS/MS RULE: PASS", page)
        self.assertIn("MS/MS RULE: NO MS/MS SPECTRUM", page)
        self.assertIn("PC_184 · phosphocholine", page)
        self.assertGreaterEqual(page.count("<svg"), 9)

    def test_nullable_exclusion_flags_do_not_hide_selected_feature(self):
        frame = _annotations().head(2).copy()
        frame["rule_excluded"] = np.nan
        frame["grid_excluded"] = np.nan
        frame["boundary_apex_excluded"] = np.nan

        page = render_annotated_feature_report(frame)

        self.assertEqual(
            page.count("class='badge primary'>SELECTED FOR LIPID ID<"), 1
        )
        self.assertIn("NOT SELECTED · ACCEPTED ALTERNATIVE", page)
        self.assertNotIn("MS/MS RULE EXCLUDED", page)
        self.assertNotIn("GRID OUTLIER", page)
        self.assertNotIn("BOUNDARY/APEX EXCLUDED", page)

    def test_report_exposes_fatty_acyl_assignments_and_marks_supporting_peaks(self):
        frame = _annotations().head(1).copy()
        frame["fatty_acyl_status"] = "supported"
        frame["fatty_acyl_detected"] = "FA(16:0); FA(18:1)"
        frame["molecular_species_candidates"] = "PC(16:0_18:1)"
        frame["molecular_species_ambiguous"] = False
        frame["fatty_acyl_evidence_json"] = json.dumps([{
            "fatty_acyl": "FA(16:0)",
            "evidence_type": "fatty_acid_anion",
            "expected_mz_or_loss": 255.2330,
            "observed_fragment_mz": 255.2328,
            "error_da": -0.0002,
            "relative_intensity_percent": 42.5,
        }])
        spectra = {"11": {
            "precursor_mz": 804.5760,
            "peaks": [[255.2328, 425.0], [184.0733, 1000.0]],
        }}

        page = render_annotated_feature_report(frame, spectra=spectra)

        self.assertIn("FA FRAGMENT EVIDENCE: SUPPORTED", page)
        self.assertIn("FA(16:0); FA(18:1)", page)
        self.assertIn("PC(16:0_18:1)", page)
        self.assertIn("no sn-position claim", page)
        self.assertIn("fatty-acyl-fragment", page)
        self.assertIn("FA-supporting fragment · m/z 255.2328", page)

    def test_peak_quality_metrics_are_a_full_width_row_below_feature_evidence(self):
        frame = _annotations().head(1).copy()
        frame["quantification_quality"] = "REVIEW"
        frame["processing_qc_flags"] = "rt_double_peak_recurrent;integration_boundary_high"
        frame["preferred_quantifier"] = "area"
        frame["median_fwhm_min"] = .08
        frame["fwhm_cv_percent"] = 22.0
        frame["median_tailing_factor"] = 1.2
        frame["rt_double_peak_fraction"] = .6
        frame["biological_detected_fraction"] = .8
        frame["sample_blank_ratio_height"] = 25.0
        frame["qc_rsd_height_percent"] = 12.5
        frame["qc_rsd_area_percent"] = 9.5
        frame["median_boundary_apex_ratio"] = .05
        frame["area_height_spearman"] = .94
        frame["median_peak_duration_min"] = .16
        frame["median_duration_fwhm_ratio"] = 2.0
        frame["max_consensus_rt_error_min"] = .025
        frame["median_mobility_fwhm"] = .0123
        frame["mobility_fwhm_cv_percent"] = 18.5
        frame["double_mobility_peak_sample_fraction"] = .2
        frame["median_ms2_apex_distance_min"] = .05
        frame["median_asymmetry_factor"] = 1.1

        page = render_annotated_feature_report(frame)

        self.assertEqual(page.count("class='peak-quality-strip'"), 1)
        self.assertEqual(page.count("class='identity-quality'"), 1)
        self.assertIn("<span class='quality-badge review'>REVIEW</span>", page)
        self.assertIn("rt_double_peak_recurrent · integration_boundary_high", page)
        self.assertIn("Median FWHM</span><b>4.8 s", page)
        self.assertIn("FWHM CV</span><b>22.0%", page)
        self.assertIn("Tailing factor</span><b>1.20", page)
        self.assertIn("Double-peak fraction</span><b>60.0%", page)
        self.assertIn("Detection rate</span><b>80.0%", page)
        self.assertIn("Sample/blank</span><b>25.0×", page)
        self.assertIn("IS-corrected QC RSD · area</span><b>+9.5%", page)
        self.assertIn("Boundary/apex ratio</span><b>0.050", page)
        self.assertNotIn("Peak height/area", page)
        self.assertIn("Height–area correlation</span><b>0.94", page)
        self.assertIn("Peak duration</span><b>9.6 s", page)
        self.assertIn("Duration/FWHM</span><b>2.00", page)
        self.assertIn("Maximum RT deviation</span><b>1.5 s", page)
        self.assertIn("Mobility FWHM</span><b>0.0123", page)
        self.assertIn("Mobility FWHM CV</span><b>18.5%", page)
        self.assertIn("Mobility double-peak fraction</span><b>20.0%", page)
        self.assertIn("MS/MS apex distance</span><b>3.0 s", page)
        self.assertIn("Asymmetry factor</span><b>1.10", page)
        self.assertEqual(page.count("class='peak-quality-metric'"), 16)
        self.assertEqual(page.count("class='plot-panel'"), 3)
        self.assertIn("position:absolute", page)

    def test_report_distinguishes_rule_failure_from_missing_spectrum(self):
        frame = _annotations().head(2).copy()
        frame.loc[0, "msms_status"] = "not_detected"
        frame.loc[0, "msms_rule_pass"] = False
        frame.loc[0, "msms_rule_target_mz"] = 184.0733
        frame.loc[0, "rule_excluded"] = True

        page = render_annotated_feature_report(frame)

        self.assertIn("MS/MS RULE: FAIL", page)
        self.assertIn("fragment m/z 184.0733 not detected", page)
        self.assertIn("candidate excluded by the active MS/MS rule filter", page)
        self.assertIn("MS/MS RULE: NO MS/MS SPECTRUM", page)

    def test_feature_rows_only_show_sample_and_qc_traces_when_roles_exist(self):
        traces = {
            "11": [
                _sample_trace("sample.d"), _sample_trace("QC.d"),
                _sample_trace("blank.d"), _sample_trace("IS.d"),
            ]
        }
        page = render_annotated_feature_report(
            _annotations().head(1), traces=traces,
            sample_roles={
                "sample.d": "biological", "QC.d": "qc",
                "blank.d": "blank", "IS.d": "calibration",
            },
        )

        self.assertIn("2 sample trace(s)", page)
        self.assertNotIn("4 sample trace(s)", page)

    def test_class_bar_counts_identities_not_rival_features(self):
        frame = _annotations()
        second_adduct = frame.iloc[[0]].copy()
        second_adduct["feat_feature_id"] = "13"
        second_adduct["adduct"] = "[M+Na]+"
        page = render_annotation_summary_charts_report(
            pd.concat([frame, second_adduct], ignore_index=True)
        )

        # PC has rival features and two adducts but only one unique lipid ID.
        before_section = page.split(
            "Unique annotations before exclusions", 1
        )[1].split("Unique annotations after all exclusions", 1)[0]
        retained_section = page.split(
            "Unique annotations after all exclusions", 1
        )[1].split("IS-corrected QC-RSD distributions", 1)[0]
        self.assertEqual(before_section.count('class="class-bar"'), 2)
        self.assertEqual(retained_section.count('class="class-bar"'), 2)
        self.assertNotRegex(page, re.compile(r'class="bar-count"[^>]*>2</text>'))
        self.assertIn("PC: 1 unique lipid IDs", page)
        self.assertIn("with MS/MS spectrum", page)
        self.assertIn("1 of 1 unique lipid IDs have an MS/MS spectrum", page)
        self.assertEqual(before_section.count('class="bar-msms-count"'), 2)
        self.assertEqual(retained_section.count('class="bar-msms-count"'), 2)

    def test_msms_exclusion_report_states_rules_and_lists_excluded_rows(self):
        frame = _annotations().iloc[[0]].copy()
        frame["msms_status"] = "not_detected"
        frame["msms_rule_pass"] = False
        frame["msms_rule_id"] = "POS_PHOSPHOCHOLINE_HEADGROUP_184"
        frame["rule_excluded"] = True
        frame["automatic_candidate"] = False
        frame["status"] = "reject"

        page = render_annotated_feature_report(
            frame, report_kind="msms_exclusions", polarity="pos"
        )

        self.assertIn("Enforced class-fragment MS/MS rules", page)
        self.assertIn("POS_PHOSPHOCHOLINE_HEADGROUP_184", page)
        self.assertIn("MS/MS-excluded lipid IDs per class", page)
        self.assertIn("NOT SELECTED · MS/MS RULE EXCLUDED", page)
        self.assertIn("rank among excluded", page)
        self.assertNotIn("SELECTED FOR LIPID ID", page)
        self.assertIn("PC(34:1)", page)

    def test_msms_exclusion_report_flags_no_rule_records_as_inconsistent(self):
        frame = _annotations().iloc[[0]].copy()
        frame["msms_status"] = "not_evaluated"
        frame["msms_rule_pass"] = pd.NA
        frame["msms_rule_id"] = ""
        frame["rule_excluded"] = True
        frame["automatic_candidate"] = False
        frame["status"] = "reject"

        page = render_annotated_feature_report(
            frame, report_kind="msms_exclusions", polarity="neg"
        )

        self.assertIn("Exclusion-record warning", page)
        self.assertIn("NOT SELECTED · INVALID MS/MS EXCLUSION", page)
        self.assertIn("MS/MS RULE: NO RULE CONFIGURED", page)
        self.assertNotIn("SELECTED FOR LIPID ID", page)

    def test_grid_exclusion_report_states_threshold_and_lists_rows(self):
        frame = _annotations().iloc[[2]].copy()
        frame["grid_coherence_score"] = 0.31
        frame["grid_excluded"] = True
        frame["automatic_candidate"] = False
        frame["status"] = "reject"

        page = render_annotated_feature_report(
            frame,
            report_kind="grid_exclusions",
            grid_coherence_threshold=0.45,
        )

        self.assertIn("score below <b>0.45</b>", page)
        self.assertIn("Grid-excluded lipid IDs per class", page)
        self.assertIn("GRID OUTLIER", page)
        self.assertIn("PE(36:2)", page)

    def test_boundary_apex_exclusion_report_states_threshold_and_lists_rows(self):
        frame = _annotations().iloc[[2]].copy()
        frame["median_boundary_apex_ratio"] = 0.082
        frame["boundary_apex_excluded"] = True
        frame["automatic_candidate"] = False
        frame["status"] = "reject"

        page = render_annotated_feature_report(
            frame,
            report_kind="boundary_apex_exclusions",
            maximum_boundary_apex_ratio=0.075,
        )

        self.assertIn("above <b>0.075</b>", page)
        self.assertIn("Boundary-excluded lipid IDs per class", page)
        self.assertIn("BOUNDARY/APEX EXCLUDED", page)
        self.assertIn("PE(36:2)", page)

    def test_run_qc_has_dedicated_tic_and_injection_order_reports(self):
        frame = _annotations().copy()
        frame["qc_rsd_height_percent"] = [10.0, 25.0, 35.0]
        frame["qc_rsd_area_percent"] = [12.0, 28.0, 32.0]
        trend_values = pd.DataFrame({
            "sample": ["QC1.d", "QC2.d", "IS1.d"],
            "sample_role": ["qc", "qc", "calibration"],
            "injection_order": [1, 2, 3],
            "state": ["DETECTED"] * 3,
            "rt": [7.0, 7.01, 7.0],
            "height": [100, 95, 110],
            "area": [400, 390, 430],
        })
        tics = [
            SummedFeatureChromatogram(
                sample_name=sample,
                rt_values=np.array([1.0, 2.0, 3.0]),
                intensities=np.array([0.0, intensity, 0.0]),
            )
            for sample, intensity in (("QC1.d", 100), ("QC2.d", 90), ("IS1.d", 110))
        ]

        page = render_annotated_feature_report(
            frame,
            sample_roles={"QC1.d": "qc", "QC2.d": "qc", "IS1.d": "calibration"},
            summed_chromatograms=tics,
            is_trends=[{
                "name": "PC(28:0) (IS)", "class": "PC", "adduct": "[M+H]+",
                "feature_id": "99", "measurements": trend_values,
            }],
        )

        self.assertNotIn("TIC · all measurements", page)
        self.assertNotIn("PC(28:0) (IS) [M+H]+ · feature #99", page)
        self.assertNotIn("All annotated features", page)
        self.assertNotIn("Distinct primary lipid annotations per class", page)

        tic_page = render_tic_report(
            tics,
            sample_roles={"QC1.d": "qc", "QC2.d": "qc", "IS1.d": "calibration"},
        )
        self.assertIn("TIC · all measurements", tic_page)
        self.assertIn("TIC · all QC", tic_page)
        self.assertIn("TIC · all IS", tic_page)

        trend_page = render_injection_order_report([{
            "name": "PC(28:0) (IS)", "class": "PC", "adduct": "[M+H]+",
            "feature_id": "99", "measurements": trend_values,
        }])
        self.assertIn("PC(28:0) (IS) [M+H]+ · feature #99", trend_page)
        self.assertIn("retention time (min)", trend_page)
        self.assertIn("peak height", trend_page)
        self.assertIn("peak area", trend_page)

        summary = render_annotation_summary_charts_report(frame)
        self.assertIn("Unique annotations before exclusions", summary)
        self.assertIn("Unique annotations after all exclusions", summary)
        self.assertIn("Total unique annotated features after all exclusions", summary)
        self.assertIn("All annotated features", summary)
        self.assertIn("<h3>PC</h3>", summary)
        self.assertIn("<h3>PE</h3>", summary)
        self.assertNotIn("TIC · all measurements", summary)

    def test_internal_standards_are_grouped_before_endogenous_lipids(self):
        frame = _annotations().copy()
        internal_standard = frame.iloc[[2]].copy()
        internal_standard["feat_feature_id"] = "31"
        internal_standard["name"] = "PC(15:0)(IS)"
        internal_standard["class"] = "PC"
        internal_standard["is_is"] = True
        frame["is_is"] = False
        frame = pd.concat([frame, internal_standard], ignore_index=True)

        page = render_annotated_feature_report(frame)

        self.assertIn("Annotated internal standards", page)
        self.assertIn("Annotated endogenous lipids", page)
        self.assertLess(page.index("PC(15:0)(IS)"), page.index("PC(34:1)"))
        self.assertLess(
            page.index("Annotated internal standards"),
            page.index("Annotated endogenous lipids"),
        )

    def test_legacy_is_name_marker_is_sorted_first_without_is_is_column(self):
        frame = _annotations().copy()
        frame.loc[2, "name"] = "PE(17:0)(IS)"

        page = render_annotated_feature_report(frame)

        self.assertLess(page.index("PE(17:0)(IS)"), page.index("PC(34:1)"))

    def test_writer_produces_a_self_contained_html_file(self):
        output = Path(tempfile.mkdtemp()) / "features.html"

        result = write_annotated_feature_report(_annotations(), output)

        self.assertEqual(result, output)
        page = output.read_text(encoding="utf-8")
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)
        self.assertNotIn("Distinct primary lipid annotations per class", page)
        self.assertIn("Annotated endogenous lipids", page)
        self.assertIn("data-laser-lazy-payload", page)
        self.assertNotIn("class='feature-row ", page)
        payload_text = re.search(
            r"data-laser-lazy-payload>([^<]+)</textarea>", page
        ).group(1)
        payload = json.loads(gzip.decompress(base64.b64decode(payload_text)))
        self.assertEqual(payload["format_version"], 2)
        self.assertEqual(len(payload["rows"]), 3)
        self.assertIn("class='feature-row ", payload["rows"][0]["html"])
        self.assertTrue(payload["fragments"])
        self.assertIn("@@LASER_FRAGMENT:", payload["rows"][0]["html"])

    def test_summary_chart_writer_produces_a_self_contained_html_file(self):
        output = Path(tempfile.mkdtemp()) / "summary.html"

        result = write_annotation_summary_charts_report(
            _annotations(), output
        )

        self.assertEqual(result, output)
        page = output.read_text(encoding="utf-8")
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)
        self.assertIn("Unique annotations before exclusions", page)
        self.assertIn("Unique annotations after all exclusions", page)
        self.assertNotIn("class='feature-row ", page)

    def test_summary_charts_separate_before_and_retained_populations(self):
        before = _annotations().copy()
        retained = before.iloc[:2].copy()

        page = render_annotation_summary_charts_report(
            retained, before_exclusions=before
        )
        before_section = page.split(
            "Unique annotations before exclusions", 1
        )[1].split("Unique annotations after all exclusions", 1)[0]
        retained_section = page.split(
            "Unique annotations after all exclusions", 1
        )[1].split("IS-corrected QC-RSD distributions", 1)[0]

        self.assertIn("PE: 1 unique lipid IDs; 1 with MS/MS spectrum", before_section)
        self.assertNotIn("PE: 1 unique lipid IDs", retained_section)
        self.assertIn("PC: 1 unique lipid IDs; 1 with MS/MS spectrum", retained_section)
        self.assertIn(
            "Total unique annotated features after all exclusions", retained_section
        )
        self.assertIn("<b>2</b>", retained_section)
        self.assertIn("Total unique lipid annotations after all exclusions", retained_section)

    def test_zero_matches_still_produces_the_standard_report(self):
        output = Path(tempfile.mkdtemp()) / "empty.html"

        result = write_annotated_feature_report(
            _annotations().iloc[0:0], output
        )

        self.assertEqual(result, output)
        self.assertIn(
            "No annotated feature matches", output.read_text(encoding="utf-8")
        )


if __name__ == "__main__":
    unittest.main()
