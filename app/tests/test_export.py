import json
import unittest

import pandas as pd

from annotator.export import build_annotated_features_export


class ExportTests(unittest.TestCase):
    def test_excel_safe_export_caps_large_tg_ranking_without_losing_candidates(self):
        ranking = [
            {
                "molecular_species": f"TG({index}:0_{20 + index}:1_18:1)",
                "supporting_fatty_acyls": ["FA(18:1)", f"FA({index}:0)"],
                "supporting_fragments": [
                    {
                        "fatty_acyl": "FA(18:1)",
                        "fragment_intensity": 400_799.6211490631,
                        "relative_intensity_percent": 99.123456789,
                    }
                ],
                "summed_fragment_intensity": 500_000 - index,
            }
            for index in range(1, 400)
        ]
        candidates = "; ".join(
            item["molecular_species"] for item in ranking
        )
        annotations = pd.DataFrame([{
            "feat_feature_id": "73",
            "name": "TG(54:9)",
            "class": "TG",
            "grid_class": "TG",
            "adduct": "[M+NH4]+",
            "feat_mz": 900.8,
            "status": "accept",
            "automatic_candidate": True,
            "molecular_species_candidates": candidates,
            "molecular_species_ranking_json": json.dumps(ranking),
        }])

        exported = build_annotated_features_export(annotations)
        ranking_text = exported.loc[0, "molecular_species_ranking_json"]
        compact = json.loads(ranking_text)

        self.assertLessEqual(len(ranking_text), 30_000)
        self.assertEqual(compact[0], ranking[0])
        self.assertLess(len(compact), len(ranking))
        self.assertIn(ranking[-1]["molecular_species"], candidates)
        self.assertEqual(exported.loc[0, "molecular_species_candidates"], candidates)

    def test_export_keeps_repeated_species_and_reports_multiple_identities(self):
        annotations = pd.DataFrame([
            {"feat_feature_id": "10", "name": "PC(34:1)", "class": "PC",
             "grid_class": "PC", "adduct": "[M+H]+", "feat_mz": 760.6,
             "status": "accept", "automatic_candidate": True,
             "grid_coherence_score": 0.91,
             "grid_coherence_neighbors": 4,
             "grid_coherence_dimensions": 3,
             "grid_coherence_assessed": True,
             "grid_coherence_outlier": False,
             "grid_coherence_pass": True,
             "msms_rule_id": "PC_RULE", "msms_rule_pass": True,
             "msms_rule_description": "PC headgroup fragment",
             "msms_status": "supported", "msms_has_spectrum": True,
             "fatty_acyl_rule_id": "NEG_FA", "fatty_acyl_status": "supported",
             "fatty_acyl_min_relative_intensity_percent": 1.0,
             "fatty_acyl_detected": "FA(16:0); FA(18:1)",
             "fatty_acyl_detected_count": 2,
             "fatty_acyl_evidence_json": "[]",
             "fatty_acyl_fragment_intensities": (
                 "FA(18:1): 1000 (100.00%); FA(16:0): 800 (80.00%)"
             ),
             "molecular_species_candidates": "PC(16:0_18:1)",
             "molecular_species_candidate_count": 1,
             "molecular_species_ambiguous": False,
             "molecular_species_ranking_json": "[]",
             "most_likely_molecular_species": "PC(16:0_18:1)",
             "most_likely_molecular_species_score": 180.0,
             "most_likely_molecular_species_ambiguous": False,
             "feat_quantification_measure": "peak_height",
             "feat_sample_A.raw": 1000, "feat_sample_B.raw": 2000,
             "feat_height::sample_A.raw": 1000,
             "feat_height::sample_B.raw": 2000,
             "feat_area::sample_A.raw": 4000,
             "feat_area::sample_B.raw": 8000},
            {"feat_feature_id": "10", "name": "PC(36:4)", "class": "PC",
             "grid_class": "PC", "adduct": "[M+H]+", "feat_mz": 760.6,
             "status": "reject", "automatic_candidate": True,
             "msms_rule_id": "PC_RULE", "msms_rule_pass": True,
             "msms_status": "supported", "msms_has_spectrum": True,
             "feat_quantification_measure": "peak_height",
             "feat_sample_A.raw": 1000, "feat_sample_B.raw": 2000,
             "feat_height::sample_A.raw": 1000,
             "feat_height::sample_B.raw": 2000,
             "feat_area::sample_A.raw": 4000,
             "feat_area::sample_B.raw": 8000},
            {"feat_feature_id": "11", "name": "PC(34:1)", "class": "PC",
             "grid_class": "PC", "adduct": "[M+H]+", "feat_mz": 760.6,
             "status": "accept", "automatic_candidate": True,
             "msms_rule_id": "PC_RULE", "msms_rule_pass": pd.NA,
             "msms_status": "no_msms", "msms_has_spectrum": False,
             "feat_quantification_measure": "peak_height",
             "feat_sample_A.raw": 3000, "feat_sample_B.raw": 4000,
             "feat_height::sample_A.raw": 3000,
             "feat_height::sample_B.raw": 4000,
             "feat_area::sample_A.raw": 12000,
             "feat_area::sample_B.raw": 16000},
        ])
        exported = build_annotated_features_export(
            annotations, ["sample_A.raw", "sample_B.raw"]
        )
        self.assertEqual(len(exported), 3)
        self.assertEqual((exported["feature_identity"] == "PC(34:1)").sum(), 2)
        pc_341 = exported[exported["feature_identity"] == "PC(34:1)"]
        self.assertEqual(
            pc_341.loc[pc_341["primary_for_identity"], "feature_id"].tolist(),
            ["10"],
        )
        self.assertEqual(set(pc_341["identity_match_rank"]), {1, 2})
        ambiguous = exported[exported["feature_id"] == "10"]
        self.assertTrue(ambiguous["multiple_identities_for_feature"].all())
        self.assertTrue((ambiguous["identity_count_for_feature"] == 2).all())
        self.assertIn("PC(34:1) [M+H]+", ambiguous.iloc[0]["all_matching_identities"])
        self.assertEqual(ambiguous.iloc[0]["intensity::sample_A.raw"], 1000)
        self.assertEqual(ambiguous.iloc[0]["height::sample_A.raw"], 1000)
        self.assertEqual(ambiguous.iloc[0]["area::sample_A.raw"], 4000)
        self.assertEqual(ambiguous.iloc[0]["quantification_measure"], "peak_height")
        self.assertIn("class_rule_pass", exported.columns)
        self.assertEqual(exported.iloc[0]["class_rule_result"], "PASS")
        self.assertEqual(
            exported.iloc[0]["class_rule_description"], "PC headgroup fragment"
        )
        self.assertEqual(
            exported.loc[
                exported["feature_id"].eq("11"), "class_rule_result"
            ].item(),
            "NO_SPECTRUM",
        )
        self.assertIn("grid_coherence_score", exported.columns)
        self.assertEqual(exported.iloc[0]["grid_coherence_score"], 0.91)
        self.assertIn("detected_fatty_acyl_groups", exported.columns)
        self.assertEqual(
            exported.iloc[0]["fatty_acyl_min_relative_intensity_percent"], 1.0
        )
        self.assertEqual(
            exported.iloc[0]["molecular_species_candidates"], "PC(16:0_18:1)"
        )
        self.assertEqual(
            exported.iloc[0]["most_likely_molecular_species"],
            "PC(16:0_18:1)",
        )
        self.assertIn(
            "FA(18:1): 1000",
            exported.iloc[0]["fatty_acyl_fragment_intensities"],
        )

    def test_export_always_has_height_and_area_columns_without_fabricating_missing_data(self):
        annotations = pd.DataFrame([{
            "feat_feature_id": "10", "name": "PC(34:1)", "class": "PC",
            "adduct": "[M+H]+", "feat_mz": 760.6,
            "feat_quantification_measure": "generic_intensity",
            "feat_sample_A.raw": 1000,
        }])

        exported = build_annotated_features_export(annotations, ["sample_A.raw"])

        self.assertIn("height::sample_A.raw", exported.columns)
        self.assertIn("area::sample_A.raw", exported.columns)
        self.assertTrue(pd.isna(exported.loc[0, "height::sample_A.raw"]))
        self.assertTrue(pd.isna(exported.loc[0, "area::sample_A.raw"]))

    def test_export_recovers_legacy_selected_measure_from_explicit_provenance(self):
        annotations = pd.DataFrame([{
            "feat_feature_id": "10", "name": "PC(34:1)", "class": "PC",
            "adduct": "[M+H]+", "feat_mz": 760.6,
            "feat_quantification_measure": "peak_height",
            "feat_sample_A.raw": 1000,
        }])

        exported = build_annotated_features_export(annotations, ["sample_A.raw"])

        self.assertEqual(exported.loc[0, "height::sample_A.raw"], 1000)
        self.assertTrue(pd.isna(exported.loc[0, "area::sample_A.raw"]))

    def test_export_keeps_raw_and_is_corrected_sample_measurements(self):
        annotations = pd.DataFrame([{
            "feat_feature_id": "10", "name": "PC(34:1)", "class": "PC",
            "adduct": "[M+H]+", "feat_mz": 760.6,
            "feat_quantification_measure": "peak_height",
            "feat_sample_A.raw": 1000,
            "feat_height::sample_A.raw": 1000,
            "feat_area::sample_A.raw": 4000,
            "is_corrected_height::sample_A.raw": 2.5,
            "is_corrected_area::sample_A.raw": 2.0,
            "is_normalization_status": "assigned",
            "is_normalization_selection": "same_class_auto",
        }])

        exported = build_annotated_features_export(annotations, ["sample_A.raw"])

        self.assertEqual(exported.loc[0, "height::sample_A.raw"], 1000)
        self.assertEqual(exported.loc[0, "area::sample_A.raw"], 4000)
        self.assertEqual(
            exported.loc[0, "is_corrected_height::sample_A.raw"], 2.5
        )
        self.assertEqual(
            exported.loc[0, "is_corrected_area::sample_A.raw"], 2.0
        )
        self.assertEqual(exported.loc[0, "is_normalization_status"], "assigned")

    def test_export_keeps_raw_and_calibrated_mass_with_provenance(self):
        annotations = pd.DataFrame([{
            "feat_feature_id": "10",
            "name": "PC(34:1)",
            "class": "PC",
            "grid_class": "PC",
            "adduct": "[M+H]+",
            "feat_mz_raw": 760.58610,
            "feat_mz": 760.58590,
            "ref_mz": 760.58595,
            "status": "accept",
            "automatic_candidate": True,
            "mass_calibration_standard_features": (
                "TuneMix 622=3; TuneMix 922=58"
            ),
            "mass_calibration_slope": 0.9999998,
            "mass_calibration_intercept": -0.00005,
            "mass_calibration_rmse_before_ppm": 1.2,
            "mass_calibration_rmse_after_ppm": 0.2,
        }])

        exported = build_annotated_features_export(annotations)

        self.assertAlmostEqual(exported.loc[0, "mz_raw"], 760.58610)
        self.assertAlmostEqual(exported.loc[0, "mz"], 760.58590)
        self.assertTrue(exported.loc[0, "mass_calibration_applied"])
        self.assertEqual(
            exported.loc[0, "mass_calibration_standard_features"],
            "TuneMix 622=3; TuneMix 922=58",
        )
        self.assertAlmostEqual(
            exported.loc[0, "mass_calibration_rmse_after_ppm"], 0.2
        )

    def test_exclusion_exports_record_filter_and_reason(self):
        annotations = pd.DataFrame([{
            "feat_feature_id": "42", "name": "PC(36:2)", "class": "PC",
            "adduct": "[M+H]+", "feat_mz": 786.6, "status": "reject",
            "automatic_candidate": False,
            "rule_excluded": True, "grid_excluded": False,
            "exclusion_reason": "class_fragment_rule_failed",
            "msms_rule_id": "POS_PHOSPHOCHOLINE_HEADGROUP_184",
            "msms_status": "not_detected", "msms_has_spectrum": True,
        }])

        exported = build_annotated_features_export(annotations)

        self.assertTrue(exported.loc[0, "msms_rule_excluded"])
        self.assertFalse(exported.loc[0, "grid_coherence_excluded"])
        self.assertEqual(
            exported.loc[0, "exclusion_reason"],
            "class_fragment_rule_failed",
        )
        self.assertFalse(exported.loc[0, "primary_for_identity"])

    def test_manual_primary_feature_is_explicit_in_export(self):
        annotations = pd.DataFrame([
            {
                "feat_feature_id": "10", "name": "PC(34:1)", "class": "PC",
                "adduct": "[M+H]+", "feat_mz": 760.6,
                "automatic_candidate": True, "msms_status": "supported",
                "primary_feature_override": False,
            },
            {
                "feat_feature_id": "11", "name": "PC(34:1)", "class": "PC",
                "adduct": "[M+H]+", "feat_mz": 760.6,
                "automatic_candidate": True, "msms_status": "no_msms",
                "primary_feature_override": True,
            },
        ])

        exported = build_annotated_features_export(annotations)

        self.assertEqual(
            exported.loc[exported["primary_for_identity"], "feature_id"].tolist(),
            ["11"],
        )
        self.assertEqual(
            exported.set_index("feature_id")["identity_match_rank"].to_dict(),
            {"10": 2, "11": 1},
        )


if __name__ == "__main__":
    unittest.main()
