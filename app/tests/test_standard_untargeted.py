from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from standard_lipidomics.untargeted import (
    UntargetedComparisonCriteria,
    analyse_standard_untargeted,
    attach_excel_sample_metadata,
    default_biological_groups,
    group_catalog,
)


class StandardUntargetedTests(unittest.TestCase):
    def _result(self) -> dict:
        features = pd.DataFrame(
            {
                "feature_id": [
                    "ann1",
                    "ann2",
                    "ann3",
                    "diff",
                    "diff_m1",
                    "diff_m2",
                    "series",
                    "same",
                    "sparse",
                ],
                "mz": [
                    700.0,
                    714.0,
                    728.0,
                    721.0,
                    722.00335483507,
                    723.00670967014,
                    735.015650064,
                    800.0,
                    850.0,
                ],
                "rt": [8.00, 8.05, 8.10, 8.06, 8.061, 8.062, 8.08, 10.0, 11.0],
                "a1": [100, 100, 100, 10, 1, 0.1, 40, 50, 0],
                "a2": [110, 110, 110, 10, 1, 0.1, 42, 52, 0],
                "a3": [90, 90, 90, 10, 1, 0.1, 38, 48, 0],
                "b1": [100, 100, 100, 1000, 100, 10, 400, 50, 500],
                "b2": [110, 110, 110, 1100, 110, 11, 420, 52, 0],
                "b3": [90, 90, 90, 900, 90, 9, 380, 48, 0],
                "qc1": [100, 100, 100, 100, 10, 1, 40, 50, 0],
            }
        )
        features.attrs["sample_group_map"] = {
            "a1": "Group_01_control",
            "a2": "Group_01_control",
            "a3": "Group_01_control",
            "b1": "Group_02_disease",
            "b2": "Group_02_disease",
            "b3": "Group_02_disease",
            "qc1": "Group_03_QC_pool",
        }
        accepted = pd.DataFrame(
            {
                "stream": ["RP_Pos"] * 3,
                "feature_id": ["ann1", "ann2", "ann3"],
                "feature_identity": ["PC(32:0)", "PC(34:0)", "PC(36:0)"],
                "lipid_class": ["PC", "PC", "PC"],
                "mz": [700.0, 714.0, 728.0],
                "rt": [8.00, 8.05, 8.10],
            }
        )

        def counterpart(feature_id: str, mz: float, rt: float) -> pd.DataFrame:
            frame = pd.DataFrame(
                {
                    "feature_id": [feature_id],
                    "mz": [mz],
                    "rt": [rt],
                    "a1": [10],
                    "a2": [10],
                    "a3": [10],
                    "b1": [1000],
                    "b2": [1100],
                    "b3": [900],
                }
            )
            frame.attrs["sample_group_map"] = {
                "a1": "Group_01_control",
                "a2": "Group_01_control",
                "a3": "Group_01_control",
                "b1": "Group_02_disease",
                "b2": "Group_02_disease",
                "b3": "Group_02_disease",
            }
            return frame

        return {
            "streams": {
                "RP_Pos": {"features": features},
                "RP_Neg": {
                    "features": counterpart(
                        "diff_neg",
                        718.985447066758,
                        8.065,
                    )
                },
                "NP_Pos": {
                    "features": counterpart(
                        "diff_np",
                        721.00050,
                        3.2,
                    )
                },
            },
            "automatic_annotations": accepted,
        }

    def test_group_catalog_excludes_qc_from_default_comparison(self):
        result = self._result()
        catalog = group_catalog(result)
        self.assertFalse(
            bool(catalog.set_index("group").loc["Group_03_QC_pool", "biological"])
        )
        self.assertEqual(
            default_biological_groups(result),
            ("Group_01_control", "Group_02_disease"),
        )

    def test_excel_metadata_is_the_authoritative_sample_group_mapping(self):
        result = self._result()
        features = result["streams"]["RP_Pos"]["features"]
        features.attrs["sample_columns"] = ["a1", "a2", "a3", "b1", "b2", "b3", "qc1"]
        metadata = pd.DataFrame(
            {
                "File Name": ["a1", "a2", "a3", "b1", "b2", "b3", "qc1"],
                "Mode": ["RP_Pos"] * 7,
                "Group Label": ["Excel_A"] * 3 + ["Excel_B"] * 3 + ["Excel_QC"],
                "Group Name": ["control"] * 3 + ["disease"] * 3 + ["pooled QC"],
                "Sample Type": ["study"] * 6 + ["QC"],
            }
        )
        with (
            patch(
                "standard_lipidomics.untargeted._metadata_workbooks",
                return_value=[Path("metadata_RP.xlsx")],
            ),
            patch(
                "standard_lipidomics.untargeted.pd.read_excel",
                return_value=metadata,
            ),
        ):
            audit = attach_excel_sample_metadata(result, ".")
        self.assertEqual(features.attrs["sample_group_map"]["a1"], "Excel_A")
        self.assertEqual(features.attrs["sample_type_map"]["qc1"], "QC")
        self.assertEqual(audit.iloc[0]["mapping_status"], "excel_metadata_validated")
        catalog = group_catalog(result).set_index("group")
        self.assertTrue(bool(catalog.loc["Excel_A", "biological"]))
        self.assertFalse(bool(catalog.loc["Excel_QC", "biological"]))

    def test_only_unannotated_significant_features_are_returned(self):
        result = analyse_standard_untargeted(
            self._result(),
            UntargetedComparisonCriteria(
                group_a="Group_01_control",
                group_b="Group_02_disease",
                p_value_threshold=0.05,
                min_abs_log2_fold_change=1.0,
                min_detection_percent=50.0,
            ),
        )
        tested = result["features"].set_index("feature_id")
        self.assertFalse({"ann1", "ann2", "ann3"} & set(tested.index))
        self.assertTrue(bool(tested.loc["diff", "significant"]))
        self.assertFalse(bool(tested.loc["same", "significant"]))
        self.assertFalse(bool(tested.loc["sparse", "significant"]))
        self.assertNotIn("q_value", tested.columns)
        self.assertIn("unadjusted", result["method"])
        self.assertEqual(tested.loc["diff", "cluster_class"], "PC")
        self.assertIn("local PC region", tested.loc["diff", "cluster_hypothesis"])
        self.assertGreaterEqual(
            tested.loc["diff", "homologous_series_neighbor_count"], 1
        )
        self.assertIn("CH2 step", tested.loc["diff", "homologous_series_hypothesis"])
        self.assertEqual(tested.loc["diff", "significant_ch2_series_size"], 2)
        self.assertIn(
            "CH2-spaced pair",
            tested.loc["diff", "significant_ch2_cluster_hypothesis"],
        )
        self.assertEqual(tested.loc["diff", "isotope_m_plus_1_feature_id"], "diff_m1")
        self.assertIn("M+2", tested.loc["diff", "isotope_pattern"])
        self.assertAlmostEqual(
            tested.loc["diff", "estimated_carbon_count_from_m_plus_1"],
            10.0 / 1.07,
            places=2,
        )
        self.assertIn(
            "exact formula unresolved",
            tested.loc["diff", "chemical_composition_hypothesis"],
        )
        self.assertTrue(bool(tested.loc["diff", "cross_phase_match"]))
        self.assertEqual(tested.loc["diff", "cross_phase_stream"], "NP_Pos")
        self.assertTrue(bool(tested.loc["diff", "opposite_polarity_match"]))
        self.assertEqual(tested.loc["diff", "opposite_polarity_stream"], "RP_Neg")
        self.assertEqual(
            tested.loc["diff", "opposite_polarity_rule"],
            "[M+H]+ ↔ [M-H]-",
        )
        self.assertAlmostEqual(
            tested.loc["diff", "cross_phase_abundance_spearman"], 1.0
        )
        self.assertEqual(tested.loc["diff", "cross_phase_abundance_pairs"], 6)
        self.assertAlmostEqual(
            tested.loc["diff", "opposite_polarity_abundance_spearman"], 1.0
        )
        self.assertEqual(tested.loc["diff", "opposite_polarity_abundance_pairs"], 6)
        self.assertIn(
            "abundance rho +1.00", tested.loc["diff", "cross_acquisition_evidence"]
        )

    def test_default_raw_p_threshold_is_conservative(self):
        criteria = UntargetedComparisonCriteria(
            group_a="Group_01_control",
            group_b="Group_02_disease",
        )
        self.assertEqual(criteria.p_value_threshold, 0.001)

    def test_mzmine_detection_states_override_gap_filled_nonzero_values(self):
        result = self._result()
        features = result["streams"]["RP_Pos"]["features"]
        for sample in ("a1", "a2", "a3", "b1", "b2", "b3"):
            features[f"{sample}::detected"] = True
        features.loc[
            features["feature_id"].eq("sparse"),
            [f"{sample}::detected" for sample in ("a1", "a2", "a3", "b1", "b2", "b3")],
        ] = False
        analysis = analyse_standard_untargeted(
            result,
            UntargetedComparisonCriteria(
                group_a="Group_01_control",
                group_b="Group_02_disease",
                p_value_threshold=0.05,
                min_abs_log2_fold_change=1.0,
                min_detection_percent=50.0,
            ),
        )
        sparse = analysis["features"].set_index("feature_id").loc["sparse"]
        self.assertEqual(sparse["detection_b_percent"], 0.0)
        self.assertEqual(sparse["detection_evidence"], "MZmine DETECTED state")
        self.assertFalse(bool(sparse["significant"]))


if __name__ == "__main__":
    unittest.main()
