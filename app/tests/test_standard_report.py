from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from standard_lipidomics.report import (
    render_standard_lipidomics_report,
    write_standard_lipidomics_report,
)


def _result(*, matlab_shape: bool = False) -> dict:
    features = pd.DataFrame({
        "feature_id": ["one", "two", "review"],
        "mz": [734.5002, 734.5003, 760.5002],
        "rt": [7.02, 7.03, 7.24],
        "rtmin": [6.98, 6.99, 7.20],
        "rtmax": [7.06, 7.07, 7.28],
        "npeaks": [12, 10, 8],
        "sample_A": [1e6, 0.8e6, 2e6],
        "sample_B": [1.1e6, 0.9e6, 0.0],
    })
    features.attrs["sample_columns"] = ["sample_A", "sample_B"]
    common = {
        "name": ["PC(32:0)", "PC(32:0)", "PC(34:1)"],
        "class": ["PC", "PC", "PC"],
        "adduct": ["[M+H]+"] * 3,
        "stream": ["RP_Pos"] * 3,
        "feat_feature_id": ["one", "two", "review"],
        "feat_mz": [734.5002, 734.5003, 760.5002],
        "ref_mz": [734.5, 734.5, 760.5],
        "feat_rt": [7.02, 7.03, 7.24],
        "ref_rt": [7.0, 7.0, 7.2],
        "ppm_error": [0.27, 0.41, 0.26],
        "grid_coherence_score": [0.9, 0.8, 0.3],
        "feat_sample_A": [1e6, 0.8e6, 2e6],
        "feat_sample_B": [1.1e6, 0.9e6, 0.0],
    }
    candidates = pd.DataFrame(common)
    if not matlab_shape:
        candidates = candidates.rename(columns={
            "name": "feature_identity", "class": "lipid_class",
            "feat_feature_id": "feature_id", "feat_mz": "mz",
            "ref_mz": "reference_mz", "feat_rt": "rt_min",
            "ref_rt": "reference_rt_min",
        })
        candidates["intensity::sample_A"] = candidates.pop("feat_sample_A")
        candidates["intensity::sample_B"] = candidates.pop("feat_sample_B")
    primary = candidates.iloc[[0]].copy()
    primary["multiple_features_for_identity"] = True
    primary["feature_count_for_identity"] = 2
    primary["all_feature_ids_for_identity"] = "one; two"
    equivalent = candidates.iloc[[1]].copy()
    automatic = candidates.iloc[:2].copy()
    return {
        "engine": "matlab" if matlab_shape else "laser",
        "streams": {"RP_Pos": {
            "features": features,
            "criteria": {"uses_rt_grid": True},
            "rt_models": pd.DataFrame(),
            "np_rt_alignment": pd.DataFrame(),
        }},
        "identified_lipids": primary,
        "equivalent_lipids": equivalent,
        "automatic_annotations": automatic,
        "all_candidates": candidates,
        "policy_audit": pd.DataFrame(),
        "np_rt_audit": pd.DataFrame(),
        "grid_coherence_audit": pd.DataFrame(),
    }


class StandardReportTests(unittest.TestCase):
    def test_report_distinguishes_selected_equivalent_and_review_rows(self):
        page = render_standard_lipidomics_report(
            _result(), settings={"ppm": 4.0}, project="example"
        )

        self.assertIn("LASER · standard lipidomics report", page)
        self.assertIn("Selected annotations", page)
        self.assertIn("Equivalent features", page)
        self.assertIn("Review candidates", page)
        self.assertIn("PRIMARY", page)
        self.assertIn("EQUIVALENT", page)
        self.assertIn("REVIEW CANDIDATE", page)
        self.assertIn("Selected among 2 feature matches", page)
        self.assertIn("positive intensity fraction", page)
        self.assertIn("100.0%", page)
        self.assertIn("Raw chromatogram traces", page)
        self.assertIn("not available", page)

    def test_report_accepts_matlab_engine_column_names_and_writes_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_standard_lipidomics_report(
                _result(matlab_shape=True),
                Path(temporary) / "report.html",
                project="matlab-project",
            )

            page = path.read_text(encoding="utf-8")
            self.assertIn("MATLAB annotation engine", page)
            self.assertIn("PC(32:0)", page)
            self.assertIn("feature one", page)

    def test_report_includes_untargeted_results_and_excel_join_audit(self):
        result = _result()
        result["untargeted_analysis"] = {
            "significant_features": pd.DataFrame({
                "stream": ["RP_Pos"], "feature_id": ["unknown-7"],
                "p_value": [0.01], "log2_fold_change_b_vs_a": [2.1],
                "cluster_hypothesis": ["local PC region"],
                "homologous_series_hypothesis": ["one nearby CH2 neighbour"],
            }),
            "group_catalog": pd.DataFrame({
                "stream": ["RP_Pos"], "group": ["Group_01_control"],
                "biological": [True], "samples": [5],
            }),
            "interpretation_boundary": "Context is not an identification.",
        }
        result["untargeted_sample_metadata_audit"] = pd.DataFrame({
            "stream": ["RP_Pos"],
            "metadata_workbook": ["metadata_RP.xlsx"],
            "sample_columns": [10], "matched_samples": [10],
            "mapping_status": ["excel_metadata_validated"],
        })
        result["untargeted_review_basket"] = pd.DataFrame({
            "decision": ["follow up"], "note": ["strong cross-mode evidence"],
            "stream": ["RP_Pos"], "feature_id": ["unknown-7"],
        })
        result["untargeted_reacquisition_list"] = pd.DataFrame({
            "feature_id": ["unknown-7"], "target_mz": [650.1234],
            "target_rt_min": [8.2], "polarity": ["POS"], "phase": ["RP"],
        })

        page = render_standard_lipidomics_report(result)

        self.assertIn("Untargeted differences", page)
        self.assertIn("unknown-7", page)
        self.assertIn("metadata_RP.xlsx", page)
        self.assertIn("excel_metadata_validated", page)
        self.assertIn("Human feature-review basket", page)
        self.assertIn("strong cross-mode evidence", page)
        self.assertIn("Targeted reacquisition list", page)


if __name__ == "__main__":
    unittest.main()
