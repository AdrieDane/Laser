"""Canonical LASER saved-project contract."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from completed_project import load_completed_project  # noqa: E402
from saved_project import (  # noqa: E402
    SAVED_PROJECT_FORMAT_VERSION,
    load_project_table,
    save_standard_project,
)


class SavedProjectTests(unittest.TestCase):
    def test_standard_result_round_trips_without_html(self):
        root = Path(tempfile.mkdtemp()) / "standard_project"
        result = {
            "engine": "matlab",
            "streams": {"RP_Pos": {
                "reference": pd.DataFrame({"name": ["PC(34:1)"], "mz": [760.6]}),
                "criteria": {"ppm_tolerance": 4.0, "uses_rt_grid": True},
            }},
            "identified_lipids": pd.DataFrame({
                "feat_feature_id": ["7"], "name": ["PC(34:1)"],
                "adduct": ["[M+H]+"], "feat_rt": [8.2], "feat_mz": [760.6],
            }),
            "equivalent_lipids": pd.DataFrame(),
            "automatic_annotations": pd.DataFrame(),
            "all_candidates": pd.DataFrame({
                "feat_feature_id": ["7", "8"],
                "name": ["PC(34:1)", "PC(36:2)"],
                "adduct": ["[M+H]+", "[M+H]+"],
                "feat_rt": [8.2, 8.6], "feat_mz": [760.6, 786.6],
            }),
        }

        paths = save_standard_project(root, result, settings={"ppm": 3.0})
        project = load_completed_project(root)

        self.assertTrue(paths["project_manifest"].is_file())
        self.assertIsNone(project["report"])
        self.assertEqual(project["workflow"], "standard")
        self.assertEqual(len(load_project_table(project, "all_candidates")), 2)
        self.assertEqual(
            len(load_project_table(project, "stream__RP_Pos__reference")), 1
        )
        manifest = json.loads(paths["project_manifest"].read_text())
        self.assertEqual(
            manifest["format_version"], SAVED_PROJECT_FORMAT_VERSION
        )
        self.assertEqual(manifest["artifact_contract_version"], 1)
        self.assertEqual(manifest["project_type"], "LASER_saved_project")
        self.assertNotIn("standard_lipidomics_report", manifest["artifacts"])
        self.assertIn("standard_stream_metadata", manifest["artifacts"])
        candidate_artifact = project["artifacts"]["all_candidates"]
        self.assertTrue(Path(candidate_artifact["columnar_path"]).is_file())
        self.assertEqual(
            Path(candidate_artifact["columnar_path"]).suffix, ".parquet"
        )


if __name__ == "__main__":
    unittest.main()
