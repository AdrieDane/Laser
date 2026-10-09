import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from matlab_annotation.configuration import DEFAULT_MATLAB_PIPELINE_REFERENCE
from matlab_annotation.xcms_project import build_rp_xcms_project
from matlab_annotation.engine import _restore_xcms_frame
from adapters.standard_engines import MatlabXcmsEngine
from application.standard_workflow import StandardAnnotationRequest
from standard_lipidomics.mass_calibration import (
    apply_standard_mass_calibration,
    calibrate_standard_xcms_bytes,
    freeze_mass_calibration,
    mass_calibration_summary,
    normalize_mass_calibrations,
    write_calibrated_xcms_files,
)
from standard_lipidomics.workflow import (
    discover_standard_streams,
    load_standard_feature_list,
)


def _model(slope: float, intercept: float = 0.0) -> dict:
    return {
        "kind": "linear",
        "coefficients": [slope, intercept],
        "slope": slope,
        "intercept": intercept,
        "n_used": 4,
        "rmse_before_ppm": 4.0,
        "rmse_after_ppm": 0.5,
    }


class StandardMassCalibrationTests(unittest.TestCase):
    @patch("adapters.standard_engines.run_matlab_annotation")
    def test_standard_engine_forwards_headless_per_stream_models(self, run):
        run.return_value = {
            "streams": {},
            **{
                name: pd.DataFrame()
                for name in (
                    "all_candidates", "automatic_annotations", "identified_lipids",
                    "equivalent_lipids", "policy_audit", "np_rt_audit",
                )
            },
        }
        fixed = freeze_mass_calibration(_model(1.000001))

        MatlabXcmsEngine().annotate(StandardAnnotationRequest(
            engine="matlab",
            project=Path("."),
            settings={"mass_calibrations": {"RP_Pos": fixed}},
        ))

        self.assertEqual(
            run.call_args.kwargs["mass_calibrations"]["RP_Pos"]["model"],
            "linear",
        )

    def test_feature_correction_preserves_raw_mz_and_attrs(self):
        features = pd.DataFrame({"feature_id": ["1"], "mz": [500.0]})
        features.attrs["sample_columns"] = ["sample_a"]

        corrected = apply_standard_mass_calibration(features, _model(1.0, 0.01))

        self.assertEqual(corrected.loc[0, "mz_raw"], 500.0)
        self.assertAlmostEqual(corrected.loc[0, "mz"], 500.01)
        self.assertEqual(corrected.attrs["sample_columns"], ["sample_a"])
        self.assertEqual(corrected.attrs["mass_calibration"]["mode"], "fixed")

    def test_frozen_models_are_accepted_by_headless_path(self):
        frozen = freeze_mass_calibration(_model(1.000001, -0.0002))
        normalized = normalize_mass_calibrations({"NP_Pos": frozen})

        self.assertEqual(normalized["NP_Pos"]["kind"], "linear")
        self.assertEqual(normalized["NP_Pos"]["coefficients"], [1.000001, -0.0002])

    def test_summary_keeps_all_four_modes_separate(self):
        summary = mass_calibration_summary({
            "RP_Pos": _model(1.000001),
            "NP_Neg": _model(0.999999),
        }).set_index("stream")

        self.assertEqual(len(summary), 4)
        self.assertEqual(summary.loc["RP_Pos", "status"], "applied")
        self.assertEqual(summary.loc["RP_Neg", "status"], "not_applied")
        self.assertNotEqual(
            summary.loc["RP_Pos", "slope"], summary.loc["NP_Neg", "slope"]
        )

    def test_xcms_project_applies_independent_positive_and_negative_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for polarity, mz in (("Pos", 500.0), ("Neg", 600.0)):
                (root / f"peakgrouplist_RP_{polarity}.txt").write_text(
                    "name\tmzmed\trtmed\tnpeaks\tsample_A\n"
                    f"1\t{mz}\t8.0\t1\t1000\n",
                    encoding="utf-8",
                )
            bundle = build_rp_xcms_project(
                root,
                reference_root=DEFAULT_MATLAB_PIPELINE_REFERENCE,
                mass_calibrations={
                    "RP_Pos": _model(1.0, 0.01),
                    "RP_Neg": _model(1.0, -0.02),
                },
            )

        positive = bundle.features["RP_Pos"]
        negative = bundle.features["RP_Neg"]
        self.assertAlmostEqual(positive.loc[0, "mz"], 500.01)
        self.assertAlmostEqual(negative.loc[0, "mz"], 599.98)
        self.assertEqual(positive.loc[0, "mz_raw"], 500.0)
        self.assertEqual(negative.loc[0, "mz_raw"], 600.0)
        self.assertTrue(np.isclose(bundle.project.phases["RP"].mz_grouped, -599.98).any())
        self.assertTrue(np.isclose(bundle.project.phases["RP"].mz_grouped, 500.01).any())
        positive_row = bundle.row_stream.index("RP_Pos")
        restored = _restore_xcms_frame(pd.DataFrame({
            "feat_feature_id": [str(positive_row + 1)],
            "feat_mz": [500.01],
            "ref_mz": [500.0],
        }), bundle, "RP_Pos")
        self.assertEqual(restored.loc[0, "feat_mz_raw"], 500.0)
        self.assertEqual(restored.loc[0, "mass_calibration_model"], "linear")
        self.assertEqual(restored.loc[0, "mass_calibration_standard_features"], "")

    def test_calibrated_xcms_export_is_self_contained_annotation_input(self):
        source = (
            b"name\tmzmed\trtmed\tnpeaks\tGroup_A\tsample_A\n"
            b"1\t500.0\t8.0\t1\t1\t1000\n"
        )

        exported = calibrate_standard_xcms_bytes(source, _model(1.0, 0.01))
        features = load_standard_feature_list(exported)

        self.assertAlmostEqual(features.loc[0, "mz"], 500.01)
        self.assertEqual(features.loc[0, "mz_raw"], 500.0)
        self.assertEqual(features.attrs["sample_columns"], ["sample_A"])
        self.assertEqual(
            features.attrs["mass_calibration"]["source"],
            "calibrated XCMS export",
        )
        self.assertNotIn("LASER_original_mz", features.attrs["sample_columns"])

    def test_calibration_module_writes_discoverable_files_and_manifest(self):
        source = (
            b"name\tmzmed\trtmed\tnpeaks\tsample_A\n"
            b"1\t500.0\t8.0\t1\t1000\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_calibrated_xcms_files(
                {"RP_Pos": source, "RP_Neg": source},
                {"RP_Pos": _model(1.0, 0.01)},
                root,
            )

            discovered = discover_standard_streams(root)
            self.assertEqual(set(discovered), {"RP_Pos", "RP_Neg"})
            self.assertEqual(paths, discovered)
            self.assertTrue((root / "LASER_standard_mass_calibration.json").is_file())
            self.assertIn(b"LASER_original_mz", paths["RP_Pos"].read_bytes())
            self.assertEqual(paths["RP_Neg"].read_bytes(), source)

            with self.assertRaises(FileExistsError):
                write_calibrated_xcms_files(
                    {"RP_Pos": source}, {"RP_Pos": _model(1.0)}, root
                )


if __name__ == "__main__":
    unittest.main()
