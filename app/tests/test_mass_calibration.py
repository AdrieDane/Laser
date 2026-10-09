import unittest

import numpy as np
import pandas as pd

from annotator.mass_calibrants import (
    MASS_REFERENCE_PANELS,
    build_ccs_reference_panel,
    build_lipid_validation_reference_panel,
    build_mass_reference_panel,
    detect_ccs_calibration_candidates,
    detect_mass_calibration_candidates,
    evaluate_lipid_validation_panel,
)
from annotator.matching import (
    MASS_CALIBRATION_MODELS,
    apply_mass_calibration,
    ccs_to_one_over_k0,
    fit_confirmed_mass_calibration,
)
from headless_workflows.four_d_calibration import automatic_calibrations


def _reference() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "name": "DG(28:0) (IS)", "class": "DG", "adduct": "[M+NH4]+",
            "ref_mz": 500.0, "ref_rt": 5.0, "ref_ccs": 220.0,
            "charge": 1, "polarity": "pos",
        },
        {
            "name": "TG(42:0) (IS)", "class": "TG", "adduct": "[M+NH4]+",
            "ref_mz": 700.0, "ref_rt": 7.0, "ref_ccs": 280.0,
            "charge": 1, "polarity": "pos",
        },
        {
            "name": "CL(56:0) (IS)", "class": "CL", "adduct": "[M+NH4]+",
            "ref_mz": 900.0, "ref_rt": 9.0, "ref_ccs": np.nan,
            "charge": 1, "polarity": "pos",
        },
    ])


UNLABELLED = {
    "DG(28:0)": "DG(28:0) (IS)",
    "TG(42:0)": "TG(42:0) (IS)",
    "CL(56:0)": "CL(56:0) (IS)",
}


class ReferencePanelTests(unittest.TestCase):
    def test_five_documented_panels_are_available(self):
        self.assertEqual(
            MASS_REFERENCE_PANELS,
            (
                "Labelled IS",
                "Labelled IS + Agilent tune mix",
                "Unlabelled IS",
                "Unlabelled IS + Agilent tune mix",
                "Agilent tune mix",
            ),
        )

    def test_tune_mix_is_only_added_to_requested_panels(self):
        lipid_only = build_mass_reference_panel(
            _reference(), polarity="pos", panel_name="Unlabelled IS",
            unlabelled_standards=UNLABELLED,
        )
        combined = build_mass_reference_panel(
            _reference(), polarity="pos",
            panel_name="Unlabelled IS + Agilent tune mix",
            unlabelled_standards=UNLABELLED,
        )
        tune_only = build_mass_reference_panel(
            _reference(), polarity="pos", panel_name="Agilent tune mix",
            unlabelled_standards=UNLABELLED,
        )

        self.assertEqual(set(lipid_only["source"]), {"unlabelled internal standard"})
        self.assertGreater(len(combined), len(lipid_only))
        self.assertEqual(set(tune_only["source"]), {"Agilent tune mix"})

    def test_bundled_independent_lipid_panels_are_available_to_laser(self):
        pos = build_lipid_validation_reference_panel("POS")
        neg = build_lipid_validation_reference_panel("neg")

        self.assertEqual(len(pos), 30)
        self.assertEqual(len(neg), 26)
        self.assertIn("DG(34:1)", set(pos["standard"]))
        self.assertTrue({
            "PC(32:1)", "LPC(20:4)", "LPE(16:0)", "PI(36:2)", "SM(d32:1)",
        }.issubset(set(neg["standard"])))
        self.assertFalse(neg["standard"].duplicated().any())
        self.assertEqual(set(pos["source"]), {"independent lipid validation"})
        self.assertTrue(pos["ref_mz"].is_monotonic_increasing)
        self.assertTrue(neg["ref_mz"].is_monotonic_increasing)

    def test_mass_search_respects_both_ppm_and_rt_windows(self):
        panel = build_mass_reference_panel(
            _reference(), polarity="pos", panel_name="Unlabelled IS",
            unlabelled_standards=UNLABELLED,
        )
        features = pd.DataFrame([
            {"feature_id": "good", "mz": 500.001, "rt": 5.1, "intensity": 1000},
            {"feature_id": "rt-fail", "mz": 500.001, "rt": 6.0, "intensity": 900},
            {"feature_id": "ppm-fail", "mz": 500.010, "rt": 5.0, "intensity": 800},
        ])

        found = detect_mass_calibration_candidates(
            features, panel, ppm_tol=5.0, rt_tol=0.2
        )

        self.assertEqual(list(found["candidates"]["feat_feature_id"]), ["good"])

    def test_np_panel_uses_np_internal_standard_rt_library(self):
        rp_panel = build_mass_reference_panel(
            _reference(), polarity="pos", panel_name="Unlabelled IS",
            unlabelled_standards=UNLABELLED, chromatography="RP",
        ).set_index("standard")
        np_panel = build_mass_reference_panel(
            _reference(), polarity="pos", panel_name="Unlabelled IS",
            unlabelled_standards=UNLABELLED, chromatography="NP",
        ).set_index("standard")

        self.assertEqual(rp_panel.loc["CL(56:0)", "ref_rt"], 9.0)
        self.assertEqual(np_panel.loc["CL(56:0)", "ref_rt"], 8.1)

    def test_np_mass_search_rejects_rp_rt_candidate(self):
        panel = build_mass_reference_panel(
            _reference(), polarity="pos", panel_name="Unlabelled IS",
            unlabelled_standards=UNLABELLED, chromatography="NP",
        )
        features = pd.DataFrame([
            {"feature_id": "np", "mz": 900.0, "rt": 8.15, "intensity": 1000},
            {"feature_id": "rp", "mz": 900.0, "rt": 9.0, "intensity": 900},
        ])

        found = detect_mass_calibration_candidates(
            features, panel, ppm_tol=5.0, rt_tol=0.2
        )

        self.assertEqual(list(found["candidates"]["feat_feature_id"]), ["np"])

    def test_np_labelled_panel_uses_rt_np_not_rp(self):
        panel = build_mass_reference_panel(
            _reference(), polarity="pos", panel_name="Labelled IS",
            unlabelled_standards={}, chromatography="NP",
        ).set_index("standard")

        self.assertEqual(panel.loc["PC(33:1)-D7", "ref_rt"], 8.629)
        self.assertEqual(panel.loc["LPC(18:1)-D7", "ref_rt"], 9.7)
        self.assertTrue(np.isnan(panel.loc["CE(16:0)-D7", "ref_rt"]))

    def test_ccs_search_retains_mass_rt_match_without_reference_ccs(self):
        panel = build_ccs_reference_panel(
            _reference(), polarity="pos", panel_name="Unlabelled IS",
            unlabelled_standards=UNLABELLED,
        )
        features = pd.DataFrame([{
            "feature_id": "cl", "mz": 900.0, "rt": 9.0,
            "mobility": 1.1, "intensity": 1000,
        }])

        found = detect_ccs_calibration_candidates(
            features, panel, ppm_tol=5.0, rt_tol=0.2
        )

        self.assertTrue(found["candidates"].empty)
        self.assertEqual(list(found["rt_candidates"]["standard"]), ["CL(56:0)"])
        self.assertEqual(found["ccs_unavailable"], ["CL(56:0)"])
        self.assertNotIn("CL(56:0)", found["missing"])


class MassCalibrationModelTests(unittest.TestCase):
    @staticmethod
    def _anchors(count: int = 6) -> pd.DataFrame:
        observed = np.linspace(300.0, 1000.0, count)
        reference = 1.000004 * observed - 0.0005
        return pd.DataFrame({
            "standard": [f"S{i}" for i in range(count)],
            "feat_feature_id": [f"F{i}" for i in range(count)],
            "feat_mz": observed,
            "ref_mz": reference,
        })

    def test_every_presented_model_fits_and_applies(self):
        for model in MASS_CALIBRATION_MODELS.values():
            with self.subTest(model=model):
                calibration = fit_confirmed_mass_calibration(
                    self._anchors(), model=model
                )
                corrected = apply_mass_calibration(
                    self._anchors()["feat_mz"], calibration
                )
                self.assertEqual(calibration["kind"], model)
                self.assertTrue(np.isfinite(corrected).all())
                self.assertLess(
                    calibration["rmse_after_ppm"],
                    calibration["rmse_before_ppm"],
                )

    def test_quadratic_rejects_a_narrow_calibration_range(self):
        anchors = self._anchors()
        anchors["feat_mz"] = np.linspace(500.0, 550.0, len(anchors))
        anchors["ref_mz"] = anchors["feat_mz"] * 1.000004

        calibration = fit_confirmed_mass_calibration(anchors, model="quadratic")

        self.assertEqual(calibration["kind"], "none")
        self.assertIn("spanning at least 100 m/z", calibration["label"])

    def test_laser_model_is_evaluated_on_independent_lipid_targets(self):
        panel = build_lipid_validation_reference_panel("pos").iloc[:4]
        observed = panel["ref_mz"].to_numpy() * (1.0 - 10e-6)
        features = pd.DataFrame({
            "feature_id": [f"V{i}" for i in range(len(panel))],
            "mz": observed,
            "rt": panel["ref_rt"].to_numpy(),
            "intensity": np.linspace(1_000_000, 4_000_000, len(panel)),
        })
        anchors = pd.DataFrame({
            "standard": [f"A{i}" for i in range(len(panel))],
            "feat_feature_id": [f"A{i}" for i in range(len(panel))],
            "feat_mz": observed,
            "ref_mz": panel["ref_mz"].to_numpy(),
        })
        calibration = fit_confirmed_mass_calibration(
            anchors, model="constant_ppm"
        )

        evaluation = evaluate_lipid_validation_panel(
            features, calibration, polarity="pos", ppm_tol=20.0
        )

        self.assertEqual(len(evaluation["details"]), 4)
        self.assertAlmostEqual(
            evaluation["median_absolute_ppm_before"], 10.0, places=4
        )
        self.assertLess(evaluation["median_absolute_ppm_after"], 0.001)
        self.assertEqual(evaluation["status"], "PASS")
        self.assertEqual(evaluation["polarity"], "pos")

    def test_lipid_validation_requires_three_detected_targets_to_pass(self):
        panel = build_lipid_validation_reference_panel("neg").iloc[:2]
        features = pd.DataFrame({
            "feature_id": ["V0", "V1"],
            "mz": panel["ref_mz"].to_numpy(),
            "rt": panel["ref_rt"].to_numpy(),
            "intensity": [1_000_000, 2_000_000],
        })
        calibration = {
            "kind": "constant_ppm",
            "model": "constant_ppm",
            "bias_ppm": 0.0,
        }

        evaluation = evaluate_lipid_validation_panel(
            features, calibration, polarity="NEG", ppm_tol=20.0
        )

        self.assertEqual(len(evaluation["details"]), 2)
        self.assertEqual(evaluation["status"], "REVIEW")
        self.assertEqual(evaluation["polarity"], "neg")


class CalibrationOrderingTests(unittest.TestCase):
    def test_confirmed_mass_model_is_applied_before_ccs_candidate_search(self):
        reference = _reference().iloc[:2].copy()
        observed_mz = np.asarray([500.005, 700.007])
        features = pd.DataFrame({
            "feature_id": ["dg", "tg"],
            "mz": observed_mz,
            "rt": [5.0, 7.0],
            "mobility": [
                float(ccs_to_one_over_k0(220.0, 500.0, 1)),
                float(ccs_to_one_over_k0(280.0, 700.0, 1)),
            ],
            "intensity": [1000.0, 900.0],
        })
        config = {
            "mass": {"mode": "auto", "model": "constant_ppm"},
            "ccs": {"mode": "auto"},
            "rt": {"mode": "none"},
            "mass_reference_panel": "Unlabelled IS",
            "ccs_reference_panel": "Unlabelled IS",
            "mass_search_ppm": 20.0,
            "mass_search_rt_min": 0.2,
            "detection_ppm": 3.0,
            "detection_rt_min": 0.2,
        }

        ccs, _, mass, audit = automatic_calibrations(
            features, reference, "pos", config
        )

        self.assertEqual(mass["kind"], "constant_ppm")
        self.assertEqual(ccs["kind"], "linear")
        self.assertEqual(audit["detected_ccs_candidates"], 2)


if __name__ == "__main__":
    unittest.main()
