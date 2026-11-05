"""RT model regression tests copied from LASER-4D a188f72."""
import unittest
import pandas as pd
from annotator.rt_calibration import (fit_confirmed_internal_standard_rt_calibration,
    apply_internal_standard_rt_calibration)

class RtCalibrationModelsTests(unittest.TestCase):
    def test_five_confirmed_standards_fit_linear_rt_reference_adjustment(self):
        confirmed = pd.DataFrame({
            "standard": ["LPC", "PC", "PE", "DG", "TG"],
            "ref_rt": [2.0, 4.0, 6.0, 8.0, 10.0],
            "feat_rt": [2.4, 4.6, 6.8, 9.0, 11.2],
        })

        calibration = fit_confirmed_internal_standard_rt_calibration(confirmed)
        adjusted = apply_internal_standard_rt_calibration([5.0], calibration)

        self.assertEqual(calibration["kind"], "linear")
        self.assertEqual(calibration["n_used"], 5)
        self.assertAlmostEqual(calibration["slope"], 1.1, places=12)
        self.assertAlmostEqual(calibration["intercept"], 0.2, places=12)
        self.assertAlmostEqual(adjusted[0], 5.7, places=12)
        self.assertGreater(calibration["rmse_before_min"], 0.0)
        self.assertAlmostEqual(calibration["rmse_after_min"], 0.0, places=12)

    def test_constant_shift_rt_model_uses_median_offset(self):
        confirmed = pd.DataFrame({
            "standard": ["early", "middle", "late"],
            "ref_rt": [2.0, 6.0, 10.0],
            "feat_rt": [2.2, 6.2, 10.8],
        })

        calibration = fit_confirmed_internal_standard_rt_calibration(
            confirmed, model="constant_shift"
        )
        adjusted = apply_internal_standard_rt_calibration([1.0, 8.0], calibration)

        self.assertEqual(calibration["kind"], "constant_shift")
        self.assertAlmostEqual(calibration["intercept"], 0.2, places=12)
        self.assertAlmostEqual(adjusted[0], 1.2, places=12)
        self.assertAlmostEqual(adjusted[1], 8.2, places=12)
        self.assertTrue(pd.notna(calibration["loo_rmse_min"]))

    def test_robust_linear_rt_model_downweights_outlier(self):
        confirmed = pd.DataFrame({
            "standard": ["A", "B", "C", "D", "E"],
            "ref_rt": [1.0, 3.0, 5.0, 7.0, 9.0],
            "feat_rt": [1.3, 3.7, 8.1, 8.5, 10.9],
        })

        calibration = fit_confirmed_internal_standard_rt_calibration(
            confirmed, model="robust_linear"
        )

        self.assertEqual(calibration["kind"], "robust_linear")
        self.assertLess(calibration["anchors"]["model_weight"].min(), 0.5)
        self.assertAlmostEqual(calibration["slope"], 1.2, places=2)
        self.assertAlmostEqual(calibration["intercept"], 0.1, delta=0.05)

    def test_monotonic_piecewise_rt_model_interpolates_and_bounds_extrapolation(self):
        confirmed = pd.DataFrame({
            "standard": list("ABCDEF"),
            "ref_rt": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
            "feat_rt": [1.2, 3.4, 5.9, 8.0, 10.3, 12.5],
        })

        calibration = fit_confirmed_internal_standard_rt_calibration(
            confirmed, model="monotonic_piecewise"
        )
        adjusted = apply_internal_standard_rt_calibration(
            [0.0, 4.0, 12.0], calibration
        )

        self.assertEqual(calibration["kind"], "monotonic_piecewise")
        self.assertEqual(calibration["reference_knots"], [1, 3, 5, 7, 9, 11])
        self.assertAlmostEqual(adjusted[0], 0.2, places=12)
        self.assertAlmostEqual(adjusted[1], 4.65, places=12)
        self.assertAlmostEqual(adjusted[2], 13.5, places=12)
        self.assertTrue(pd.notna(calibration["loo_rmse_min"]))

    def test_monotonic_piecewise_rt_model_rejects_elution_order_inversion(self):
        confirmed = pd.DataFrame({
            "standard": list("ABCDEF"),
            "ref_rt": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
            "feat_rt": [1.2, 3.4, 8.1, 7.9, 10.3, 12.5],
        })

        calibration = fit_confirmed_internal_standard_rt_calibration(
            confirmed, model="monotonic_piecewise"
        )

        self.assertEqual(calibration["kind"], "none")
        self.assertIn("elution order", calibration["label"])

    def test_monotonic_piecewise_rt_model_rejects_duplicate_reference_rt(self):
        confirmed = pd.DataFrame({
            "standard": list("ABCDEFG"),
            "ref_rt": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 11.0],
            "feat_rt": [1.2, 3.4, 5.9, 8.0, 10.3, 12.5, 12.8],
        })

        calibration = fit_confirmed_internal_standard_rt_calibration(
            confirmed, model="monotonic_piecewise"
        )

        self.assertEqual(calibration["kind"], "none")
        self.assertIn("distinct reference RT", calibration["label"])
