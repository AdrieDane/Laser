"""048 engineering controls; no fitting, new cutoffs or QE accuracy experiment."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import catboost_experiment_data as data
import catboost_evidence_model as backend
import run_qe_catboost_model_experiment as runner
from verify_qe_catboost_experiment import fixed_winners, select, counts
from report_qe_catboost_decision import decide


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_frozen_resources(REPO)

    def test_independent_graph_matches_native_with_duplicates_ties(self):
        from matlab_annotation.guarded_class_operating_point import (
            clear_identity_winners,
        )
        from catboost_operating_points import select_candidates

        rng = np.random.default_rng(48)
        for trial in range(30):
            n = 120
            pool = pd.DataFrame(
                dict(
                    stream="RP_Pos",
                    **{"class": "PE"},
                    name=["PE" + str(i) for i in rng.integers(0, 30, n)],
                    adduct="[M+H]+",
                    feat_feature_id=rng.integers(0, 50, n).astype(str),
                    feat_intensity=rng.integers(0, 5, n).astype(float),
                )
            )
            pool.index = pool.index * 2 + trial
            scores = pd.Series(rng.integers(0, 21, n) / 20, index=pool.index)
            a, b = fixed_winners(pool, scores), clear_identity_winners(pool, scores)
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            for cut in (0, 0.025, 0.5, 0.95, 1):
                assert select(pool, scores, cut, {}).index.equals(
                    select_candidates(pool, scores, baseline_threshold=cut).index
                )

    def test_no_fit_gate_blocks_stageA(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t)
            data.dump(
                path / "receipt.json",
                dict(all_passed=True, actual_training_fit_authorized=False),
            )
            with (
                patch.object(data, "PREFLIGHT", path),
                patch.object(backend, "fit_four_modes") as fit,
            ):
                with self.assertRaises(AssertionError):
                    runner.initialize()
                fit.assert_not_called()

    def test_no_output_until_all_five_calibrations_frozen(self):
        with tempfile.TemporaryDirectory() as t, patch.object(data, "OUT", Path(t)):
            data.dump(data.OUT / "protocol.json", {})
            h = data.sha(data.OUT / "protocol.json")
            for o in range(4):
                p = data.OUT / "calibration" / f"fold_{o}"
                p.mkdir(parents=True)
                data.dump(p / "operating_points.json", dict(protocol_sha256=h))
            with self.assertRaises(FileNotFoundError):
                runner.freeze_all_calibrations()
            p = data.OUT / "calibration/fold_4"
            p.mkdir()
            data.dump(p / "operating_points.json", dict(protocol_sha256=h))
            with self.assertRaises(FileNotFoundError):
                runner.require_all_calibrations()
            self.assertTrue(runner.freeze_all_calibrations())
            data.dump(
                p / "operating_points.json", dict(protocol_sha256=h, changed=True)
            )
            with self.assertRaises(AssertionError):
                runner.require_all_calibrations()

    def test_model_hash_required_before_unpickle(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "model.pkl"
            data.dump(p, {"not": "a model"})
            data.dump(p.with_suffix(".complete.json"), dict(model_sha256="wrong"))
            with (
                patch.object(data, "parent_path", return_value=p),
                patch.object(pd, "read_pickle") as read,
            ):
                with self.assertRaises(AssertionError):
                    runner.read_model(0, None)
                read.assert_not_called()

    def test_pause_at_model_boundary_does_not_calibrate_or_annotate(self):
        class ImmediateExecutor:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def map(self, func, items):
                return map(func, items)

        with (
            patch.object(runner, "initialize", return_value={"numerical_workers": 2}),
            patch.object(runner, "ProcessPoolExecutor", ImmediateExecutor),
            patch.object(runner, "fit_parent", return_value={"paused": True}),
            patch.object(runner, "pause_requested", return_value=True),
            patch.object(runner, "checkpoint") as checkpoint,
            patch.object(runner, "calibrate") as calibrate,
            patch.object(runner, "annotate") as annotate,
        ):
            runner.main()
            checkpoint.assert_called_once_with("model_parents")
            calibrate.assert_not_called()
            annotate.assert_not_called()

    def test_disabled_one_counts_nothing_even_score_one(self):
        table = pd.DataFrame(
            dict(
                class_selection_score=[1.0],
                training_positive=[True],
                calibration_identity_in_manual=[True],
                training_label_known=[True],
            )
        )
        self.assertEqual(counts(table, 1), dict(correct=0, extras=0, different_peak=0))

    def test_no_outer_winner_splice_even_large_pooled_gain(self):
        summary = dict(
            correct_before=800,
            correct_after=950,
            extras_before=10,
            extras_after=10,
            different_feature_before=5,
            different_feature_after=5,
        )
        priority = pd.DataFrame(
            dict(
                optimization_priority=["prioritize"] * 2,
                manual_rows_before=[100, 100],
                manual_rows_after=[100, 100],
                manual_same_peak_before=[91, 60],
                manual_same_peak_after=[89, 90],
            )
        )
        std = dict(
            correct_before=10,
            correct_after=10,
            extras_before=0,
            extras_after=0,
            different_feature_before=0,
            different_feature_after=0,
        )
        decision = decide(summary, priority, std)
        self.assertFalse(decision["retain_whole_method"])
        self.assertFalse(decision["guards"]["no_already90_priority_class_falls_below"])
        self.assertFalse(
            decision["guards"]["no_priority_class_loses_over_one_percentage_point"]
        )


if __name__ == "__main__":
    unittest.main()
