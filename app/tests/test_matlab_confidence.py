from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from matlab_annotation.composition_plausibility import (
    composition_features, fit_composition_priors,
)
from matlab_annotation.confidence import (
    PostRegressionCompletionSettings,
    complete_post_regression_annotations,
)


def _model(path: Path) -> Path:
    numeric = {
        name: {"mean": 0.0, "scale": 1.0, "coefficient": 0.0}
        for name in (
            "abs_ppm", "abs_rt_seconds", "coherence",
            "stream_intensity_z", "class_intensity_z",
            "feature_candidates", "identity_features",
        )
    }
    payload = {
        "format": "laser-post-regression-confidence-v1",
        "candidate_limits": {
            "ppm": 3.0,
            "rt_seconds": 10.0,
            "require_grid_coherence_pass": True,
            "monoisotopic_only": True,
        },
        "intercept": -2.0,
        "numeric": numeric,
        "categorical": {
            "stream": {"RP_Pos": 0.0},
            "lipid_class": {"PC": 4.0, "PE": 2.05},
            "adduct": {"[M+H]+": 0.0},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _candidates() -> pd.DataFrame:
    return pd.DataFrame({
        "feat_feature_id": ["1", "1", "2", "3"],
        "name": ["PC(34:1)", "PE(37:1)", "PC(36:2)", "PC(38:4)"],
        "class": ["PC", "PE", "PC", "PC"],
        "adduct": ["[M+H]+"] * 4,
        "matlab_code": [1, 2, 3, 4],
        "isotopologue": [""] * 4,
        "ppm_error": [0.2, 0.2, 3.0, 0.2],
        "rt_error": [0.01, 0.01, 0.01, 0.01],
        "grid_coherence_score": [0.9] * 4,
        "grid_coherence_pass": [True, True, True, False],
        "feat_intensity": [1000.0, 1000.0, 1000.0, 1000.0],
        "is_is": [False] * 4,
    })


class MatlabConfidenceCompletionTests(unittest.TestCase):
    def test_v2_replaces_a_call_without_increasing_the_count(self):
        candidates = _candidates().iloc[[0, 1]].copy()
        candidates["feat_feature_id"] = ["1", "2"]
        with tempfile.TemporaryDirectory() as temporary:
            path = _model(Path(temporary) / "model.json")
            payload = json.loads(path.read_text())
            payload["categorical"]["lipid_class"] = {"PC": 4.0, "PE": -4.0}
            reference = json.loads(json.dumps(payload))
            payload["format"] = "laser-post-regression-confidence-v2"
            payload["categorical"]["lipid_class"] = {"PC": -4.0, "PE": 4.0}
            payload["count_reference_model"] = reference
            path.write_text(json.dumps(payload))
            accepted, proposals, audit = complete_post_regression_annotations(
                pd.DataFrame(), candidates, stream="RP_Pos",
                settings=PostRegressionCompletionSettings(model_path=path),
            )
        self.assertEqual(accepted["name"].tolist(), ["PE(37:1)"])
        self.assertEqual(int(audit.iloc[0]["baseline_count_target"]), 1)
        self.assertEqual(int(proposals["automatic_accepted"].sum()), 1)
        self.assertEqual(proposals.iloc[0]["auto_annotation_status"],
                         "review_only_blue_candidate")

    def test_v2_retains_supported_baseline_calls_instead_of_forcing_new_calls(self):
        candidates = _candidates().iloc[[0, 1]].copy()
        candidates["feat_feature_id"] = ["1", "2"]
        with tempfile.TemporaryDirectory() as temporary:
            path = _model(Path(temporary) / "model.json")
            payload = json.loads(path.read_text())
            payload["categorical"]["lipid_class"] = {"PC": 4.0, "PE": 4.0}
            reference = json.loads(json.dumps(payload))
            payload["format"] = "laser-post-regression-confidence-v2"
            payload["categorical"]["lipid_class"] = {"PC": 4.0, "PE": -4.0}
            payload["count_reference_model"] = reference
            path.write_text(json.dumps(payload))
            accepted, _proposals, audit = complete_post_regression_annotations(
                pd.DataFrame(), candidates, stream="RP_Pos",
                settings=PostRegressionCompletionSettings(model_path=path),
            )
        self.assertEqual(set(accepted["name"]), {"PC(34:1)", "PE(37:1)"})
        self.assertEqual(int(audit.iloc[0]["baseline_calls_retained"]), 1)
        retained = accepted[accepted["post_regression_count_retained"]]
        self.assertEqual(retained["auto_annotation_status"].tolist(),
                         ["baseline_count_retained"])

    def test_composition_is_graded_and_does_not_ban_novel_or_odd_chains(self):
        training = pd.DataFrame({
            "lipid_class": ["PC"] * 20,
            "carbon": [32, 34, 36, 38, 40] * 4,
            "db": [0, 1, 2, 3, 4] * 4,
        })
        priors = fit_composition_priors(training)
        candidates = pd.DataFrame({
            "lipid_class": ["PC", "PC", "PC", "novel_class"],
            "carbon": [36, 36, 37, 37], "db": [2, 12, 2, 2],
        })
        features = composition_features(candidates, priors)
        self.assertLess(features.iloc[0]["composition_distance"],
                        features.iloc[1]["composition_distance"])
        self.assertEqual(features.iloc[2]["odd_carbon"], 1)
        self.assertGreater(features.iloc[2]["composition_distance"], 0)
        self.assertEqual(features.iloc[3]["composition_distance"], 0)
        self.assertFalse(features.isna().any().any())

    def test_v2_reserves_standards_without_count_inflation(self):
        candidates = _candidates().iloc[[0, 1]].copy()
        candidates["feat_feature_id"] = ["1", "2"]
        existing = candidates.iloc[[0]].copy()
        existing["is_is"] = True
        existing["auto_annotation_status"] = "matlab_mapper_accepted"
        with tempfile.TemporaryDirectory() as temporary:
            path = _model(Path(temporary) / "model.json")
            payload = json.loads(path.read_text())
            payload["categorical"]["lipid_class"] = {"PC": 4.0, "PE": 4.0}
            reference = json.loads(json.dumps(payload))
            payload["format"] = "laser-post-regression-confidence-v2"
            payload["categorical"]["lipid_class"] = {"PC": -4.0, "PE": 4.0}
            payload["count_reference_model"] = reference
            path.write_text(json.dumps(payload))
            accepted, _proposals, audit = complete_post_regression_annotations(
                existing, candidates, stream="RP_Pos",
                settings=PostRegressionCompletionSettings(model_path=path),
            )
        self.assertEqual(len(accepted), 2)
        self.assertEqual(int(audit.iloc[0]["baseline_count_target"]), 2)
        self.assertEqual(int(accepted["is_is"].fillna(False).sum()), 1)

    def test_v2_does_not_inflate_a_zero_baseline_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _model(Path(temporary) / "model.json")
            payload = json.loads(path.read_text())
            reference = json.loads(json.dumps(payload))
            reference["categorical"]["lipid_class"] = {"PC": -4.0, "PE": -4.0}
            payload["format"] = "laser-post-regression-confidence-v2"
            payload["count_reference_model"] = reference
            path.write_text(json.dumps(payload))
            accepted, proposals, audit = complete_post_regression_annotations(
                pd.DataFrame(), _candidates(), stream="RP_Pos",
                settings=PostRegressionCompletionSettings(model_path=path),
            )
        self.assertTrue(accepted.empty)
        self.assertFalse(proposals["automatic_accepted"].any())
        self.assertEqual(int(audit.iloc[0]["baseline_count_target"]), 0)

    def test_promotes_only_strict_unambiguous_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = _model(Path(temporary) / "model.json")
            accepted, candidates, audit = complete_post_regression_annotations(
                pd.DataFrame(),
                _candidates(),
                stream="RP_Pos",
                settings=PostRegressionCompletionSettings(
                    model_path=model,
                    confidence_threshold=0.5,
                    ambiguity_margin=0.1,
                ),
            )

        self.assertEqual(accepted["name"].tolist(), ["PC(34:1)"])
        self.assertEqual(
            accepted.iloc[0]["auto_annotation_status"],
            "curated_cohort_supported",
        )
        self.assertEqual(int(candidates["automatic_accepted"].sum()), 1)
        self.assertEqual(int(audit.iloc[0]["promoted_annotations"]), 1)

    def test_protected_internal_standard_is_not_demoted(self):
        existing = _candidates().iloc[[2]].copy()
        existing["is_is"] = True
        existing["auto_annotation_status"] = "matlab_mapper_accepted"
        with tempfile.TemporaryDirectory() as temporary:
            model = _model(Path(temporary) / "model.json")
            accepted, _candidates_out, audit = (
                complete_post_regression_annotations(
                    existing,
                    _candidates(),
                    stream="RP_Pos",
                    settings=PostRegressionCompletionSettings(model_path=model),
                )
            )

        self.assertIn("PC(36:2)", accepted["name"].tolist())
        self.assertEqual(int(audit.iloc[0]["final_annotations"]), 2)


if __name__ == "__main__":
    unittest.main()
