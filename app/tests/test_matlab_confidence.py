from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

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
