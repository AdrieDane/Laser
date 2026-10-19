from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from matlab_annotation.fitted_grid_completion import (
    FittedGridCompletionSettings, propose_missing_pairs,
    refine_fitted_grid_annotations, score_independent_grid_support,
)


LIMITS = {"ppm": 3.0, "rt_seconds": 10.0, "require_grid_coherence_pass": True}


def fixture():
    baseline = pd.DataFrame({
        "feat_feature_id": ["1", "2", "3"],
        "name": ["PC(34:1)", "PC(38:1)", "PC(36:6)"],
        "class": ["PC"] * 3, "adduct": ["[M+H]+"] * 3,
        "carbon": [34, 38, 36], "db": [1, 1, 6],
        "matlab_code": [1, 2, 3], "matlab_compound": [1] * 3,
        "ppm_error": [0.0] * 3, "rt_error": [0.0] * 3,
        "feat_intensity": [1000.0] * 3,
        "is_is": [False] * 3, "isotopologue": [""] * 3,
        "grid_coherence_score": [0.9] * 3, "grid_coherence_pass": [True] * 3,
        "automatic_accepted": [True] * 3,
        "auto_annotation_status": ["curated_cohort_supported"] * 3,
    })
    reference = pd.DataFrame({
        "name": ["PC(35:1)"], "class": ["PC"], "adduct": ["[M+H]+"],
        "carbon": [35], "db": [1], "matlab_code": [4],
        "matlab_compound": [1], "ref_mz": [714.0], "ref_rt": [10.25],
    })
    features = pd.DataFrame({
        "feature_id": ["4"], "mz": [714.0], "rt": [10.25],
        "intensity": [10000.0],
    })
    return baseline, reference, features


def model(path):
    payload = {
        "format": "laser-fitted-grid-confidence-v1",
        "candidate_limits": LIMITS,
        "intercept": 2.0, "numeric": {},
        "categorical": {"class_carbon": {"PC|36": -10.0}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class FittedGridCompletionTests(unittest.TestCase):
    def test_recovers_omitted_odd_carbon_pair_at_the_same_count(self):
        baseline, reference, features = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = model(Path(directory) / "model.json")
            accepted, candidates, audit = refine_fitted_grid_annotations(
                baseline, baseline, reference, features, stream="RP_Pos",
                settings=FittedGridCompletionSettings(model_path=path),
            )
        self.assertEqual(len(accepted), len(baseline))
        self.assertIn("PC(35:1)", accepted["name"].tolist())
        self.assertNotIn("PC(36:6)", accepted["name"].tolist())
        self.assertFalse(accepted["feat_feature_id"].duplicated().any())
        self.assertEqual(int(audit.iloc[0]["selected_completed_pairs"]), 1)
        self.assertEqual(int(candidates["automatic_accepted"].sum()), 3)

    def test_new_proposals_cannot_support_one_another(self):
        baseline, reference, features = fixture()
        proposals = propose_missing_pairs(
            baseline, reference, features, stream="RP_Pos", limits=LIMITS,
        )
        second = proposals.copy()
        second["carbon"] = 36
        second["name"] = "PC(36:1)"
        scored = score_independent_grid_support(
            pd.concat([proposals, second], ignore_index=True), baseline.iloc[[0]],
            limits=LIMITS, threshold=0.45,
        )
        self.assertFalse(scored["grid_coherence_pass"].any())
        self.assertFalse(scored["grid_coherence_assessed"].any())

    def test_preserves_the_mass_rt_and_positional_safeguards(self):
        baseline, reference, features = fixture()
        bad_mass = features.assign(feature_id="5", mz=714.0 * (1 + 3.1e-6))
        bad_rt = features.assign(feature_id="6", rt=10.25 + 10.1 / 60.0)
        positional = reference.assign(name="1-acylLPC(35:1)", **{"class": "1-acylLPC"})
        wrong_adduct = reference.assign(name="PC(35:1)", adduct="[M+Na]+")
        proposed = propose_missing_pairs(
            baseline, pd.concat([reference, positional, wrong_adduct], ignore_index=True),
            pd.concat([features, bad_mass, bad_rt], ignore_index=True),
            stream="RP_Pos", limits=LIMITS,
        )
        self.assertEqual(proposed["feat_feature_id"].tolist(), ["4"])
        self.assertEqual(proposed["name"].tolist(), ["PC(35:1)"])

    def test_similarly_supported_features_do_not_create_a_new_full_identity(self):
        baseline, reference, features = fixture()
        competing = features.assign(feature_id="5", mz=714.0001)
        with tempfile.TemporaryDirectory() as directory:
            path = model(Path(directory) / "model.json")
            accepted, _candidates, audit = refine_fitted_grid_annotations(
                baseline, baseline, reference,
                pd.concat([features, competing], ignore_index=True),
                stream="RP_Pos",
                settings=FittedGridCompletionSettings(model_path=path),
            )
        self.assertEqual(set(accepted["name"]), set(baseline["name"]))
        self.assertEqual(int(audit.iloc[0]["baseline_calls_retained"]), 1)

    def test_declared_standard_remains_reserved(self):
        baseline, reference, features = fixture()
        baseline.loc[2, "is_is"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = model(Path(directory) / "model.json")
            accepted, _candidates, _audit = refine_fitted_grid_annotations(
                baseline, baseline, reference, features, stream="RP_Pos",
                settings=FittedGridCompletionSettings(model_path=path),
            )
        self.assertEqual(len(accepted), 3)
        self.assertIn("3", accepted["feat_feature_id"].tolist())
        self.assertEqual(int(accepted["is_is"].fillna(False).sum()), 1)


if __name__ == "__main__":
    unittest.main()
