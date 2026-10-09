import unittest

import numpy as np
import pandas as pd

from annotator.grid_coherence import score_grid_coherence


def _candidate(
    feature_id, carbon, db, ppm, rt, ccs, lipid_class="PC",
):
    return {
        "feat_feature_id": str(feature_id),
        "name": f"{lipid_class}({carbon}:{db})",
        "class": lipid_class,
        "grid_class": lipid_class,
        "adduct": "[M+H]+",
        "carbon": carbon,
        "db": db,
        "ppm_error": ppm,
        "rt_error": rt,
        "ccs_error_percent": ccs,
    }


class GridCoherenceTests(unittest.TestCase):
    def test_local_outlier_inside_hard_windows_is_flagged(self):
        data = pd.DataFrame([
            _candidate(1, 32, 0, 1.0, 0.10, 2.0),
            _candidate(2, 34, 0, 1.1, 0.11, 2.1),
            _candidate(3, 36, 0, 0.9, 0.09, 1.9),
            _candidate(4, 34, 1, 4.8, 0.48, -4.8),
        ])
        scored = score_grid_coherence(
            data, 5.0, 0.5, 5.0, threshold=0.35
        ).set_index("feat_feature_id")
        self.assertTrue(bool(scored.loc["1", "grid_coherence_pass"]))
        self.assertTrue(bool(scored.loc["4", "grid_coherence_assessed"]))
        self.assertTrue(bool(scored.loc["4", "grid_coherence_outlier"]))
        self.assertLess(scored.loc["4", "grid_coherence_score"], 0.35)

    def test_constant_dataset_offset_is_coherent(self):
        data = pd.DataFrame([
            _candidate(1, 32, 0, 2.0, 0.30, 4.0),
            _candidate(2, 34, 0, 2.1, 0.31, 4.1),
            _candidate(3, 36, 0, 1.9, 0.29, 3.9),
        ])
        scored = score_grid_coherence(data, 5.0, 0.5, 5.0)
        self.assertTrue(scored["grid_coherence_assessed"].all())
        self.assertTrue(scored["grid_coherence_pass"].all())
        self.assertGreater(float(scored["grid_coherence_score"].min()), 0.9)

    def test_duplicate_same_node_does_not_supply_neighbour_support(self):
        data = pd.DataFrame([
            _candidate(1, 34, 0, 1.0, 0.10, 2.0),
            _candidate(2, 34, 0, 1.2, 0.11, 2.1),
            _candidate(3, 36, 0, 1.0, 0.10, 2.0),
        ])
        scored = score_grid_coherence(data, 5.0, 0.5, 5.0)
        self.assertFalse(scored["grid_coherence_assessed"].any())
        self.assertTrue(scored["grid_coherence_pass"].all())

    def test_missing_ccs_falls_back_to_mz_and_rt(self):
        data = pd.DataFrame([
            _candidate(1, 32, 0, 1.0, 0.10, np.nan),
            _candidate(2, 34, 0, 1.1, 0.11, np.nan),
            _candidate(3, 36, 0, 0.9, 0.09, np.nan),
        ])
        scored = score_grid_coherence(
            data, 5.0, 0.5, 5.0, use_ccs=True
        )
        self.assertTrue(scored["grid_coherence_assessed"].all())
        self.assertTrue(scored["grid_coherence_pass"].all())
        self.assertTrue(scored["grid_coherence_dimensions"].eq(2).all())

    def test_sparse_edge_candidate_is_unassessed_and_passes(self):
        data = pd.DataFrame([
            _candidate(1, 30, 0, 4.9, 0.49, 4.9),
            _candidate(2, 50, 10, -4.9, -0.49, -4.9),
            _candidate(3, 70, 20, 0.0, 0.0, 0.0),
        ])
        scored = score_grid_coherence(data, 5.0, 0.5, 5.0)
        self.assertFalse(scored["grid_coherence_assessed"].any())
        self.assertTrue(scored["grid_coherence_pass"].all())


def _reference_scores(
    data, ppm_tolerance, rt_tolerance, ccs_tolerance_percent,
    use_ccs=True, threshold=0.35, min_neighbors=2, max_neighbors=6,
    max_grid_distance=2.5,
):
    """Straightforward per-candidate reference implementation.

    Deliberately written as plain nested loops so it can be read against the
    module docstring. The production scorer must agree with it exactly.
    """
    scored = data.copy()
    scored["_ppm"] = scored["ppm_error"] / ppm_tolerance
    scored["_rt"] = scored["rt_error"] / rt_tolerance
    scored["_ccs"] = (
        scored["ccs_error_percent"] / ccs_tolerance_percent
        if use_ccs and "ccs_error_percent" in scored else np.nan
    )
    scored["_origin"] = np.sqrt(
        scored["_ppm"].fillna(0.0) ** 2
        + scored["_rt"].fillna(0.0) ** 2
        + scored["_ccs"].fillna(0.0) ** 2
    )
    results = {}
    for _, group in scored.groupby(["grid_class", "adduct"], sort=False):
        representatives = (
            group.sort_values("_origin", kind="stable")
            .drop_duplicates(["carbon", "db"], keep="first")
        )
        if len(representatives) <= min_neighbors:
            continue
        for index, row in group.iterrows():
            neighbours = representatives[
                ~(
                    representatives["carbon"].eq(row["carbon"])
                    & representatives["db"].eq(row["db"])
                )
            ].copy()
            neighbours["_distance"] = np.sqrt(
                ((neighbours["carbon"] - row["carbon"]) / 2.0) ** 2
                + (neighbours["db"] - row["db"]) ** 2
            )
            neighbours = (
                neighbours[neighbours["_distance"].le(max_grid_distance)]
                .sort_values("_distance", kind="stable")
                .head(max_neighbors)
            )
            similarities, weights, dimensions = [], [], []
            for _, neighbour in neighbours.iterrows():
                differences = [
                    float(row[column] - neighbour[column])
                    for column in ("_ppm", "_rt", "_ccs")
                    if np.isfinite(row[column]) and np.isfinite(neighbour[column])
                ]
                if len(differences) < 2:
                    continue
                similarities.append(
                    float(np.exp(-0.5 * float(np.linalg.norm(differences)) ** 2))
                )
                weights.append(float(np.exp(-0.5 * neighbour["_distance"] ** 2)))
                dimensions.append(len(differences))
            if len(similarities) < min_neighbors:
                continue
            results[index] = (
                float(np.average(similarities, weights=weights)),
                len(similarities),
                int(round(float(np.average(dimensions, weights=weights)))),
            )
    return results


class GridCoherenceVectorisationTests(unittest.TestCase):
    """The scorer is array-based; it must still match a plain-loop reference."""

    def _dataset(self, seed):
        rng = np.random.default_rng(seed)
        rows = []
        feature_id = 0
        for lipid_class in ("PC", "PE", "TG"):
            for carbon in range(28, 46, 2):
                for db in range(0, 5):
                    if rng.random() < 0.35:
                        continue
                    feature_id += 1
                    rows.append(_candidate(
                        feature_id, carbon, db,
                        float(rng.normal(0, 2)),
                        float(rng.normal(0, 0.2)),
                        (np.nan if rng.random() < 0.2
                         else float(rng.normal(0, 2))),
                        lipid_class=lipid_class,
                    ))
        # Repeated features at one node must not inflate neighbour support.
        rows.append(_candidate(9001, 34, 1, 0.4, 0.04, 1.0))
        rows.append(_candidate(9002, 34, 1, 0.5, 0.05, 1.1))
        return pd.DataFrame(rows)

    def test_matches_plain_loop_reference_across_settings(self):
        for seed in (1, 2, 3):
            data = self._dataset(seed)
            for use_ccs in (True, False):
                with self.subTest(seed=seed, use_ccs=use_ccs):
                    scored = score_grid_coherence(
                        data, 5.0, 0.5, 5.0, use_ccs=use_ccs, threshold=0.35
                    )
                    expected = _reference_scores(
                        data, 5.0, 0.5, 5.0, use_ccs=use_ccs, threshold=0.35
                    )
                    self.assertEqual(
                        int(scored["grid_coherence_assessed"].sum()),
                        len(expected),
                    )
                    self.assertGreater(len(expected), 20)
                    for index, (score, neighbours, dims) in expected.items():
                        self.assertAlmostEqual(
                            float(scored.at[index, "grid_coherence_score"]),
                            score, places=12,
                        )
                        self.assertEqual(
                            int(scored.at[index, "grid_coherence_neighbors"]),
                            neighbours,
                        )
                        self.assertEqual(
                            int(scored.at[index, "grid_coherence_dimensions"]),
                            dims,
                        )

    def test_candidate_without_grid_coordinates_is_unassessed(self):
        data = self._dataset(1)
        data.loc[data.index[0], "carbon"] = np.nan
        scored = score_grid_coherence(data, 5.0, 0.5, 5.0)
        self.assertFalse(bool(scored.at[data.index[0], "grid_coherence_assessed"]))
        self.assertTrue(bool(scored.at[data.index[0], "grid_coherence_pass"]))


if __name__ == "__main__":
    unittest.main()
