import unittest
import warnings

import numpy as np
import pandas as pd

from annotator.grid_coherence_v2 import (
    MIN_LOCAL_SPREAD_NATIVE,
    TYPICAL_SPREAD_MULTIPLIER,
    score_grid_coherence,
)


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

    def test_one_ccs_neighbour_cannot_define_a_ccs_cluster(self):
        data = pd.DataFrame([
            _candidate(1, 32, 0, 1.0, 0.10, 1.0),
            _candidate(2, 34, 0, 1.0, 0.10, 8.0),
            _candidate(3, 36, 0, 1.0, 0.10, np.nan),
            _candidate(4, 38, 0, 1.0, 0.10, np.nan),
        ])
        scored = score_grid_coherence(data, 5.0, 0.5, 10.0).set_index("feat_feature_id")
        self.assertTrue(bool(scored.loc["2", "grid_coherence_assessed"]))
        self.assertGreater(float(scored.loc["2", "grid_coherence_score"]), 0.9)
        self.assertNotEqual(scored.loc["2", "grid_coherence_worst_axis"], "CCS")

    def test_sparse_edge_candidate_is_unassessed_and_passes(self):
        data = pd.DataFrame([
            _candidate(1, 30, 0, 4.9, 0.49, 4.9),
            _candidate(2, 50, 10, -4.9, -0.49, -4.9),
            _candidate(3, 70, 20, 0.0, 0.0, 0.0),
        ])
        scored = score_grid_coherence(data, 5.0, 0.5, 5.0)
        self.assertFalse(scored["grid_coherence_assessed"].any())
        self.assertTrue(scored["grid_coherence_pass"].all())

    def test_shifted_tg_cluster_flags_single_axis_outliers_inside_windows(self):
        """A tight observed TG band must outweigh permissive reference windows."""
        rng = np.random.default_rng(9)
        rows = []
        for carbon in range(50, 72, 2):
            for db in range(1, 6):
                row = _candidate(
                    f"{carbon}-{db}", carbon, db,
                    0.8 + 0.01 * (carbon - 60) + rng.normal(0, 0.4),
                    0.08 + 0.003 * (carbon - 60) + 0.002 * db
                    + rng.normal(0, 0.015),
                    3.0 + 0.03 * (carbon - 60) + 0.02 * db
                    + rng.normal(0, 0.5),
                    lipid_class="TG",
                )
                row["adduct"] = "[M+NH4]+"
                rows.append(row)
        data = pd.DataFrame(rows).set_index("feat_feature_id")
        outliers = {
            "62-3": ("rt_error", 0.12),
            "66-2": ("ccs_error_percent", 4.0),
            "58-4": ("ppm_error", 2.5),
        }
        for feature, (column, offset) in outliers.items():
            data.loc[feature, column] += offset

        scored = score_grid_coherence(data, 5.0, 0.25, 10.0)

        self.assertTrue(scored.loc[list(outliers), "grid_coherence_assessed"].all())
        self.assertTrue(scored.loc[list(outliers), "grid_coherence_outlier"].all())
        self.assertLess(scored.loc[list(outliers), "grid_coherence_score"].max(), 0.6)
        self.assertEqual(
            scored.loc[list(outliers), "grid_coherence_worst_axis"].to_dict(),
            {"62-3": "RT", "66-2": "CCS", "58-4": "m/z"},
        )
        self.assertTrue(
            scored.loc[list(outliers), "grid_coherence_local_distance"].gt(1).all()
        )
        self.assertEqual(scored["grid_coherence_method"].unique().tolist(),
                         ["local_median_mad_v3"])
        self.assertTrue(scored.drop(index=list(outliers))["grid_coherence_pass"].all())

    def test_rt_score_follows_observed_seven_second_shift(self):
        """A candidate at -2 s should not fit neighbours clustered at +7 s."""
        rows = []
        for carbon in range(50, 72, 2):
            for db in range(1, 6):
                rt_seconds = 7.0 + 0.3 * (carbon - 60) / 2 + 0.2 * (db - 3)
                row = _candidate(
                    f"{carbon}-{db}", carbon, db, 0.8, rt_seconds / 60,
                    2.0, lipid_class="TG",
                )
                row["adduct"] = "[M+NH4]+"
                rows.append(row)
        data = pd.DataFrame(rows).set_index("feat_feature_id")
        data.loc["62-3", "rt_error"] = -2 / 60
        for rt_window in (0.25, 0.5):
            with self.subTest(rt_window=rt_window):
                scored = score_grid_coherence(data, 5.0, rt_window, 5.0)
                self.assertLess(float(scored.at["62-3", "grid_coherence_score"]), 0.35)
                self.assertEqual(scored.at["62-3", "grid_coherence_worst_axis"], "RT")
                self.assertTrue(bool(scored.at["62-3", "grid_coherence_outlier"]))
                self.assertTrue(scored.drop(index="62-3")["grid_coherence_pass"].all())
                if rt_window == 0.25:
                    first = float(scored.at["62-3", "grid_coherence_score"])
                else:
                    self.assertAlmostEqual(
                        float(scored.at["62-3", "grid_coherence_score"]), first,
                    )


def _reference_scores(
    data, ppm_tolerance, rt_tolerance, ccs_tolerance_percent,
    use_ccs=True, threshold=0.35, min_neighbors=2, max_neighbors=6,
    max_grid_distance=2.5,
):
    """Plain-loop reference for the observed-cluster score."""
    scored = data.copy()
    scored["_ppm"] = scored["ppm_error"] / ppm_tolerance
    scored["_rt"] = scored["rt_error"] / rt_tolerance
    scored["_ccs"] = (
        scored["ccs_error_percent"] / ccs_tolerance_percent
        if use_ccs and "ccs_error_percent" in scored else np.nan
    )
    results = {}
    for _, group in scored.groupby(["grid_class", "adduct"], sort=False):
        representatives = (
            group.groupby(["carbon", "db"], sort=False)[["_ppm", "_rt", "_ccs"]]
            .median().reset_index()
        )
        if len(representatives) <= min_neighbors:
            continue
        provisional = {}
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
            comparisons = []
            for _, neighbour in neighbours.iterrows():
                shared = [
                    column for column in ("_ppm", "_rt", "_ccs")
                    if np.isfinite(row[column]) and np.isfinite(neighbour[column])
                ]
                if len(shared) < 2:
                    continue
                comparisons.append((neighbour, len(shared)))
            if len(comparisons) < min_neighbors:
                continue
            centres = np.full(3, np.nan)
            spreads = np.full(3, np.nan)
            for axis, column in enumerate(("_ppm", "_rt", "_ccs")):
                if not np.isfinite(row[column]):
                    continue
                values = np.asarray([
                    neighbour[column] for neighbour, _ in comparisons
                    if np.isfinite(neighbour[column])
                ], dtype=float)
                if len(values) < min_neighbors:
                    continue
                centres[axis] = float(np.median(values))
                spreads[axis] = float(1.4826 * np.median(np.abs(values - centres[axis])))
            if np.isfinite(centres).sum() < 2:
                continue
            provisional[index] = (
                row[["_ppm", "_rt", "_ccs"]].to_numpy(float),
                centres, spreads,
                len(comparisons),
                int(round(float(np.mean([dims for _, dims in comparisons])))),
            )
        if not provisional:
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            typical = np.nanmedian(
                np.array([item[2] for item in provisional.values()]), axis=0
            )
        typical = np.where(np.isfinite(typical), typical, 0.0)
        floor = np.array(MIN_LOCAL_SPREAD_NATIVE) / np.array((
            ppm_tolerance, rt_tolerance,
            ccs_tolerance_percent if use_ccs else 1.0,
        ))
        for index, (residuals, centres, spreads, count, dimensions) in provisional.items():
            scale = np.maximum(spreads, np.maximum(TYPICAL_SPREAD_MULTIPLIER * typical, floor))
            distance = np.nanmax(np.abs((residuals - centres) / scale))
            results[index] = (
                float(np.exp(-0.5 * distance ** 2)), count, dimensions,
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
