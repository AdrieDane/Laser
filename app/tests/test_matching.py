"""Focused parity checks for vectorized m/z candidate matching."""

from unittest.mock import patch
import unittest
from time import perf_counter

import numpy as np
import pandas as pd

from annotator.matching import _match_by_mz


def _reference(masses: list[float]) -> pd.DataFrame:
    count = len(masses)
    return pd.DataFrame({
        "name": [f"lipid-{index}" for index in range(count)],
        "class": ["PC"] * count,
        "adduct": ["[M+H]+"] * count,
        "ref_mz": masses,
        "ref_rt": [5.0] * count,
        "n_RT": [1] * count,
        "carbon": [34] * count,
        "db": [1] * count,
        "base": [""] * count,
        "is_is": [False] * count,
        "polarity": ["POS"] * count,
    })


class VectorizedMzMatchingTests(unittest.TestCase):
    def test_vectorized_search_preserves_all_pairs_and_order(self):
        features = pd.DataFrame({
            "feature_id": ["a", "b", "c"],
            "mz": [100.0, 200.0, 900.0],
            "rt": [1.0, 2.0, 3.0],
        })
        reference = _reference([200.001, 99.9995, 100.0005, 700.0])
        result = _match_by_mz(features, reference, 10.0)
        self.assertEqual(
            list(zip(result["feat_feature_id"], result["name"], strict=True)),
            [("a", "lipid-1"), ("a", "lipid-2"), ("b", "lipid-0")],
        )

    def test_searchsorted_is_called_once_per_boundary_not_per_feature(self):
        features = pd.DataFrame({
            "feature_id": np.arange(1_000).astype(str),
            "mz": np.linspace(100.0, 1_000.0, 1_000),
            "rt": np.ones(1_000),
        })
        original = np.searchsorted
        with patch("annotator.matching.np.searchsorted", wraps=original) as search:
            _match_by_mz(features, _reference([500.0]), 5.0)
        self.assertEqual(search.call_count, 2)

    def test_buffycoats_and_perox_scale_candidate_search_stays_interactive(self):
        reference = _reference(np.linspace(100.0, 1_000.0, 2_000).tolist())
        for label, row_count in (("Buffycoats", 60_000), ("Perox", 120_000)):
            features = pd.DataFrame({
                "feature_id": np.arange(row_count).astype(str),
                "mz": np.linspace(100.0, 1_000.0, row_count),
                "rt": np.ones(row_count),
            })
            started = perf_counter()
            result = _match_by_mz(features, reference, 5.0)
            elapsed = perf_counter() - started
            with self.subTest(workload=label):
                self.assertFalse(result.empty)
                self.assertLess(
                    elapsed,
                    3.0,
                    f"{label} candidate search took {elapsed:.3f} seconds",
                )


if __name__ == "__main__":
    unittest.main()
