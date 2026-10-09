from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from annotator.composition_ranges import (
    DEFAULT_COMPOSITION_RANGE_SETTINGS,
    ClassCompositionRange,
    CompositionRange,
    CompositionRangeSettings,
    FOUR_CHAIN,
    SINGLE_ACYL,
    SPHINGOLIPID,
    THREE_CHAIN,
    TWO_CHAIN,
    canonical_lipid_class,
    composition_range_settings_from_dict,
)
from standard_lipidomics.policy import apply_composition_range_filter
from matlab_annotation.composition_filter import filter_compound_set_targets
from matlab_annotation.vendor.target_model import CompoundSetTargetTable


class CompositionRangeSettingsTests(unittest.TestCase):
    def test_bundled_defaults_are_the_combined_example_data_extrema(self):
        rules = DEFAULT_COMPOSITION_RANGE_SETTINGS.class_lookup()
        self.assertEqual(len(rules), 66)
        groups = DEFAULT_COMPOSITION_RANGE_SETTINGS.group_lookup()
        self.assertEqual(
            groups[SINGLE_ACYL], CompositionRange(8, 42, 0, 7)
        )
        self.assertEqual(
            groups[TWO_CHAIN], CompositionRange(26, 66, 0, 13)
        )
        self.assertEqual(
            groups[SPHINGOLIPID], CompositionRange(18, 50, 0, 7)
        )
        self.assertEqual(
            groups[THREE_CHAIN], CompositionRange(36, 80, 0, 18)
        )
        self.assertEqual(
            groups[FOUR_CHAIN], CompositionRange(64, 86, 1, 22)
        )
        expected = {
            "1-acyl LPC": (14, 32, 0, 7),
            "LPC[O+P]": (14, 26, 0, 2),
            "PC": (26, 66, 0, 13),
            "PC[O]": (26, 50, 0, 8),
            "Cer[d]": (32, 46, 0, 4),
            "TG": (36, 80, 0, 18),
            "TG[O]": (46, 72, 0, 14),
            "CL": (64, 86, 1, 22),
        }
        for lipid_class, values in expected.items():
            with self.subTest(lipid_class=lipid_class):
                limits = rules[lipid_class].limits
                self.assertEqual(
                    (
                        limits.carbon_min, limits.carbon_max,
                        limits.db_min, limits.db_max,
                    ),
                    values,
                )

    def test_every_inclusive_boundary_is_allowed_and_one_step_beyond_is_not(self):
        settings = DEFAULT_COMPOSITION_RANGE_SETTINGS
        for rule in settings.class_ranges:
            limits = rule.limits
            with self.subTest(lipid_class=rule.lipid_class):
                self.assertTrue(settings.decide(
                    rule.lipid_class, limits.carbon_min, limits.db_min
                ).allowed)
                self.assertTrue(settings.decide(
                    rule.lipid_class, limits.carbon_max, limits.db_max
                ).allowed)
                self.assertFalse(settings.decide(
                    rule.lipid_class, limits.carbon_min - 1, limits.db_min
                ).allowed)
                self.assertFalse(settings.decide(
                    rule.lipid_class, limits.carbon_max + 1, limits.db_max
                ).allowed)
                self.assertFalse(settings.decide(
                    rule.lipid_class, limits.carbon_min, limits.db_max + 1
                ).allowed)

    def test_master_group_and_per_class_opt_outs_are_independent(self):
        group = CompositionRange(30, 40, 0, 5)
        class_limits = CompositionRange(20, 50, 0, 10)
        base = CompositionRangeSettings(
            enabled=True,
            use_group_ranges=True,
            group_ranges=((TWO_CHAIN, group),),
            class_ranges=(ClassCompositionRange(
                "PC", TWO_CHAIN, class_limits,
            ),),
        )
        self.assertFalse(base.decide("PC", 25, 2).allowed)

        without_groups = CompositionRangeSettings(
            enabled=True,
            use_group_ranges=False,
            group_ranges=base.group_ranges,
            class_ranges=base.class_ranges,
        )
        self.assertTrue(without_groups.decide("PC", 25, 2).allowed)

        class_without_group = CompositionRangeSettings(
            enabled=True,
            use_group_ranges=True,
            group_ranges=base.group_ranges,
            class_ranges=(ClassCompositionRange(
                "PC", TWO_CHAIN, class_limits, use_group_range=False,
            ),),
        )
        self.assertTrue(class_without_group.decide("PC", 25, 2).allowed)

        class_disabled = CompositionRangeSettings(
            enabled=True,
            use_group_ranges=True,
            group_ranges=base.group_ranges,
            class_ranges=(ClassCompositionRange(
                "PC", TWO_CHAIN, class_limits, enabled=False,
            ),),
        )
        self.assertTrue(class_disabled.decide("PC", 100, 30).allowed)

        all_disabled = CompositionRangeSettings(
            enabled=False,
            use_group_ranges=True,
            group_ranges=base.group_ranges,
            class_ranges=base.class_ranges,
        )
        self.assertTrue(all_disabled.decide("PC", 100, 30).allowed)

    def test_headless_overlay_preserves_defaults_and_changes_one_class(self):
        settings = composition_range_settings_from_dict({
            "use_group_ranges": False,
            "classes": {
                "PC": {
                    "enabled": False,
                    "carbon_min": 30,
                    "carbon_max": 44,
                }
            },
        })
        self.assertFalse(settings.use_group_ranges)
        self.assertFalse(settings.class_lookup()["PC"].enabled)
        self.assertEqual(settings.class_lookup()["PC"].limits.carbon_min, 30)
        self.assertEqual(settings.class_lookup()["PC"].limits.carbon_max, 44)
        self.assertEqual(
            settings.class_lookup()["TG"].limits,
            DEFAULT_COMPOSITION_RANGE_SETTINGS.class_lookup()["TG"].limits,
        )

    def test_matlab_and_reference_class_aliases_are_canonical(self):
        aliases = {
            "1-acylLPC": "1-acyl LPC",
            "2-acyl LPE": "2-acyl LPE",
            "PC-O": "PC[O]",
            "PC-O+P": "PC[O+P]",
            "Cer-d": "Cer[d]",
            "TG-O": "TG[O]",
        }
        for label, expected in aliases.items():
            self.assertEqual(canonical_lipid_class(label), expected)
        self.assertEqual(
            canonical_lipid_class("FA", species="FA(18:1)-OH"), "FA-OH"
        )

    def test_reference_filter_protects_internal_standards(self):
        reference = pd.DataFrame({
            "name": [
                "PC(34:1)", "PC(67:10)", "PC(80:20) (IS)",
                "PC(O-51:1)", "PC(51:1)",
            ],
            "class": ["PC"] * 5,
            "base": ["", "", "", "O", ""],
            "adduct": ["[M+H]+"] * 5,
            "carbon": [34, 67, 80, 51, 51],
            "db": [1, 10, 20, 1, 1],
            "is_is": [False, False, True, False, False],
        })
        filtered, audit = apply_composition_range_filter(
            reference, DEFAULT_COMPOSITION_RANGE_SETTINGS, stream="RP_Pos"
        )
        self.assertEqual(set(filtered["name"]), {
            "PC(34:1)", "PC(51:1)", "PC(80:20) (IS)",
        })
        reasons = audit.set_index("name")["policy_result"]
        self.assertEqual(
            reasons["PC(67:10)"], "outside_major_group_range"
        )
        self.assertEqual(
            reasons["PC(80:20) (IS)"], "internal_standard_protected"
        )
        self.assertEqual(reasons["PC(O-51:1)"], "outside_class_range")


class MatlabTargetCompositionFilterTests(unittest.TestCase):
    def test_filter_runs_before_matlab_target_competition_and_keeps_is(self):
        table = CompoundSetTargetTable(
            codes=np.array([1, 2, -1]),
            theoretical_mz=np.array([700.0, 701.0, 702.0]),
            stages=np.array([1, 1, 1]),
            sibling_positions=np.array([1, 1, 1]),
            composition_bases=(1, 1, 1, 1, 1, 3),
        )

        class FakeDecoder:
            def __init__(self, project, compound, phase):
                pass

            def decode(self, code):
                if code == 1:
                    return SimpleNamespace(
                        lipid_class="PC", carbon=34, double_bonds=1,
                        species="PC(34:1)",
                    )
                return SimpleNamespace(
                    lipid_class="PC", carbon=67, double_bonds=10,
                    species="PC(67:10)",
                )

        with patch(
            "matlab_annotation.composition_filter.CodeDecoder", FakeDecoder
        ):
            filtered, audit = filter_compound_set_targets(
                object(), table, (1,), phase="RP",
                settings=DEFAULT_COMPOSITION_RANGE_SETTINGS,
            )
        np.testing.assert_array_equal(filtered.codes, np.array([1, -1]))
        self.assertEqual(audit.endogenous_filtered, 1)
        self.assertEqual(audit.internal_standards_protected, 1)
        self.assertEqual(filtered.composition_bases, table.composition_bases)


if __name__ == "__main__":
    unittest.main()
