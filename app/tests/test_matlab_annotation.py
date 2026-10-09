import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from annotator.fa_model import expected_compositions
from annotator.np_rt import combine_np_ether_plasmalogen_labels
from matlab_annotation import (
    AUTOMATIC_REGRESSION_INLIERS,
    BIOLOGICAL_ABUNDANCE_PRIOR,
    ITERATIVE_POSITIONAL_REFIT,
    POST_REGRESSION_POSITIONAL_RESOLUTION,
    UNIFIED_EVIDENCE_ANCHORS,
    AutomaticAnchorSettings,
    DEFAULT_MATLAB_PIPELINE_REFERENCE,
    MatlabAnnotationError,
    decode_annotation_code,
    discover_matlab_annotation_project,
    lipid_class_from_template,
    run_matlab_annotation,
)
from matlab_annotation.automatic import (
    MATLAB_NP_ANNOTATION_ADDUCTS,
    _anchor_selection_allows,
    _canonical_positional_anchor_species,
    _is_expected_abundance_anchor,
)
from matlab_annotation.xcms_project import (
    build_np_xcms_project,
    build_rp_xcms_project,
    load_bundled_compound_catalog,
    with_memberships,
)
from matlab_annotation.decode import CodeDecoder, composition_dimensions
from matlab_annotation.engine import (
    _build_positional_refit_project,
    _calculate_matlab_grid_coherence,
    _copy_grid_coherence,
    _drop_orphan_two_acyl_lpc,
    _internal_standard_audit,
    _internal_standard_project_dir,
    _label_internal_standards,
    _monoisotopic_codes,
    _observed_rt_window,
    _mark_positional_seed_anchors,
    _merge_provisional_positional_audit,
    _reconcile_positional_review_anchors,
    _restore_xcms_anchor_frame,
    _sibling_groups,
)
from matlab_annotation.isomer_resolver import (
    PositionalIsomerSettings,
    resolve_positional_lyso_isomers,
)
from matlab_annotation.vendor.internal_standard_model import (
    build_simple_internal_standard_targets,
)
from matlab_annotation.vendor.reference_loader import load_reference_data
from matlab_annotation.vendor.legacy_loader import load_legacy_project
from matlab_annotation.vendor.rtdb_model import map_retention_time_observations
from matlab_annotation.vendor.naming import (
    compound_display_template,
    resolved_compound_name,
)


REFERENCE_PROJECTS = (
    Path.home()
    / "Library/Mobile Documents/com~apple~CloudDocs/Work/AI/example_data_CFM"
)
PROJECT = REFERENCE_PROJECTS / "26P0026_ALD_rat_brain"


class BundledReferenceTests(unittest.TestCase):
    """The MATLAB engine's reference data must ship with LASER."""

    def test_provisional_positional_decisions_are_merged_by_pair_id(self):
        final = pd.DataFrame(
            {"pair_id": ["LPC-1"], "status": ["final_membership"]}
        )
        provisional = pd.DataFrame(
            {
                "pair_id": ["LPC-1"],
                "status": ["provisional_membership"],
                "selected_one_feature_id": ["17"],
                "assignment_score": [0.93],
            }
        )

        merged = _merge_provisional_positional_audit(final, provisional)

        self.assertEqual(merged.loc[0, "status"], "final_membership")
        self.assertEqual(
            merged.loc[0, "provisional_status"], "provisional_membership"
        )
        self.assertEqual(merged.loc[0, "provisional_selected_one_feature_id"], "17")
        self.assertAlmostEqual(merged.loc[0, "provisional_assignment_score"], 0.93)

    def test_reference_bundle_contains_every_required_input(self):
        root = DEFAULT_MATLAB_PIPELINE_REFERENCE
        self.assertTrue((root / "Code" / "param.json").is_file())
        for name in (
            "lipids.tsv", "lipids_info.tsv", "internal_standards_all.tsv",
            "metabolites.tsv", "metabolites_info.tsv", "stable_isotopes.tsv",
        ):
            self.assertTrue((root / "DataBase" / name).is_file(), name)
        self.assertTrue(list((root / "RTDB").glob("lipidomics_*.tsv")))
        self.assertTrue((root / "Project" / "internal_standards.tsv").is_file())

    def test_bundled_tsv_constructs_the_complete_matlab_catalogue(self):
        catalog = load_bundled_compound_catalog(DEFAULT_MATLAB_PIPELINE_REFERENCE)
        self.assertEqual(catalog.count, 228)
        self.assertEqual(catalog.classes[:5], ("TG",) * 5)
        self.assertEqual(catalog.legacy_fields["composition"][1, 1], "'")
        self.assertEqual(catalog.legacy_fields["composition"][4, 2], "O-")
        self.assertEqual(catalog.legacy_fields["nR"][0], 3)
        self.assertEqual(AutomaticAnchorSettings().ppm_tolerance, 3.0)
        self.assertEqual(
            AutomaticAnchorSettings().grid_coherence_rt_tolerance_seconds,
            10.0,
        )
        self.assertTrue(AutomaticAnchorSettings().use_grid_coherence)
        self.assertEqual(AutomaticAnchorSettings().grid_coherence_threshold, 0.45)
        self.assertEqual(AutomaticAnchorSettings().expected_abundance_top_k, 15)
        self.assertEqual(
            AutomaticAnchorSettings().anchor_selection_mode,
            UNIFIED_EVIDENCE_ANCHORS,
        )
        self.assertEqual(
            AutomaticAnchorSettings().positional_isomer_workflow_mode,
            POST_REGRESSION_POSITIONAL_RESOLUTION,
        )
        self.assertEqual(
            set(AutomaticAnchorSettings().excluded_anchor_species),
            {
                "1-acylLPC(20:5)",
                "1-acylLPC(22:1)",
                "1-acylLPC(22:5)",
            },
        )

    def test_experimental_positional_refit_replaces_only_positional_memberships(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "peakgrouplist_RP_Pos.txt").write_text(
                "name\tmzmed\trtmed\tnpeaks\tsample_A\n"
                "101\t552.4\t8.0\t1\t1000\n",
                encoding="utf-8",
            )
            bundle = build_rp_xcms_project(
                root, reference_root=DEFAULT_MATLAB_PIPELINE_REFERENCE
            )

        memberships = [
            [list(row) for row in compound_rows]
            for compound_rows in bundle.project.compounds.inliers
        ]
        memberships[0].append(["RP", 1, 900.0, 700.0])
        memberships[35].append(["RP", 1, 552.4, 470.0])
        memberships[36].append(["NP", 2, 552.4, 300.0])
        seeded = with_memberships(bundle.project, tuple(memberships))
        candidates = pd.DataFrame({
            "class": ["1-acylLPC", "2-acylLPC"],
            "matlab_compound": [36, 37],
            "matlab_code": [1, 2],
        })
        resolved = pd.DataFrame({
            "class": ["1-acylLPC", "2-acylLPC"],
            "matlab_compound": [36, 37],
            "matlab_code": [1, 2],
            "feat_feature_id": ["101", "102"],
            "feat_mz": [552.4, 552.4],
            "feat_rt": [8.0, 7.7],
            "name": ["1-acylLPC(20:0)", "2-acylLPC(20:0)"],
            "adduct": ["[M+H]+", "[M+H]+"],
            "isotopologue": ["", ""],
        })

        refit, audit, anchors = _build_positional_refit_project(
            seeded, resolved, candidates, phase="RP"
        )

        self.assertEqual(refit.compounds.inliers[0], memberships[0])
        self.assertEqual(
            refit.compounds.inliers[35], [["RP", 1, 552.4, 480.0]]
        )
        self.assertEqual(
            refit.compounds.inliers[36], [
                ["NP", 2, 552.4, 300.0],
                ["RP", 2, 552.4, 462.0],
            ],
        )
        self.assertEqual(set(audit["status"]), {"refit_membership"})
        self.assertEqual(set(anchors["row"]), {100, 101})
        self.assertEqual(ITERATIVE_POSITIONAL_REFIT, "iterative_refit")

    def test_abundance_prior_accepts_matlab_grid_class_spellings(self):
        self.assertEqual(
            expected_compositions("1-acylLPC"),
            expected_compositions("LPC"),
        )
        self.assertEqual(expected_compositions("2-acylLPE"), {})
        self.assertEqual(
            expected_compositions("PC-O"),
            expected_compositions("PC", "O"),
        )
        self.assertEqual(
            expected_compositions("SM-d"),
            expected_compositions("SM", "d"),
        )
        self.assertTrue(expected_compositions("AC-OH"))

    def test_only_expected_abundant_endogenous_species_are_anchor_eligible(self):
        def decoded(lipid_class, carbon, db, *, internal_standard=False):
            return SimpleNamespace(
                lipid_class=lipid_class,
                carbon=carbon,
                double_bonds=db,
                internal_standard=internal_standard,
            )

        self.assertTrue(_is_expected_abundance_anchor(
            decoded("1-acylLPC", 18, 1), top_k=15
        ))
        self.assertFalse(_is_expected_abundance_anchor(
            decoded("1-acylLPC", 17, 0), top_k=15
        ))
        self.assertFalse(_is_expected_abundance_anchor(
            decoded("2-acylLPC", 18, 1), top_k=15
        ))
        self.assertTrue(_is_expected_abundance_anchor(
            decoded("PC", 34, 1), top_k=15
        ))
        self.assertFalse(_is_expected_abundance_anchor(
            decoded("PC", 33, 0), top_k=15
        ))
        self.assertFalse(_is_expected_abundance_anchor(
            decoded("unknown", 18, 1), top_k=15
        ))
        self.assertTrue(_is_expected_abundance_anchor(
            decoded("unknown", None, None, internal_standard=True), top_k=15
        ))

    def test_anchor_selection_toggle_changes_only_abundance_eligibility(self):
        uncommon = SimpleNamespace(
            lipid_class="1-acylLPC",
            species="1-acylLPC(17:0)",
            carbon=17,
            double_bonds=0,
            internal_standard=False,
        )
        self.assertFalse(_anchor_selection_allows(
            uncommon, mode=BIOLOGICAL_ABUNDANCE_PRIOR, top_k=15
        ))
        self.assertTrue(_anchor_selection_allows(
            uncommon, mode=AUTOMATIC_REGRESSION_INLIERS, top_k=15
        ))
        self.assertTrue(_anchor_selection_allows(
            uncommon, mode=UNIFIED_EVIDENCE_ANCHORS, top_k=15
        ))
        uncommon_lpe = SimpleNamespace(
            lipid_class="1-acylLPE", species="1-acylLPE(17:0)",
            carbon=17, double_bonds=0, internal_standard=False,
        )
        self.assertFalse(_anchor_selection_allows(
            uncommon_lpe, mode=UNIFIED_EVIDENCE_ANCHORS, top_k=15
        ))
        with self.assertRaisesRegex(ValueError, "anchor-selection mode"):
            _anchor_selection_allows(uncommon, mode="invalid", top_k=15)

    def test_positional_anchor_exclusions_apply_to_both_isomer_spellings(self):
        self.assertEqual(
            _canonical_positional_anchor_species("2-acylLPC(22:5)"),
            "1-acylLPC(22:5)",
        )
        self.assertEqual(
            _canonical_positional_anchor_species("2-acylLPE(18:1)"),
            "1-acylLPE(18:1)",
        )

    def test_automatic_anchor_provenance_restores_stream_feature_id(self):
        anchors = pd.DataFrame({
            "row": [0, 1],
            "species": ["1-acylLPC(18:2)", "2-acylLPC(18:2)"],
            "adduct": ["[M+H]+", "[M+H]+"],
        })
        bundle = SimpleNamespace(
            row_stream=("RP_Pos", "RP_Neg"),
            row_feature_id=("41118", "22345"),
        )

        restored = _restore_xcms_anchor_frame(
            anchors, bundle, "RP_Pos"
        )

        self.assertEqual(restored.iloc[0]["feat_feature_id"], "41118")
        self.assertEqual(restored.iloc[0]["standard_stream"], "RP_Pos")
        self.assertEqual(restored.iloc[0]["species"], "1-acylLPC(18:2)")

    def test_positional_review_anchors_follow_resolved_feature_identity(self):
        seeds = pd.DataFrame({
            "standard_stream": ["RP_Pos", "RP_Pos"],
            "feat_feature_id": ["early", "pc"],
            "species": ["1-acylLPC(20:0)", "PC(34:1)"],
            "adduct": ["[M+H]+", "[M+H]+"],
        })
        candidates = pd.DataFrame({
            "standard_stream": ["RP_Pos"] * 4,
            "feat_feature_id": ["late", "early", "unpaired", "excluded"],
            "name": [
                "1-acylLPC(20:0)", "2-acylLPC(20:0)",
                "1-acylLPC(18:0)",
                "1-acylLPC(22:5)",
            ],
            "adduct": ["[M+H]+"] * 4,
            "positional_isomer_selected": [True] * 4,
            "positional_isomer_seed_anchor": [True, False, False, True],
            "positional_isomer_status": [
                "paired_rt_order_supported", "paired_rt_order_supported",
                "one_acyl_primary_only_no_two_acyl",
                "paired_rt_order_supported",
            ],
        })

        reconciled = _reconcile_positional_review_anchors(
            seeds, candidates,
            excluded_species=("1-acylLPC(22:5)",),
        )

        keys = set(zip(
            reconciled["species"], reconciled["feat_feature_id"], strict=True
        ))
        self.assertIn(("PC(34:1)", "pc"), keys)
        self.assertIn(("1-acylLPC(20:0)", "late"), keys)
        self.assertNotIn(("2-acylLPC(20:0)", "early"), keys)
        self.assertNotIn(("1-acylLPC(18:0)", "unpaired"), keys)
        self.assertNotIn(("1-acylLPC(20:0)", "early"), keys)
        self.assertNotIn(("1-acylLPC(22:5)", "excluded"), keys)
        roles = reconciled.set_index("feat_feature_id")["anchor_role"].to_dict()
        self.assertEqual(roles["late"], "resolved_positional_seed")

    @staticmethod
    def _positional_candidates():
        rows = []
        for feature, rt, intensity in (("early", 6.02, 800), ("late", 6.31, 1000)):
            for position, expected in ((1, 6.30), (2, 6.00)):
                rows.append({
                    "feat_feature_id": feature,
                    "name": f"{position}-acylLPC(18:2)",
                    "class": f"{position}-acylLPC",
                    "adduct": "[M+H]+",
                    "isotopologue": "",
                    "is_is": False,
                    "carbon": 18,
                    "db": 2,
                    "feat_mz": 520.34,
                    "ref_mz": 520.34,
                    "ppm_error": 0.2,
                    "feat_rt": rt,
                    "rt_expected": expected,
                    "ref_rt": expected,
                    "rt_error": rt - expected,
                    "feat_intensity": intensity,
                    "matlab_code": position,
                    "automatic_accepted": False,
                    "automatic_candidate": True,
                    "auto_annotation_status": "review_candidate",
                })
        return pd.DataFrame(rows)

    def test_positional_lyso_resolver_jointly_assigns_early_and_late_peaks(self):
        candidates = self._positional_candidates()
        # Reproduce the failure mode: independent mapper decisions crossed the
        # two labels before the coupled resolver sees the complete pool.
        annotations = candidates.iloc[[0, 3]].copy()
        annotations["automatic_accepted"] = True

        accepted, decorated, audit = resolve_positional_lyso_isomers(
            annotations, candidates
        )

        selected = accepted.set_index("name")["feat_feature_id"].to_dict()
        self.assertEqual(selected["2-acylLPC(18:2)"], "early")
        self.assertEqual(selected["1-acylLPC(18:2)"], "late")
        self.assertTrue(accepted["positional_isomer_selected"].all())
        self.assertTrue(
            accepted["auto_annotation_status"].eq(
                "matlab_positional_isomer_paired"
            ).all()
        )
        self.assertEqual(int(decorated["positional_isomer_selected"].sum()), 2)
        selected_rows = decorated[decorated["positional_isomer_selected"]]
        rt_errors = selected_rows.set_index("name")[
            "positional_isomer_rt_error_seconds"
        ].to_dict()
        self.assertAlmostEqual(rt_errors["2-acylLPC(18:2)"], 1.2, places=6)
        self.assertAlmostEqual(rt_errors["1-acylLPC(18:2)"], 0.6, places=6)
        self.assertEqual(audit.iloc[0]["status"], "paired_rt_order_supported")
        self.assertAlmostEqual(
            float(audit.iloc[0]["observed_gap_seconds"]), 17.4, places=6
        )
        self.assertAlmostEqual(
            float(audit.iloc[0]["expected_gap_seconds"]), 18.0, places=6
        )

    def test_positional_seed_keeps_verified_second_peak_and_finds_first(self):
        rows = []
        for feature, rt in (("first", 6.00), ("second", 6.30), ("third", 6.60)):
            rows.append({
                    "feat_feature_id": feature,
                    "name": "2-acylLPC(20:0)",
                    "class": "2-acylLPC", "base": "",
                    "adduct": "[M+H]+", "isotopologue": "", "is_is": False,
                    "carbon": 20, "db": 0,
                    "feat_mz": 552.4024, "ref_mz": 552.4024,
                    "ppm_error": 0.1, "feat_rt": rt,
                    "rt_expected": 6.30, "ref_rt": 6.30,
                    "rt_error": rt - 6.30, "feat_intensity": 1000,
                    "matlab_code": 2,
                    "automatic_accepted": False,
                    "automatic_candidate": True,
                    "auto_annotation_status": "review_candidate",
                    "positional_isomer_seed_anchor": False,
                    "positional_isomer_seed_species": (
                        "1-acylLPC(20:0)" if feature == "second" else ""
                    ),
                })
        candidates = pd.DataFrame(rows)
        reference = pd.DataFrame({
            "name": ["1-acylLPC(20:0)", "2-acylLPC(20:0)"],
            "class": ["1-acylLPC", "2-acylLPC"], "base": ["", ""],
            "adduct": ["[M+H]+", "[M+H]+"],
            "carbon": [20, 20], "db": [0, 0],
            "ref_mz": [552.4024, 552.4024], "ref_rt": [6.30, 6.00],
            "matlab_code": [1, 2], "matlab_compound": [36, 37],
        })

        accepted, _, audit = resolve_positional_lyso_isomers(
            pd.DataFrame(), candidates, reference=reference
        )

        selected = accepted.set_index("name")["feat_feature_id"].to_dict()
        self.assertEqual(selected["2-acylLPC(20:0)"], "first")
        self.assertEqual(selected["1-acylLPC(20:0)"], "second")
        self.assertEqual(audit.iloc[0]["status"], "paired_rt_order_supported")

    def test_marks_exact_positional_seed_feature_identity(self):
        candidates = pd.DataFrame({
            "feat_feature_id": ["43748", "43748", "43745"],
            "name": [
                "1-acylLPC(20:0)", "2-acylLPC(20:0)",
                "2-acylLPC(20:0)",
            ],
            "class": ["1-acylLPC", "2-acylLPC", "2-acylLPC"],
            "adduct": ["[M+H]+"] * 3,
            "carbon": [20] * 3,
            "db": [0] * 3,
            "ppm_error": [0.2, 0.2, 0.1],
            "feat_intensity": [20_000, 20_000, 2_000],
        })
        anchors = pd.DataFrame({
            "row": [43747], "species": ["1-acylLPC(20:0)"],
            "adduct": ["[M+H]+"],
        })

        marked = _mark_positional_seed_anchors(candidates, anchors)

        self.assertEqual(marked["positional_isomer_seed_anchor"].tolist(), [
            True, False, False,
        ])
        self.assertEqual(
            marked["positional_isomer_seed_species"].tolist(),
            ["1-acylLPC(20:0)", "1-acylLPC(20:0)", ""],
        )

    def test_excluded_lpc_species_cannot_become_seed_anchor(self):
        candidates = pd.DataFrame({
            "feat_feature_id": ["one", "two"],
            "name": ["1-acylLPC(22:5)", "2-acylLPC(22:5)"],
            "class": ["1-acylLPC", "2-acylLPC"],
            "adduct": ["[M+H]+"] * 2,
            "carbon": [22] * 2, "db": [5] * 2,
            "ppm_error": [0.1, 0.1],
            "feat_intensity": [20_000, 2_000],
        })
        anchors = pd.DataFrame({
            "row": [0], "species": ["1-acylLPC(22:5)"],
            "adduct": ["[M+H]+"],
        })

        marked = _mark_positional_seed_anchors(
            candidates, anchors,
            excluded_species=("1-acylLPC(22:5)",),
        )

        self.assertFalse(marked["positional_isomer_seed_anchor"].any())
        self.assertTrue(
            marked["positional_isomer_seed_species"].astype(str).eq("").all()
        )

    def test_positional_seed_cannot_leak_between_isobaric_lpe_and_lpc(self):
        candidates = pd.DataFrame({
            "feat_feature_id": ["1", "1", "1", "1"],
            "name": [
                "1-acylLPE(18:0)", "2-acylLPE(18:0)",
                "1-acylLPC(15:0)", "2-acylLPC(15:0)",
            ],
            "class": [
                "1-acylLPE", "2-acylLPE", "1-acylLPC", "2-acylLPC",
            ],
            "adduct": ["[M-H]-"] * 4,
            "carbon": [18, 18, 15, 15], "db": [0, 0, 0, 0],
            "ppm_error": [0.1] * 4, "feat_intensity": [1000] * 4,
            "feat_rt": [6.0] * 4, "rt_expected": [6.0] * 4,
        })
        anchors = pd.DataFrame({
            "row": [0], "species": ["1-acylLPE(18:0)"],
            "adduct": ["[M-H]-"],
        })

        marked = _mark_positional_seed_anchors(candidates, anchors)

        lpe = marked[marked["class"].eq("1-acylLPE")].iloc[0]
        lpc = marked[marked["class"].eq("1-acylLPC")].iloc[0]
        self.assertEqual(lpe["positional_isomer_seed_species"], "1-acylLPE(18:0)")
        self.assertTrue(bool(lpe["positional_isomer_seed_anchor"]))
        self.assertEqual(lpc["positional_isomer_seed_species"], "")
        self.assertFalse(bool(lpc["positional_isomer_seed_anchor"]))

    def test_lpc_abundance_seed_is_restricted_to_rt_envelope(self):
        candidates = pd.DataFrame({
            "feat_feature_id": ["correct", "wrong"],
            "name": ["1-acylLPC(18:0)", "1-acylLPC(18:0)"],
            "class": ["1-acylLPC", "1-acylLPC"],
            "adduct": ["[M+H]+"] * 2,
            "carbon": [18, 18], "db": [0, 0],
            "ppm_error": [0.2, 0.1],
            "feat_intensity": [1_000, 100_000],
            "feat_rt": [6.0, 6.5], "rt_expected": [6.0, 6.0],
        })
        anchors = pd.DataFrame({
            "row": [0], "species": ["1-acylLPC(18:0)"],
            "adduct": ["[M+H]+"],
        })

        marked = _mark_positional_seed_anchors(candidates, anchors)

        selected = marked[marked["positional_isomer_seed_anchor"]]
        self.assertEqual(selected["feat_feature_id"].tolist(), ["correct"])

    def test_two_acyl_is_not_accepted_without_a_credible_one_acyl_peak(self):
        candidates = self._positional_candidates()
        candidates = candidates[candidates["feat_feature_id"].eq("early")]
        annotations = candidates.iloc[[0]].copy()
        annotations["automatic_accepted"] = True

        accepted, decorated, audit = resolve_positional_lyso_isomers(
            annotations, candidates
        )

        self.assertTrue(accepted.empty)
        self.assertFalse(decorated["positional_isomer_selected"].any())
        self.assertEqual(audit.iloc[0]["status"], "unresolved_missing_one_acyl")

    def test_one_acyl_primary_is_accepted_without_two_acyl_peak(self):
        candidates = self._positional_candidates()
        candidates = candidates[candidates["feat_feature_id"].eq("late")]

        accepted, decorated, audit = resolve_positional_lyso_isomers(
            pd.DataFrame(), candidates
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted.iloc[0]["name"], "1-acylLPC(18:2)")
        self.assertEqual(accepted.iloc[0]["feat_feature_id"], "late")
        self.assertEqual(
            accepted.iloc[0]["auto_annotation_status"],
            "matlab_one_acyl_primary",
        )
        selected = decorated[decorated["positional_isomer_selected"]]
        self.assertEqual(selected["name"].tolist(), ["1-acylLPC(18:2)"])
        self.assertEqual(
            audit.iloc[0]["status"], "one_acyl_primary_only_no_two_acyl"
        )

    def test_lpc_primary_prefers_abundance_within_rt_guardrails(self):
        candidates = self._positional_candidates()
        alternative = candidates[
            candidates["feat_feature_id"].eq("late")
        ].copy()
        alternative["feat_feature_id"] = "highest"
        alternative["feat_rt"] = 6.29
        alternative["feat_intensity"] = 2_000
        candidates = pd.concat([candidates, alternative], ignore_index=True)

        accepted, _, audit = resolve_positional_lyso_isomers(
            pd.DataFrame(), candidates
        )

        selected = accepted.set_index("name")["feat_feature_id"].to_dict()
        self.assertEqual(selected["1-acylLPC(18:2)"], "highest")
        self.assertEqual(selected["2-acylLPC(18:2)"], "early")
        self.assertEqual(audit.iloc[0]["status"], "paired_rt_order_supported")

    def test_lpe_still_requires_a_complete_pair(self):
        candidates = self._positional_candidates()
        candidates = candidates[candidates["feat_feature_id"].eq("late")].copy()
        candidates["name"] = candidates["name"].str.replace("LPC", "LPE")
        candidates["class"] = candidates["class"].str.replace("LPC", "LPE")

        accepted, _, audit = resolve_positional_lyso_isomers(
            pd.DataFrame(), candidates
        )

        self.assertTrue(accepted.empty)
        self.assertEqual(audit.iloc[0]["status"], "unresolved_single_peak")

    def test_lpe_exact_seed_can_retain_a_supported_one_acyl_singleton(self):
        candidates = self._positional_candidates()
        candidates = candidates[
            candidates["feat_feature_id"].eq("late")
        ].copy()
        candidates["name"] = candidates["name"].str.replace("LPC", "LPE")
        candidates["class"] = candidates["class"].str.replace("LPC", "LPE")
        candidates["positional_isomer_seed_species"] = "1-acylLPE(18:2)"
        candidates["positional_isomer_seed_anchor"] = True

        accepted, _, audit = resolve_positional_lyso_isomers(
            pd.DataFrame(), candidates
        )

        self.assertEqual(accepted["name"].tolist(), ["1-acylLPE(18:2)"])
        self.assertEqual(
            audit.iloc[0]["status"], "one_acyl_primary_only_lpe_supported"
        )

    def test_neighbouring_pairs_supply_only_a_bounded_local_rt_correction(self):
        rows = []
        for carbon, residual_seconds in ((16, 8.0), (18, 8.0), (20, 15.0)):
            for position, expected in ((2, 6.0), (1, 6.2)):
                rows.append({
                    "feat_feature_id": f"{carbon}-{position}",
                    "name": f"{position}-acylLPE({carbon}:0)",
                    "class": f"{position}-acylLPE", "base": "",
                    "adduct": "[M-H]-", "carbon": carbon, "db": 0,
                    "isotopologue": "", "is_is": False,
                    "feat_rt": expected + residual_seconds / 60.0,
                    "rt_expected": expected, "ref_rt": expected,
                    "feat_mz": 500.0 + carbon, "ref_mz": 500.0 + carbon,
                    "ppm_error": 0.1, "feat_intensity": 1000.0,
                })

        accepted, _, audit = resolve_positional_lyso_isomers(
            pd.DataFrame(), pd.DataFrame(rows)
        )

        target = accepted[accepted["carbon"].eq(20)]
        self.assertEqual(set(target["class"]), {"1-acylLPE", "2-acylLPE"})
        target_audit = audit[audit["carbon"].eq(20)].iloc[0]
        self.assertEqual(target_audit["status"], "paired_rt_order_supported")
        self.assertEqual(target_audit["local_support_groups"], 2)
        self.assertAlmostEqual(
            float(target_audit["local_rt_adjustment_seconds"]), 8.0
        )

    def test_positional_lyso_resolver_restores_missing_sibling_from_grid(self):
        candidates = self._positional_candidates()
        candidates = candidates[candidates["name"].eq("1-acylLPC(18:2)")]
        reference = pd.DataFrame({
            "name": ["1-acylLPC(18:2)", "2-acylLPC(18:2)"],
            "class": ["1-acylLPC", "2-acylLPC"],
            "base": ["", ""],
            "adduct": ["[M+H]+", "[M+H]+"],
            "carbon": [18, 18], "db": [2, 2],
            "ref_mz": [520.34, 520.34],
            "ref_rt": [6.30, 6.00],
            "matlab_code": [1, 2],
            "matlab_compound": [36, 37],
        })

        accepted, decorated, audit = resolve_positional_lyso_isomers(
            candidates.iloc[[0]].copy(), candidates, reference=reference
        )

        selected = accepted.set_index("name")["feat_feature_id"].to_dict()
        self.assertEqual(selected["2-acylLPC(18:2)"], "early")
        self.assertEqual(selected["1-acylLPC(18:2)"], "late")
        restored = decorated[
            decorated["positional_isomer_candidate_source"].eq(
                "rt_grid_counterpart"
            )
        ]
        self.assertEqual(len(restored), 2)
        self.assertEqual(audit.iloc[0]["status"], "paired_rt_order_supported")

    def test_positional_lyso_resolver_rejects_implausible_pair_gap(self):
        candidates = self._positional_candidates()
        candidates.loc[candidates["feat_feature_id"].eq("late"), "feat_rt"] = 6.18
        annotations = candidates.iloc[[1, 2]].copy()

        accepted, _, audit = resolve_positional_lyso_isomers(
            annotations,
            candidates,
            settings=PositionalIsomerSettings(maximum_gap_error_seconds=5.0),
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted.iloc[0]["name"], "1-acylLPC(18:2)")
        self.assertEqual(
            audit.iloc[0]["status"], "one_acyl_primary_only_gap_mismatch"
        )

    def test_rtdb_spacing_maps_positional_lyso_traits(self):
        """RTDB's ``1-acyl LPC`` spelling must map to the catalogue key.

        The catalogue template is rendered as ``1-acylLPC`` while the
        reference RTDB contains a space between the positional prefix and
        class.  Both spellings describe the same compound family.
        """
        root = DEFAULT_MATLAB_PIPELINE_REFERENCE
        reference = load_reference_data(
            code_dir=root / "Code", database_dir=root / "DataBase",
            rtdb_dir=root / "RTDB", project_dir=root / "Project",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "peakgrouplist_RP_Pos.txt").write_text(
                "name\tmzmed\trtmed\tnpeaks\tsample_A\n"
                "feature-1\t678.5068\t10.7\t1\t1000\n",
                encoding="utf-8",
            )
            project = build_rp_xcms_project(
                path, reference_root=root
            ).project
        mapped = map_retention_time_observations(project, reference)
        traits = {item.trait for item in mapped}
        self.assertIn("1-acyl LPC(18:1)", traits)
        self.assertIn("2-acyl LPC(18:1)", traits)
        self.assertIn("1-acyl LPE(18:1)", traits)
        self.assertIn("2-acyl LPE(18:1)", traits)

    def test_grid_coherence_scoring_does_not_filter_before_policy_stage(self):
        annotations = pd.DataFrame({
            "feat_feature_id": ["1", "2", "3", "4"],
            "name": ["PC(32:0)", "PC(34:0)", "PC(36:0)", "PC(34:1)"],
            "class": ["PC"] * 4, "base": [""] * 4,
            "adduct": ["[M+H]+"] * 4,
            "carbon": [32, 34, 36, 34], "db": [0, 0, 0, 1],
            "ppm_error": [0.2, 0.3, 0.1, 2.9],
            "rt_error": [0.01, 0.011, 0.009, 0.16],
            "automatic_accepted": [True] * 4,
        })
        scored, audit = _calculate_matlab_grid_coherence(
            annotations,
            settings=AutomaticAnchorSettings(grid_coherence_threshold=0.9),
        )
        self.assertEqual(len(scored), len(annotations))
        self.assertTrue(scored["automatic_accepted"].all())
        self.assertTrue(bool(scored.loc[3, "grid_coherence_outlier"]))
        self.assertTrue(bool(audit.iloc[0]["used_for_annotation"]))

    def test_grid_coherence_uses_full_candidate_pool_for_accepted_rows(self):
        candidates = pd.DataFrame({
            "feat_feature_id": ["1", "2", "3", "4"],
            "name": ["TG(32:0)", "TG(34:0)", "TG(36:0)", "TG(38:0)"],
            "class": ["TG"] * 4, "base": [""] * 4,
            "adduct": ["[M+NH4]+"] * 4,
            "matlab_code": [1, 2, 3, 4],
            "carbon": [32, 34, 36, 38], "db": [0, 0, 0, 0],
            "ppm_error": [0.1, 0.2, 0.1, 0.2],
            "rt_error": [0.01, 0.011, 0.009, 0.012],
        })
        accepted = candidates.iloc[:1].copy()
        scored, _ = _calculate_matlab_grid_coherence(
            candidates, settings=AutomaticAnchorSettings()
        )
        self.assertTrue(scored["grid_coherence_assessed"].all())
        copied = _copy_grid_coherence(accepted, scored)
        self.assertTrue(bool(copied.iloc[0]["grid_coherence_assessed"]))
        self.assertTrue(np.isfinite(copied.iloc[0]["grid_coherence_score"]))

    def test_two_acyl_lpc_cannot_survive_primary_coherence_filtering(self):
        annotations = pd.DataFrame({
            "name": [
                "1-acylLPC(18:1)", "2-acylLPC(18:1)",
                "1-acylLPC(20:0)", "2-acylLPC(20:0)",
            ],
            "class": [
                "1-acylLPC", "2-acylLPC", "1-acylLPC", "2-acylLPC",
            ],
            "carbon": [18, 18, 20, 20], "db": [1, 1, 0, 0],
            "adduct": ["[M+H]+"] * 4, "isotopologue": [""] * 4,
        })
        # Simulate coherence having removed only the 20:0 primary.
        filtered = annotations[~annotations["name"].eq("1-acylLPC(20:0)")]

        guarded, removed = _drop_orphan_two_acyl_lpc(filtered)

        self.assertEqual(removed, 1)
        self.assertEqual(set(guarded["name"]), {
            "1-acylLPC(18:1)", "2-acylLPC(18:1)",
        })

    def test_two_acyl_lpe_cannot_survive_primary_coherence_filtering(self):
        annotations = pd.DataFrame({
            "name": ["2-acylLPE(18:1)"], "class": ["2-acylLPE"],
            "carbon": [18], "db": [1], "adduct": ["[M-H]-"],
            "isotopologue": [""],
        })

        guarded, removed = _drop_orphan_two_acyl_lpc(annotations)

        self.assertEqual(removed, 1)
        self.assertTrue(guarded.empty)

    def test_positional_coherence_uses_only_resolved_peak_series(self):
        rows = []
        for carbon in (16, 18, 20, 22):
            status = (
                "one_acyl_primary_only_no_two_acyl"
                if carbon == 22 else "paired_rt_order_supported"
            )
            rows.extend([
                {
                    "feat_feature_id": f"late-{carbon}",
                    "name": f"1-acylLPC({carbon}:0)",
                    "class": "1-acylLPC", "base": "",
                    "adduct": "[M+H]+", "carbon": carbon, "db": 0,
                    "ppm_error": 2.9, "rt_error": 0.15,
                    "positional_isomer_selected": True,
                    "positional_isomer_status": status,
                },
                {
                    "feat_feature_id": f"early-{carbon}",
                    "name": f"1-acylLPC({carbon}:0)",
                    "class": "1-acylLPC", "base": "",
                    "adduct": "[M+H]+", "carbon": carbon, "db": 0,
                    "ppm_error": 0.1, "rt_error": 0.0,
                    "positional_isomer_selected": False,
                    "positional_isomer_status": status,
                },
            ])
        scored, _ = _calculate_matlab_grid_coherence(
            pd.DataFrame(rows),
            settings=AutomaticAnchorSettings(grid_coherence_threshold=0.45),
        )
        resolved = scored[
            scored["positional_isomer_selected"].fillna(False).astype(bool)
        ]
        self.assertTrue(resolved["grid_coherence_assessed"].all())
        self.assertGreater(float(resolved["grid_coherence_score"].min()), 0.9)
        self.assertTrue(resolved["grid_coherence_pass"].all())

    def test_bundled_reference_contains_the_standard_spike_panel(self):
        root = DEFAULT_MATLAB_PIPELINE_REFERENCE
        reference = load_reference_data(
            code_dir=root / "Code",
            database_dir=root / "DataBase",
            rtdb_dir=root / "RTDB",
            project_dir=root / "Project",
        )
        quantities = {
            item.name: item.quantity for item in reference.internal_standards
        }
        self.assertEqual(len(quantities), 26)
        self.assertEqual(quantities["PC(14:0)2"], 2000)
        self.assertEqual(quantities["PE(14:0)2"], 500)
        self.assertEqual(quantities["TAG(14:0)3"], 500)

        project = load_bundled_compound_catalog(root)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "peakgrouplist_RP_Pos.txt").write_text(
                "name\tmzmed\trtmed\tnpeaks\tsample_A\n"
                "feature-1\t678.5068\t10.7\t1\t1000\n",
                encoding="utf-8",
            )
            xcms = build_rp_xcms_project(path, reference_root=root).project
        targets = build_simple_internal_standard_targets(
            xcms, reference, "RP", np.arange(1, project.count + 1)
        )
        self.assertTrue({
            "PC(14:0)2", "PE(14:0)2", "TAG(14:0)3"
        } <= set(targets.names))

    def test_project_internal_standard_file_overrides_bundled_fallback(self):
        root = DEFAULT_MATLAB_PIPELINE_REFERENCE
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "internal_standards.tsv").write_text(
                "\tnM\nPC(14:0)2\t10000\n", encoding="utf-8"
            )
            selected = _internal_standard_project_dir(project, root)
        self.assertEqual(selected, project.resolve())

    def test_internal_standard_is_labelled_and_audited(self):
        root = DEFAULT_MATLAB_PIPELINE_REFERENCE
        reference = load_reference_data(
            code_dir=root / "Code", database_dir=root / "DataBase",
            rtdb_dir=root / "RTDB", project_dir=root / "Project",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "peakgrouplist_RP_Pos.txt").write_text(
                "name\tmzmed\trtmed\tnpeaks\tsample_A\n"
                "feature-1\t678.5068\t10.7\t1\t1000\n",
                encoding="utf-8",
            )
            project = build_rp_xcms_project(
                path, reference_root=root
            ).project
        targets = build_simple_internal_standard_targets(
            project, reference, "RP", np.arange(1, project.compounds.count + 1)
        )
        index = targets.names.index("PC(14:0)2")
        compound = int(targets.compound_ids[index])
        code = int(targets.codes[index])
        frame = pd.DataFrame({
            "is_is": [True], "name": ["PC(28:0)"],
            "matlab_compound": [compound], "matlab_code": [code],
        })
        labelled = _label_internal_standards(frame, project, reference)
        self.assertEqual(labelled.loc[0, "name"], "PC(28:0) (IS)")
        self.assertEqual(
            labelled.loc[0, "internal_standard_name"], "PC(14:0)2"
        )
        self.assertEqual(labelled.loc[0, "internal_standard_quantity"], 2000)
        audit = _internal_standard_audit(
            project, reference,
            pd.DataFrame({"compound": [compound], "code": [code]}),
            labelled, labelled,
        )
        pc = audit[audit["internal_standard_name"].eq("PC(14:0)2")].iloc[0]
        self.assertEqual(pc["reported_traits"], "PC(28:0)")
        self.assertEqual(pc["status"], "accepted")

    def test_xcms_project_uses_signed_sorted_masses_and_preserves_feature_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = "name\tmzmed\trtmed\tnpeaks\tsample_A\n"
            (root / "peakgrouplist_RP_Pos.txt").write_text(
                common + "pos-7\t300.1\t10.0\t1\t1000\n", encoding="utf-8"
            )
            (root / "peakgrouplist_RP_Neg.txt").write_text(
                common + "neg-9\t500.2\t11.0\t1\t2000\n", encoding="utf-8"
            )
            bundle = build_rp_xcms_project(
                root, reference_root=DEFAULT_MATLAB_PIPELINE_REFERENCE
            )
        phase = bundle.project.phases["RP"]
        np.testing.assert_allclose(phase.mz_grouped, [-500.2, 300.1])
        np.testing.assert_allclose(phase.rt_grouped, [660.0, 600.0])
        self.assertEqual(bundle.row_stream, ("RP_Neg", "RP_Pos"))
        self.assertEqual(bundle.row_feature_id, ("neg-9", "pos-7"))
        self.assertEqual(bundle.project.metadata.sample_count, 1)

    def test_np_xcms_project_is_an_independent_phase_with_np_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = "name\tmzmed\trtmed\tnpeaks\tsample_A\n"
            (root / "peakgrouplist_NP_Pos.txt").write_text(
                common + "np-pos\t300.1\t8.0\t1\t1000\n", encoding="utf-8"
            )
            (root / "peakgrouplist_NP_Neg.txt").write_text(
                common + "np-neg\t500.2\t9.0\t1\t2000\n", encoding="utf-8"
            )
            # An RP file in the same folder must never enter the NP model.
            (root / "peakgrouplist_RP_Pos.txt").write_text(
                common + "rp-pos\t700.3\t10.0\t1\t3000\n", encoding="utf-8"
            )
            bundle = build_np_xcms_project(
                root, reference_root=DEFAULT_MATLAB_PIPELINE_REFERENCE
            )

        self.assertEqual(tuple(bundle.project.phases), ("NP",))
        np.testing.assert_allclose(
            bundle.project.phases["NP"].mz_grouped, [-500.2, 300.1]
        )
        self.assertEqual(bundle.row_stream, ("NP_Neg", "NP_Pos"))
        self.assertEqual(bundle.row_feature_id, ("np-neg", "np-pos"))
        self.assertNotIn("rp-pos", bundle.row_feature_id)
        self.assertIn("[M+CHO2]-", MATLAB_NP_ANNOTATION_ADDUCTS)
        self.assertIn("[M+Cl]-", MATLAB_NP_ANNOTATION_ADDUCTS)

    def test_automatic_engine_rejects_unknown_phase(self):
        with self.assertRaisesRegex(MatlabAnnotationError, "NP and/or RP"):
            run_matlab_annotation("unused", phases=("HILIC",))

    def test_automatic_engine_runs_both_phases_independently_by_default(self):
        def phase_result(project_dir, *, phase, **kwargs):
            annotation = pd.DataFrame({
                "name": ["PC(34:1)"],
                "isotopologue": [""],
                "feat_intensity": [1_000 if phase == "RP" else 900],
            })
            return {
                "streams": {f"{phase}_Pos": {"annotations": annotation}},
                "xcms_input_dir": str(project_dir),
                "matlab_reference_dir": str(
                    kwargs.get("reference_dir", DEFAULT_MATLAB_PIPELINE_REFERENCE)
                ),
                "matlab_phase": phase,
                "matlab_compound_audit": pd.DataFrame({
                    "phase": [phase], "status": ["annotated"],
                }),
                "matlab_positional_isomer_audit": pd.DataFrame({
                    "status": ["evaluated"],
                }),
                "anchor_settings": {},
                "internal_standard_source": "internal_standards.tsv",
                "internal_standards_expected": [],
            }

        with (
            patch(
                "matlab_annotation.engine.discover_standard_streams",
                return_value={"RP_Pos": "rp", "NP_Pos": "np"},
            ),
            patch(
                "matlab_annotation.engine._run_matlab_annotation_phase",
                side_effect=phase_result,
            ) as worker,
        ):
            result = run_matlab_annotation("unused")

        self.assertEqual(
            [call.kwargs["phase"] for call in worker.call_args_list],
            ["RP", "NP"],
        )
        self.assertEqual(set(result["streams"]), {"RP_Pos", "NP_Pos"})
        self.assertEqual(result["matlab_phases"], ("RP", "NP"))
        self.assertEqual(result["matlab_skipped_phases"], ())
        self.assertEqual(len(result["identified_lipids"]), 1)
        self.assertEqual(len(result["equivalent_lipids"]), 1)
        self.assertEqual(
            set(result["matlab_compound_audit"]["phase"]), {"RP", "NP"}
        )
        self.assertEqual(
            set(result["matlab_positional_isomer_audit"]["phase"]),
            {"RP", "NP"},
        )

    def test_vendored_subset_needs_no_optional_heavy_dependency(self):
        # ``native_store`` pulls in h5py and is deliberately not vendored.
        vendor = Path(__file__).resolve().parents[1] / "matlab_annotation/vendor"
        self.assertFalse((vendor / "native_store.py").exists())
        self.assertFalse((vendor / "gui.py").exists())
        sources = "\n".join(
            path.read_text() for path in vendor.glob("*.py")
        )
        self.assertNotIn("import h5py", sources)

    def test_upstream_provenance_is_recorded(self):
        upstream = (
            Path(__file__).resolve().parents[1]
            / "matlab_annotation" / "UPSTREAM.md"
        )
        self.assertTrue(upstream.is_file())
        text = upstream.read_text()
        self.assertIn("AI_Lipidomics_Pipeline", text)
        # A 40-character commit hash pins what was copied.
        self.assertRegex(text, r"\b[0-9a-f]{40}\b")


class DecodeTests(unittest.TestCase):
    def test_lipid_class_moves_the_subclass_marker_onto_the_class(self):
        self.assertEqual(lipid_class_from_template("TG(x:x)"), "TG")
        self.assertEqual(lipid_class_from_template("PC(O-x:x)"), "PC-O")
        self.assertEqual(lipid_class_from_template("PE(P-x:x)"), "PE-P")
        self.assertEqual(lipid_class_from_template("HexCer(dx:x)"), "HexCer-d")

    def test_np_pc_and_pe_ether_isomers_use_one_o_plus_p_identity(self):
        source = pd.DataFrame([
            {"name": "PE(O-36:2)", "display_name": "PE(O-36:2) [M-H]-", "class": "PE-O", "db": 2},
            {"name": "PE(P-36:1)", "display_name": "PE(P-36:1) [M-H]-", "class": "PE-P", "db": 1},
            {"name": "PC(P-38:4)", "display_name": "PC(P-38:4) [M+H]+", "class": "PC-P", "db": 4},
            {"name": "PC(36:2)", "display_name": "PC(36:2) [M+H]+", "class": "PC", "db": 2},
        ])

        combined = combine_np_ether_plasmalogen_labels(source)

        self.assertEqual(combined.loc[0, "name"], "PE(O+P-36:2)")
        self.assertEqual(combined.loc[1, "name"], "PE(O+P-36:2)")
        self.assertEqual(combined.loc[0, "class"], "PE-O+P")
        self.assertEqual(combined.loc[1, "class"], "PE-O+P")
        self.assertEqual(combined.loc[1, "db"], 2)
        self.assertEqual(combined.loc[2, "name"], "PC(O+P-38:5)")
        self.assertEqual(combined.loc[2, "class"], "PC-O+P")
        self.assertEqual(combined.loc[3, "name"], "PC(36:2)")
        self.assertEqual(
            combined.loc[1, "np_unresolved_source_name"], "PE(P-36:1)"
        )

    def test_np_o_plus_p_rule_applies_to_lyso_and_other_ether_classes(self):
        source = pd.DataFrame([
            {"name": "LPC(O-18:1)", "class": "LPC-O", "db": 1},
            {"name": "LPC(P-18:0)", "class": "LPC-P", "db": 0},
            {"name": "TG(P-52:3)", "class": "TG-P", "db": 3},
            {"name": "PE(O+P-36:2)", "class": "PE", "base": "O", "db": 2},
        ])

        combined = combine_np_ether_plasmalogen_labels(source)

        self.assertEqual(combined.loc[0, "name"], "LPC(O+P-18:1)")
        self.assertEqual(combined.loc[1, "name"], "LPC(O+P-18:1)")
        self.assertEqual(combined.loc[1, "class"], "LPC-O+P")
        self.assertEqual(combined.loc[2, "name"], "TG(O+P-52:4)")
        self.assertEqual(combined.loc[2, "class"], "TG-O+P")
        self.assertEqual(combined.loc[3, "class"], "PE-O+P")
        self.assertEqual(combined.loc[3, "base"], "")



@unittest.skipUnless(
    (PROJECT / "Processing/Annotation/MS.mat").is_file(),
    "reference MATLAB projects are not available on this machine",
)
class ProjectBackedTests(unittest.TestCase):
    """Checks that need a real MATLAB project.

    These run against the annotated reference datasets in ``example_data_CFM``
    and are skipped where those are not present.
    """

    @classmethod
    def setUpClass(cls):
        cls.annotation_dir = PROJECT / "Processing" / "Annotation"
        cls.project = load_legacy_project(cls.annotation_dir)

    def test_discovery_accepts_the_root_and_the_annotation_folder(self):
        self.assertEqual(
            discover_matlab_annotation_project(PROJECT), self.annotation_dir
        )
        self.assertEqual(
            discover_matlab_annotation_project(self.annotation_dir),
            self.annotation_dir,
        )

    def test_discovery_walks_up_from_a_sibling_project_folder(self):
        # Mirrors how peak-group-list discovery already resolves a project:
        # pointing anywhere inside it is enough.
        self.assertEqual(
            discover_matlab_annotation_project(PROJECT / "ProjectDocumentation"),
            self.annotation_dir,
        )

    def test_discovery_returns_none_outside_any_matlab_project(self):
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "some" / "folder"
            nested.mkdir(parents=True)
            self.assertIsNone(discover_matlab_annotation_project(nested))

    def test_decoded_names_reproduce_matlab_display_strings(self):
        # The structured decode must stay exactly equivalent to the vendored
        # naming path; drift here would silently rename lipids.
        rng = np.random.default_rng(0)
        checked = 0
        for compound in range(1, self.project.compounds.count + 1):
            attr = self.project.compounds.composition_attr[compound - 1]
            template = compound_display_template(self.project, compound - 1)
            for phase in ("NP", "RP"):
                dimensions = composition_dimensions(attr, phase)
                total = int(np.prod([len(d) for d in dimensions], dtype=np.int64))
                if total < 1:
                    continue
                for code in rng.integers(1, total + 1, size=min(4, total)):
                    expected = resolved_compound_name(
                        template, attr, phase, int(code)
                    )
                    decoded = decode_annotation_code(
                        self.project, compound, phase, int(code)
                    )
                    self.assertEqual(decoded.display_name, expected)
                    checked += 1
        self.assertGreater(checked, 500)

    def test_cached_decoder_matches_the_one_shot_decode(self):
        decoder = CodeDecoder(self.project, 1, "RP")
        for code in (1, 2, 41, 4100, 100000):
            self.assertEqual(
                decoder.decode(code),
                decode_annotation_code(self.project, 1, "RP", code),
            )

    def test_monoisotopic_codes_are_unlabelled_and_cover_the_surface(self):
        decoder = CodeDecoder(self.project, 1, "RP")
        codes = _monoisotopic_codes(decoder)
        expected = (
            len(decoder.axis(0)) * len(decoder.axis(1))
            * len(decoder.axis(3)) * len(decoder.axis(4))
        )
        self.assertEqual(len(codes), expected)
        self.assertEqual(len(set(codes.tolist())), expected)
        for code in codes[:: max(len(codes) // 20, 1)]:
            self.assertEqual(decoder.decode(int(code)).isotopologue, "")

    def test_sibling_groups_cover_each_set_once_in_either_order(self):
        groups = _sibling_groups(self.project, "RP", only_accepted=False)
        self.assertEqual(len(groups), len(set(groups)))
        # Compounds 36 and 37 are siblings; each resolves to the same set with
        # itself in front, and the sweep must not run that set twice.
        self.assertEqual(len({36, 37} & set(groups)), 1)

    def test_accepted_only_sweep_matches_the_projects_own_coverage(self):
        accepted = np.asarray(
            self.project.compounds.accepted["RP"]
        ).reshape(-1)
        flagged = set((np.flatnonzero(np.nan_to_num(accepted) != 0) + 1).tolist())
        groups = set(_sibling_groups(self.project, "RP", only_accepted=True))

        self.assertTrue(groups <= flagged)
        self.assertLess(len(groups), len(_sibling_groups(
            self.project, "RP", only_accepted=False
        )))


class RtWindowTests(unittest.TestCase):
    def test_review_window_covers_the_widest_deviation_the_mapper_kept(self):
        annotations = pd.DataFrame({"rt_error": [0.01, -0.22, 0.13]})
        self.assertAlmostEqual(_observed_rt_window(annotations), 0.22)

    def test_review_window_falls_back_without_annotations(self):
        self.assertEqual(_observed_rt_window(pd.DataFrame()), 0.5)


if __name__ == "__main__":
    unittest.main()
