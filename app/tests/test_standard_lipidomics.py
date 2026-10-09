import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from standard_lipidomics.workflow import (
    DEFAULT_STANDARD_PPM_TOLERANCE,
    MAX_STANDARD_PPM_TOLERANCE,
    StandardStream,
    annotate_standard_batch,
    fit_automatic_class_rt_models,
    discover_standard_streams,
    load_standard_feature_list,
    parse_standard_stream,
)
from annotator.reference import load_reference


def reference_frame():
    return pd.DataFrame({
        "name": ["PC(34:1)", "PE(34:1)"],
        "class": ["PC", "PE"],
        "formula": ["C42H82NO8P", "C39H76NO8P"],
        "adduct": ["[M+H]+", "[M-H]-"],
        "charge": [1, -1],
        "mz": [760.585, 716.524],
        "RT_RP_median_min": [8.0, 7.0],
        "n_RT": [10, 10],
    })


class StandardLipidomicsTests(unittest.TestCase):
    def test_project_root_auto_discovers_annotation_streams(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            annotation = project / "Processing" / "Annotation"
            annotation.mkdir(parents=True)
            expected = annotation / "peakgrouplist_RP_Pos.txt"
            expected.write_text("name\tmzmed\trtmed\n", encoding="utf-8")

            self.assertEqual(
                discover_standard_streams(project),
                {"RP_Pos": expected},
            )

    def test_default_standard_mass_window_is_four_ppm(self):
        reference_mz = 760.585
        inside = reference_mz * (1.0 + 3.9e-6)
        outside = reference_mz * (1.0 + 4.1e-6)
        source = (
            "name\tmzmed\trtmed\tnpeaks\tsample_a\n"
            f"inside\t{inside:.9f}\t8.0\t10\t100\n"
            f"outside\t{outside:.9f}\t8.0\t10\t100\n"
        ).encode()

        result = annotate_standard_batch(
            {"RP_Pos": source},
            load_reference_frame(reference_frame()),
            align_rp_rt=False,
            adapt_class_rt=False,
        )

        self.assertEqual(DEFAULT_STANDARD_PPM_TOLERANCE, 4.0)
        self.assertEqual(MAX_STANDARD_PPM_TOLERANCE, 20.0)
        self.assertEqual(
            set(result["all_candidates"]["feature_id"].astype(str)),
            {"inside"},
        )
        annotate_standard_batch(
            {"RP_Pos": source},
            load_reference_frame(reference_frame()),
            ppm_tolerance=20.0,
        )
        with self.assertRaisesRegex(ValueError, "no larger than 20 ppm"):
            annotate_standard_batch(
                {"RP_Pos": source},
                load_reference_frame(reference_frame()),
                ppm_tolerance=20.1,
            )

    def test_grid_coherence_is_enabled_at_point_45_by_default(self):
        source = (
            b"name\tmzmed\trtmed\tnpeaks\tGroup_01_wt\tsample_a\n"
            b"1\t760.585\t8.0\t10\t1\t100\n"
        )
        result = annotate_standard_batch(
            {"RP_Pos": source}, load_reference_frame(reference_frame()),
            adapt_class_rt=False,
        )

        self.assertTrue(result["grid_coherence_enabled"])
        self.assertEqual(result["grid_coherence_threshold"], 0.45)
        self.assertIn("grid_coherence_audit", result)
        self.assertIn("grid_coherence_score", result["all_candidates"])
        self.assertTrue(bool(
            result["streams"]["RP_Pos"]["grid_coherence_audit"]
            .iloc[0]["used_for_annotation"]
        ))

    def test_xcms_group_structure_is_preserved_as_sample_metadata(self):
        source = (
            b"name\tmzmed\trtmed\tnpeaks\tGroup_01_control\tGroup_02_case\tGroup_03_QC_pool\ta1\ta2\tb1\tb2\tqc1\n"
            b"1\t760.585\t8.0\t5\t2\t2\t1\t100\t110\t200\t210\t150\n"
        )
        features = load_standard_feature_list(source)
        self.assertNotIn("Group_01_control", features.columns)
        self.assertEqual(features.attrs["sample_group_mapping_status"], "validated")
        self.assertEqual(features.attrs["sample_group_map"]["a2"], "Group_01_control")
        self.assertEqual(features.attrs["sample_group_map"]["qc1"], "Group_03_QC_pool")

    def test_grid_coherence_can_filter_assessed_outliers(self):
        ref = pd.DataFrame({
            "name": ["PC(32:0)", "PC(34:0)", "PC(36:0)", "PC(34:1)"],
            "class": ["PC"] * 4,
            "formula": ["C40H80NO8P"] * 4,
            "adduct": ["[M+H]+"] * 4,
            "charge": [1] * 4,
            "mz": [700.0, 728.0, 756.0, 726.0],
            "RT_RP_median_min": [8.0, 8.2, 8.4, 8.2],
            "n_RT": [20] * 4,
        })
        rows = [
            f"feature-{index}\t{mz}\t{rt}\t10\t100"
            for index, (mz, rt) in enumerate(
                [(700.0, 8.01), (728.0, 8.21), (756.0, 8.39),
                 (726.0, 9.00)],
                start=1,
            )
        ]
        source = (
            "name\tmzmed\trtmed\tnpeaks\tsample_a\n"
            + "\n".join(rows) + "\n"
        ).encode()
        result = annotate_standard_batch(
            {"RP_Pos": source}, load_reference_frame(ref),
            adapt_class_rt=False, use_grid_coherence=True,
            grid_coherence_threshold=0.90,
        )

        audit = result["streams"]["RP_Pos"]["grid_coherence_audit"].iloc[0]
        self.assertTrue(bool(audit["used_for_annotation"]))
        self.assertGreater(int(audit["filtered_annotations"]), 0)
        self.assertTrue(
            result["streams"]["RP_Pos"]["all_pairs"]["grid_coherence_outlier"]
            .fillna(False).astype(bool).any()
        )

    def test_stream_metadata_is_inferred_from_matlab_filename(self):
        self.assertEqual(
            parse_standard_stream("peakgrouplist_RP_Pos.txt"),
            StandardStream("RP_Pos", "RP", "Pos"),
        )
        self.assertEqual(
            parse_standard_stream("anything-np_neg.tsv").name, "NP_Neg"
        )

    def test_xcms_group_counts_are_not_sample_intensities(self):
        source = io.BytesIO(
            b"name\tmzmed\trtmed\tnpeaks\tGroup_01_wt\tsample_a\tsample_b\n"
            b"17\t760.585\t8.0\t2\t2\t100\t80\n"
        )
        result = load_standard_feature_list(source)
        self.assertNotIn("Group_01_wt", result.columns)
        self.assertEqual(result.attrs["sample_columns"], ["sample_a", "sample_b"])
        self.assertEqual(result.loc[0, "intensity"], 100)

    def test_batch_keeps_rp_rt_and_marks_np_as_mass_only(self):
        rp = (
            b"name\tmzmed\trtmed\tnpeaks\tGroup_01_wt\tsample_a\n"
            b"1\t760.585\t8.0\t1\t1\t100\n"
        )
        np_stream = (
            b"name\tmzmed\trtmed\tnpeaks\tGroup_01_wt\tsample_a\n"
            b"2\t760.585\t2.5\t1\t1\t100\n"
        )
        result = annotate_standard_batch(
            {"RP_Pos": rp, "NP_Pos": np_stream},
            load_reference_frame(reference_frame()),
            align_rp_rt=False,
        )
        identified = result["identified_lipids"]
        self.assertEqual(set(identified["stream"]), {"RP_Pos"})
        modes = dict(zip(
            result["all_candidates"]["stream"],
            result["all_candidates"]["rt_evidence_mode"],
        ))
        self.assertEqual(modes["RP_Pos"], "automatic_class_RP_reference")
        self.assertEqual(modes["NP_Pos"], "mass_only_no_NP_reference")
        np_candidates = result["all_candidates"][
            result["all_candidates"]["stream"].eq("NP_Pos")
        ]
        self.assertEqual(
            set(np_candidates["auto_annotation_status"]), {"mass_only"}
        )
        self.assertFalse(np_candidates["automatic_accepted"].any())
        self.assertNotIn(
            "intensity::Group_01_wt", identified.columns
        )
        self.assertIn("intensity::sample_a", identified.columns)

    def test_class_rt_offset_requires_supported_cross_validated_improvement(self):
        pairs = pd.DataFrame({
            "feat_feature_id": [str(i) for i in range(6)],
            "name": [f"TG({40 + i}:2)" for i in range(6)],
            "class": ["TG"] * 6,
            "adduct": ["[M+NH4]+"] * 6,
            "feat_rt": [8.30, 8.51, 8.69, 8.91, 9.11, 9.29],
            "ref_rt": [8.00, 8.20, 8.40, 8.60, 8.80, 9.00],
            "n_RT": [20] * 6,
            "carbon": [40, 42, 44, 46, 48, 50],
            "db": [2] * 6,
        })

        models = fit_automatic_class_rt_models(pairs, 0.1)

        self.assertEqual(len(models), 1)
        self.assertTrue(bool(models.loc[0, "validated"]))
        self.assertEqual(models.loc[0, "model_type"], "validated_class_offset")
        self.assertAlmostEqual(models.loc[0, "offset_min"], 0.30, places=2)
        self.assertGreater(models.loc[0, "improvement_min"], 0.2)

    def test_ambiguous_isobars_are_candidates_but_not_automatic_ids(self):
        source = (
            b"name\tmzmed\trtmed\tnpeaks\tGroup_01_wt\tsample_a\n"
            b"1\t760.585\t8.0\t10\t1\t100\n"
        )
        ref = pd.DataFrame({
            "name": ["PC(34:1)", "PE(37:1)"],
            "class": ["PC", "PE"],
            "formula": ["C42H82NO8P", "C42H82NO8P"],
            "adduct": ["[M+H]+", "[M+H]+"],
            "charge": [1, 1],
            "mz": [760.585, 760.585],
            "RT_RP_median_min": [8.0, 8.0],
            "n_RT": [20, 20],
        })

        result = annotate_standard_batch(
            {"RP_Pos": source}, load_reference_frame(ref),
            align_rp_rt=False, adapt_class_rt=False,
        )

        self.assertTrue(result["identified_lipids"].empty)
        self.assertIn(
            "ambiguous_isobar",
            set(result["all_candidates"]["auto_annotation_status"]),
        )
        self.assertFalse(result["all_candidates"]["automatic_accepted"].any())

    def test_accepted_cross_stream_measurements_become_primary_and_equivalent(self):
        positive = (
            b"name\tmzmed\trtmed\tnpeaks\tGroup_01_wt\tsample_a\n"
            b"1\t760.585\t8.0\t10\t1\t100\n"
        )
        negative = (
            b"name\tmzmed\trtmed\tnpeaks\tGroup_01_wt\tsample_a\n"
            b"2\t758.570\t8.0\t10\t1\t100\n"
        )
        ref = pd.DataFrame({
            "name": ["PC(34:1)", "PC(34:1)"],
            "class": ["PC", "PC"],
            "formula": ["C42H82NO8P", "C42H82NO8P"],
            "adduct": ["[M+H]+", "[M-H]-"],
            "charge": [1, -1],
            "mz": [760.585, 758.570],
            "RT_RP_median_min": [8.0, 8.0],
            "n_RT": [20, 20],
        })

        result = annotate_standard_batch(
            {"RP_Pos": positive, "RP_Neg": negative},
            load_reference_frame(ref),
            align_rp_rt=False,
            adapt_class_rt=False,
        )

        self.assertEqual(len(result["identified_lipids"]), 1)
        self.assertEqual(len(result["equivalent_lipids"]), 1)
        self.assertEqual(
            set(result["automatic_annotations"]["quantitative_role"]),
            {"primary", "equivalent"},
        )
        self.assertTrue(
            result["automatic_annotations"]["automatic_annotation"].all()
        )


def load_reference_frame(frame):
    """Mirror load_reference enrichment without filesystem fixtures."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "reference.csv"
        frame.to_csv(path, index=False)
        return load_reference(path)


if __name__ == "__main__":
    unittest.main()
