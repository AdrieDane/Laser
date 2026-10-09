import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from annotator.np_rt import (
    attach_np_rt_reference,
    fit_np_rt_anchor_alignment,
    load_np_rt_reference,
    np_rt_identity,
)
from annotator.reference import load_reference
from standard_lipidomics.workflow import annotate_standard_batch


class NormalPhaseRtTests(unittest.TestCase):
    def test_np_stream_without_any_mass_pair_returns_an_empty_audited_result(self):
        reference_frame = pd.DataFrame({
            "name": ["PC(34:1)"],
            "class": ["PC"],
            "formula": [""],
            "adduct": ["[M+H]+"],
            "charge": [1],
            "mz": [800.0],
            "RT_RP_median_min": [12.0],
            "n_RT": [10],
        })
        np_reference = pd.DataFrame({
            "name": ["PC(34:1)"],
            "class": ["PC"],
            "np_rt_seconds": [300.0],
            "np_rt_min": [5.0],
            "np_rt_source": ["test"],
        })
        source = io.BytesIO(
            b"name\tmzmed\trtmed\tnpeaks\tsample_a\n"
            b"1\t100.0\t5.0\t10\t100\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            reference_frame.to_csv(path, index=False)
            reference = load_reference(path)

        result = annotate_standard_batch(
            {"NP_Pos": source},
            reference,
            np_rt_reference=np_reference,
        )

        self.assertTrue(result["identified_lipids"].empty)
        alignment = result["streams"]["NP_Pos"]["np_rt_alignment"].iloc[0]
        self.assertFalse(bool(alignment["validated"]))
        self.assertEqual(alignment["validation_reason"], "no_mass_pairs")

    def test_loader_converts_seconds_to_minutes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retention_times.csv"
            pd.DataFrame({
                "name": ["PC(34:1)"],
                "class": ["PC"],
                "IS": [""],
                "NP_RT": [480.0],
                "RP_RT": [900.0],
            }).to_csv(path, index=False)

            reference = load_np_rt_reference(path)

            self.assertAlmostEqual(reference.loc[0, "np_rt_min"], 8.0)
            self.assertEqual(reference.loc[0, "np_rt_seconds"], 480.0)

    def test_np_identity_combines_ether_and_positional_lyso_names(self):
        self.assertEqual(
            np_rt_identity("PC(O-34:2)", "PC[O+P]"),
            "PC(O+P-34:2)",
        )
        self.assertEqual(
            np_rt_identity("PC(P-34:1)", "PC[O+P]"),
            "PC(O+P-34:2)",
        )
        self.assertEqual(
            np_rt_identity("1-acyl LPC(18:1)", "LPC"),
            "LPC(18:1)",
        )

    def test_reference_attachment_uses_combined_np_identity(self):
        reference = pd.DataFrame({
            "name": ["PC(O-34:2)", "PC(P-34:1)"],
            "class": ["PC", "PC"],
            "adduct": ["[M+H]+", "[M+H]+"],
            "matlab_policy_class": ["PC[O+P]", "PC[O+P]"],
            "ref_rt": [12.0, 12.0],
            "n_RT": [10, 10],
        })
        np_reference = pd.DataFrame({
            "name": ["PC(O+P-34:2)"],
            "class": ["PC[O+P]"],
            "np_rt_seconds": [480.0],
            "np_rt_min": [8.0],
            "np_rt_source": ["test"],
        })

        attached, audit = attach_np_rt_reference(
            reference, np_reference, "NP_Pos"
        )

        self.assertEqual(len(attached), 2)
        self.assertTrue(attached["ref_rt"].eq(8.0).all())
        self.assertTrue(audit["np_rt_reference_available"].all())

    def test_anchor_alignment_finds_shift_cluster_before_identity_choice(self):
        rows = []
        for index in range(18):
            rows.append({
                "feat_feature_id": f"true-{index}",
                "name": f"PC({30 + index}:1)",
                "np_rt_identity": f"PC({30 + index}:1)",
                "class": ["PC", "PE", "SM"][index % 3],
                "adduct": "[M+H]+",
                "feat_rt": 4.0 + index * 0.2 + 0.35,
                "ref_rt": 4.0 + index * 0.2,
            })
            rows.append({
                "feat_feature_id": f"noise-{index}",
                "name": f"PC({30 + index}:1)",
                "np_rt_identity": f"PC({30 + index}:1)",
                "class": ["PC", "PE", "SM"][index % 3],
                "adduct": "[M+H]+",
                "feat_rt": 4.0 + index * 0.2 - 0.55 + index * 0.01,
                "ref_rt": 4.0 + index * 0.2,
            })

        model = fit_np_rt_anchor_alignment(
            pd.DataFrame(rows),
            stream="NP_Pos",
            rt_tolerance_min=0.30,
        ).iloc[0]

        self.assertTrue(bool(model["validated"]))
        self.assertAlmostEqual(model["offset_min"], 0.35, places=2)
        self.assertGreaterEqual(model["anchor_nodes"], 12)

    def test_validated_np_grid_resolves_mass_isobars(self):
        names = [f"Anchor{i}({20 + i}:1)" for i in range(12)]
        classes = ["PC", "PE", "SM"] * 4
        masses = [500.0 + i * 10 for i in range(12)]
        rt_seconds = [240.0 + i * 24 for i in range(12)]
        names += ["PC(34:1)", "PE(37:1)"]
        classes += ["PC", "PE"]
        masses += [800.0, 800.0]
        rt_seconds += [420.0, 510.0]
        reference_frame = pd.DataFrame({
            "name": names,
            "class": classes,
            "formula": [""] * len(names),
            "adduct": ["[M+H]+"] * len(names),
            "charge": [1] * len(names),
            "mz": masses,
            "RT_RP_median_min": [12.0] * len(names),
            "n_RT": [10] * len(names),
        })
        np_reference = pd.DataFrame({
            "name": names,
            "class": classes,
            "np_rt_seconds": rt_seconds,
            "np_rt_min": np.asarray(rt_seconds) / 60.0,
            "np_rt_source": ["test"] * len(names),
        })
        rows = [
            (
                str(index + 1),
                mass,
                seconds / 60.0 + 0.20,
            )
            for index, (mass, seconds) in enumerate(
                zip(masses[:12], rt_seconds[:12])
            )
        ]
        rows.append(("99", 800.0, 420.0 / 60.0 + 0.20))
        source = io.BytesIO(
            (
                "name\tmzmed\trtmed\tnpeaks\tsample_a\n"
                + "".join(
                    f"{feature}\t{mass}\t{rt}\t10\t100\n"
                    for feature, mass, rt in rows
                )
            ).encode()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            reference_frame.to_csv(path, index=False)
            reference = load_reference(path)

        result = annotate_standard_batch(
            {"NP_Pos": source},
            reference,
            np_rt_reference=np_reference,
            np_rt_tolerance_min=0.30,
        )
        candidates = result["all_candidates"]
        feature = candidates[candidates["feature_id"].astype(str).eq("99")]

        self.assertEqual(set(feature["feature_identity"]), {"PC(34:1)"})
        self.assertTrue(bool(feature.iloc[0]["automatic_accepted"]))
        self.assertEqual(
            feature.iloc[0]["auto_annotation_status"],
            "high_confidence_np_grid",
        )
        alignment = result["streams"]["NP_Pos"]["np_rt_alignment"].iloc[0]
        self.assertTrue(bool(alignment["validated"]))
        self.assertAlmostEqual(alignment["offset_min"], 0.20, places=2)

    def test_rejected_alignment_preserves_mass_only_candidates(self):
        names = ["PC(34:1)", "PE(37:1)"]
        reference_frame = pd.DataFrame({
            "name": names,
            "class": ["PC", "PE"],
            "formula": ["", ""],
            "adduct": ["[M+H]+", "[M+H]+"],
            "charge": [1, 1],
            "mz": [800.0, 800.0],
            "RT_RP_median_min": [12.0, 12.0],
            "n_RT": [10, 10],
        })
        np_reference = pd.DataFrame({
            "name": names,
            "class": ["PC", "PE"],
            "np_rt_seconds": [300.0, 600.0],
            "np_rt_min": [5.0, 10.0],
            "np_rt_source": ["test", "test"],
        })
        source = io.BytesIO(
            b"name\tmzmed\trtmed\tnpeaks\tsample_a\n"
            b"1\t800.0\t7.5\t10\t100\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            reference_frame.to_csv(path, index=False)
            reference = load_reference(path)

        result = annotate_standard_batch(
            {"NP_Pos": source},
            reference,
            np_rt_reference=np_reference,
        )
        candidates = result["all_candidates"]
        alignment = result["streams"]["NP_Pos"]["np_rt_alignment"].iloc[0]

        self.assertFalse(bool(alignment["validated"]))
        self.assertEqual(
            set(candidates["feature_identity"]),
            {"PC(34:1)", "PE(37:1)"},
        )
        self.assertTrue(
            candidates["rt_evidence_mode"].eq(
                "unvalidated_NP_reference_mass_only"
            ).all()
        )
        self.assertFalse(candidates["automatic_accepted"].any())


if __name__ == "__main__":
    unittest.main()
