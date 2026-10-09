from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from annotator.standard_annotated_inputs import (
    append_standard_annotations,
    write_standard_annotated_inputs,
)


class StandardAnnotatedInputTests(unittest.TestCase):
    def test_appends_annotations_without_changing_original_rows_or_order(self):
        source = (
            b"name\tmzmed\trtmed\tsample_A\r\n"
            b"feature-2\t734.500200000\t7.0200\t00100\r\n"
            b"feature-1\t760.5\t7.2\t2.0e+06\r\n"
            b"feature-3\t786.6003\t7.410\t\r\n"
        )
        annotations = pd.DataFrame({
            "stream": ["RP_Pos", "RP_Pos", "RP_Pos"],
            "feat_feature_id": ["feature-1", "feature-1", "feature-2"],
            "name": ["PC(34:1)", "PC(34:1)", "PC(32:0)"],
            "adduct": ["[M+H]+", "[M+Na]+", "[M+H]+"],
        })

        rendered = append_standard_annotations(
            source, annotations, stream="RP_Pos"
        ).decode("utf-8")

        self.assertIn("\r\n", rendered)
        lines = rendered.splitlines()
        self.assertEqual(
            lines[0],
            "name\tmzmed\trtmed\tsample_A\tLASER_annotation\tLASER_adduct",
        )
        self.assertEqual(
            lines[1],
            "feature-2\t734.500200000\t7.0200\t00100\tPC(32:0)\t[M+H]+",
        )
        self.assertEqual(
            lines[2],
            "feature-1\t760.5\t7.2\t2.0e+06\t"
            "PC(34:1); PC(34:1)\t[M+H]+; [M+Na]+",
        )
        self.assertEqual(
            lines[3], "feature-3\t786.6003\t7.410\t\t\t"
        )

    def test_writes_one_named_copy_per_input_stream(self):
        source = b"name\tmzmed\trtmed\n1\t500\t5\n"
        annotations = pd.DataFrame({
            "stream": ["NP_Neg"], "feature_id": ["1"],
            "feature_identity": ["PE(34:1)"], "adduct": ["[M-H]-"],
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "peakgrouplist_NP_Neg.txt"
            original.write_bytes(source)
            paths = write_standard_annotated_inputs(
                {"NP_Neg": original}, annotations, root / "output"
            )

            self.assertEqual(
                paths["NP_Neg"].name,
                "peakgrouplist_NP_Neg_LASER_annotated.txt",
            )
            self.assertIn("PE(34:1)\t[M-H]-", paths["NP_Neg"].read_text())


if __name__ == "__main__":
    unittest.main()
