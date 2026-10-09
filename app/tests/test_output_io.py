from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from errors import OutputWriteError
from output_io import (
    StructuredStageLogger,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    write_completion_marker,
)


class OutputIoTests(unittest.TestCase):
    def test_atomic_writers_leave_complete_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_write_text(root / "a.txt", "complete")
            atomic_write_json(root / "a.json", {"value": 1})
            atomic_write_csv(root / "a.csv", pd.DataFrame({"x": [1]}))
            atomic_write_csv(
                root / "a.tsv", pd.DataFrame({"x": [1]}), sep="\t",
            )
            self.assertEqual((root / "a.txt").read_text(), "complete")
            self.assertIn('"value": 1', (root / "a.json").read_text())
            self.assertEqual(pd.read_csv(root / "a.csv")["x"].tolist(), [1])
            self.assertEqual(pd.read_csv(root / "a.tsv", sep="\t")["x"].tolist(), [1])

    def test_failed_replace_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.txt"
            path.write_text("previous")
            with patch("output_io.os.replace", side_effect=OSError("offline")):
                with self.assertRaises(OutputWriteError):
                    atomic_write_text(path, "partial")
            self.assertEqual(path.read_text(), "previous")

    def test_strict_json_rejects_nan_without_replacing_previous_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text('{"state": "previous"}\n')
            with self.assertRaises(OutputWriteError):
                atomic_write_json(path, {"value": float("nan")}, allow_nan=False)
            self.assertEqual(path.read_text(), '{"state": "previous"}\n')

    def test_structured_log_and_completion_marker_are_machine_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = atomic_write_text(root / "result.txt", "done")
            logger = StructuredStageLogger(root / "run.jsonl")
            logger.record("annotation", "started")
            logger.record("annotation", "completed", rows=12)
            marker = write_completion_marker(root, [result])
            self.assertIn(sha256_file(result), marker.read_text())
            self.assertEqual(len((root / "run.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
