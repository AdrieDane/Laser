"""The scrollable contact sheet of annotated lipid traces."""

import re
import sqlite3
import tempfile
import zlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from annotator.feature_traces import build_feature_trace_databases  # noqa: F401
from annotator.trace_gallery import (
    collect_tiles,
    sample_colours,
    render_gallery,
    write_gallery_page,
    write_trace_gallery,
)


def _trace_db(path: Path, feature_ids, samples=("S1.d", "S2.d", "S3.d")):
    """Minimal database in the shape load_feature_traces expects."""
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE traces (feature_id TEXT, sample_name TEXT, "
        "peak_height REAL, is_representative INTEGER, rt_values BLOB, "
        "rt_intensities BLOB, mobility_values BLOB, mobility_intensities BLOB)"
    )
    for feature_id in feature_ids:
      for index, sample in enumerate(samples):
        rt = np.linspace(9.5, 10.5, 24).astype("<f4")
        scale = 1e5 / (index + 1)
        intensity = (np.exp(-((rt - 10.0) ** 2) / 0.02) * scale).astype("<f4")
        mobility = np.linspace(1.20, 1.32, 18).astype("<f4")
        mobility_intensity = (
            np.exp(-((mobility - 1.26) ** 2) / 4e-4) * scale
        ).astype("<f4")
        def pack(values):
            return zlib.compress(values.tobytes())
        connection.execute(
            "INSERT INTO traces VALUES (?,?,?,?,?,?,?,?)",
            (str(feature_id), sample, scale, 1 if index == 0 else 0,
             pack(rt), pack(intensity), pack(mobility), pack(mobility_intensity)),
        )
    connection.commit()
    connection.close()


def _annotations():
    return pd.DataFrame({
        "feat_feature_id": ["11", "12", "13"],
        "name": ["PC(34:1)", "PC(36:2)", "PE(34:1)"],
        "class": ["PC", "PC", "PE"],
        "adduct": "[M+H]+",
        "carbon": [34.0, 36.0, 34.0],
        "db": [1.0, 2.0, 1.0],
        "feat_mz": [760.5851, 786.6007, 718.5381],
        "feat_rt": [10.0, 10.4, 9.8],
        "rt_expected": [10.02, 10.38, 9.75],
        "msms_status": ["supported", "no_msms", "not_detected"],
        "score": [0.4, 0.5, 0.6],
    })


class TraceGalleryTests(unittest.TestCase):

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.database = self.directory / "traces.sqlite"
        _trace_db(self.database, ["11", "12", "13"])

    def test_a_tile_is_produced_for_each_annotated_lipid(self):
        tiles = collect_tiles(_annotations(), self.database)

        self.assertEqual(len(tiles), 3)
        self.assertEqual(
            {panel.feature_id for tile in tiles for panel in tile.panels},
            {"11", "12", "13"},
        )

    def test_every_sample_is_drawn_in_its_own_colour(self):
        tiles = collect_tiles(_annotations(), self.database)

        self.assertEqual(len(tiles[0].panels[0].samples), 3)
        page = render_gallery(tiles)
        # One polyline per sample, each with a distinct stroke.
        strokes = set(re.findall(r'<polyline points="[^"]*" stroke="(#[0-9a-f]{6})"', page))
        self.assertGreaterEqual(len(strokes), 3)

    def test_a_sample_keeps_one_colour_across_the_whole_page(self):
        colours = sample_colours(["S1.d", "S2.d", "S3.d"])

        self.assertEqual(len(set(colours.values())), 3)
        # Colour follows the sample, not its position in any one panel.
        self.assertEqual(colours, sample_colours(["S3.d", "S1.d", "S2.d"]))

    def test_each_panel_carries_labelled_axes(self):
        page = render_gallery(collect_tiles(_annotations(), self.database))

        self.assertIn('class="axis"', page)
        self.assertIn('class="tick-label"', page)
        self.assertIn("RT (min)", page)

    def test_intensity_ticks_are_scientific(self):
        page = render_gallery(collect_tiles(_annotations(), self.database))

        self.assertRegex(page, r">\d\.\de\d<")

    def test_competing_features_are_all_shown_with_the_pick_marked(self):
        # Two features claim PC(34:1); the better score is the reported one.
        rivals = pd.DataFrame({
            "feat_feature_id": ["11", "12"],
            "name": "PC(34:1)",
            "class": "PC",
            "adduct": "[M+H]+",
            "carbon": 34.0, "db": 1.0,
            "feat_mz": 760.5851, "feat_rt": [10.0, 10.3],
            "rt_expected": 10.02,
            "msms_status": "supported",
            "score": [0.2, 0.9],
        })

        tiles = collect_tiles(rivals, self.database)

        self.assertEqual(len(tiles), 1)
        self.assertEqual(len(tiles[0].panels), 2)
        self.assertEqual(tiles[0].panels[0].feature_id, "11")
        self.assertTrue(tiles[0].panels[0].picked)
        self.assertFalse(tiles[0].panels[1].picked)
        self.assertEqual(tiles[0].candidate_count, 2)

    def test_samples_share_one_intensity_scale_within_a_panel(self):
        # A weaker injection must look weaker; per-sample scaling would make
        # ten injections of different abundance look identical.
        page = render_gallery(collect_tiles(_annotations(), self.database))
        polylines = re.findall(r'<polyline points="([^"]*)"', page)
        apexes = [
            min(float(pair.split(",")[1]) for pair in line.split())
            for line in polylines[:3]
        ]

        # Three injections at 1x, 1/2x and 1/3x height must reach different tops.
        self.assertEqual(len(set(round(a, 1) for a in apexes)), 3)

    def test_tiles_are_ordered_so_homologues_are_adjacent(self):
        tiles = collect_tiles(_annotations(), self.database)

        self.assertEqual(
            [tile.label for tile in tiles],
            ["PC(34:1)", "PC(36:2)", "PE(34:1)"],
        )

    def test_the_class_column_survives_itertuples(self):
        # "class" is a Python keyword, so itertuples renames it; reading it back
        # by name would silently collapse every tile into one unnamed group.
        tiles = collect_tiles(_annotations(), self.database)

        self.assertEqual({tile.lipid_class for tile in tiles}, {"PC", "PE"})

    def test_the_feature_id_is_shown_on_the_tile(self):
        page = render_gallery(collect_tiles(_annotations(), self.database))

        self.assertIn("#11", page)

    def test_msms_status_becomes_a_visible_edge(self):
        flags = {
            tile.label: tile.flag
            for tile in collect_tiles(_annotations(), self.database)
        }

        self.assertEqual(flags["PC(34:1)"], "confirmed")
        self.assertEqual(flags["PC(36:2)"], "")
        self.assertEqual(flags["PE(34:1)"], "contradicted")

    def test_a_lipid_matched_twice_is_one_tile(self):
        doubled = pd.concat([_annotations(), _annotations().head(1)])

        self.assertEqual(len(collect_tiles(doubled, self.database)), 3)

    def test_the_page_is_self_contained(self):
        page = render_gallery(collect_tiles(_annotations(), self.database))

        self.assertNotIn("<script", page)
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)
        self.assertIn("<svg", page)

    def test_features_without_a_stored_trace_are_skipped(self):
        frame = _annotations()
        frame.loc[0, "feat_feature_id"] = "absent"

        self.assertEqual(len(collect_tiles(frame, self.database)), 2)

    def test_chromatogram_and_mobilogram_are_separate_pages(self):
        tiles = collect_tiles(_annotations(), self.database)
        chromatogram = render_gallery(tiles, dimension="chromatogram")
        mobilogram = render_gallery(tiles, dimension="mobilogram")

        self.assertIn("RT (min)", chromatogram)
        self.assertIn("1/K0", mobilogram)
        self.assertIn("predicted retention time", chromatogram)
        self.assertNotIn("predicted retention time", mobilogram)

    def test_a_mobilogram_page_is_skipped_when_there_is_no_mobility(self):
        flat = self.directory / "flat.sqlite"
        connection = sqlite3.connect(flat)
        connection.execute(
            "CREATE TABLE traces (feature_id TEXT, sample_name TEXT, "
            "peak_height REAL, is_representative INTEGER, rt_values BLOB, "
            "rt_intensities BLOB, mobility_values BLOB, mobility_intensities BLOB)"
        )
        rt = np.linspace(9.5, 10.5, 8).astype("<f4")
        for feature_id in ("11", "12", "13"):
            connection.execute(
                "INSERT INTO traces VALUES (?,?,?,?,?,?,?,?)",
                (feature_id, "S1.d", 1e5, 1, zlib.compress(rt.tobytes()),
                 zlib.compress(rt.tobytes()), None, None),
            )
        connection.commit()
        connection.close()
        tiles = collect_tiles(_annotations(), flat)

        self.assertIsNotNone(write_gallery_page(
            tiles, self.directory / "c.html", title="c",
            dimension="chromatogram"))
        self.assertIsNone(write_gallery_page(
            tiles, self.directory / "m.html", title="m",
            dimension="mobilogram"))

    def test_no_traces_at_all_writes_nothing(self):
        empty = self.directory / "empty.sqlite"
        _trace_db(empty, [])

        result = write_trace_gallery(
            _annotations(), empty, self.directory / "out.html"
        )

        self.assertIsNone(result)

    def test_writing_produces_a_readable_page(self):
        destination = write_trace_gallery(
            _annotations(), self.database, self.directory / "gallery.html"
        )

        self.assertIsNotNone(destination)
        text = destination.read_text(encoding="utf-8")
        self.assertIn("PC(34:1)", text)
        self.assertIn("#11", text)

    def test_lipid_names_are_escaped(self):
        frame = _annotations()
        frame.loc[0, "name"] = "PC<script>alert(1)</script>"

        page = render_gallery(collect_tiles(frame, self.database))

        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)


if __name__ == "__main__":
    unittest.main()
