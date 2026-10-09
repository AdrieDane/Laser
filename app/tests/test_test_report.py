"""The standard tabbed test report."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_report import (  # noqa: E402
    ReportTab,
    TestReport,
    embedded_document,
    feature_filter_bar_chart,
    render_report,
    species_bar_chart,
    write_report,
)
from performance_budgets import budget_result  # noqa: E402


def _species_table():
    return pd.DataFrame({
        "class": ["PC", "PE", "TG", "TOTAL"],
        "identified_species": [113, 79, 183, 375],
        "identified_features": [163, 95, 199, 457],
        "all_unique_features": [180, 110, 230, 520],
        "excluded_unique_features": [17, 15, 31, 63],
    })


class TabbedReportTests(unittest.TestCase):

    def _report(self):
        return TestReport("LASER test report", "POS_20260803", [
            ReportTab("chromatograms", "Chromatograms", "<p>traces</p>"),
            ReportTab("species", "Species per class", "<p>counts</p>"),
            ReportTab("laser", "LASER settings", "<p>criteria</p>"),
        ])

    def test_the_first_tab_is_selected(self):
        page = render_report(self._report())

        self.assertIn('id="tab-chromatograms" checked', page)
        self.assertEqual(page.count(" checked"), 1)

    def test_each_tab_reveals_only_its_own_panel(self):
        page = render_report(self._report())

        for key in ("chromatograms", "species", "laser"):
            self.assertIn(f"#tab-{key}:checked ~ .panel-{key}{{display:block}}", page)

    def test_the_page_is_self_contained_and_defers_hidden_feature_payloads(self):
        # The embedded script has no external dependency and only hydrates the
        # currently selected tab.  Reports remain portable offline evidence.
        page = render_report(self._report())

        self.assertIn("LASER_LAZY_DEFER_HIDDEN", page)
        self.assertIn("DecompressionStream", page)
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)
        self.assertTrue(
            budget_result("report_size", len(page.encode("utf-8")))[
                "within_budget"
            ]
        )

    def test_an_empty_report_still_renders(self):
        page = render_report(TestReport("Empty"))

        self.assertIn("Nothing to show", page)

    def test_writing_produces_a_file(self):
        path = write_report(self._report(), Path(tempfile.mkdtemp()) / "r.html")

        self.assertTrue(path.exists())
        self.assertIn("Chromatograms", path.read_text(encoding="utf-8"))


class SpeciesChartTests(unittest.TestCase):

    def test_feature_filter_chart_separates_all_and_excluded_counts(self):
        chart = feature_filter_bar_chart(_species_table())

        self.assertIn("all unique candidate features", chart)
        self.assertIn("fully criteria-excluded subset", chart)
        self.assertIn("all 180 · excluded 17", chart)
        self.assertNotIn(">TOTAL<", chart)

    def test_the_total_row_is_excluded_from_the_bars(self):
        # TOTAL dwarfs every class and would flatten the chart.
        chart = species_bar_chart(_species_table())

        self.assertNotIn(">TOTAL<", chart)

    def test_only_unique_lipid_ids_are_drawn_with_a_legend(self):
        chart = species_bar_chart(_species_table())

        self.assertIn("unique lipid IDs", chart)
        self.assertNotIn("annotated features", chart)
        self.assertEqual(chart.count("<rect"), 3 + 1)

    def test_classes_are_ordered_by_species_count(self):
        chart = species_bar_chart(_species_table())
        order = [
            name for name in ("TG", "PC", "PE")
            if f">{name}<" in chart
        ]
        positions = [chart.index(f">{name}<") for name in order]

        self.assertEqual(positions, sorted(positions))
        self.assertEqual(order[0], "TG")

    def test_no_annotated_classes_says_so(self):
        empty = pd.DataFrame({
            "class": ["TOTAL"], "identified_species": [0],
            "identified_features": [0],
        })

        self.assertIn("No annotated classes", species_bar_chart(empty))


class EmbeddingTests(unittest.TestCase):

    def test_only_the_body_of_another_report_is_taken(self):
        source = Path(tempfile.mkdtemp()) / "qc.html"
        source.write_text(
            "<!doctype html><head><style>h1{color:red}</style></head>"
            "<body><h1>QC</h1></body>", encoding="utf-8"
        )

        embedded = embedded_document(source)

        self.assertIn("<h1>QC</h1>", embedded)
        self.assertNotIn("<body", embedded)

    def test_embedded_styles_cannot_restyle_the_host_page(self):
        source = Path(tempfile.mkdtemp()) / "qc.html"
        source.write_text(
            "<html><head><style>table{border:9px}</style></head>"
            "<body><table></table></body></html>", encoding="utf-8"
        )

        embedded = embedded_document(source)

        self.assertIn(".embedded table{", embedded)

    def test_scripts_in_an_embedded_document_are_dropped(self):
        source = Path(tempfile.mkdtemp()) / "qc.html"
        source.write_text(
            "<body><script>alert(1)</script><p>ok</p></body>", encoding="utf-8"
        )

        self.assertNotIn("<script", embedded_document(source))

    def test_a_missing_document_is_reported_not_raised(self):
        self.assertIn("Not produced", embedded_document(Path("/nope.html")))


if __name__ == "__main__":
    unittest.main()
