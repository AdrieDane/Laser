import unittest

import numpy as np
import pandas as pd

from standard_lipidomics.grid_browser import (
    apply_review_criteria,
    build_class_grid_state,
    build_standard_grid_figure,
    prepare_standard_grid_data,
    select_plotted_species,
    standard_sample_columns,
    _feature_trace_colour_map,
    _feature_trace_range,
    _benchmark_visible_plot_candidates,
    _review_criteria_signature,
    _same_id_trace_candidates,
)


class StandardGridBrowserTests(unittest.TestCase):
    def setUp(self):
        self.reference = pd.DataFrame({
            "name": ["PC(32:0)", "PC(34:0)", "PC(32:1)", "PC(34:1)"],
            "class": ["PC"] * 4,
            "base": [None] * 4,
            "adduct": ["[M+H]+"] * 4,
            "carbon": [32, 34, 32, 34],
            "db": [0, 0, 1, 1],
            "ref_rt": [7.0, 7.5, 6.7, 7.2],
            "ref_mz": [734.5, 762.5, 732.5, 760.5],
        })
        self.all_pairs = pd.DataFrame({
            "name": ["PC(32:0)", "PC(34:1)", "PC(32:1)"],
            "class": ["PC"] * 3,
            "base": [None] * 3,
            "adduct": ["[M+H]+"] * 3,
            "carbon": [32, 34, 32],
            "db": [0, 1, 1],
            "feat_feature_id": ["1", "2", "3"],
            "feat_rt": [7.21, 7.35, 9.90],
            "feat_mz": [734.5002, 760.5001, 732.5003],
            "ref_mz": [734.5, 760.5, 732.5],
            "ref_rt": [7.0, 7.2, 6.7],
            "rt_expected": [7.0, 7.2, 6.7],
            "rt_error": [0.21, 0.15, 3.20],
            "ppm_error": [0.27, 0.13, 3.90],
        })
        # The run accepted the first two; the third is a far-off RT near-miss
        # the browser must still be able to show as review-only.
        self.annotations = self.all_pairs.head(2).copy()
        self.annotations["automatic_accepted"] = [True, True]
        self.annotations["auto_annotation_status"] = [
            "mass_rt_supported", "class_grid_supported",
        ]
        self.stream_result = {
            "reference": self.reference,
            "annotations": self.annotations,
            "all_pairs": self.all_pairs,
            "rt_models": pd.DataFrame(),
            "criteria": {
                "ppm_tolerance": 4.0,
                "rt_tolerance_min": 0.5,
                "uses_rt_grid": True,
            },
        }

    def _grid_state(self, confirmed=frozenset()):
        reference, candidates = prepare_standard_grid_data(self.stream_result)
        return build_class_grid_state(
            reference, candidates,
            lipid_class="PC", adduct="[M+H]+", confirmed=set(confirmed),
        )

    def test_viewer_criteria_signature_is_run_and_stream_specific(self):
        first = _review_criteria_signature(
            (101, "RP_Pos"), 3.0, 0.5, True, 0.45,
        )
        same_values_new_project = _review_criteria_signature(
            (202, "RP_Pos"), 3.0, 0.5, True, 0.45,
        )
        same_project_new_stream = _review_criteria_signature(
            (101, "NP_Pos"), 3.0, 0.5, True, 0.45,
        )

        self.assertNotEqual(first, same_values_new_project)
        self.assertNotEqual(first, same_project_new_stream)

    def test_validated_class_offset_moves_full_reference_grid(self):
        self.stream_result["rt_models"] = pd.DataFrame({
            "class": ["PC"],
            "adduct": ["[M+H]+"],
            "model_type": ["validated_class_offset"],
            "offset_min": [0.2],
            "validated": [True],
        })

        grid, candidates = prepare_standard_grid_data(self.stream_result)

        self.assertTrue((grid["grid_class"] == "PC").all())
        self.assertAlmostEqual(grid.loc[0, "grid_rt"], 7.2)
        self.assertEqual(set(candidates["grid_class"]), {"PC"})

    def test_candidate_superset_marks_run_accepted_rows_only(self):
        _, candidates = prepare_standard_grid_data(self.stream_result)

        accepted = dict(zip(candidates["feat_feature_id"],
                            candidates["automatic_candidate"]))
        self.assertEqual(accepted, {"1": True, "2": True, "3": False})
        self.assertEqual(
            candidates.loc[
                candidates["feat_feature_id"].eq("3"), "auto_annotation_status"
            ].iloc[0],
            "not_reviewed_by_the_run",
        )

    def test_candidate_pool_coherence_is_not_overwritten_by_annotation_lookup(self):
        all_pairs = self.all_pairs.copy()
        all_pairs["grid_coherence_score"] = [0.91, 0.72, 0.18]
        all_pairs["grid_coherence_neighbors"] = [3, 2, 4]
        all_pairs["grid_coherence_dimensions"] = [2, 2, 2]
        all_pairs["grid_coherence_assessed"] = True
        all_pairs["grid_coherence_outlier"] = [False, False, True]
        all_pairs["grid_coherence_pass"] = [True, True, False]
        all_pairs["grid_coherence_threshold"] = 0.35
        self.stream_result["all_pairs"] = all_pairs

        _, candidates = prepare_standard_grid_data(self.stream_result)

        self.assertEqual(
            candidates.set_index("feat_feature_id").loc[
                "3", "grid_coherence_score"
            ],
            0.18,
        )
        self.assertTrue(bool(
            candidates.set_index("feat_feature_id").loc[
                "3", "grid_coherence_outlier"
            ]
        ))

    def test_np_stream_joins_acceptance_on_the_reference_identity(self):
        annotations = self.annotations.copy()
        annotations["reference_identity"] = annotations["name"]
        annotations["name"] = ["PC 32:0", "PC 34:1"]
        self.stream_result["annotations"] = annotations

        _, candidates = prepare_standard_grid_data(self.stream_result)

        self.assertEqual(
            set(candidates.loc[candidates["automatic_candidate"], "name"]),
            {"PC(32:0)", "PC(34:1)"},
        )

    def test_figure_separates_accepted_original_widened_and_anchor_points(self):
        figure = build_standard_grid_figure(self._grid_state(), pd.DataFrame())

        trace_names = {trace.name for trace in figure.data if trace.name}
        self.assertIn("accepted annotation", trace_names)
        self.assertIn("candidate", trace_names)
        self.assertIn("widened-window candidate", trace_names)
        self.assertIn("engine-selected RT anchor", trace_names)
        self.assertIn("manually confirmed review anchor", trace_names)
        self.assertNotIn("suggested anchor", trace_names)
        self.assertIn("grid nodes", trace_names)
        traces = {trace.name: trace for trace in figure.data if trace.name}
        self.assertEqual(traces["accepted annotation"].marker.color, "#2e9f64")
        self.assertEqual(traces["candidate"].marker.color, "#2677c9")
        self.assertEqual(
            traces["widened-window candidate"].marker.color, "#83bbed"
        )
        self.assertEqual(
            traces["engine-selected RT anchor"].marker.color, "#ffd400"
        )
        self.assertEqual(
            traces["manually confirmed review anchor"].marker.color,
            "#af52de",
        )

    def test_run_candidate_includes_original_grid_coherence_filter(self):
        all_pairs = self.all_pairs.copy()
        all_pairs["grid_coherence_score"] = [0.91, 0.72, 0.18]
        all_pairs["grid_coherence_assessed"] = True
        all_pairs["grid_coherence_outlier"] = [False, False, True]
        all_pairs["grid_coherence_pass"] = [True, True, False]
        self.stream_result["all_pairs"] = all_pairs
        reference, candidates = prepare_standard_grid_data(self.stream_result)

        state = build_class_grid_state(
            reference, candidates,
            lipid_class="PC", adduct="[M+H]+", confirmed=set(),
            run_criteria={
                "ppm_tolerance": 4.0,
                "rt_tolerance": 5.0,
                "use_rt": True,
                "use_grid_coherence": True,
                "grid_coherence_threshold": 0.45,
            },
            review={
                "ppm_tolerance": 4.0,
                "rt_tolerance": 5.0,
                "use_rt": True,
                "use_grid_coherence": False,
                "grid_coherence_threshold": 0.45,
            },
        )

        provenance = state["candidates"].set_index("feat_feature_id")[
            "run_candidate"
        ].to_dict()
        self.assertEqual(provenance, {"1": True, "2": True, "3": False})

    def test_live_grid_coherence_toggle_hides_only_assessed_low_scores(self):
        state = self._grid_state()
        candidates = state["candidates"].copy()
        candidates["grid_coherence_assessed"] = [True, True, False]
        candidates["grid_coherence_score"] = [0.91, 0.20, np.nan]

        filtered = apply_review_criteria(
            candidates,
            ppm_tolerance=4.0,
            rt_tolerance=5.0,
            use_rt=True,
            use_grid_coherence=True,
            grid_coherence_threshold=0.45,
        )

        self.assertEqual(set(filtered["feat_feature_id"]), {"1", "3"})

    def test_figure_highlights_significant_unannotated_features(self):
        differential = pd.DataFrame({
            "feature_id": ["unknown-1"], "rt": [7.25], "mz": [745.2],
            "group_a": ["control"], "group_b": ["disease"],
            "log2_fold_change_b_vs_a": [2.5], "p_value": [0.001],
            "detection_a_percent": [100.0], "detection_b_percent": [100.0],
            "significant": [True],
            "cluster_hypothesis": ["local PC region (3/4 neighbours)"],
        })
        figure = build_standard_grid_figure(
            self._grid_state(), pd.DataFrame(),
            differential_features=differential,
        )
        trace = next(
            item for item in figure.data
            if item.name == "differential unannotated"
        )
        self.assertEqual(list(trace.x), [7.25])
        self.assertEqual(trace.marker.symbol, "diamond")
        self.assertIn("local PC region", str(trace.customdata))

    def test_rp_positional_anchors_use_exact_engine_identity(self):
        reference = pd.DataFrame({
            "name": ["1-acylLPC(18:2)", "2-acylLPC(18:2)"],
            "class": ["1-acylLPC", "2-acylLPC"], "base": ["", ""],
            "adduct": ["[M+H]+", "[M+H]+"],
            "carbon": [18, 18], "db": [2, 2],
            "ref_rt": [6.3, 6.0], "ref_mz": [520.34, 520.34],
        })
        candidates = pd.DataFrame({
            "name": [
                "1-acylLPC(18:2)", "1-acylLPC(18:2)",
                "2-acylLPC(18:2)",
            ],
            "class": ["1-acylLPC", "1-acylLPC", "2-acylLPC"],
            "base": ["", "", ""],
            "adduct": ["[M+H]+", "[M+H]+", "[M+H]+"],
            "carbon": [18, 18, 18], "db": [2, 2, 2],
            "feat_feature_id": ["41118", "closer", "41118"],
            "feat_rt": [6.31, 6.3, 6.3],
            "feat_mz": [520.34, 520.34, 520.34],
            "ref_mz": [520.34, 520.34, 520.34],
            "ref_rt": [6.3, 6.3, 6.0],
            "rt_expected": [6.3, 6.3, 6.0],
            "rt_error": [0.01, 0.0, 0.3],
            "ppm_error": [0.0, 0.0, 0.0],
        })
        stream_result = {
            "reference": reference,
            "annotations": pd.DataFrame(),
            "all_pairs": candidates,
            "rt_models": pd.DataFrame(),
            "automatic_anchors": pd.DataFrame({
                "feat_feature_id": ["41118"],
                "species": ["1-acylLPC(18:2)"],
                "adduct": ["[M+H]+"],
            }),
        }
        prepared_reference, prepared = prepare_standard_grid_data(stream_result)

        anchor_flags = prepared.set_index(
            ["name", "feat_feature_id"]
        )["automatic_rt_anchor"]
        self.assertTrue(bool(anchor_flags[
            ("1-acylLPC(18:2)", "41118")
        ]))
        self.assertFalse(bool(anchor_flags[
            ("1-acylLPC(18:2)", "closer")
        ]))
        self.assertFalse(bool(anchor_flags[
            ("2-acylLPC(18:2)", "41118")
        ]))

        one_acyl = build_class_grid_state(
            prepared_reference, prepared,
            lipid_class="1-acylLPC", adduct="[M+H]+", confirmed=set(),
        )
        self.assertTrue(bool(one_acyl["plot_df"].iloc[0]["anchor"]))
        self.assertEqual(
            one_acyl["plot_df"].iloc[0]["feat_feature_id"], "41118"
        )
        trace_names = {
            trace.name for trace in build_standard_grid_figure(
                one_acyl, pd.DataFrame()
            ).data if trace.name
        }
        self.assertIn("engine-selected RT anchor", trace_names)

        two_acyl = build_class_grid_state(
            prepared_reference, prepared,
            lipid_class="2-acylLPC", adduct="[M+H]+", confirmed=set(),
        )
        self.assertFalse(bool(two_acyl["plot_df"].iloc[0]["anchor"]))

    def test_benchmark_status_colours_override_reviewer_point_palette(self):
        self.stream_result["benchmark_status_by_identity"] = {
            "PC|PC(32:0)|[M+H]+": "overlap_same_feature",
            "PC|PC(34:1)|[M+H]+": "laser_only",
            "PC|PC(32:1)|[M+H]+": "matlab_only",
        }
        state = self._grid_state()
        figure = build_standard_grid_figure(state, pd.DataFrame())
        traces = {
            str(trace.name): trace
            for trace in figure.data
            if trace.name in {
                "both MATLAB + LASER", "LASER-only", "MATLAB-only",
            }
        }
        self.assertEqual(traces["both MATLAB + LASER"].marker.color, "#16a34a")
        self.assertEqual(traces["LASER-only"].marker.color, "#f97316")
        self.assertEqual(traces["MATLAB-only"].marker.color, "#7c3aed")

    def test_benchmark_status_never_leaks_across_identity_on_shared_feature(self):
        reference = pd.DataFrame({
            "name": ["PE(67:10)"],
            "class": ["PE"], "base": [""],
            "adduct": ["[M+H]+"], "carbon": [67], "db": [10],
            "ref_rt": [15.1], "ref_mz": [1162.9146],
        })
        candidate = pd.DataFrame({
            "name": ["PE(67:10)"],
            "class": ["PE"], "base": [""],
            "adduct": ["[M+H]+"], "carbon": [67], "db": [10],
            "feat_feature_id": ["76122"], "feat_rt": [15.1125],
            "feat_mz": [1162.91461], "ref_mz": [1162.9146],
            "ref_rt": [15.1], "rt_expected": [15.1],
            "rt_error": [0.0125], "ppm_error": [0.01],
        })
        stream_result = {
            "reference": reference,
            "annotations": pd.DataFrame(),
            "all_pairs": candidate,
            "rt_models": pd.DataFrame(),
            "benchmark_status_by_identity": {
                "PC|PC(64:10)|[M+H]+": "matlab_only",
            },
            # Legacy payloads may still carry a status for this shared XCMS
            # feature. It must never colour a different lipid identity.
            "benchmark_status_by_feature": {
                "76122": "matlab_only",
            },
        }

        _, candidates = prepare_standard_grid_data(stream_result)

        self.assertEqual(candidates.iloc[0]["benchmark_status"], "")

        stream_result["benchmark_status_by_identity"] = {
            "PE|PE(67:10)|[M+H]+": "laser_only",
        }
        _, candidates = prepare_standard_grid_data(stream_result)
        self.assertEqual(candidates.iloc[0]["benchmark_status"], "laser_only")

    def test_positional_rt_residual_is_not_replaced_by_generic_grid(self):
        reference = self.reference.copy()
        reference["name"] = [
            "1-acylLPC(32:0)", "1-acylLPC(34:0)",
            "1-acylLPC(32:1)", "1-acylLPC(34:1)",
        ]
        reference["class"] = "1-acylLPC"
        candidate = self.all_pairs.iloc[[0]].copy()
        candidate["name"] = "1-acylLPC(32:0)"
        candidate["class"] = "1-acylLPC"
        candidate["feat_rt"] = 7.0
        candidate["rt_expected"] = 6.8
        candidate["rt_error"] = 0.2
        candidate["positional_isomer_rt_expected"] = 6.8
        candidate["positional_isomer_rt_error_seconds"] = 12.0
        stream_result = {
            "reference": reference,
            "annotations": pd.DataFrame(),
            "all_pairs": candidate,
            "rt_models": pd.DataFrame(),
        }
        prepared_reference, prepared_candidates = prepare_standard_grid_data(
            stream_result
        )

        state = build_class_grid_state(
            prepared_reference,
            prepared_candidates,
            lipid_class="1-acylLPC",
            adduct="[M+H]+",
            confirmed=set(),
            review={
                "ppm_tolerance": 3.0,
                "rt_tolerance": 0.15,
                "use_rt": True,
            },
        )

        self.assertAlmostEqual(
            float(state["all_candidates"].iloc[0]["grid_resid"]), 0.2
        )
        self.assertTrue(state["candidates"].empty)

    def test_benchmark_toggle_hides_only_statusless_review_candidates(self):
        self.stream_result["benchmark_status_by_identity"] = {
            "PC|PC(32:0)|[M+H]+": "overlap_same_feature",
        }
        state = self._grid_state()

        visible = _benchmark_visible_plot_candidates(
            state["plot_df"], show_unbenchmarked=False,
        )

        # The first two rows are automatic candidates and remain visible even
        # when they have no benchmark status; the third is review-only and has
        # no MATLAB/LASER comparison entry, so it is the blue point hidden by
        # the toggle.
        self.assertEqual(set(visible["feat_feature_id"]), {"1", "2"})

    def test_figure_omits_ccs_and_msms_but_exposes_coherence_hover(self):
        figure = build_standard_grid_figure(self._grid_state(), pd.DataFrame())

        self.assertEqual(figure.layout.xaxis.title.text, "Retention time (min)")
        self.assertEqual(figure.layout.yaxis.title.text, "m/z")
        rendered = figure.to_plotly_json()
        for absent in ("CCS", "MS/MS", "mobility"):
            self.assertNotIn(absent, str(rendered))
        self.assertIn("grid coherence", str(rendered))

    def test_trace_count_is_stable_when_a_projection_is_added(self):
        state = self._grid_state()
        plain = build_standard_grid_figure(state, pd.DataFrame())
        projected = build_standard_grid_figure(
            state, pd.DataFrame(),
            projected_alternatives=state["candidates"].head(1),
            projected_best=state["plot_df"].head(1),
        )

        self.assertEqual(len(plain.data), len(projected.data))

    def test_confirmed_anchor_shifts_the_predicted_grid(self):
        baseline = self._grid_state()
        anchored = self._grid_state(confirmed={"PC(32:0)"})

        self.assertEqual(baseline["correction"]["kind"], "none")
        self.assertEqual(anchored["correction"]["kind"], "shift")
        self.assertAlmostEqual(
            anchored["correction"]["shift"],
            float(
                baseline["plot_df"]
                .set_index("name").loc["PC(32:0)", "grid_resid_db"]
            ),
        )

    def test_review_criteria_only_narrow_the_displayed_candidates(self):
        state = self._grid_state()
        candidates = state["candidates"]

        wide = apply_review_criteria(
            candidates, ppm_tolerance=4.0, rt_tolerance=5.0, use_rt=True
        )
        tight = apply_review_criteria(
            candidates, ppm_tolerance=1.0, rt_tolerance=0.5, use_rt=True
        )

        self.assertEqual(len(wide), len(candidates))
        self.assertEqual(set(tight["feat_feature_id"]), {"1", "2"})
        # The run's own acceptance flags are untouched by the display filter.
        self.assertTrue(tight["automatic_candidate"].all())

    def test_mass_only_stream_ignores_the_rt_window(self):
        state = self._grid_state()

        kept = apply_review_criteria(
            state["candidates"], ppm_tolerance=4.0, rt_tolerance=0.01,
            use_rt=False,
        )

        self.assertEqual(len(kept), 3)

    def test_sample_columns_exclude_xcms_metadata_and_group_counts(self):
        features = pd.DataFrame({
            "feature_id": ["1"],
            "mz": [734.5],
            "rt": [7.2],
            "npeaks": [3],
            "intensity": [1e6],
            "group_WT": [4],
            "SampleA": [1.0],
            "SampleB": [2.0],
        })

        self.assertEqual(
            standard_sample_columns(features), ["SampleA", "SampleB"]
        )

    def test_feature_trace_colours_cover_more_than_six_candidates(self):
        feature_ids = [str(index) for index in range(9)]
        colours = _feature_trace_colour_map(feature_ids)

        self.assertEqual(set(colours), set(feature_ids))
        self.assertEqual(colours["0"], colours["6"])
        self.assertEqual(colours["1"], colours["7"])

    def test_trace_diagnostics_dedupe_features_and_use_strict_window(self):
        rows = pd.DataFrame({
            "feat_feature_id": ["a", "a", "b", "c"],
            "ppm_error": [0.2, 0.2, 2.9, 5.0],
            "rt_error": [0.01, 0.01, 9.0 / 60.0, 30.0 / 60.0],
            "automatic_candidate": [True, True, False, False],
        })
        selected, omitted = _same_id_trace_candidates(
            rows, rows.iloc[0]
        )
        self.assertEqual(list(selected["feat_feature_id"]), ["a", "b"])
        self.assertEqual(omitted, 1)

    def test_trace_range_caps_broad_grouped_window(self):
        row = pd.Series({
            "feat_rt": 10.0, "feat_rtmin": 0.0, "feat_rtmax": 20.0,
        })
        lower, upper = _feature_trace_range(row)
        self.assertAlmostEqual(lower, 9.825)
        self.assertAlmostEqual(upper, 10.175)

    def test_assigned_primary_feature_wins_when_the_window_retains_it(self):
        # Two features carry PC(32:0); the far one is the assigned primary.
        extra = self.all_pairs.iloc[[0]].copy()
        extra["feat_feature_id"] = ["9"]
        extra["feat_rt"] = [7.05]
        self.stream_result["all_pairs"] = pd.concat(
            [self.all_pairs, extra], ignore_index=True
        )
        reference, candidates = prepare_standard_grid_data(self.stream_result)

        state = build_class_grid_state(
            reference, candidates, lipid_class="PC", adduct="[M+H]+",
            confirmed=set(), primary_overrides={"PC(32:0)": "9"},
        )

        plotted = state["plot_df"].set_index("name")
        self.assertEqual(plotted.loc["PC(32:0)", "feat_feature_id"], "9")

    def test_hidden_primary_feature_falls_back_to_a_visible_one(self):
        extra = self.all_pairs.iloc[[0]].copy()
        extra["feat_feature_id"] = ["9"]
        extra["feat_rt"] = [12.0]          # far outside any sane RT window
        self.stream_result["all_pairs"] = pd.concat(
            [self.all_pairs, extra], ignore_index=True
        )
        reference, candidates = prepare_standard_grid_data(self.stream_result)
        state = build_class_grid_state(
            reference, candidates, lipid_class="PC", adduct="[M+H]+",
            confirmed=set(), primary_overrides={"PC(32:0)": "9"},
        )

        visible = apply_review_criteria(
            state["candidates"], ppm_tolerance=4.0, rt_tolerance=0.5,
            use_rt=True,
        )
        plotted = select_plotted_species(visible).set_index("name")

        # The species stays on the grid through its remaining feature rather
        # than disappearing with its hidden assignment.
        self.assertIn("PC(32:0)", plotted.index)
        self.assertEqual(plotted.loc["PC(32:0)", "feat_feature_id"], "1")

    def test_mesh_is_drawn_over_the_visible_candidates_only(self):
        # An isobaric near-miss at a far composition must not stretch the mesh.
        far = self.all_pairs.iloc[[0]].copy()
        far["name"] = ["PC(60:0)"]
        far["carbon"] = [60]
        far["feat_rt"] = [20.0]
        self.stream_result["all_pairs"] = pd.concat(
            [self.all_pairs, far], ignore_index=True
        )
        reference, candidates = prepare_standard_grid_data(self.stream_result)

        unfiltered = build_class_grid_state(
            reference, candidates, lipid_class="PC", adduct="[M+H]+",
            confirmed=set(),
        )
        reviewed = build_class_grid_state(
            reference, candidates, lipid_class="PC", adduct="[M+H]+",
            confirmed=set(),
            review={"ppm_tolerance": 4.0, "rt_tolerance": 0.5, "use_rt": True},
        )

        drawn = [c for c in reviewed["mesh"]["carbons"]]
        self.assertLess(max(drawn), max(unfiltered["mesh"]["carbons"]))
        self.assertLessEqual(max(drawn), 36)  # 34 observed + the mesh's padding

    def test_review_window_never_moves_the_anchored_grid(self):
        wide = build_class_grid_state(
            *prepare_standard_grid_data(self.stream_result),
            lipid_class="PC", adduct="[M+H]+", confirmed={"PC(32:0)"},
            review={"ppm_tolerance": 4.0, "rt_tolerance": 5.0, "use_rt": True},
        )
        tight = build_class_grid_state(
            *prepare_standard_grid_data(self.stream_result),
            lipid_class="PC", adduct="[M+H]+", confirmed={"PC(32:0)"},
            review={"ppm_tolerance": 1.0, "rt_tolerance": 0.3, "use_rt": True},
        )

        self.assertAlmostEqual(
            wide["correction"]["shift"], tight["correction"]["shift"]
        )

    def test_browser_survives_a_stream_without_any_mass_match(self):
        self.stream_result["all_pairs"] = pd.DataFrame()
        self.stream_result["annotations"] = pd.DataFrame()

        reference, candidates = prepare_standard_grid_data(self.stream_result)
        state = build_class_grid_state(
            reference, candidates,
            lipid_class="PC", adduct="[M+H]+", confirmed=set(),
        )
        figure = build_standard_grid_figure(state, pd.DataFrame())

        self.assertTrue(state["plot_df"].empty)
        self.assertTrue(np.isfinite(figure.layout.height))


if __name__ == "__main__":
    unittest.main()
