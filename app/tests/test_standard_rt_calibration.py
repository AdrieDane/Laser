"""RT export, version history and GUI integration on synthetic standards."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from annotator.rt_calibration import (
    apply_internal_standard_rt_calibration, fit_confirmed_internal_standard_rt_calibration,
)
from annotator.mass_calibrants import build_mass_reference_panel
from annotator.matching import POSITIVE_CCS_INTERNAL_STANDARDS
from annotator.reference import load_reference
from resource_paths import REFERENCE_DATA_DIRECTORY
from standard_lipidomics.project_store import save_calibration_workflow
from standard_lipidomics.rt_calibration import calibrate_standard_rt_bytes, inverse_rt
from standard_lipidomics.workflow import load_standard_feature_list, annotate_standard_stream
from matlab_annotation.engine import _calculate_matlab_grid_coherence
from matlab_annotation.automatic import AutomaticAnchorSettings
from test_grid_coherence_v2 import _candidate
from test_standard_lipidomics import reference_frame, load_reference_frame

APP = Path(__file__).resolve().parents[1] / "app.py"


def model(kind="linear"):
    anchors = pd.DataFrame({"standard": list("ABCDEF"), "ref_rt": [1, 3, 5, 7, 9, 11],
                            "feat_rt": [1.2, 3.4, 5.9, 8, 10.3, 12.5]})
    return fit_confirmed_internal_standard_rt_calibration(anchors, model=kind)


@pytest.mark.parametrize("kind", ["constant_shift", "linear", "robust_linear", "monotonic_piecewise"])
def test_xcms_rt_inverse_matches_copied_reference_model(kind):
    calibration = model(kind)
    points = np.array([0, 1, 2, 5, 8, 11, 12, np.nan], dtype=float)
    predicted = apply_internal_standard_rt_calibration(points, calibration)
    np.testing.assert_allclose(inverse_rt(predicted, calibration), points, equal_nan=True)


@pytest.mark.parametrize("separator", ["\t", ","])
def test_rt_export_preserves_raw_values_samples_and_line_endings(separator):
    raw = separator.join(["name", "mzmed", "rtmed", "rtmin", "rtmax", "npeaks", "sample_a"]) + "\r\n"
    raw += separator.join(["feature1", "500.0000", "5.9", "5.8", "6.0", "1", "1.234567890123"]) + "\r\n"
    exported = calibrate_standard_rt_bytes(raw.encode(), model("monotonic_piecewise"))
    header, row = [line.split(separator) for line in exported.decode().splitlines()]
    assert row[0:2] == ["feature1", "500.0000"]
    assert row[6] == "1.234567890123"
    assert row[header.index("LASER_original_rtmed")] == "5.9"
    assert float(row[2]) == pytest.approx(5.0)
    assert exported.count(b"\r\n") == 2
    restored = load_standard_feature_list(exported)
    assert restored.iloc[0].rt == pytest.approx(5.0)
    assert restored.iloc[0].rt_raw == pytest.approx(5.9)
    assert restored.attrs["sample_columns"] == ["sample_a"]
    with pytest.raises(ValueError, match="already RT-calibrated"):
        calibrate_standard_rt_bytes(exported, model())


def test_saved_rt_model_and_export_follow_selected_archive(tmp_path):
    xcms = tmp_path / "XCMS"
    xcms.mkdir()
    raw = b"name\tmzmed\trtmed\tnpeaks\tsample_a\n1\t500\t5.9\t1\t100\n"
    path = xcms / "peakgrouplist_RP_Pos.txt"
    path.write_bytes(raw)
    save_calibration_workflow(tmp_path, {"RP_Pos": path}, {}, rt_calibrations={"RP_Pos": model("monotonic_piecewise")})
    _, archive = save_calibration_workflow(tmp_path, {"RP_Pos": path}, {})
    manifest = json.loads((archive / "LASER_standard_mass_calibration.json").read_text())
    assert len(manifest["streams"]["RP_Pos"]["rt_calibration"]["anchors"]) == 6
    assert load_standard_feature_list(archive / path.name).iloc[0].rt == pytest.approx(5.0)
    assert load_standard_feature_list(tmp_path / "Calibration" / path.name).iloc[0].rt == pytest.approx(5.9)
    assert path.read_bytes() == raw


def test_v2_reaches_both_annotation_engines():
    raw = b"name\tmzmed\trtmed\tnpeaks\tsample_a\n1\t760.585\t8\t1\t100\n"
    result = annotate_standard_stream(raw, "RP_Pos", load_reference_frame(reference_frame()), grid_coherence_version="v2")
    assert result["criteria"]["grid_coherence_version"] == "v2"
    assert result["all_pairs"]["grid_coherence_method"].eq("local_median_mad_v3").all()
    candidates = pd.DataFrame([_candidate(i, 32 + 2*i, 0, 1.0, .1, 0) for i in range(4)])
    scored, _ = _calculate_matlab_grid_coherence(candidates, settings=AutomaticAnchorSettings(grid_coherence_version="v2"))
    assert scored["grid_coherence_method"].eq("local_median_mad_v3").all()


def test_gui_rt_only_search_apply_and_save(tmp_path):
    xcms = tmp_path / "XCMS"
    xcms.mkdir()
    reference = load_reference(REFERENCE_DATA_DIRECTORY / "4d_lipid_reference.csv")
    panel = build_mass_reference_panel(reference, polarity="pos", panel_name="Unlabelled IS",
        unlabelled_standards=POSITIVE_CCS_INTERNAL_STANDARDS)
    panel = panel.dropna(subset=["ref_rt"]).drop_duplicates("ref_mz")
    raw = "name\tmzmed\trtmed\tnpeaks\tsample_a\n"
    for i, row in enumerate(panel.itertuples()):
        raw += f"{i}\t{row.ref_mz:.12f}\t{row.ref_rt + .2:.12f}\t1\t1000\n"
    (xcms / "peakgrouplist_RP_Pos.txt").write_text(raw)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.button(key="standard-module-calibration").click().run()
    next(w for w in app.text_input if w.label == "Project folder").set_value(str(tmp_path))
    next(w for w in app.checkbox if w.label == "m/z calibration").uncheck()
    next(w for w in app.checkbox if w.label == "RT calibration").check()
    next(b for b in app.button if b.label == "Review XCMS calibration").click().run()
    assert not app.exception
    assert any(s.label == "RT calibration model" for s in app.selectbox)
    next(b for b in app.button if b.label == "Search RT calibration masses").click().run()
    assert not app.exception
    next(b for b in app.button if b.label == "Apply RT calibration").click().run()
    assert not app.exception
    next(b for b in app.button if b.label == "Save calibration").click().run()
    assert not app.exception
    saved = load_standard_feature_list(tmp_path / "Calibration" / "peakgrouplist_RP_Pos.txt")
    assert saved.attrs["rt_calibration"]["kind"] == "linear"
    np.testing.assert_allclose(saved.rt_raw - saved.rt, .2, atol=1e-8)


def test_gui_grid_choice_is_offered_for_both_engines(tmp_path):
    xcms = tmp_path / "XCMS"
    xcms.mkdir()
    (xcms / "peakgrouplist_RP_Pos.txt").write_text("name\tmzmed\trtmed\n")
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.button(key="standard-module-annotation").click().run()
    app.text_input(key="standard-annotation-project").set_value(str(tmp_path)).run()
    assert app.selectbox(key="standard-matlab-grid-version").options == ["Grid coherence V1", "Grid coherence V2"]
    next(r for r in app.radio if r.label == "Annotation engine").set_value("LASER reference matcher").run()
    assert app.selectbox(key="standard-laser-grid-version").options == ["Grid coherence V1", "Grid coherence V2"]
