"""Project discovery, calibration history and the five GUI entry points."""
from pathlib import Path
import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from standard_lipidomics.project_layout import available_calibrations, xcms_project_root
from standard_lipidomics.project_store import (
    load_annotation_workflow, load_untargeted_workflow,
    save_annotation_workflow, save_calibration_workflow, save_untargeted_workflow,
)
from standard_lipidomics.workflow import discover_standard_streams
from test_standard_project_store import _annotation_result, _features

APP = Path(__file__).resolve().parents[1] / "app.py"
RAW = b"name\tmzmed\trtmed\tnpeaks\tsample_A\n1\t500.0\t8.0\t1\t1000\n"


def project_inputs(root):
    folder = root / "XCMS"
    folder.mkdir()
    path = folder / "peakgrouplist_RP_Pos.txt"
    path.write_bytes(RAW)
    return {"RP_Pos": path}


def test_calibration_versions_keep_raw_files_and_export_metadata(tmp_path):
    sources = project_inputs(tmp_path)
    (tmp_path / "XCMS" / "internal_standards.tsv").write_text("standards")
    model = {"mode": "fixed", "model": "linear", "slope": 1.0, "intercept": 0.01,
             "n_used": 4, "rmse_before_ppm": 20.0, "rmse_after_ppm": 0.0}
    save_calibration_workflow(tmp_path, sources, {"RP_Pos": model})
    _, archive = save_calibration_workflow(tmp_path, sources, {})
    choices = available_calibrations(tmp_path)
    assert list(choices.values()) == [tmp_path / "Calibration", archive]
    assert b"LASER_original_mz" in (archive / "peakgrouplist_RP_Pos.txt").read_bytes()
    assert (tmp_path / "Calibration" / "peakgrouplist_RP_Pos.txt").read_bytes() == RAW
    assert sources["RP_Pos"].read_bytes() == RAW
    assert (archive / "internal_standards.tsv").read_text() == "standards"
    assert xcms_project_root(archive) == tmp_path
    assert discover_standard_streams(archive)["RP_Pos"].parent == archive
    save_calibration_workflow(tmp_path, sources, {}, existing="overwrite")
    assert len(available_calibrations(tmp_path)) == 2
    assert json.loads((archive / "LASER_standard_mass_calibration.json").read_text())["streams"]["RP_Pos"]["source"] == str(sources["RP_Pos"])


def test_failed_calibration_save_preserves_previous_version(tmp_path):
    sources = project_inputs(tmp_path)
    save_calibration_workflow(tmp_path, sources, {})
    before = (tmp_path / "Calibration" / "LASER_standard_mass_calibration.json").read_bytes()
    with pytest.raises(ValueError):
        save_calibration_workflow(tmp_path, {}, {}, existing="overwrite")
    assert (tmp_path / "Calibration" / "LASER_standard_mass_calibration.json").read_bytes() == before


def test_discovery_stays_in_project_and_ignores_outputs(tmp_path):
    unrelated = tmp_path / "metadata"
    unrelated.mkdir()
    (unrelated / "peakgrouplist_RP_Pos.txt").write_bytes(RAW)
    project = tmp_path / "project"
    project.mkdir()
    (project / "Calibration").mkdir()
    (project / "Calibration" / "peakgrouplist_RP_Pos.txt").write_bytes(RAW)
    assert discover_standard_streams(project) == {}
    input_folder = project / "xcms exports"
    input_folder.mkdir()
    expected = input_folder / "peakgrouplist_RP_Pos.txt"
    expected.write_bytes(RAW)
    assert discover_standard_streams(project) == {"RP_Pos": expected}


def test_legacy_annotation_and_untargeted_still_open_after_annotation_save(tmp_path):
    save_annotation_workflow(tmp_path, _annotation_result())
    laser = tmp_path / "LASER"
    laser.rename(tmp_path / "old")
    laser.mkdir()
    (tmp_path / "old").rename(laser / "Annotation")
    features = pd.DataFrame({"stream": ["RP_Pos"], "feature_id": ["f1"]})
    analysis = {"features": features, "significant_features": features.copy(), "criteria": {}}
    save_untargeted_workflow(tmp_path, {"streams": {"RP_Pos": {"features": _features()}}}, analysis)
    (tmp_path / "Untargeted").rename(laser / "Untargeted")
    assert not load_annotation_workflow(tmp_path)[0]["identified_lipids"].empty
    save_annotation_workflow(tmp_path, _annotation_result(), existing="overwrite")
    assert load_untargeted_workflow(tmp_path)[0]["untargeted_analysis"]["features"].iloc[0].feature_id == "f1"


def test_gui_has_five_modules_and_returns_home():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert {button.label for button in app.button} == {
        "Calibration", "Annotation", "Untargeted analysis", "Open completed annotation",
        "Open completed untargeted analysis"}
    for page in ("calibration", "annotation", "untargeted", "open_annotation", "open_untargeted"):
        app.button(key=f"standard-module-{page}").click().run()
        assert not app.exception
        assert any(widget.label == "Project folder" for widget in app.text_input)
        next(button for button in app.button if button.label == "Back to start").click().run()
        assert not app.exception


def test_gui_selects_current_or_archived_calibration(tmp_path):
    sources = project_inputs(tmp_path)
    save_calibration_workflow(tmp_path, sources, {})
    _, archive = save_calibration_workflow(tmp_path, sources, {})
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.button(key="standard-module-annotation").click().run()
    app.text_input(key="standard-annotation-project").set_value(str(tmp_path)).run()
    next(r for r in app.radio if r.label == "Annotation input").set_value("Calibrated XCMS data").run()
    choice = next(s for s in app.selectbox if s.label == "Calibration version")
    assert len(choice.options) == 2
    choice.select_index(1).run()
    assert not app.exception
    assert app.session_state["standard_selected_input_directory"] == str(archive)


def test_calibration_has_save_archive_and_overwrite_controls(tmp_path):
    sources = project_inputs(tmp_path)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.button(key="standard-module-calibration").click().run()
    app.session_state["standard_calibration_input_directory"] = str(tmp_path)
    app.session_state["standard_calibration_finished"] = {
        "sources": sources, "models": {}, "saved": False,
        "output_directory": str(tmp_path / "Calibration"), "paths": {}, "applied": (),
    }
    app.run()
    radio = app.radio(key="standard-calibration-existing")
    assert radio.options == ["Archive old project", "Overwrite old project"]
    next(b for b in app.button if b.label == "Save calibration").click().run()
    assert not app.exception
    assert (tmp_path / "Calibration" / "LASER_standard_mass_calibration.json").is_file()
    next(b for b in app.button if b.label == "Save calibration").click().run()
    assert len(available_calibrations(tmp_path)) == 2


@pytest.mark.parametrize("raw", [b"name\n1\n", b"name\tmzmed\trtmed\r\nf1\t500.000\t8.00\r\n\r\nf2\t600.0\t9.0"])
def test_saved_annotation_retains_the_exact_selected_input(raw):
    from annotator.standard_annotated_inputs import append_standard_annotations, remove_standard_annotation_columns
    exported = append_standard_annotations(raw, pd.DataFrame(), stream="RP_Pos")
    assert remove_standard_annotation_columns(exported) == raw


def test_archived_annotation_can_be_opened_directly(tmp_path):
    save_annotation_workflow(tmp_path, _annotation_result())
    _, archive = save_annotation_workflow(tmp_path, _annotation_result())
    assert not load_annotation_workflow(archive)[0]["identified_lipids"].empty
    assert xcms_project_root(archive) == tmp_path


def test_reopened_annotation_uses_saved_input_after_calibration_overwrite(tmp_path):
    from annotator.standard_annotated_inputs import append_standard_annotations
    sources = project_inputs(tmp_path)
    calibration = tmp_path / "Calibration"
    save_calibration_workflow(tmp_path, sources, {})
    result = _annotation_result()
    exported = append_standard_annotations(RAW, pd.DataFrame(), stream="RP_Pos")
    save_annotation_workflow(tmp_path, result, annotated_inputs={"RP_Pos": exported},
        inputs={"project": str(tmp_path), "xcms_source": str(calibration)})
    (calibration / "peakgrouplist_RP_Pos.txt").write_bytes(RAW.replace(b"500.0", b"501.0"))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.button(key="standard-module-open_annotation").click().run()
    app.text_input(key="standard-open-project-folder").set_value(str(tmp_path)).run()
    next(b for b in app.button if b.label == "Open completed annotation").click().run()
    assert not app.exception
    assert app.session_state["standard_input_sources"]["RP_Pos"] == RAW
