"""Project-folder persistence for annotation and untargeted workflows."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from errors import ArtifactContractError

from standard_lipidomics.project_store import (
    load_annotation_workflow,
    load_untargeted_workflow,
    save_annotation_workflow,
    save_untargeted_workflow,
    stage_project_metadata,
    workflow_folder,
)
from standard_lipidomics.workflow import discover_standard_streams
from standard_lipidomics.report import render_standard_lipidomics_report
from test_standard_report import _result as report_result


def _features() -> pd.DataFrame:
    frame = pd.DataFrame({
        "feature_id": ["f1", "f2"], "mz": [700.0, 800.0],
        "rt": [8.0, 9.0], "a1": [10, 20], "b1": [100, 20],
    })
    frame.attrs["sample_columns"] = ["a1", "b1"]
    frame.attrs["sample_group_map"] = {"a1": "control", "b1": "treated"}
    frame.attrs["sample_type_map"] = {"a1": "study", "b1": "study"}
    return frame


def _annotation_result() -> dict:
    return {
        "engine": "matlab",
        "streams": {"RP_Pos": {"features": _features(),
                                "criteria": {"ppm_tolerance": 3.0}}},
        "identified_lipids": pd.DataFrame({"feature_id": ["f1"],
                                             "feature_identity": ["PC(34:1)"]}),
        "equivalent_lipids": pd.DataFrame(),
        "automatic_annotations": pd.DataFrame({"stream": ["RP_Pos"],
                                                 "feature_id": ["f1"]}),
        "all_candidates": pd.DataFrame({"feature_id": ["f1"]}),
    }


def test_annotation_saves_in_project_and_reopens_without_annotation(tmp_path):
    project = tmp_path / "project"
    metadata = project / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "peakgrouplist_RP_Pos.txt").write_text("input")
    result = _annotation_result()
    paths, archive = save_annotation_workflow(
        project, result, inputs={"project": str(project)},
        annotated_inputs={"RP_Pos": b"annotated"},
    )
    assert archive is None
    assert paths["project_manifest"] == project / "LASER" / "LASER_PROJECT.json"
    assert (project / "LASER" / "annotated_peaklists" /
            "peakgrouplist_RP_Pos_annotated.txt").read_bytes() == b"annotated"
    assert (metadata / "peakgrouplist_RP_Pos.txt").read_text() == "input"
    restored, _ = load_annotation_workflow(project)
    assert restored["streams"]["RP_Pos"]["features"].attrs["sample_group_map"]["a1"] == "control"
    assert restored["streams"]["RP_Pos"]["criteria"]["ppm_tolerance"] == 3.0
    assert restored["identified_lipids"].iloc[0]["feature_identity"] == "PC(34:1)"


def test_resave_archives_previous_export_and_overwrite_removes_previous(tmp_path):
    project = tmp_path / "project"
    result = _annotation_result()
    save_annotation_workflow(project, result)
    result["identified_lipids"].loc[0, "feature_identity"] = "PC(36:2)"
    _, archive = save_annotation_workflow(project, result)
    assert archive is not None and archive.is_dir()
    previous = json.loads((archive / "LASER_PROJECT.json").read_text())
    assert previous["workflow"] == "standard"
    assert load_annotation_workflow(project)[0]["identified_lipids"].iloc[0]["feature_identity"] == "PC(36:2)"
    _, overwritten = save_annotation_workflow(project, result, existing="overwrite")
    assert overwritten is None
    assert len(list((project / "Archive" / "LASER").iterdir())) == 1


def test_failed_resave_restores_previous_annotation(tmp_path):
    project = tmp_path / "project"
    save_annotation_workflow(project, _annotation_result())
    with pytest.raises(ArtifactContractError):
        save_annotation_workflow(project, {"streams": {}})
    reopened, _ = load_annotation_workflow(project)
    assert reopened["identified_lipids"].iloc[0]["feature_identity"] == "PC(34:1)"


def test_untargeted_result_reopens_from_its_own_folder(tmp_path):
    project = tmp_path / "project"
    features = pd.DataFrame({"stream": ["RP_Pos"], "feature_id": ["f2"],
                             "significant": [True], "p_value": [0.001]})
    analysis = {
        "criteria": {"group_a": "control", "group_b": "treated"},
        "features": features, "significant_features": features.copy(),
        "group_catalog": pd.DataFrame({"group": ["control", "treated"]}),
        "skipped_streams": pd.DataFrame(columns=["stream", "reason"]),
        "method": "test method",
    }
    result = {"streams": {"RP_Pos": {"features": _features()}},
              "automatic_annotations": pd.DataFrame()}
    paths, archive = save_untargeted_workflow(project, result, analysis)
    assert archive is None
    assert paths["project_manifest"] == workflow_folder(project, "Untargeted") / "LASER_PROJECT.json"
    reopened, _ = load_untargeted_workflow(project)
    assert reopened["untargeted_analysis"]["features"].iloc[0]["feature_id"] == "f2"
    assert reopened["streams"]["RP_Pos"]["features"].attrs["sample_group_map"]["b1"] == "treated"
    assert reopened["untargeted_analysis"]["by_stream"]["RP_Pos"].iloc[0]["significant"]


def test_metadata_folder_is_xcms_input(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "peakgrouplist_RP_Pos.txt").write_text("input")
    assert discover_standard_streams(tmp_path)["RP_Pos"] == metadata / "peakgrouplist_RP_Pos.txt"


def test_stage_metadata_copies_required_inputs_without_moving_sources(tmp_path):
    source = tmp_path / "Processing" / "Annotation"
    source.mkdir(parents=True)
    peaklist = source / "peakgrouplist_RP_Pos.txt"
    peaklist.write_bytes(b"peaklist")
    workbook = source / "samples.xlsx"
    workbook.write_bytes(b"workbook")
    standards = source / "internal_standards.tsv"
    standards.write_bytes(b"standards")
    copied = stage_project_metadata(tmp_path, {"RP_Pos": peaklist})
    assert copied["RP_Pos"].read_bytes() == b"peaklist"
    assert (tmp_path / "metadata" / "samples.xlsx").read_bytes() == b"workbook"
    assert (tmp_path / "metadata" / "internal_standards.tsv").read_bytes() == b"standards"
    assert peaklist.read_bytes() == b"peaklist"


def test_reopened_annotation_can_render_report_and_resave(tmp_path):
    project = tmp_path / "project"
    save_annotation_workflow(project, report_result())
    reopened, _ = load_annotation_workflow(project)
    page = render_standard_lipidomics_report(reopened)
    assert "PC(32:0)" in page
    _, archive = save_annotation_workflow(project, reopened)
    assert archive is not None
    assert "PC(32:0)" in render_standard_lipidomics_report(
        load_annotation_workflow(project)[0]
    )
