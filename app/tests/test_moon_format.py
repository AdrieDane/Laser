"""Moon annotation documents (.moon): export, import and comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from moon_format import (
    FORMAT,
    MAPPED_STATUS,
    PeakIndex,
    annotations_to_moon,
    moon_to_crosswalk,
    moon_to_frame,
    read_moon,
    read_peaklist,
    stream_name,
    validate_moon,
    write_moon,
)
from standard_lipidomics.validation import compare_manual_annotations


TOOL = Path(__file__).resolve().parents[2] / "tools/moon_crosswalk.py"


def write_list(path: Path, header: list[str], rows: list[list]) -> Path:
    lines = ["\t".join(header)] + ["\t".join(str(c) for c in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path


@pytest.fixture
def project(tmp_path):
    """RP_Pos: XCMS 'name' IDs not equal to the row; RT in minutes. NP_Neg: a
    'feature' column (Moon uses it) and RT in seconds. 12 rows -> FT01..FT12."""
    ann = tmp_path / "proj" / "Processing" / "Annotation"
    ann.mkdir(parents=True)
    rp = [[f"{100 + k}", f"{600 + k}.1234", f"{5 + k / 10:.4f}", "1", "10"] for k in range(1, 13)]
    write_list(ann / "peakgrouplist_RP_Pos.txt", ["name", "mzmed", "rtmed", "npeaks", "S1"], rp)
    np_rows = [[f"FT{k:03d}", str(k), f"{700 + k}.5", f"{300 + k}", "10"] for k in range(1, 4)]
    write_list(ann / "peakgrouplist_NP_Neg.txt", ["feature", "id", "mzmed", "rtmed", "S1"], np_rows)
    return tmp_path / "proj"


def laser_row(stream, fid, name, adduct="[M+H]+", **extra):
    row = dict(stream=stream, name=name, adduct=adduct, feat_feature_id=fid, isotopologue="",
               is_is="False", quantitative_role="primary", auto_annotation_status="fitted_grid",
               ppm_error=0.5, matlab_compound=25.0, matlab_code=73.0)
    row.update(extra)
    return row


def test_peaklist_ids_follow_moon_and_laser(project):
    ann = project / "Processing" / "Annotation"
    rows = read_peaklist(ann / "peakgrouplist_RP_Pos.txt", "RP_Pos")
    assert [(r.laser_id, r.feature, r.row) for r in rows[:2]] == [("101", "FT01", 1), ("102", "FT02", 2)]
    assert rows[0].rt_seconds == pytest.approx(5.1 * 60) and rows[0].rt_raw == pytest.approx(5.1)
    np_rows = read_peaklist(ann / "peakgrouplist_NP_Neg.txt", "NP_Neg")
    assert [(r.laser_id, r.feature) for r in np_rows] == [("FT001", "FT001"), ("FT002", "FT002"), ("FT003", "FT003")]
    assert np_rows[0].rt_seconds == 301.0          # already seconds (max >= 60)


def test_stream_names():
    assert [stream_name(s) for s in ("RP_Pos", "RP+", "np negative", "NP_Neg", "HILIC_Pos", "")] == [
        "RP_Pos", "RP_Pos", "NP_Neg", "NP_Neg", None, None]


def test_export_maps_features_and_reports_skips(project):
    rows = [
        laser_row("RP_Pos", "103", "PC(34:1)", feat_mz="603.1234", feat_rt="5.3"),
        laser_row("RP_Pos", "105.0", "PC(28:0) (IS)", feat_mz="605.1234"),
        laser_row("RP_Pos", "103", "PC(34:1)"),                               # same pair again: dropped silently
        laser_row("RP_Pos", "103", "PE(34:1)"),                               # other label on the feature
        laser_row("RP_Pos", "104", "PC(34:2)", isotopologue="M+1"),
        laser_row("RP_Pos", "999", "PC(36:1)"),
        laser_row("RP_Pos", "106", "PC(36:2)", feat_mz="700.0"),
        laser_row("XX", "107", "PC(36:3)"),
        laser_row("NP_Neg", "FT002", "PE(36:2)", adduct="[M-H]-", is_is="True"),
    ]
    doc, report = annotations_to_moon(pd.DataFrame(rows), PeakIndex.from_project(project),
                                      laser_info={"engine": "matlab"}, created="2026-11-14T10:00:00")
    assert validate_moon(doc) == []
    assert doc["format"] == FORMAT and doc["source"] == "Laser" and doc["set_as"] == "inlier"
    assert doc["interpolation"] is False and doc["laser"]["engine"] == "matlab"
    feats = {(f["mode"], f["feature"]): f for f in doc["features"]}
    assert set(feats) == {("RP_Pos", "FT03"), ("RP_Pos", "FT05"), ("NP_Neg", "FT002")}
    pc = feats[("RP_Pos", "FT03")]
    assert pc == {"mode": "RP_Pos", "feature": "FT03", "name": "PC(34:1)", "adduct": "[M+H]+",
                  "internal_standard": False, "mz": 603.1234, "rt": 318.0, "set_as": "inlier",
                  "laser": {"role": "primary", "status": "fitted_grid", "ppm": 0.5,
                            "matlab_compound": 25.0, "matlab_code": 73.0}}
    std = feats[("RP_Pos", "FT05")]
    assert std["name"] == "PC(28:0)" and std["internal_standard"] and std["laser"]["name"] == "PC(28:0) (IS)"
    assert feats[("NP_Neg", "FT002")]["internal_standard"] is True
    assert report["skipped_by_reason"] == {"duplicate_feature": 1, "isotope": 1, "unknown_feature": 1,
                                           "mz_mismatch": 1, "unknown_stream": 1}
    assert report["features"] == 3 and report["rows"] == len(rows)


def test_write_read_round_trip(project, tmp_path):
    doc, _ = annotations_to_moon([laser_row("RP_Pos", "101", "LPC(O+P-18:2) µ")], project)
    for name in ("export.moon", "EXPORT.MOON", "export.json"):
        path = write_moon(doc, tmp_path / name)
        text = path.read_text(encoding="utf-8")
        assert "µ" in text and "\n  " in text and "$schema" not in text
        assert read_moon(path) == doc
    bom = tmp_path / "bom.moon"
    bom.write_bytes(b"\xef\xbb\xbf" + (tmp_path / "export.moon").read_bytes())
    assert read_moon(bom) == doc


@pytest.mark.parametrize("doc, message", [
    ([], "JSON object"),
    ({"features": {}}, "list"),
    ({"format": "moon-annotation/2", "features": []}, "unknown format"),
    ({"features": ["x"]}, "expected an object"),
    ({"features": [{"mode": "RP_Pos", "feature": "FT1", "name": "PC(34:1)", "set_as": "data"}]}, "missing adduct"),
    ({"features": [{"mode": "RP_Pos", "feature": "FT1", "name": "PC(34:1)", "adduct": "M+H"}]}, "adduct"),
    ({"features": [{"mode": "RP_Pos", "feature": "FT1", "name": "X", "adduct": "[M+H]+", "role": "main"}]}, "role"),
    ({"features": [{"mode": "LC_Pos", "feature": "FT1", "name": "X", "adduct": "[M+H]+"}]}, "unknown mode"),
])
def test_invalid_documents(doc, message, tmp_path):
    assert any(message in p for p in validate_moon(doc))
    path = tmp_path / "bad.moon"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        read_moon(path)
    with pytest.raises(ValueError):
        write_moon(doc, tmp_path / "never.moon")
    assert not (tmp_path / "never.moon").exists()


def test_inlier_without_adduct_is_valid():
    assert validate_moon({"set_as": "inlier", "features": [
        {"mode": "RP_Pos", "feature": "FT1", "name": "PC(34:1)"}]}) == []


def matlab_export():
    """Like Moon getannotation --all: per-feature source/role/matlab, classes."""
    return {
        "format": FORMAT, "project": "X:/p/Processing/Annotation",
        "features": [
            {"mode": "RP_Pos", "feature": "FT03", "name": "PC(34:1)", "adduct": "[M+H]+", "isotope": "M+0",
             "source": "Moon", "role": "primary", "internal_standard": False,
             "matlab": {"column": "RP", "row": 7, "i_compound": 25, "code": 101}},
            {"mode": "RP_Pos", "feature": "FT5", "name": "PC(28:0) (IS)", "adduct": "[M+H]+",
             "source": "Laser", "internal_standard": True, "matlab": {"i_compound": 25}},
            {"mode": "RP_Pos", "feature": "FT03", "name": "PC(34:1)", "adduct": "[M+H]+", "isotope": "(13C)",
             "role": "other"},
            {"mode": "RP_Pos", "feature": "FT99", "name": "PE(36:1)", "adduct": "[M+H]+"},
            {"mode": "NP_Neg", "feature": "FT002", "name": "PE(36:2)", "adduct": "[M-H]-"},
        ],
        "classes": [{"class": "PC", "i_compound": 25, "column": "RP", "accepted": True,
                     "interpolation": False, "inliers": [], "outliers": []}],
    }


def test_import_to_laser_frame(project):
    frame = moon_to_frame(matlab_export(), project)
    assert list(frame.feat_feature_id) == ["103", "105", "103", "", "FT002"]
    assert list(frame.moon_status) == ["mapped", "mapped", "mapped", "unknown_feature", "mapped"]
    assert list(frame["class"]) == ["PC", "PC", "PC", "PE", "PE"]
    assert frame.loc[1, "name"] == "PC(28:0)" and bool(frame.loc[1, "is_is"])
    assert frame.loc[2, "isotopologue"] == "(13C)" and frame.loc[0, "isotopologue"] == ""
    assert frame.loc[0, "feat_rt"] == pytest.approx(5.3) and frame.loc[0, "moon_matlab_row"] == 7


def test_crosswalk_feeds_manual_comparison(project):
    crosswalk = moon_to_crosswalk(matlab_export(), project)
    assert len(crosswalk) == 4                                   # isotope left out
    assert list(crosswalk.crosswalk_status) == [MAPPED_STATUS, MAPPED_STATUS, "unmapped", MAPPED_STATUS]
    laser = pd.DataFrame([
        dict(stream="RP_Pos", name="PC(34:1)", **{"class": "PC"}, adduct="[M+H]+", feat_feature_id="103"),
        dict(stream="RP_Pos", name="PC(28:0) (IS)", **{"class": "PC"}, adduct="[M+H]+", feat_feature_id="106",
             is_is=True),
    ])
    result = compare_manual_annotations(crosswalk, laser)
    outcomes = dict(zip(result.manual.original_name, result.manual.outcome))
    # The IS keeps its identity through the flag (name without "(IS)")
    assert outcomes == {"PC(34:1)": "same_manual_peak", "PC(28:0)": "same_lipid_different_peak",
                        "PE(36:1)": "manual_peak_unresolved", "PE(36:2)": "manual_lipid_not_selected"}


def test_laser_export_reads_back_with_same_ids(project):
    rows = [laser_row("RP_Pos", "102", "PC(32:0)"), laser_row("NP_Neg", "FT003", "PE(38:4)", adduct="[M-H]-")]
    doc, _ = annotations_to_moon(rows, project)
    frame = moon_to_frame(doc, project)
    assert list(zip(frame.stream, frame.feat_feature_id, frame.name, frame.source)) == [
        ("RP_Pos", "102", "PC(32:0)", "Laser"), ("NP_Neg", "FT003", "PE(38:4)", "Laser")]


def test_cli_writes_moon(project, tmp_path, capsys):
    import standard_cli

    args = argparse.Namespace(moon=tmp_path / "out" / "laser.moon", project=project, engine="matlab", phase="both")
    (tmp_path / "out").mkdir()
    standard_cli._write_moon(args, pd.DataFrame([laser_row("RP_Pos", "101", "PC(31:0)"),
                                                 laser_row("RP_Pos", "999", "PC(30:0)")]))
    doc = read_moon(tmp_path / "out" / "laser.moon")
    assert [f["feature"] for f in doc["features"]] == ["FT01"]
    assert doc["laser"]["engine"] == "matlab" and doc["laser"]["laser_version"]
    report = json.loads((tmp_path / "out" / "laser_report.json").read_text(encoding="utf-8"))
    assert report["skipped_by_reason"] == {"unknown_feature": 1}
    assert "1 rows not written" in capsys.readouterr().out


def test_cli_option_is_optional():
    import standard_cli

    assert standard_cli._parser().parse_args(["x"]).moon is None


def test_crosswalk_tool(project, tmp_path):
    moon = tmp_path / "matlab.moon"
    moon.write_text(json.dumps(matlab_export()), encoding="utf-8")
    out = tmp_path / "curation" / "ds" / "curated_feature_crosswalk.csv"
    run = subprocess.run([sys.executable, str(TOOL), str(moon), str(project), "--output", str(out)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    table = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(table.xcms_feature_id) == ["103", "105", "", "FT002"]
    again = subprocess.run([sys.executable, str(TOOL), str(moon), str(project), "--output", str(out)],
                           capture_output=True, text=True)
    assert again.returncode != 0 and "never overwritten" in again.stderr
