from __future__ import annotations

from dataclasses import asdict
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest
from scipy.io import savemat

from tools.dataset_scanner import core
from tools.dataset_scanner.readers import Workbook, load_rt_predictions
from tools.dataset_scanner.server import Session, make_handler


def make_workbook(path, sheets):
    """Minimal native test fixtures, not authored user spreadsheets."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    book = ET.Element("workbook", xmlns=ns)
    entries = ET.SubElement(book, "sheets")
    rels = ET.Element(
        "Relationships",
        xmlns="http://schemas.openxmlformats.org/package/2006/relationships",
    )
    with zipfile.ZipFile(path, "w") as archive:
        for index, (name, rows) in enumerate(sheets.items(), 1):
            ET.SubElement(
                entries,
                "sheet",
                name=name,
                sheetId=str(index),
                **{
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id": f"rId{index}"
                },
            )
            ET.SubElement(
                rels,
                "Relationship",
                Id=f"rId{index}",
                Target=f"worksheets/sheet{index}.xml",
            )
            sheet = ET.Element("worksheet", xmlns=ns)
            data = ET.SubElement(sheet, "sheetData")
            for rownum, values in enumerate(rows, 1):
                row = ET.SubElement(data, "row", r=str(rownum))
                for column, value in enumerate(values):
                    if column >= 26:
                        raise ValueError("Fixture column range too large")
                    cell = ET.SubElement(
                        row, "c", r=f"{chr(65 + column)}{rownum}", t="inlineStr"
                    )
                    ET.SubElement(ET.SubElement(cell, "is"), "t").text = str(value)
            archive.writestr(f"xl/worksheets/sheet{index}.xml", ET.tostring(sheet))
        archive.writestr("xl/workbook.xml", ET.tostring(book))
        archive.writestr("xl/_rels/workbook.xml.rels", ET.tostring(rels))


def make_complete_dataset(source):
    project = source / "2026" / "26P9999_ScannerTest"
    project.mkdir(parents=True)
    raw = [
        [
            "name",
            "adduct",
            "m/z",
            "RT",
            "class",
            "column",
            "Sample 0",
            "Sample 1",
            "Sample 2",
            "Sample 3",
        ],
        ["", "", "", "", "", "", "group", "group", "group", "group"],
    ]
    normalized = [["name", "chem formula"]]
    mat = {
        phase: dict(mz_grouped=[], RT_grouped=[], RTp_grouped=[], annotation=[])
        for phase in ("RP", "NP")
    }
    for i in range(60):
        formula = f"C{30 + i}H{60 + i}O2"
        name = f"ScannerTest({i})"
        normalized.append([name, formula])
        neutral = core.formula_mass(formula)
        for stream in core.STREAMS:
            sign = 1 if stream.endswith("Pos") else -1
            mz = (neutral + sign * core.PROTON) * (1 + 0.5e-6)
            rt = 600 + i * 4 + (50 if sign < 0 else 0)
            raw.append(
                [
                    name,
                    "[M+H]+" if sign > 0 else "[M-H]-",
                    sign * mz,
                    rt,
                    f"TestClass{i % 3}",
                    stream[:2],
                    100,
                    200,
                    400,
                    800,
                ]
            )
            phase = mat[stream[:2]]
            phase["mz_grouped"].append(sign * mz)
            phase["RT_grouped"].append(rt)
            phase["RTp_grouped"].append(rt - 1)
            phase["annotation"].append([1, i + 1])
    make_workbook(
        project / "MS_data.xlsx",
        {"Primary list (raw)": raw, "Primary list (normalized)": normalized},
    )
    for phase in ("RP", "NP"):
        rows = [["Mode", "File Name", "Sample Name", "Protein", "Sample Type"]]
        for polarity in ("Pos", "Neg"):
            stream = f"{phase}_{polarity}"
            entries = [
                r
                for r in raw[2:]
                if r[5] == phase and r[1].endswith("+" if polarity == "Pos" else "-")
            ]
            content = "name\tmzmed\trtmed\tf0\tf1\tf2\tf3\n" + "\n".join(
                "\t".join(map(str, [f"original-{i}", abs(r[2]), r[3] / 60, *r[6:]]))
                for i, r in enumerate(entries)
            )
            (project / f"peakgrouplist_{stream}.txt").write_text(content)
            rows.extend(
                [[stream, f"f{i}", f"Sample_{i}", 1, "study"] for i in range(4)]
            )
        make_workbook(project / f"26P9999_{phase}.xlsx", {"metadata": rows})
        for key in mat[phase]:
            mat[phase][key] = np.asarray(mat[phase][key])
    savemat(project / "MS_data.mat", {"MS_data": mat}, do_compression=True)
    return project


def stream_fixture(tmp_path, *, ppm=1, rt_error=1, duplicate=False):
    rows, mass, predicted, peak_rows = [], {}, [], []
    samples = {f"f{i}": (f"Sample {i}", 1.0) for i in range(4)}
    for index in range(6):
        name = f"PE({30 + index}:1)"
        theory = 700 + index
        observed = theory * (1 + ppm * 1e-6)
        rt = 600 + 4 * index
        values = [100, 200, 400, 800]
        rows.append(
            dict(
                stream="RP_Pos",
                name=name,
                adduct="[M+H]+",
                lipid_class="PE",
                mz=observed,
                rt_seconds=rt,
                profile={f"Sample {i}": v for i, v in enumerate(values)},
                sheet="Primary list (raw)",
                excel_row=index + 3,
            )
        )
        mass[(core.mass_name(name), "[M+H]+")] = theory
        predicted.append((observed, rt, rt - rt_error))
        peak_rows.append([f"original-{index}", observed, rt / 60, *values])
        if duplicate:
            peak_rows.append([f"other-{index}", observed, rt / 60, *values])
    path = tmp_path / "peakgrouplist_RP_Pos.txt"
    path.write_text(
        "name\tmzmed\trtmed\tf0\tf1\tf2\tf3\n"
        + "\n".join("\t".join(map(str, r)) for r in peak_rows)
    )
    return rows, mass, predicted, path, samples


def screen(args, **overrides):
    rows, masses, predictions, path, samples = args
    return core.screen_stream(
        "RP_Pos",
        rows,
        {},
        overrides.get("masses", masses),
        overrides.get("predictions", predictions),
        path,
        samples,
        core.Limits(min_annotations=3, min_classes=1),
    )


def test_good_stream_has_real_feature_ids_and_separate_mass_rt_evidence(tmp_path):
    result, proofs = screen(stream_fixture(tmp_path))
    assert result["status"] == "PASS"
    assert result["feature_match_percent"] == result["mass_within_limit_percent"] == 100
    assert result["rt_within_limit_percent"] == 100
    assert result["median_absolute_ppm"] == pytest.approx(1)
    assert result["median_absolute_rt_seconds"] == pytest.approx(1)
    assert {r["original_feature_id"] for r in proofs} == {
        f"original-{i}" for i in range(6)
    }


@pytest.mark.parametrize(
    "setting,kwargs",
    [
        ("mass", dict(ppm=4)),
        ("RT", dict(rt_error=20)),
        ("ambiguous", dict(duplicate=True)),
    ],
)
def test_bad_or_ambiguous_evidence_is_not_a_pass(tmp_path, setting, kwargs):
    result, _ = screen(stream_fixture(tmp_path, **kwargs))
    assert result["status"] == "REVIEW"


def test_unknown_reference_or_missing_saved_RT_is_not_a_good_measurement(tmp_path):
    args = stream_fixture(tmp_path)
    assert screen(args, masses={})[0]["mass_assessed"] == 0
    assert screen(args, masses={})[0]["status"] == "REVIEW"
    assert screen(args, predictions=[])[0]["rt_assessed"] == 0
    assert screen(args, predictions=[])[0]["status"] == "REVIEW"


def test_conflicting_saved_predictions_cannot_cherry_pick_a_good_residual(tmp_path):
    args = stream_fixture(tmp_path)
    predictions = args[2] + [(mz, rt, pred + 100) for mz, rt, pred in args[2]]
    assert screen(args, predictions=predictions)[0]["rt_assessed"] == 0


def test_XCMS_minutes_are_explicitly_converted_to_seconds(tmp_path):
    args = stream_fixture(tmp_path)
    result, _ = screen(args)
    assert result["median_absolute_rt_seconds"] == pytest.approx(1)
    path = args[3]
    data = path.read_text().replace("10.0\t", "600.0\t")
    path.write_text(data)
    assert screen(args)[0]["matched_manual_features"] == 5


def test_duplicate_original_feature_ids_are_invalid(tmp_path):
    args = stream_fixture(tmp_path)
    p = args[3]
    p.write_text(p.read_text().replace("original-1", "original-0"))
    with pytest.raises(ValueError, match="duplicate"):
        screen(args)


def test_formula_and_charge_parsing_do_not_guess_unknown_chemistry():
    assert core.formula_mass("C2H6O") == pytest.approx(46.041864813)
    assert core.formula_mass("C2H6Unknown") is None
    assert core.formula_mass("(13C)2H6O") is None
    assert core.canonical_adduct("[M-2H](2-)") == "[M-2H]2-"
    assert core.mass_name("PC[O](36:2)") == core.mass_name("PC(O-36:2)")


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("endian", ["<", ">"])
def test_native_MAT_reader_matches_scipy_generated_arrays(tmp_path, compressed, endian):
    p = tmp_path / "MS_data.mat"
    count = 7
    phase = dict(
        mz_grouped=np.arange(count, dtype=endian + "f8") + 700,
        RT_grouped=np.arange(count, dtype=endian + "f8") + 600,
        RTp_grouped=np.arange(count, dtype=endian + "f8") + 599,
        annotation=np.column_stack(
            [np.ones(count, dtype="uint16"), np.arange(count, dtype="uint16")]
        ),
        x_grouped=np.ones((count, 10)),
    )
    savemat(p, dict(MS_data=dict(RP=phase, NP=phase)), do_compression=compressed)
    output = load_rt_predictions(p)
    assert output["RP"] == [(700 + i, 600 + i, 599 + i) for i in range(count)]
    assert output["NP"] == output["RP"]


def test_HDF5_or_corrupt_MAT_files_are_explicitly_unsupported(tmp_path):
    p = tmp_path / "MS_data.mat"
    p.write_bytes(b"\x89HDF" + b"\0" * 128)
    with pytest.raises(ValueError, match="v7.3/HDF5"):
        load_rt_predictions(p)


def test_XLSX_reader_skips_blank_header_line_and_supports_multiple_charge(tmp_path):
    p = tmp_path / "MS_data.xlsx"
    make_workbook(
        p,
        {
            "Primary list (raw)": [
                ["name", "adduct", "m/z", "RT", "class", "column", "S 1"],
                ["", "", "", "", "", "", "group"],
                ["GD1(d36:1)", "[M-2H](2-)", -700, 600, "GD1[d]", "RP", 100],
            ]
        },
    )
    rows, _ = core.curated_rows(p)
    assert len(rows) == 1 and rows[0]["stream"] == "RP_Neg" and rows[0]["mz"] == 700
    with Workbook(p) as book:
        assert "Primary list (raw)" in book.sheets


def test_discovery_is_restricted_to_2024_2025_2026_and_excludes_archives(tmp_path):
    for year in ["2023", "2024", "2025", "2026", "2027", "_Statistics_Archive"]:
        d = tmp_path / year / "24P0001_Test"
        d.mkdir(parents=True)
        (d / "MS_data.xlsx").write_bytes(b"test")
    _, groups, issues = core.discovery(tmp_path)
    assert len(groups) == 3 and not issues
    assert {p.relative_to(tmp_path).parts[0] for p in groups} == set(core.YEARS)
    with pytest.raises(ValueError, match="Only year folders"):
        core.discovery(tmp_path / "2023")


def test_discovery_does_not_follow_linked_years_or_project_folders(tmp_path):
    year = tmp_path / "2026"
    year.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "MS_data.xlsx").write_bytes(b"test")
    try:
        (year / "26P0001_Link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink privileges unavailable")
    assert not core.discovery(tmp_path)[1]


def test_ambiguous_processing_versions_are_not_chosen_by_date(tmp_path):
    a = tmp_path / "A" / "MS_data.mat"
    b = tmp_path / "B" / "MS_data.mat"
    with pytest.raises(ValueError, match="Multiple versions"):
        core.choose(
            [a, b], tmp_path, lambda p: p.name == "MS_data.mat", "RT predictions"
        )
    assert (
        core.choose(
            [a, b], a.parent, lambda p: p.name == "MS_data.mat", "RT predictions"
        )
        == a
    )


def copy_fixture(tmp_path):
    source = tmp_path / "source"
    p = source / "2026" / "26P0001_Test"
    p.mkdir(parents=True)
    f = p / "peakgrouplist_RP_Pos.txt"
    f.write_bytes(b"original bytes\r\n")
    d = dict(
        id="a" * 12,
        project=p.name,
        source_relative="2026/26P0001_Test",
        status="PASS",
        modes=[dict(stream=s, status="PASS", reasons=[]) for s in core.STREAMS],
        files=[core.file_record(f, source, "RP_Pos original features")],
        total_bytes=f.stat().st_size,
    )
    plan = dict(
        schema=core.SCHEMA,
        scan_id="test",
        source_root=str(source),
        allowed_years=list(core.YEARS),
        limits=asdict(core.Limits()),
        datasets=[d],
    )
    return plan, f, tmp_path / "destination"


def test_copy_requires_explicit_second_stage_approval_and_does_not_create_destination(
    tmp_path,
):
    plan, _, destination = copy_fixture(tmp_path)
    with pytest.raises(ValueError, match="explicit approval"):
        core.copy_approved(plan, destination, ["a" * 12])
    assert not destination.exists()


def test_copy_preserves_bytes_and_verifies_manifest(tmp_path):
    plan, f, destination = copy_fixture(tmp_path)
    original = f.read_bytes()
    completed = core.copy_approved(plan, destination, ["a" * 12], confirm=True)
    target = Path(completed[0])
    assert (target / f.name).read_bytes() == original == f.read_bytes()
    manifest = json.loads((target / "collection_manifest.json").read_text())
    assert manifest["files"][0]["sha256"] == hashlib.sha256(original).hexdigest()
    with pytest.raises(FileExistsError):
        core.copy_approved(plan, destination, ["a" * 12], confirm=True)


def test_changed_source_since_scan_is_rejected_before_any_destination_creation(
    tmp_path,
):
    plan, f, destination = copy_fixture(tmp_path)
    f.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="changed since scan"):
        core.copy_approved(plan, destination, ["a" * 12], confirm=True)
    assert not destination.exists()


@pytest.mark.parametrize("status", ["REVIEW", "INCOMPLETE", "DUPLICATE"])
def test_nonpassing_datasets_cannot_be_copied(tmp_path, status):
    plan, _, destination = copy_fixture(tmp_path)
    plan["datasets"][0]["status"] = status
    with pytest.raises(ValueError, match="passed all four"):
        core.copy_approved(plan, destination, ["a" * 12], confirm=True)
    assert not destination.exists()


@pytest.mark.parametrize(
    "path", ["../secret", "/secret", "C:/secret", "2026/../secret", "2026\\secret"]
)
def test_unsafe_plan_paths_are_rejected(path):
    with pytest.raises(ValueError):
        core.safe_relative(path)


def test_older_year_or_unsafe_dataset_id_in_edited_plan_is_rejected(tmp_path):
    plan, _, destination = copy_fixture(tmp_path)
    plan["datasets"][0]["id"] = "../../escape"
    with pytest.raises(ValueError, match="dataset ID"):
        core.copy_approved(plan, destination, ["../../escape"], confirm=True)
    plan["datasets"][0]["id"] = "a" * 12
    plan["datasets"][0]["files"][0]["source_relative"] = (
        "2023/old/peakgrouplist_RP_Pos.txt"
    )
    with pytest.raises(ValueError, match="2024–2026"):
        core.copy_approved(plan, destination, ["a" * 12], confirm=True)


def test_source_destination_overlap_is_not_allowed(tmp_path):
    plan, _, _ = copy_fixture(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        core.copy_approved(
            plan, Path(plan["source_root"]) / "new", ["a" * 12], confirm=True
        )


def test_cancel_is_not_misreported_as_an_incomplete_or_passing_dataset(tmp_path):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        core.check_cancel(cancel)
    plan, _, destination = copy_fixture(tmp_path)
    with pytest.raises(InterruptedError):
        core.copy_approved(plan, destination, ["a" * 12], confirm=True, cancel=cancel)
    assert not destination.exists()


def test_incomplete_dataset_report_still_lists_all_four_modes(tmp_path):
    result = core.inspect_dataset(
        tmp_path, [], tmp_path, tmp_path, {}, progress=lambda message: None
    )
    assert result["status"] == "INCOMPLETE"
    assert {m["stream"] for m in result["modes"]} == set(core.STREAMS)
    assert all(m["status"] == "NOT_CHECKED" for m in result["modes"])


def test_browser_service_rejects_missing_token_cross_origin_and_arbitrary_files(
    tmp_path,
):
    resources = Path(__file__).resolve().parents[2] / "tools/dataset_scanner"
    session = Session(resources, tmp_path / "reports")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session, "test-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/test-token/api/status") as response:
            assert json.load(response)["status"] == "idle"
        for path in ["/api/status", "/test-token/../../secret"]:
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(base + path)
        for headers in [
            {},
            {"X-Scanner-Token": "test-token", "Origin": "https://outside.example"},
        ]:
            request = urllib.request.Request(
                base + "/test-token/api/stop",
                data=b"{}",
                headers={"Content-Type": "application/json", **headers},
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_scan_report_does_not_create_collection_or_claim_unavailable_values(tmp_path):
    plan, _, destination = copy_fixture(tmp_path)
    plan["datasets"][0].update(
        status="INCOMPLETE", reasons=["Missing NP_Pos"], proofs=[], modes=[]
    )
    report = tmp_path / "report"
    core.write_reports(plan, report)
    assert not destination.exists()
    assert (report / "scan_plan.json").is_file()
    data = (report / "scan_summary.csv").read_text(encoding="utf-8-sig")
    assert "NOT_CHECKED" in data and len(data.splitlines()) == 5


def test_complete_two_stage_workflow_scans_reports_then_copies_only_required_files(
    tmp_path,
):
    source = tmp_path / "source"
    project = make_complete_dataset(source)
    # Unnecessary raw/plots/archives must never enter the selected collection.
    for name in (
        "instrument.raw",
        "EICs.Rdata",
        "plot.png",
        "whole-project.zip",
        "peakgrouplist_RP_Pos_annotated.txt",
    ):
        (project / name).write_bytes(b"not required")
    original = {p.name: core.sha256(p) for p in project.iterdir() if p.is_file()}
    plan = core.scan(source)
    assert len(plan["datasets"]) == 1
    d = plan["datasets"][0]
    assert d["status"] == "PASS" and len(d["files"]) == 8
    assert all(m["status"] == "PASS" for m in d["modes"])
    assert d["total_bytes"] == sum(f["size"] for f in d["files"])
    report = tmp_path / "reports"
    destination = tmp_path / "collected"
    core.write_reports(plan, report)
    assert not destination.exists()
    result = core.copy_approved(plan, destination, [d["id"]], confirm=True)
    target = Path(result[0])
    assert {p.name for p in target.iterdir()} == {f["name"] for f in d["files"]} | {
        "collection_manifest.json"
    }
    assert all(core.sha256(target / f["name"]) == f["sha256"] for f in d["files"])
    assert all(
        core.sha256(project / name) == digest for name, digest in original.items()
    )


def test_csv_report_strings_cannot_be_executed_as_excel_formulas():
    x = core.csv_safe(dict(name='=HYPERLINK("bad")', error_ppm=-1.3, other="PE(36:1)"))
    assert (
        x["name"].startswith("'=")
        and x["error_ppm"] == -1.3
        and x["other"] == "PE(36:1)"
    )


def test_limits_reject_fractional_sample_counts_and_undefined_correlations():
    with pytest.raises(ValueError, match="whole number"):
        core.Limits(min_profile_samples=3.5).validate()
    with pytest.raises(ValueError, match="At least 3"):
        core.Limits(min_profile_samples=2).validate()


def test_cancelling_during_copy_keeps_partial_separate_from_completed_dataset(tmp_path):
    plan, f, destination = copy_fixture(tmp_path)
    cancel = threading.Event()
    with pytest.raises(InterruptedError):
        core.copy_approved(
            plan,
            destination,
            ["a" * 12],
            confirm=True,
            cancel=cancel,
            progress=lambda message: (
                cancel.set() if message.startswith("Copying ") else None
            ),
        )
    partials = list(destination.iterdir())
    assert len(partials) == 1 and partials[0].name.startswith(".partial_")
    assert not (partials[0] / "collection_manifest.json").exists()
    assert f.read_bytes() == b"original bytes\r\n"


def test_insufficient_destination_space_fails_before_creating_output(
    tmp_path, monkeypatch
):
    from collections import namedtuple

    plan, _, destination = copy_fixture(tmp_path)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(core.shutil, "disk_usage", lambda path: usage(1000, 999, 1))
    with pytest.raises(ValueError, match="Not enough"):
        core.copy_approved(plan, destination, ["a" * 12], confirm=True)
    assert not destination.exists()


def test_duplicate_collections_are_flagged_and_not_counted_twice(tmp_path):
    import shutil

    source = tmp_path / "source"
    project = make_complete_dataset(source)
    shutil.copytree(project, project.parent / "26P9998_Copy")
    plan = core.scan(source)
    assert sorted(d["status"] for d in plan["datasets"]) == ["DUPLICATE", "PASS"]
    duplicate = next(d for d in plan["datasets"] if d["status"] == "DUPLICATE")
    assert duplicate["duplicate_of"]


def test_discovery_shows_current_folder_before_reading_and_live_counts(
    tmp_path, monkeypatch
):
    project = tmp_path / "2024" / "24P0001_Test"
    (project / "nested").mkdir(parents=True)
    (project / "MS_data.xlsx").write_bytes(b"test")
    (project / "nested" / "irrelevant.raw").write_bytes(b"not selected")
    events = []
    original = core.os.scandir

    def tracked(path):
        assert events[-1].details["current_path"] == str(path)
        return original(path)

    monkeypatch.setattr(core.os, "scandir", tracked)
    _, groups, issues = core.discovery(tmp_path, progress=events.append)
    assert not issues and list(groups) == [project]
    assert events[-1].details["folders_checked"] == 3
    assert events[-1].details["files_seen"] == 2
    assert events[-1].details["projects_found"] == 1
    assert events[-1].details["relevant_files_found"] == 1
    assert all(isinstance(e, str) for e in events)


def test_discovery_reports_unreadable_folder_without_claiming_full_scan(
    tmp_path, monkeypatch
):
    year = tmp_path / "2024"
    year.mkdir()
    monkeypatch.setattr(
        core.os,
        "scandir",
        lambda path: (_ for _ in ()).throw(
            PermissionError(13, "Access denied", str(path))
        ),
    )
    _, groups, issues = core.discovery(tmp_path)
    assert not groups and len(issues) == 1 and "Cannot read" in issues[0]


def test_hash_progress_counts_real_bytes_without_changing_hash(tmp_path):
    path = tmp_path / "data.bin"
    content = b"test" * (2 * 1024**2 + 1)
    path.write_bytes(content)
    events = []
    assert (
        core.sha256(path, progress=events.append) == hashlib.sha256(content).hexdigest()
    )
    assert events[-1].details["file_bytes_done"] == len(content)
    assert events[-1].details["file_bytes_total"] == len(content)
    assert all(e.details["current_path"] == str(path) for e in events)


def test_status_poll_is_not_misrepresented_as_scan_progress(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.dataset_scanner import server

    clock = [10.0]
    monkeypatch.setattr(server, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    session = Session(tmp_path, tmp_path / "reports")
    session.started = 10.0
    session.progress(core.ProgressEvent("Reading folder", current_path="G:/slow"))
    clock[0] = 80.0
    a = session.snapshot()
    assert a["progress_age_seconds"] == a["elapsed_seconds"] == 70
    assert session.snapshot()["progress_age_seconds"] == 70
    session.request_cancel()
    assert session.snapshot()["progress_age_seconds"] == 70
    assert session.snapshot()["cancel_requested"]
    session.progress(core.ProgressEvent("Read next folder", current_path="G:/next"))
    assert session.snapshot()["progress_age_seconds"] == 0
    for i in range(100):
        session.progress(f"step {i}")
    assert len(session.snapshot()["recent_activity"]) == 50


def test_blocked_worker_still_exposes_status_and_cancellation(tmp_path, monkeypatch):
    from tools.dataset_scanner import server as module

    entered, release = threading.Event(), threading.Event()

    def slow_scan(source, reference, limits, cancel, progress):
        progress(
            core.ProgressEvent(
                "Reading network folder",
                phase="Finding project folders",
                current_path="G:/slow",
            )
        )
        entered.set()
        assert release.wait(5)
        core.check_cancel(cancel)
        raise AssertionError("Test expects cancellation")

    monkeypatch.setattr(module, "scan", slow_scan)
    session = Session(tmp_path, tmp_path / "reports")
    session.start(("scan", "G:/slow", {}))
    try:
        assert entered.wait(3)
        snapshot = session.snapshot()
        assert snapshot["worker_alive"] and snapshot["status"] == "scanning"
        assert snapshot["activity"]["current_path"] == "G:/slow"
        session.request_cancel()
        assert session.snapshot()["cancel_requested"]
        assert "Waiting" in session.snapshot()["message"]
        with pytest.raises(ValueError, match="already running"):
            session.start(("scan", "G:/second", {}))
    finally:
        release.set()
        session.worker.join(timeout=3)
    assert session.snapshot()["status"] == "cancelled"
    assert not session.snapshot()["worker_alive"]


def test_activity_instrumentation_keeps_screening_results_unchanged(tmp_path):
    source = tmp_path / "source"
    make_complete_dataset(source)
    events = []
    before = core.scan(source)
    after = core.scan(source, progress=events.append)
    for volatile in ("scan_id", "created_utc"):
        before.pop(volatile)
        after.pop(volatile)
    assert after == before
    assert any(e.details.get("datasets_done") == 1 for e in events)
    assert any(e.details.get("phase") == "Reading saved RT predictions" for e in events)
    assert any(e.details.get("annotations_checked") == 60 for e in events)
