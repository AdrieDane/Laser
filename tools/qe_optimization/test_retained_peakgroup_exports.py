"""Export checks are formatting-only; no model or annotation fits."""

import pandas as pd
import pytest

from export_qe_retained_peakgroups import check_export, scores_text


def test_exact_bytes_crlf_leading_zero_and_no_final_newline():
    original = b"name\tmzmed\tx\r\n001\t1.0000\t2e-6\r\n2\t2\tNA"
    exported = (b"name\tlaser_annotation\tadduct\tmzmed\tx\r\n"
                b"001\tPE(34:1)\t[M-H]-\t1.0000\t2e-6\r\n2\t\t\t2\tNA")
    rows, count = check_export(original, exported, {"001": ("PE(34:1)", "[M-H]-")}, "RP_Neg")
    assert count == 2 and rows == [dict(stream="RP_Neg", feat_feature_id="001",
                                      name="PE(34:1)", adduct="[M-H]-")]


def test_bom_preserved_and_nonfirst_id_column():
    raw = b"\xef\xbb\xbfother\tname\tx\na\t42\t001\n"
    out = b"\xef\xbb\xbfother\tlaser_annotation\tadduct\tname\tx\na\tPC(32:0)\t[M+H]+\t42\t001\n"
    rows, count = check_export(raw, out, {"42": ("PC(32:0)", "[M+H]+")}, "NP_Pos")
    assert count == 1 and rows[0]["feat_feature_id"] == "42"


@pytest.mark.parametrize("change", ("values", "rows", "identity", "adduct", "missing", "header"))
def test_reject_changes_or_dropped_calls(change):
    raw = b"name\tmz\n1\t1.0000\n2\t2\n"
    output = b"name\tlaser_annotation\tadduct\tmz\n1\tPE(34:1)\t[M-H]-\t1.0000\n2\t\t\t2\n"
    expected = {"1": ("PE(34:1)", "[M-H]-")}
    if change == "values":
        output = output.replace(b"1.0000", b"1.0")
    elif change == "rows":
        output = output.replace(b"2\t\t\t2\n", b"")
    elif change == "identity":
        output = output.replace(b"PE(34:1)", b"PC(34:1)")
    elif change == "adduct":
        output = output.replace(b"[M-H]-", b"[M+H]+")
    elif change == "missing":
        expected["3"] = ("PE(36:1)", "[M-H]-")
    else:
        output = output.replace(b"laser_annotation", b"annotation")
    with pytest.raises(ValueError):
        check_export(raw, output, expected, "NP_Neg")


def test_unselected_rows_must_stay_blank():
    raw = b"name\tmz\n1\t500\n"
    out = b"name\tlaser_annotation\tadduct\tmz\n1\tPC(32:0)\t[M+H]+\t500\n"
    with pytest.raises(ValueError):
        check_export(raw, out, {}, "RP_Pos")


def test_duplicate_source_ids_refused():
    raw = b"name\tmz\n1\t500\n1\t501\n"
    out = b"name\tlaser_annotation\tadduct\tmz\n1\t\t\t500\n1\t\t\t501\n"
    with pytest.raises(ValueError):
        check_export(raw, out, {}, "RP_Pos")


def test_clear_scores_full_manual_denominator_and_zero_manual():
    table = pd.DataFrame([
        dict(stream="NP_Neg", manual_rows=100, manual_same_peak=89,
             unresolved_manual_rows=2, laser_not_in_manual_list=7,
             laser_same_manual_lipid_different_peak=3, unique_laser_annotations=99),
        dict(stream="RP_Pos", manual_rows=0, manual_same_peak=0,
             unresolved_manual_rows=0, laser_not_in_manual_list=4,
             laser_same_manual_lipid_different_peak=0, unique_laser_annotations=4),
    ])
    rendered = scores_text(table, "Test")
    assert "89.0% (89/100)" in rendered and "Total extra annotations: 7" in rendered
    assert "Manual annotations not correctly matched: 11" in rendered
    assert "n/a (0 manual annotations)" in rendered
    assert "Unresolved manual peaks (included above): 2" in rendered
