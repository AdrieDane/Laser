import pandas as pd
import pytest

from matlab_annotation.standard_rt_library import (
    fit_standard_retention_library,
    predict_standard_retention,
)


def inputs():
    data = pd.DataFrame(
        [
            dict(
                cohort=f"g{g}",
                stream="RP_Pos",
                standard_compound_id=f"S{i}|100",
                adduct="[M+H]+",
                rt_seconds=200 + 80 * i + g,
            )
            for g in range(4)
            for i in range(7)
        ]
    )
    library = fit_standard_retention_library(data)
    targets = pd.DataFrame(
        [
            dict(
                name=f"S{i} (IS)",
                adduct="[M+H]+",
                ref_mz=300 + i,
                standard_compound_id=f"S{i}|100",
            )
            for i in range(7)
        ]
    )
    auto = targets.assign(
        feat_mz=targets.ref_mz,
        feat_rt=[(200 + 80 * i + 12) / 60 for i in range(7)],
        feat_intensity=1e6,
        is_is=True,
    )
    return data, library, targets, auto


def test_other_standards_align_a_frozen_training_library():
    _, library, targets, auto = inputs()
    result = predict_standard_retention(targets, auto, library, stream="RP_Pos")
    assert result.standard_adduct_supported.all()
    assert result.standard_alignment_anchors.iloc[1:6].eq(6).all()
    assert (
        result.standard_rt_seconds.iloc[[0, 6]].isna().all()
    )  # Extrapolation is deliberately limited.
    assert result.standard_rt_seconds.iloc[3] == pytest.approx(452.0, abs=0.2)


def test_target_and_all_companion_adducts_cannot_calibrate_themselves():
    _, library, targets, auto = inputs()
    # Supply the complete declaration catalog for matching other standards.
    expected = predict_standard_retention(
        targets, auto.drop(index=3), library, stream="RP_Pos"
    ).iloc[3]
    wrong = auto.iloc[[3]].assign(feat_rt=9999.0, feat_intensity=1e12)
    poisoned = pd.concat(
        [auto.drop(index=3), pd.concat([wrong] * 20)], ignore_index=True
    )
    actual = predict_standard_retention(
        targets, poisoned, library, stream="RP_Pos"
    ).iloc[3]
    pd.testing.assert_series_equal(expected, actual)


def test_sparse_or_inconsistent_alignment_and_unseen_adduct_stays_unavailable():
    _, library, targets, auto = inputs()
    result = predict_standard_retention(
        targets, auto.iloc[:3], library, stream="RP_Pos"
    )
    assert result.standard_rt_seconds.isna().all()
    extra = targets.copy()
    extra["adduct"] = "[M+Na]+"
    result = predict_standard_retention(extra, auto, library, stream="RP_Pos")
    assert not result.standard_adduct_supported.any()
    assert result.standard_rt_seconds.isna().all()
    with pytest.raises(ValueError):
        fit_standard_retention_library(pd.DataFrame())


def test_narrow_NP_standard_range_estimates_only_a_time_offset():
    data, _, targets, auto = inputs()
    data["stream"] = "NP_Neg"
    data["adduct"] = "[M-H]-"
    data["rt_seconds"] = (
        500
        + data.standard_compound_id.str.extract(r"S(\d+)")[0].astype(int) * 10
        + data.cohort.str[1:].astype(int)
    )
    targets["adduct"] = "[M-H]-"
    auto["adduct"] = "[M-H]-"
    auto["feat_rt"] = [(500 + 10 * i + 40) / 60 for i in range(7)]
    library = fit_standard_retention_library(data)
    result = predict_standard_retention(targets, auto, library, stream="NP_Neg")
    assert result.standard_alignment_anchors.eq(6).all()
    assert result.standard_rt_seconds.iloc[3] == pytest.approx(570.0, abs=0.1)


def test_declared_standard_reporting_consensus_is_phase_specific_and_does_not_merge_isomers():
    data, _, targets, auto = inputs()
    key = "LPC(14:0)|100"
    data.loc[data.standard_compound_id.eq("S3|100"), "standard_compound_id"] = key
    data["reporting_name"] = (
        data.standard_compound_id.str.split("|", regex=False).str[0] + " (IS)"
    )
    data["reporting_class"] = "other"
    data.loc[data.standard_compound_id.eq(key), "reporting_name"] = (
        "1-acyl LPC(14:0) (IS)"
    )
    data.loc[data.standard_compound_id.eq(key), "reporting_class"] = "1-acylLPC"
    npdata = data.assign(stream="NP_Pos")
    npdata.loc[npdata.standard_compound_id.eq(key), "reporting_name"] = "LPC(14:0) (IS)"
    npdata.loc[npdata.standard_compound_id.eq(key), "reporting_class"] = "LPC"
    library = fit_standard_retention_library(
        pd.concat([data, npdata], ignore_index=True)
    )
    targets.loc[3, ["name", "standard_compound_id"]] = ["LPC(14:0) (IS)", key]
    auto.loc[3, ["name", "standard_compound_id"]] = ["LPC(14:0) (IS)", key]
    rp = predict_standard_retention(targets, auto, library, stream="RP_Pos")
    npresult = predict_standard_retention(targets, auto, library, stream="NP_Pos")
    assert rp.standard_report_name.iloc[3] == "1-acylLPC(14:0) (IS)"
    assert npresult.standard_report_name.iloc[3] == "LPC(14:0) (IS)"
    assert rp.standard_rt_seconds.iloc[3] == pytest.approx(452.0, abs=0.2)
    # Mixed sn1/sn2 reference labels cannot become an unconditional sn1 label.
    data.loc[
        data.cohort.eq("g0") & data.standard_compound_id.eq(key), "reporting_name"
    ] = "2-acylLPC(14:0) (IS)"
    mixed = fit_standard_retention_library(data)
    assert "reporting_name" not in mixed["models"]["RP_Pos|" + key]
