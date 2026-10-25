import pandas as pd
import pytest

from report_qe_class_scope import report


def test_full_tested_scope_fills_only_absent_class_counts_not_percentages(tmp_path):
    rows = [
        dict(dataset=f"d{i}", stream=s)
        for i in range(33)
        for s in ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")
    ]
    pd.DataFrame(rows).to_csv(
        tmp_path / "by_dataset_mode_before_after.csv", index=False
    )
    single = dict(rows[0], lipid_class="BMP", correct_gain=1, extra_change=2)
    for side, correct in (("before", 2), ("after", 3)):
        single.update(
            {
                f"manual_rows_{side}": 5,
                f"manual_same_peak_{side}": correct,
                f"correct_manual_percent_{side}": 20 * correct,
                f"laser_not_in_manual_list_{side}": correct,
                f"unresolved_manual_rows_{side}": 0,
                f"laser_same_manual_lipid_different_peak_{side}": 0,
                f"unique_laser_annotations_{side}": 2 * correct,
            }
        )
    pd.DataFrame([single]).to_csv(
        tmp_path / "by_dataset_mode_class_before_after.csv", index=False
    )
    output = tmp_path / "report.csv"
    report(tmp_path, "BMP", output)
    result = pd.read_csv(output)
    assert len(result) == 132 and result.manual_rows_before.sum() == 5
    assert result.correct_manual_percent_before.notna().sum() == 1
    assert result.manual_same_peak_after.sum() == 3
    incomplete = pd.DataFrame(rows).iloc[:-1]
    incomplete.to_csv(tmp_path / "by_dataset_mode_before_after.csv", index=False)
    with pytest.raises(AssertionError):
        report(tmp_path, "BMP", tmp_path / "invalid.csv")
