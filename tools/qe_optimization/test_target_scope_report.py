import pandas as pd
import pytest

from report_qe_target_scope import FIELDS, target_tables


def test_target_exclusion_changes_scope_not_annotation_counts_or_source():
    rows = []
    for cls, manual, correct in (("PE", 10, 9), ("X4", 2, 0)):
        row = dict(dataset="example", stream="NP_Neg", lipid_class=cls)
        for side in ("before", "after"):
            row.update({f + "_" + side: 0 for f in FIELDS})
            row["manual_rows_" + side] = manual
            row["manual_same_peak_" + side] = correct
            row["laser_not_in_manual_list_" + side] = 3 if cls == "PE" else 0
        rows.append(row)
    source = pd.DataFrame(rows)
    original = source.copy(deep=True)
    table = target_tables(source, {"X4"})["by_mode"]
    assert table.manual_rows_after.iloc[0] == 10
    assert table.correct_manual_percent_after.iloc[0] == 90
    assert table.laser_not_in_manual_list_after.iloc[0] == 3
    assert table.correct_gain.iloc[0] == 0
    pd.testing.assert_frame_equal(source, original)
    with pytest.raises(ValueError, match="existing classes"):
        target_tables(source, {"unknown"})
