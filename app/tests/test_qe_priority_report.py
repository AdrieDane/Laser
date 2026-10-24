import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/qe_optimization/report_qe_priority_classes.py"
)
SPEC = importlib.util.spec_from_file_location("qe_priority_report", PATH)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def test_priority_groups_shared_acquisitions_and_preserves_sparse_and_excluded_classes():
    rows = pd.DataFrame(
        [
            ("24P0044_fish_A", "TG", 30),
            ("24P0044_fish_B", "TG", 30),
            ("24P0055_brain", "TG", 30),
            ("24P0066_liver", "TG", 30),
            ("24P0044_fish_A", "SMALL", 10),
            ("24P0055_brain", "SMALL", 10),
            ("24P0066_liver", "SMALL", 10),
            ("24P0044_fish_A", "X4", 2),
        ],
        columns=["dataset", "lipid_class", "manual_rows_after"],
    )
    rows["stream"] = "RP_Pos"
    rows["manual_rows_before"] = rows.manual_rows_after
    result = REPORT.priority_table(rows).set_index("lipid_class")
    assert result.loc["TG", "independent_project_groups"] == 3
    assert result.loc["TG", "manual_entries"] == 120
    assert result.loc["TG", "optimization_priority"] == "prioritize"
    assert result.loc["SMALL", "optimization_priority"] == "defer_sparse"
    assert result.loc["X4", "optimization_priority"] == "user_excluded_X4"
    assert result.manual_entries.sum() == rows.manual_rows_after.sum()
    with pytest.raises(ValueError, match="identical"):
        REPORT.priority_table(rows.assign(manual_rows_before=0))
