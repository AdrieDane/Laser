import ast
from pathlib import Path

import pandas as pd
import pytest

from verify_qe_competing_ion_family_pilot_v2 import assert_same_links


def test_edge_order_is_ignored_but_endpoints_counts_and_duplicates_are_exact():
    links = pd.DataFrame(dict(source_index=[2, 1, 2], alternative_index=[3, 7, 3]))
    assert_same_links(links, links.iloc[[1, 2, 0]].reset_index(drop=True))
    wrong = links.copy()
    wrong.loc[0, "alternative_index"] = 4
    for variant in (
        wrong,
        links.iloc[:2],
        links.drop_duplicates(),
        pd.concat([links, links.iloc[[0]]], ignore_index=True),
    ):
        with pytest.raises(AssertionError):
            assert_same_links(links, variant)


def test_v2_repeats_numeric_and_physical_checks_without_relaxation():
    directory = Path(__file__).parent
    old = (directory / "verify_qe_competing_ion_family_pilot.py").read_text()
    new = (directory / "verify_qe_competing_ion_family_pilot_v2.py").read_text()
    old = old.replace(
        'pd.testing.assert_frame_equal(saved["links"], links, check_exact=True)',
        'assert_same_links(saved["links"], links)',
    )

    def function(source):
        return next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == "one"
        )

    assert ast.dump(function(old)) == ast.dump(function(new))


def test_gold_analysis_same_as_original_except_receipt_and_output_paths():
    directory = Path(__file__).parent
    old = (directory / "diagnose_qe_competing_ion_family_pilot.py").read_text()
    expected = old.replace(
        "competing_ion_family_diagnostic_041_v1",
        "competing_ion_family_diagnostic_041_v2",
    ).replace(
        "competing_ion_family_independent_check_v1/receipt.json",
        "competing_ion_family_independent_check_v2/receipt.json",
    )
    new = (directory / "diagnose_qe_competing_ion_family_pilot_v2.py").read_text()
    assert expected.strip() == new.strip()
