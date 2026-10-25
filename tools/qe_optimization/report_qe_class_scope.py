"""Full dataset/mode companion for one class; only complete measured runs."""

import argparse
from pathlib import Path

import pandas as pd


def report(comparison, lipid_class, destination):
    whole = pd.read_csv(comparison / "by_dataset_mode_before_after.csv")
    assert len(whole) == 132 and whole.dataset.nunique() == 33
    assert not whole.duplicated(["dataset", "stream"]).any()
    assert set(whole.stream) == {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}
    source = pd.read_csv(comparison / "by_dataset_mode_class_before_after.csv")
    selected = source.loc[source.lipid_class.eq(lipid_class)].copy()
    assert not selected.duplicated(["dataset", "stream"]).any()
    result = whole[["dataset", "stream"]].merge(
        selected, on=["dataset", "stream"], how="left", validate="one_to_one"
    )
    absent = result.lipid_class.isna()
    result.loc[absent, "lipid_class"] = lipid_class
    numeric = [c for c in result if c not in {"dataset", "stream", "lipid_class"}]
    # An absent class row means zero observations/calls in a TESTED mode, not
    # an untested mode. Percentages with zero manual denominator remain NA.
    for c in numeric:
        if not c.startswith("correct_manual_percent"):
            result.loc[absent, c] = 0
            result[c] = result[c].astype(int)
    for side in ("before", "after"):
        n, d = result[f"manual_same_peak_{side}"], result[f"manual_rows_{side}"]
        result[f"correct_manual_percent_{side}"] = 100 * n / d.where(d.ne(0))
    assert len(result) == 132
    assert (
        result.manual_same_peak_before.sum() == selected.manual_same_peak_before.sum()
    )
    assert result.manual_same_peak_after.sum() == selected.manual_same_peak_after.sum()
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(
        f"Complete {lipid_class} companion: 33 datasets /132 tested modes: {destination}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--class", dest="lipid_class", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report(a.comparison, a.lipid_class, a.output)
