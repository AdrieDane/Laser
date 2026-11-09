"""Check LASER CLI output against synthetic_truth.csv.

Usage: python check_synthetic_output.py <base_dir> <output_subdir>
<base_dir> must contain synthetic_project/ made by make_synthetic_project.py.
"""
import re
import sys

import pandas as pd

S, out = sys.argv[1], sys.argv[2]
truth = pd.read_csv(f"{S}/synthetic_project/synthetic_truth.csv", dtype=str)


def norm(s):
    return re.sub(r"^(1-acyl |2-acyl )", "", str(s).replace(" (IS)", "")).strip()


for stream in ("RP_Pos", "RP_Neg"):
    a = pd.read_csv(f"{S}/{out}/annotated_input_files/peakgrouplist_{stream}_annotated.txt", sep="\t", dtype=str)
    orig = pd.read_csv(f"{S}/synthetic_project/XCMS/peakgrouplist_{stream}.txt", sep="\t", dtype=str)
    assert a.drop(columns=["laser_annotation","adduct"]).equals(orig), "original columns changed!"
    t = truth[truth.stream == stream].set_index("feature_id")
    called = a[a.laser_annotation.notna()].set_index("name")
    n = len(t)
    iso_ids = set(a.name.iloc[n:2 * n])
    on_iso = called.index.isin(iso_ids).sum()
    print(f"  {stream}: calls on 13C isotope features: {on_iso}")
    planted = called.index.isin(t.index)
    exact = sum(norm(called.loc[f,"laser_annotation"]) == norm(t.loc[f,"name"]) and called.loc[f,"adduct"] == t.loc[f,"adduct"] for f in called.index[planted])
    print(f"{stream}: planted {len(t)}, annotated {len(called)} (on planted {planted.sum()}, on noise {(~planted).sum()}), same lipid+adduct on planted feature {exact}; original columns byte-identical: yes")
