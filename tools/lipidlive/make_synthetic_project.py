"""Build a fully synthetic RP XCMS project from LASER's bundled reference data.

No real measurements are used: m/z and RT come from the bundled reference
table with small random noise; sample intensities are random numbers.
Usage: python make_synthetic_project.py <repo_root> <output_project_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
rng = np.random.default_rng(20261109)
ref = pd.read_csv(repo / "reference_data" / "4d_lipid_reference.csv")
ref = ref.dropna(subset=["RT_RP_median_min", "mz"])
(out / "XCMS").mkdir(parents=True, exist_ok=True)
samples = [f"S{i}" for i in range(1, 7)]
truth = []
for stream, sign in (("RP_Pos", 1), ("RP_Neg", -1)):
    pool = ref[np.sign(ref["charge"]) == sign]
    standards = pool[pool["name"].str.contains(r"\(IS\)", regex=True)]
    lipids = pool.drop(standards.index).sample(n=250, random_state=1)
    picked = pd.concat([standards, lipids])
    mz = picked["mz"].to_numpy() * (1 + rng.normal(0, 1e-6, len(picked)))
    rt = picked["RT_RP_median_min"].to_numpy() + rng.normal(0, 0.03, len(picked))
    # 13C (M+1) isotope peak for every planted lipid, co-eluting exactly.
    iso_mz = mz + 1.0033548 / np.abs(picked["charge"].to_numpy())
    iso_rt = rt.copy()
    n_noise = 300
    mz = np.concatenate([mz, iso_mz, rng.uniform(300, 1000, n_noise)])
    rt = np.concatenate([rt, iso_rt, rng.uniform(1, 18, n_noise)])
    n = len(mz)
    table = pd.DataFrame({
        "name": [f"FT{i:05d}" for i in range(1, n + 1)],
        "mzmed": mz.round(6), "mzmin": (mz - 0.001).round(6), "mzmax": (mz + 0.001).round(6),
        "rtmed": rt.round(4), "rtmin": (rt - 0.05).round(4), "rtmax": (rt + 0.05).round(4),
        "npeaks": 6, "Group_A": 3, "Group_B": 3,
    })
    for s in samples:
        table[s] = rng.lognormal(12, 1, n).round(1)
    table.to_csv(out / "XCMS" / f"peakgrouplist_{stream}.txt", sep="\t", index=False)
    t = pd.DataFrame({"stream": stream, "feature_id": table["name"].iloc[: len(picked)],
                      "name": picked["name"].to_numpy(), "adduct": picked["adduct"].to_numpy()})
    truth.append(t)
pd.concat(truth).to_csv(out / "synthetic_truth.csv", index=False)
print("written", out)
