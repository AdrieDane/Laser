"""Research-only within-run context; no matrix names or curator labels.

Context describes OTHER strong automatic compositions and declared-standard
response. It is not a hard carbon whitelist or proof of a lipid's identity.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class


def _root(cls):
    text = canonical_lipid_class(cls).split("[")[0]
    return re.sub(r"^[12]-acyl", "", text)


def run_context_evidence(pool, automatic):
    """Numerical, label-free context with missing evidence explicitly unknown."""
    required = {"class", "carbon", "db", "feat_intensity"}
    if not required.issubset(pool):
        raise ValueError("Context candidates require class/C/DB/intensity")
    output = pd.DataFrame(index=pool.index)
    columns = [
        "q_log_over_global_standard_response",
        "q_log_over_class_standard_response",
        "q_context_carbon_median",
        "q_context_carbon_p90",
        "q_context_db_median",
        "q_context_db_p90",
        "q_carbon_minus_context_median",
        "q_db_minus_context_median",
    ]
    for c in columns:
        output[c] = np.nan
    output["q_global_standard_calibrators"] = 0
    output["q_class_standard_calibrators"] = 0
    output["q_other_composition_anchors"] = 0
    if automatic.empty or not required.issubset(automatic):
        return output
    auto = automatic.copy()
    standard = (
        auto.get("is_is", pd.Series(False, index=auto.index)).fillna(False).astype(bool)
    )
    if "name" in auto:
        standard |= auto.name.astype(str).str.endswith(" (IS)") | auto.name.astype(
            str
        ).str.startswith("IS: ")
    auto["canonical_class"] = auto["class"].map(canonical_lipid_class)
    auto["class_root"] = auto["class"].map(_root)
    for c in ["carbon", "db", "feat_intensity"]:
        auto[c] = pd.to_numeric(auto[c], errors="coerce")
    # Only explicitly known, commensurate nM spike concentrations are used.
    # An absent/unknown unit cannot silently become a concentration estimate.
    if {"internal_standard_quantity", "internal_standard_unit"}.issubset(auto):
        quantity = pd.to_numeric(auto.internal_standard_quantity, errors="coerce")
        unit = auto.internal_standard_unit.astype(str).str.strip().str.casefold()
        known = (
            standard
            & quantity.gt(0)
            & auto.feat_intensity.gt(0)
            & unit.isin(["nm", "nmol/l"])
        )
        spikes = auto.loc[known].copy()
        spikes["log_response"] = np.log(spikes.feat_intensity / quantity[known])
        if "internal_standard_name" in spikes:
            spikes = spikes.drop_duplicates("internal_standard_name")
        intensity = np.log1p(
            pd.to_numeric(pool.feat_intensity, errors="coerce").clip(lower=0)
        )
        if len(spikes) >= 3:
            output["q_global_standard_calibrators"] = len(spikes)
            output["q_log_over_global_standard_response"] = intensity - float(
                spikes.log_response.median()
            )
        roots = pool["class"].map(_root)
        for root, g in spikes.groupby("class_root"):
            indices = pool.index[roots.eq(root)]
            output.loc[indices, "q_class_standard_calibrators"] = len(g)
            output.loc[indices, "q_log_over_class_standard_response"] = intensity.loc[
                indices
            ] - float(g.log_response.median())
    confidence = pd.to_numeric(
        auto.get("post_regression_confidence", pd.Series(np.nan, index=auto.index)),
        errors="coerce",
    )
    mass = pd.to_numeric(
        auto.get("ppm_error", pd.Series(np.nan, index=auto.index)), errors="coerce"
    )
    strong = auto[~standard & confidence.ge(0.9) & mass.abs().lt(1.5)].copy()
    strong = strong[np.isfinite(strong[["carbon", "db", "feat_intensity"]]).all(axis=1)]
    strong = strong.sort_values("feat_intensity", ascending=False, kind="stable")
    strong = strong.drop_duplicates(["canonical_class", "carbon", "db"])
    canonical = pool["class"].map(canonical_lipid_class)
    for cls, indices in canonical.groupby(canonical, sort=False).groups.items():
        own = strong[strong.canonical_class.eq(cls)].head(15)
        if own.empty:
            continue
        c, d = own.carbon.to_numpy(float), own.db.to_numpy(float)
        for row in pool.loc[indices, ["carbon", "db"]].drop_duplicates().itertuples():
            tc, td = float(row.carbon), float(row.db)
            keep = ~((c == tc) & (d == td))
            matching = pd.Index(indices)[
                pool.loc[indices, "carbon"].eq(tc) & pool.loc[indices, "db"].eq(td)
            ]
            n = int(keep.sum())
            output.loc[matching, "q_other_composition_anchors"] = n
            if n < 5:
                continue
            cm, dm = float(np.median(c[keep])), float(np.median(d[keep]))
            output.loc[matching, "q_context_carbon_median"] = cm
            output.loc[matching, "q_context_carbon_p90"] = float(
                np.quantile(c[keep], 0.9)
            )
            output.loc[matching, "q_context_db_median"] = dm
            output.loc[matching, "q_context_db_p90"] = float(np.quantile(d[keep], 0.9))
            output.loc[matching, "q_carbon_minus_context_median"] = tc - cm
            output.loc[matching, "q_db_minus_context_median"] = td - dm
    return output
