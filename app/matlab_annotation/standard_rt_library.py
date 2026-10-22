"""Research-only RT priors for explicitly declared chemical standards.

Offline training excludes the evaluated acquisition group. Runtime alignment
uses OTHER automatically detected standards, not a curated workbook or test RT.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .retention_atlas import _affine_alignment

FORMAT = "laser-declared-standard-rt-v1"


def _standard_warp(predicted, observed):
    # A broad RT span is needed to estimate a slope, NOT a simple time offset.
    # Fixing the slope avoids an unidentified/stretching fit in clustered NP
    # standards; the same residual checks still govern runtime acceptance.
    if np.ptp(predicted) < 120:
        return 1.0, float(
            np.clip(np.median(np.asarray(observed) - predicted), -90.0, 90.0)
        )
    return _affine_alignment(predicted, observed)


def ion_key(value):
    value = re.sub(r"\]\((\d+)([+-])\)$", r"]\1\2", str(value).strip())
    return {"[M+HCOO]-": "[M+CHO2]-"}.get(value, value)


def nominal_standard_name(value):
    """Match a DECLARED spike's sum composition, retaining report labels separately.

    This is NOT endogenous isomer equivalence and is never used by benchmark
    comparison. Phase-specific positional reporting still needs strong training
    consensus and an independent RT match before it can be emitted.
    """
    return re.sub(r"^[12]-acyl\s*", "", str(value).strip())


def standard_report_name(value):
    return re.sub(r"^([12]-acyl)\s+(LPC|LPE)", r"\1\2", str(value).strip())


def fit_standard_retention_library(observations, *, minimum_cohorts=3):
    required = {"cohort", "stream", "standard_compound_id", "adduct", "rt_seconds"}
    if not required.issubset(observations) or minimum_cohorts < 3:
        raise ValueError(
            "Standard RT training requires explicit groups/identities and >=3 groups"
        )
    raw = observations.copy()
    if "dataset" not in raw:
        raw["dataset"] = raw.cohort
    raw["adduct"] = raw.adduct.map(ion_key)
    raw["rt_seconds"] = pd.to_numeric(raw.rt_seconds, errors="coerce")
    if not np.isfinite(raw.rt_seconds).all() or not raw.rt_seconds.gt(0).all():
        raise ValueError("Standard RT observations must be finite positive seconds")
    # Coeluting adducts are not independent calibration points.
    data = raw.groupby(
        ["dataset", "cohort", "stream", "standard_compound_id"], as_index=False
    ).rt_seconds.median()
    data["aligned_rt"] = data.rt_seconds
    for _ in range(3):
        balanced = data.groupby(
            ["cohort", "stream", "standard_compound_id"], as_index=False
        ).aligned_rt.median()
        centers = balanced.groupby(
            ["stream", "standard_compound_id"]
        ).aligned_rt.median()
        for (_dataset, stream), group in data.groupby(
            ["dataset", "stream"], sort=False
        ):
            p = np.asarray(
                [centers[(stream, s)] for s in group.standard_compound_id], float
            )
            if len(p) < 4:
                continue
            slope, intercept = _standard_warp(p, group.rt_seconds.to_numpy(float))
            data.loc[group.index, "aligned_rt"] = (group.rt_seconds - intercept) / slope
    models = {}
    data = data.groupby(
        ["cohort", "stream", "standard_compound_id"], as_index=False
    ).aligned_rt.median()
    for (stream, std), group in data.groupby(
        ["stream", "standard_compound_id"], sort=True
    ):
        n = int(group.cohort.nunique())
        if n < minimum_cohorts:
            continue
        rt = float(group.aligned_rt.median())
        support = (
            raw[raw.stream.eq(stream) & raw.standard_compound_id.eq(std)]
            .groupby("adduct")
            .cohort.nunique()
        )
        models[stream + "|" + std] = dict(
            rt_seconds=rt,
            training_cohorts=n,
            training_MAD_seconds=float(np.median(np.abs(group.aligned_rt - rt))),
            adduct_support={str(k): int(v) for k, v in support.items()},
        )
        if {"reporting_name", "reporting_class"}.issubset(raw):
            subset = raw[
                raw.stream.eq(stream) & raw.standard_compound_id.eq(std)
            ].copy()
            subset["reporting_name"] = subset.reporting_name.map(standard_report_name)
            votes = subset[
                ["cohort", "reporting_name", "reporting_class"]
            ].drop_duplicates()
            unique = votes.groupby("cohort").size().eq(1)
            votes = votes[votes.cohort.isin(unique[unique].index)]
            counts = votes.groupby(
                ["reporting_name", "reporting_class"]
            ).cohort.nunique()
            if len(counts):
                name, cls = counts.idxmax()
                count = int(counts.max())
                if count >= 3 and count / n >= 0.9:
                    models[stream + "|" + std].update(
                        reporting_name=name,
                        reporting_class=cls,
                        reporting_support_cohorts=count,
                        reporting_fraction=count / n,
                    )
    return dict(
        format=FORMAT,
        models=models,
        minimum_cohorts=minimum_cohorts,
        training_cohorts=int(data.cohort.nunique()),
    )


def predict_standard_retention(targets, automatic, library, *, stream):
    """Predict RT from a frozen training library and other declared standards.

    Four distinct OTHER standards must support a robust run alignment, with
    median residual <=3s and p90 <=6s. A >=120s span allows a fitted slope;
    otherwise only a fixed-slope time offset is estimated. Target and companion
    adducts never calibrate themselves. New adduct channels require >=3 training
    groups and >=50% prevalence among groups curating this chemical standard.
    Unavailable/unstable predictions stay unknown, never zero.
    """
    if library.get("format") != FORMAT:
        raise ValueError("Unsupported standard RT library")
    required = {"name", "adduct", "ref_mz", "standard_compound_id"}
    if not required.issubset(targets) or not targets.index.is_unique:
        raise ValueError(
            "Standard targets need unique indices and chemical identities/masses"
        )
    output = pd.DataFrame(index=targets.index)
    output["standard_rt_seconds"] = np.nan
    output["standard_alignment_anchors"] = 0
    output["standard_alignment_median_seconds"] = np.nan
    output["standard_training_cohorts"] = 0
    output["standard_adduct_supported"] = False
    output["standard_report_name"] = ""
    output["standard_report_class"] = ""
    model = {s: m for s, m in library["models"].items() if s.startswith(stream + "|")}
    catalog = targets.copy()
    catalog["adduct_key"] = catalog.adduct.map(ion_key)
    catalog["standard_lookup_name"] = catalog.name.map(nominal_standard_name)
    for index, row in catalog.iterrows():
        m = model.get(stream + "|" + row.standard_compound_id)
        if m is None:
            continue
        output.loc[index, "standard_training_cohorts"] = m["training_cohorts"]
        n = m["adduct_support"].get(row.adduct_key, 0)
        output.loc[index, "standard_adduct_supported"] = (
            n >= 3 and n / m["training_cohorts"] >= 0.5
        )
        output.loc[index, "standard_report_name"] = m.get("reporting_name", "")
        output.loc[index, "standard_report_class"] = m.get("reporting_class", "")
    needed = {"name", "adduct", "feat_mz", "feat_rt"}
    if automatic.empty or not needed.issubset(automatic):
        return output
    known = automatic.copy()
    standard = (
        known.get("is_is", pd.Series(False, index=known.index))
        .fillna(False)
        .astype(bool)
    )
    standard |= known.name.astype(str).str.endswith(" (IS)") | known.name.astype(
        str
    ).str.startswith("IS: ")
    known = known[standard].copy()
    known["adduct_key"] = known.adduct.map(ion_key)
    known["standard_lookup_name"] = known.name.map(nominal_standard_name)
    known = known.merge(
        catalog[
            ["standard_lookup_name", "adduct_key", "ref_mz", "standard_compound_id"]
        ],
        on=["standard_lookup_name", "adduct_key"],
        how="inner",
        suffixes=("_old", ""),
        validate="many_to_many",
    )
    if known.empty:
        return output
    known["mass_residual"] = (
        (pd.to_numeric(known.feat_mz, errors="coerce") - known.ref_mz)
        / known.ref_mz
        * 1e6
    )
    known["library_rt"] = [
        model.get(stream + "|" + s, {}).get("rt_seconds", np.nan)
        for s in known.standard_compound_id
    ]
    known = known[
        known.mass_residual.abs().lt(3)
        & np.isfinite(known.library_rt)
        & pd.to_numeric(known.feat_rt, errors="coerce").gt(0)
    ]
    if "feat_intensity" in known:
        known = known.sort_values("feat_intensity", ascending=False, kind="stable")
    known = known.drop_duplicates("standard_compound_id")
    for std, indices in catalog.groupby(
        "standard_compound_id", sort=False
    ).groups.items():
        m = model.get(stream + "|" + std)
        if m is None or m["training_MAD_seconds"] > 5:
            continue
        peers = known[known.standard_compound_id.ne(std)]
        p = peers.library_rt.to_numpy(float)
        rt = pd.to_numeric(peers.feat_rt, errors="coerce").to_numpy(float) * 60
        if len(p) < 4:
            continue
        slope, intercept = _standard_warp(p, rt)
        residual = rt - (slope * p + intercept)
        if np.median(np.abs(residual)) > 3 or np.quantile(np.abs(residual), 0.9) > 6:
            continue
        expected = float(m["rt_seconds"])
        if not p.min() - 60 <= expected <= p.max() + 60:
            continue
        predicted = expected * slope + intercept
        order = np.argsort(np.abs(p - expected), kind="stable")[:5]
        nearby = order[np.abs(p[order] - expected) <= 180]
        if len(nearby) >= 3:
            correction = float(np.median(residual[nearby]))
            mad = float(np.median(np.abs(residual[nearby] - correction)))
            if abs(correction) <= 15 and mad <= 3:
                predicted += correction
        if predicted <= 0:
            continue
        output.loc[indices, "standard_rt_seconds"] = predicted
        output.loc[indices, "standard_alignment_anchors"] = len(p)
        output.loc[indices, "standard_alignment_median_seconds"] = float(
            np.median(np.abs(residual))
        )
    return output
