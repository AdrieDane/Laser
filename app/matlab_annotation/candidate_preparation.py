"""Reusable research preparation of001-007 candidates/evidence, not deployment.

All chemical/RT/prior models are explicit inputs; callers must enforce project
exclusion. Only baseline AUTOMATIC tables and original raw features are used.
No files, curation, fitted classifier or acceptance cutoff is read here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from annotator.peak_evidence import peak_evidence
from .chemical_targets import compile_additional_targets, _charge
from .class_candidates import (
    prepare_class_candidates,
    candidate_evidence,
    attach_composition_prior_evidence,
)
from .cross_mode_evidence import cross_mode_evidence
from .dual_candidates import combine_rt_sources
from .run_context_evidence import run_context_evidence


def _quality(pool, features, *, spreads):
    if pool.empty:
        return pool.copy()
    pool = pool.copy()
    chosen = features.set_index(features.feature_id.astype(str)).loc[
        pool.feat_feature_id.astype(str)
    ]
    columns = [c for c in features.attrs.get("sample_columns", ()) if c in features]
    if columns:
        matrix = chosen[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = np.isfinite(matrix) & (matrix >= 0)
        count = valid.sum(axis=1)
        pool["q_observed_detection_fraction"] = np.divide(
            ((matrix > 0) & valid).sum(axis=1),
            count,
            out=np.full(len(pool), np.nan),
            where=count > 0,
        )
        pool["q_log_observed_median"] = np.log1p(
            np.nanmedian(np.where(valid, matrix, np.nan), axis=1)
        )
    if spreads:
        for a, b, label, factor in (
            ("rtmax", "rtmin", "q_peakgroup_RT_spread_seconds", 60.0),
            ("mzmax", "mzmin", "q_peakgroup_mass_spread_ppm", 1e6),
        ):
            if a in chosen and b in chosen:
                span = (
                    pd.to_numeric(chosen[a], errors="coerce")
                    - pd.to_numeric(chosen[b], errors="coerce")
                ).to_numpy(float) * factor
                if "mass" in label:
                    span /= pool.feat_mz.to_numpy(float)
                pool[label] = span
    return pool


def prepare_candidate_pool(result, atlas, envelope):
    """Same strict3ppm/10s union+003 lattice targets as original007.

    Deliberately retains003's ORIGINAL embedded-reference standard-protection
    semantics for parity, not a silent new declared-catalogue policy. Standards
    are reserved/handled downstream by the unchanged005 adjudicator.
    """
    parts = []
    for stream, payload in result["streams"].items():
        for key in ("reference", "annotations", "all_pairs"):
            if {"training_positive", "training_label_known", "same_manual_peak"} & set(
                payload[key]
            ):
                raise ValueError("Supervised labels cannot enter physical preparation")
        auto, reference, features = (
            payload["annotations"],
            payload["reference"],
            payload["features"],
        )
        learned, _ = prepare_class_candidates(
            auto, reference, features, stream=stream, atlas=atlas
        )
        learned = _quality(learned, features, spreads=False)
        previous = combine_rt_sources(
            learned, payload["all_pairs"], auto, atlas, stream=stream
        )
        previous = _quality(previous, features, spreads=True)
        previous["q_chemical_target_generated"] = 0.0
        targets, ranges, _ = compile_additional_targets(
            reference, envelope, stream=stream
        )
        extra = pd.DataFrame()
        if len(targets):
            extra, _ = prepare_class_candidates(
                auto, targets, features, stream=stream, atlas=atlas, ranges=ranges
            )
            if len(extra):
                extra = _quality(extra, features, spreads=True)
                extra["q_chemical_target_generated"] = 1.0
                extra["q_atlas_source"] = 1.0
                extra["q_fitted_source"] = 0.0
                extra["q_previously_selected"] = 0.0
                extra["q_atlas_RT_error_seconds"] = (
                    extra.feat_rt * 60 - extra.atlas_rt_seconds
                )
                extra["q_absolute_atlas_RT_error_seconds"] = (
                    extra.q_atlas_RT_error_seconds.abs()
                )
        pool = (
            pd.concat([previous, extra], ignore_index=True)
            .drop_duplicates(["name", "adduct", "feat_feature_id"])
            .reset_index(drop=True)
        )
        parts.append(pool)
    return pd.concat(parts, ignore_index=True)


def enrich_candidate_pools(pools, result, priors, *, raw_quality_cache=None):
    """Same004 raw/context and007 OTHER-mode evidence; optional raw-only cache.

    Cache keys are (stream, absolutecharge), values unique original-feature-index
    q_ measurement tables. Missing features are MEASURED, never invented/zeroed.
    No cached prior/atlas/classifier/curation evidence may be supplied as raw
    quality. Returned model evidence contains no IDs, names or supervised labels.
    """
    cache = {} if raw_quality_cache is None else dict(raw_quality_cache)
    queries = pd.concat(
        [p[["stream", "feat_feature_id", "adduct"]] for p in pools.values()],
        ignore_index=True,
    ).drop_duplicates()
    for stream, group in queries.groupby("stream", sort=False):
        charges = group.adduct.map(_charge)
        for charge, indices in charges.groupby(charges).groups.items():
            key = (stream, int(charge))
            table = cache.get(key, pd.DataFrame())
            if len(table) and (
                not table.index.is_unique
                or any(not str(c).startswith("q_") for c in table)
            ):
                raise ValueError(
                    "Raw quality cache must contain unique original-feature q_ measurements only"
                )
            wanted = set(group.loc[indices, "feat_feature_id"].astype(str))
            missing = wanted - set(table.index.astype(str))
            if missing:
                measured = peak_evidence(
                    result["streams"][stream]["features"],
                    sorted(missing),
                    polarity=stream.split("_")[1],
                    charge=int(charge),
                ).set_index("feature_id")
                measured = measured[[c for c in measured if c.startswith("q_")]]
                table = pd.concat([table, measured]) if len(table) else measured
            cache[key] = table
    prepared = {}
    for model, physical in pools.items():
        parts, evidences = [], []
        for stream, pool in physical.groupby("stream", sort=False):
            pool = pool.copy()
            charges = pool.adduct.map(_charge)
            for charge, indices in charges.groupby(charges).groups.items():
                measured = cache[(stream, int(charge))].loc[
                    pool.loc[indices, "feat_feature_id"].astype(str)
                ]
                for c in measured:
                    pool.loc[indices, "q_raw_" + c.removeprefix("q_")] = measured[
                        c
                    ].to_numpy()
            context = run_context_evidence(
                pool, result["streams"][stream]["annotations"]
            )
            for c in context:
                pool[c] = context[c]
            parts.append(pool)
            evidences.append(
                attach_composition_prior_evidence(
                    candidate_evidence(pool), priors[model]
                )
            )
        prepared[model] = dict(
            pool=pd.concat(parts, ignore_index=True),
            evidence=pd.concat(evidences, ignore_index=True),
        )
    keys = ["stream", "class", "carbon", "db", "feat_feature_id", "feat_rt"]
    queries = (
        pd.concat([d["pool"][keys] for d in prepared.values()], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    automatic = pd.concat(
        [
            p["annotations"].copy().assign(stream=s)
            for s, p in result["streams"].items()
        ],
        ignore_index=True,
    )
    extra = cross_mode_evidence(
        queries, {s: p["features"] for s, p in result["streams"].items()}, automatic
    )
    extra.index = pd.MultiIndex.from_frame(queries[keys])
    assert extra.index.is_unique
    for data in prepared.values():
        pool, evidence = data["pool"], data["evidence"]
        aligned = extra.reindex(pd.MultiIndex.from_frame(pool[keys]))
        for c in extra:
            assert c not in pool and c not in evidence
            pool[c] = aligned[c].to_numpy()
            evidence[c] = aligned[c].to_numpy()
        assert not {
            "name",
            "feat_feature_id",
            "dataset",
            "cohort",
            "training_positive",
            "training_label_known",
        } & set(evidence)
    return prepared, cache
