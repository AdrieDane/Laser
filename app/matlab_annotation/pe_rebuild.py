"""Reconsider raw RP-positive PE peaks using independent, cross-mode evidence.

Candidate discovery is deliberately separate from the old fitted-grid filters.
The scorer learns curator agreement, not structural certainty or analytical FDR.
No curated data, sample labels, dataset names or feature IDs enter its features.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from annotator.composition_ranges import (
    CompositionRangeSettings,
    DEFAULT_COMPOSITION_RANGE_SETTINGS,
)
from resource_paths import REFERENCE_DATA_DIRECTORY
from .confidence import _protected_rows


@dataclass(frozen=True)
class PERebuildSettings:
    enabled: bool = True
    score_threshold: float = 0.10
    feature_margin: float = 0.03
    ppm_tolerance: float = 3.0
    search_seconds: float = 40.0
    model_path: Path = REFERENCE_DATA_DIRECTORY / "standard_pe_rebuild.json"


def _validate(s: PERebuildSettings) -> None:
    if not 0 <= s.score_threshold <= 1 or not 0 <= s.feature_margin <= 1:
        raise ValueError("PE selection score and margin must be in [0, 1]")
    if not np.isfinite(s.ppm_tolerance) or not 0 < s.ppm_tolerance <= 3:
        raise ValueError("PE corrected mass tolerance must be in (0, 3] ppm")
    if not np.isfinite(s.search_seconds) or not 0 < s.search_seconds <= 60:
        raise ValueError("PE discovery window must be in (0, 60] seconds")


def _pe_keys(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series("", index=frame.index, dtype=object)
    # A theoretical MATLAB code can be absent for a regenerated node, or
    # represented as float after concatenation. Identity does not depend on it.
    return (
        frame[["feat_feature_id", "name", "adduct"]]
        .fillna("")
        .astype(str)
        .agg("\x1f".join, axis=1)
    )


def _mass_center(annotations: pd.DataFrame) -> tuple[float, int]:
    if annotations.empty or not {"class", "ppm_error", "rt_error"}.issubset(
        annotations
    ):
        return 0.0, 0
    strong = annotations[
        annotations["class"].ne("PE")
        & annotations.ppm_error.abs().lt(3)
        & annotations.rt_error.abs().mul(60).lt(5)
    ].dropna(subset=["ppm_error"])
    # A target PE can never calibrate its own mass. Require many independently
    # accepted peaks; this is a bounded residual offset, not wider mass tolerance.
    if len(strong) < 20:
        return 0.0, len(strong)
    return float(strong.ppm_error.median()), len(strong)


def pe_search_reference(
    reference: pd.DataFrame,
    ranges: CompositionRangeSettings,
    *,
    adduct: str = "[M+H]+",
) -> pd.DataFrame:
    """Fill sparse RTDB nodes locally without consulting observed target peaks.

    Newly enumerated compositions have no MATLAB code: inheriting another
    species' dense code would give the review grid an incorrect identity.
    User carbon/DB limits always apply, including disabled-range behavior.
    """
    required = {"class", "adduct", "name", "carbon", "db", "ref_mz", "ref_rt"}
    if reference.empty or not required.issubset(reference):
        return reference.iloc[:0].copy()
    refs = reference[
        reference["class"].eq("PE")
        & reference.adduct.eq(adduct)
        & ~reference["name"].astype(str).str.endswith(" (IS)")
    ].drop_duplicates(["carbon", "db"])
    valid = refs.dropna(subset=["ref_rt", "ref_mz", "carbon", "db"])
    if len(valid) < 6:
        return refs.iloc[:0].copy()
    center = valid.iloc[(valid.carbon.sub(38).abs() + valid.db.sub(2).abs()).argmin()]
    lookup = {(int(r["carbon"]), int(r["db"])): r for r in refs.to_dict("records")}
    rows = []
    # Safe bounded discovery domain; narrower user ranges are respected. These
    # are method search limits, not a universal claim about possible PE biology.
    for carbon in range(26, 53):
        for db in range(min(15, carbon // 2 - 2) + 1):
            if not ranges.decide("PE", carbon, db).allowed:
                continue
            original = lookup.get((carbon, db))
            row = dict(center) if original is None else dict(original)
            if original is None or not np.isfinite(original["ref_rt"]):
                near = (
                    valid.assign(
                        distance=(valid.carbon - carbon).abs() / 2
                        + (valid.db - db).abs()
                    )
                    .sort_values("distance", kind="stable")
                    .head(12)
                )
                c, d = near.carbon.to_numpy() - carbon, near.db.to_numpy() - db
                design = np.column_stack(
                    [np.ones(len(near)), c, d, c * c, d * d, c * d]
                )
                rt = float(
                    np.linalg.lstsq(design, near.ref_rt.to_numpy(), rcond=None)[0][0]
                )
            else:
                rt = float(original["ref_rt"])
            if original is None:
                for col in ("matlab_code", "matlab_compound"):
                    row[col] = np.nan
            row.update(
                name=f"PE({carbon}:{db})",
                carbon=carbon,
                db=db,
                ref_mz=float(center.ref_mz)
                + (carbon - center.carbon) * 14.015650064
                - (db - center.db) * 2.015650064,
                ref_rt=rt,
                pe_reference_extrapolated=original is None
                or not np.isfinite(original["ref_rt"]),
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _mass_order(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    masses = pd.to_numeric(features.mz, errors="coerce").to_numpy(float)
    valid = np.flatnonzero(np.isfinite(masses) & (masses > 0))
    order = valid[np.argsort(masses[valid], kind="stable")]
    return masses[order], order


def _join_raw_pe(
    refs: pd.DataFrame,
    features: pd.DataFrame,
    negative_features: pd.DataFrame,
    negative_annotations: pd.DataFrame,
    mass_center: float,
    negative_center: float,
    settings: PERebuildSettings,
) -> pd.DataFrame:
    masses, order = _mass_order(features)
    neg_masses, neg_order = (
        _mass_order(negative_features)
        if not negative_features.empty
        else (np.array([]), np.array([], dtype=int))
    )
    neg = negative_annotations
    target_adduct = str(refs.adduct.iloc[0])
    counterpart_adduct = "[M+H]+" if target_adduct == "[M-H]-" else "[M-H]-"
    if not neg.empty:
        neg = neg[
            neg["class"].eq("PE")
            & neg.adduct.eq(counterpart_adduct)
            & ~neg["name"].astype(str).str.endswith(" (IS)")
        ].drop_duplicates("name", keep=False)
    neg_lookup = {r.name: r for r in neg.itertuples(index=False)}
    rows = []
    for target in refs.to_dict("records"):
        mz, rt = float(target["ref_mz"]), float(target["ref_rt"])
        if not np.isfinite(rt) or not np.isfinite(mz) or mz <= 0:
            continue
        lo = np.searchsorted(
            masses, mz * (1 + (mass_center - settings.ppm_tolerance) * 1e-6)
        )
        hi = np.searchsorted(
            masses,
            mz * (1 + (mass_center + settings.ppm_tolerance) * 1e-6),
            side="right",
        )
        # Catalogue [M+H]+ minus [M-H]-; both singly charged species.
        nmz = mz + (2.014552 if target_adduct == "[M-H]-" else -2.014552)
        nl = np.searchsorted(neg_masses, nmz * (1 + (negative_center - 3) * 1e-6))
        nh = np.searchsorted(
            neg_masses, nmz * (1 + (negative_center + 3) * 1e-6), side="right"
        )
        negatives = negative_features.iloc[neg_order[nl:nh]]
        for position in order[lo:hi]:
            feature = features.iloc[position]
            observed_rt = float(feature.rt)
            ppm = (float(feature.mz) - mz) / mz * 1e6
            if (
                not np.isfinite(observed_rt)
                or abs(observed_rt - rt) * 60 > settings.search_seconds
                or abs(ppm - mass_center) >= settings.ppm_tolerance
                or not np.isfinite(float(feature.intensity))
                or float(feature.intensity) <= 0
            ):
                continue
            row = dict(target)
            row.update({f"feat_{col}": value for col, value in feature.items()})
            row.update(
                feat_feature_id=str(feature.feature_id),
                rt_expected=rt,
                rt_error=observed_rt - rt,
                ppm_error=ppm,
                pe_mass_center_ppm=mass_center,
                pe_corrected_mass_error_ppm=ppm - mass_center,
                isotopologue="",
                is_is=False,
                display_name=target["name"],
                automatic_accepted=False,
                automatic_candidate=False,
                pe_negative_rt=np.nan,
                pe_negative_intensity=np.nan,
            )
            n = neg_lookup.get(target["name"])
            if n is not None:
                row.update(
                    pe_negative_rt=float(n.feat_rt),
                    pe_negative_intensity=float(n.feat_intensity),
                )
            for window in (2, 5, 10, 20):
                near = (
                    negatives[negatives.rt.sub(observed_rt).abs().mul(60).le(window)]
                    if not negatives.empty
                    else negatives
                )
                best = (
                    near.sort_values("intensity", ascending=False, kind="stable").iloc[
                        0
                    ]
                    if not near.empty
                    else None
                )
                row[f"pe_negative_raw_intensity_{window}s"] = (
                    float(best.intensity) if best is not None else 0.0
                )
                row[f"pe_negative_raw_rt_error_{window}s"] = (
                    (observed_rt - float(best.rt)) * 60 if best is not None else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def prepare_pe_rebuild_candidates(
    annotations: pd.DataFrame,
    reference: pd.DataFrame,
    features: pd.DataFrame,
    negative_annotations: pd.DataFrame,
    negative_features: pd.DataFrame,
    *,
    settings: PERebuildSettings = PERebuildSettings(),
    ranges: CompositionRangeSettings = DEFAULT_COMPOSITION_RANGE_SETTINGS,
    target_adduct: str = "[M+H]+",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _validate(settings)
    center, count = _mass_center(annotations)
    neg_center, neg_count = _mass_center(negative_annotations)
    context = dict(
        pe_mass_center_ppm=center,
        pe_mass_support=count,
        pe_negative_mass_center_ppm=neg_center,
        pe_negative_mass_support=neg_count,
    )
    if target_adduct not in {"[M+H]+", "[M-H]-"}:
        raise ValueError(
            "PE rebuild supports singly charged protonated/deprotonated ions only"
        )
    refs = pe_search_reference(reference, ranges, adduct=target_adduct)
    if refs.empty or features.empty:
        return pd.DataFrame(), pd.DataFrame(), context
    pool = _join_raw_pe(
        refs,
        features,
        negative_features,
        negative_annotations,
        center,
        neg_center,
        settings,
    )
    if pool.empty:
        return pool, pd.DataFrame(), context
    pool["pe_local_rt_error_seconds"] = np.nan
    pool["pe_support_distance"] = np.nan
    pool["pe_support_count"] = 0
    pool["pe_support_compositions"] = ""
    anchors = (
        pool[
            pool.pe_corrected_mass_error_ppm.abs().lt(1.5)
            & pool.rt_error.abs().mul(60).lt(20)
        ]
        .sort_values("feat_intensity", ascending=False, kind="stable")
        .drop_duplicates("name")
        .head(30)
        .copy()
    )
    for name, group in pool.groupby("name", sort=False):
        carbon, db = group.carbon.iloc[0], group.db.iloc[0]
        support = anchors[anchors["name"].ne(name)].copy()
        support["distance"] = np.sqrt(
            ((support.carbon - carbon) / 2) ** 2 + (support.db - db) ** 2
        )
        support = support.sort_values("distance", kind="stable").head(12)
        if len(support) < 5:
            continue
        design = np.column_stack(
            [np.ones(len(support)), support.carbon - carbon, support.db - db]
        )
        weights = np.exp(-support.distance.to_numpy() / 3)
        shift = np.linalg.solve(
            (design.T * weights) @ design + np.diag([0.01, 0.2, 0.2]),
            (design.T * weights) @ (support.rt_error.to_numpy() * 60),
        )[0]
        pool.loc[group.index, "pe_local_rt_error_seconds"] = group.rt_error * 60 - shift
        pool.loc[group.index, "pe_support_distance"] = float(support.distance.mean())
        pool.loc[group.index, "pe_support_count"] = len(support)
        pool.loc[group.index, "pe_support_compositions"] = ";".join(support["name"])
    return pool, anchors, context


def pe_rebuild_evidence(pool: pd.DataFrame) -> pd.DataFrame:
    """Observable evidence only, normalized inside one stream, never by truth."""
    x = pool
    out = pd.DataFrame(index=x.index)
    out["mass_error"] = x.pe_corrected_mass_error_ppm.abs()
    out["carbon"], out["db"] = x.carbon, x.db
    out["support_distance"] = x.pe_support_distance
    out["rt_prior_abs"] = x.rt_error.abs() * 60
    out["local_rt_abs"] = x.pe_local_rt_error_seconds.abs()
    out["neg_accepted_abs"] = (x.feat_rt - x.pe_negative_rt).abs() * 60
    li = np.log1p(x.feat_intensity.clip(lower=0))
    out["log_intensity"] = li
    out["relative_intensity"] = li - li.max()
    out["within_identity_intensity"] = li - li.groupby(x["name"]).transform("max")
    for window in (10, 20):
        top = (
            li.where(x.rt_error.abs().mul(60).le(window))
            .groupby(x["name"])
            .transform("max")
        )
        out[f"within_window_intensity{window}"] = li - top
    out["odd_carbon"] = x.carbon.mod(2)
    out["db_fraction"] = x.db / x.carbon
    ni = np.log1p(x.pe_negative_intensity)
    out["accepted_negative_relative_intensity"] = ni - ni.max()
    for window in (2, 5, 10, 20):
        raw = x[f"pe_negative_raw_intensity_{window}s"]
        ni = np.log1p(raw)
        out[f"neg_i{window}"] = ni - ni.max()
        out[f"neg_e{window}"] = x[f"pe_negative_raw_rt_error_{window}s"].abs()
        out[f"neg_present{window}"] = raw.gt(0).astype(float)
        out[f"neg_log_i{window}"] = ni
        ratio = ni - li
        out[f"neg_ratio{window}"] = ratio - ratio.where(raw.gt(0)).median()
    return out.replace([np.inf, -np.inf], np.nan).fillna(-50.0)


def score_pe_rebuild(evidence: pd.DataFrame, payload: dict) -> np.ndarray:
    """Portable depth-limited tree inference; sklearn is calibration-only."""
    if payload.get("format") != "laser-pe-rebuild-v1":
        raise ValueError("Unsupported PE rebuild model")
    x = evidence[payload["features"]].to_numpy(dtype=np.float32)
    score = np.full(len(x), float(payload["intercept"]))
    for tree in payload["trees"]:
        left, right = np.array(tree["left"]), np.array(tree["right"])
        feat, threshold = np.array(tree["feature"]), np.array(tree["threshold"])
        values = np.array(tree["value"])
        nodes = np.zeros(len(x), dtype=int)
        for _ in range(5):
            active = np.flatnonzero(left[nodes] >= 0)
            if not len(active):
                break
            n = nodes[active]
            nodes[active] = np.where(
                x[active, feat[n]] <= threshold[n], left[n], right[n]
            )
        if (left[nodes] >= 0).any():
            raise ValueError("PE rebuild model exceeds supported tree depth")
        score += float(payload["learning_rate"]) * values[nodes]
    if not np.isfinite(score).all():
        raise ValueError("PE rebuild model produced non-finite evidence scores")
    return 1 / (1 + np.exp(-np.clip(score, -50, 50)))


def complete_pe_stream(
    annotations: pd.DataFrame,
    all_pairs: pd.DataFrame,
    reference: pd.DataFrame,
    features: pd.DataFrame,
    negative_annotations: pd.DataFrame,
    negative_features: pd.DataFrame,
    *,
    stream: str = "RP_Pos",
    settings: PERebuildSettings = PERebuildSettings(),
    ranges: CompositionRangeSettings = DEFAULT_COMPOSITION_RANGE_SETTINGS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate(settings)
    audit = dict(
        stream=stream,
        method="pe_raw_evidence_rebuild",
        enabled=settings.enabled,
        baseline_annotations=len(annotations),
        added_pe=0,
    )
    empty = pd.DataFrame()
    if stream not in {"RP_Pos", "RP_Neg", "NP_Pos"} or not settings.enabled:
        audit["status"] = (
            "not_applicable"
            if stream not in {"RP_Pos", "RP_Neg", "NP_Pos"}
            else "disabled"
        )
        return annotations.copy(), all_pairs.copy(), pd.DataFrame([audit]), empty, empty
    try:
        payload = json.loads(settings.model_path.read_text())
        if payload.get("format") != "laser-pe-rebuild-v1":
            raise ValueError("Unsupported PE rebuild model")
        if payload.get("stream", "RP_Pos") != stream:
            raise ValueError("PE model does not match the requested measurement stream")
    except (OSError, ValueError) as error:
        audit.update(status="model_unavailable", detail=str(error))
        return annotations.copy(), all_pairs.copy(), pd.DataFrame([audit]), empty, empty
    pool, anchors, context = prepare_pe_rebuild_candidates(
        annotations,
        reference,
        features,
        negative_annotations,
        negative_features,
        settings=settings,
        ranges=ranges,
        target_adduct="[M-H]-" if stream.endswith("Neg") else "[M+H]+",
    )
    audit.update(context)
    if pool.empty:
        audit["status"] = "no_pe_candidates"
        return (
            annotations.copy(),
            all_pairs.copy(),
            pd.DataFrame([audit]),
            anchors,
            pool,
        )
    pool["pe_completion_score"] = score_pe_rebuild(pe_rebuild_evidence(pool), payload)
    reserved = (
        set(annotations.feat_feature_id.astype(str)) if not annotations.empty else set()
    )
    declared = _protected_rows(all_pairs)
    reserved |= (
        set(declared.feat_feature_id.astype(str)) if not declared.empty else set()
    )
    occupied_names = (
        set(annotations.loc[annotations["class"].eq("PE"), "name"].astype(str))
        if not annotations.empty
        else set()
    )
    pool["pe_decision"] = "below_pe_score"
    pool["pe_final_selected"] = False
    # Existing calls are never reassigned by this additive PE pass. Their
    # features may provide evidence, but cannot be new annotation targets.
    free = ~pool.feat_feature_id.isin(reserved) & ~pool["name"].isin(occupied_names)
    pool.loc[~free, "pe_decision"] = "existing_annotation_or_standard"
    eligible = pool[free & pool.pe_support_count.ge(5)].copy()
    selected = []
    for _, group in eligible.groupby("name", sort=True):
        ranked = group.sort_values(
            ["pe_completion_score", "feat_intensity", "feat_feature_id"],
            ascending=[False, False, True],
            kind="stable",
        )
        best = ranked.iloc[0]
        if best.pe_completion_score < settings.score_threshold:
            continue
        if (
            len(ranked) > 1
            and best.pe_completion_score - ranked.pe_completion_score.iloc[1]
            < settings.feature_margin
        ):
            pool.loc[ranked.index, "pe_decision"] = "ambiguous_peak_choice"
            continue
        selected.append(best.name)
    chosen = (
        pool.loc[selected]
        .sort_values("pe_completion_score", ascending=False, kind="stable")
        .drop_duplicates("feat_feature_id")
        .copy()
    )
    pool.loc[chosen.index, "pe_final_selected"] = True
    pool.loc[chosen.index, "pe_decision"] = "accepted"
    chosen["pe_final_selected"] = True
    chosen["pe_decision"] = "accepted"
    chosen["automatic_accepted"], chosen["automatic_candidate"] = True, True
    chosen["auto_annotation_status"] = "pe_raw_evidence_supported"
    chosen["status"] = "pe_raw_evidence_supported"
    chosen["rt_evidence_mode"] = "pe_rebuilt_local_and_cross_polarity"
    chosen["standard_stream"], chosen["chromatography"] = stream, stream.split("_")[0]
    chosen["pe_completion_supported"] = True
    chosen["rank_for_feature"] = 1
    # Retain raw ppm/RT errors for truthful display; the PE-specific evidence
    # is separate from the previous fitted-grid coherence and confidence.
    chosen["grid_coherence_assessed"] = False
    chosen["grid_coherence_pass"] = True
    chosen["grid_coherence_outlier"] = False
    chosen["post_regression_count_retained"] = False
    accepted = (
        pd.concat([annotations, chosen], ignore_index=True, sort=False)
        if not chosen.empty
        else annotations.copy()
    )
    keys = set(_pe_keys(chosen))
    # Replace only identical review rows, keeping all unrelated candidates.
    candidates = (
        pd.concat(
            [all_pairs[~_pe_keys(all_pairs).isin(keys)], chosen],
            ignore_index=True,
            sort=False,
        )
        if not chosen.empty
        else all_pairs.copy()
    )
    pool["automatic_accepted"] = pool.pe_final_selected
    pool["automatic_candidate"] = pool.pe_final_selected
    audit.update(
        status="completed",
        eligible_pe=len(pool),
        strong_anchors=len(anchors),
        added_pe=len(chosen),
        final_annotations=len(accepted),
        score_threshold=settings.score_threshold,
        feature_margin=settings.feature_margin,
        model_training_datasets=";".join(payload.get("training_datasets", [])),
    )
    return accepted, candidates, pd.DataFrame([audit]), anchors, pool


def pe_model_path_for_stream(stream: str) -> Path:
    names = {
        "RP_Pos": "standard_pe_rebuild.json",
        "RP_Neg": "standard_pe_rebuild_rp_negative.json",
        "NP_Pos": "standard_pe_rebuild_np_positive.json",
    }
    return REFERENCE_DATA_DIRECTORY / names[stream]


def rebuild_rp_positive_pe(
    annotations: pd.DataFrame,
    all_pairs: pd.DataFrame,
    reference: pd.DataFrame,
    features: pd.DataFrame,
    negative_annotations: pd.DataFrame,
    negative_features: pd.DataFrame,
    *,
    stream: str = "RP_Pos",
    settings: PERebuildSettings = PERebuildSettings(),
    ranges: CompositionRangeSettings = DEFAULT_COMPOSITION_RANGE_SETTINGS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compatibility entry point; the explicit single-stream API stays a no-op elsewhere."""
    if stream != "RP_Pos":
        return (
            annotations.copy(),
            all_pairs.copy(),
            pd.DataFrame(
                [
                    dict(
                        stream=stream,
                        status="not_applicable",
                        added_pe=0,
                    )
                ]
            ),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    return complete_pe_stream(
        annotations,
        all_pairs,
        reference,
        features,
        negative_annotations,
        negative_features,
        stream=stream,
        settings=settings,
        ranges=ranges,
    )
