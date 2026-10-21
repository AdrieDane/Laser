"""Experimental target-independent mass/RT calibration; no manual labels.

Research only: this tools/ module is not packaged with the application.
Neither default nor optional annotation workflows call it. Calibration quality is checked against
held-out automatic anchors; sparse or unsupported regions retain the prior
calibration. Original theoretical masses and measured coordinates are retained.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError

from annotator.peak_evidence import peak_evidence
from matlab_annotation.experimental_pe import _protected
from matlab_annotation.pe_rebuild import (
    _pe_keys,
    _mass_center,
    _mass_order,
    _validate,
    PERebuildSettings,
    pe_search_reference,
    pe_rebuild_evidence,
    score_pe_rebuild,
)
from annotator.composition_ranges import (
    CompositionRangeSettings,
    DEFAULT_COMPOSITION_RANGE_SETTINGS,
)
from resource_paths import REFERENCE_DATA_DIRECTORY


def robust_fit(x, y, weights, penalty, floor):
    """Ridge-stabilized Huber reweighting, deterministic and label-free."""
    x, y, weights = (
        np.asarray(x, float),
        np.asarray(y, float),
        np.asarray(weights, float),
    )
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(weights).all()
    ):
        raise ValueError("Calibration inputs must be finite")
    if len(y) < x.shape[1] + 2 or (weights <= 0).any():
        raise ValueError("Insufficient independent calibration observations")
    coef = np.zeros(x.shape[1])
    w = weights.copy()
    for _ in range(20):
        fitted = np.linalg.solve((x.T * w) @ x + np.diag(penalty), (x.T * w) @ y)
        residual = y - x @ fitted
        scale = max(floor, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        w = weights * np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-12))
        if np.max(np.abs(coef - fitted)) < 1e-7:
            return fitted
        coef = fitted
    return coef


def _mass_design(mz, rt):
    m, t = (np.asarray(mz) - 800.0) / 200.0, (np.asarray(rt) - 12.0) / 5.0
    return np.column_stack([np.ones(len(m)), m, t, m * m, m * t])


def fit_mass_calibration(annotations: pd.DataFrame, baseline_center: float):
    """Leave-class-out verification; the target PE family never supplies mass anchors."""
    required = {
        "class",
        "ppm_error",
        "rt_error",
        "feat_mz",
        "feat_rt",
        "feat_feature_id",
    }
    if not required.issubset(annotations):
        return None, pd.DataFrame(), dict(status="missing_mass_observations")
    ordinary = ~annotations["class"].astype(str).str.match(r"^PE(?:$|[- ])")
    valid = (
        ordinary
        & annotations.ppm_error.abs().lt(3)
        & annotations.rt_error.abs().mul(60).lt(5)
    )
    a = annotations[valid].drop_duplicates("feat_feature_id").copy()
    a = a.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["ppm_error", "feat_mz", "feat_rt", "class"]
    )
    audit = dict(mass_anchors=len(a), mass_classes=a["class"].nunique())
    if len(a) < 50 or a["class"].nunique() < 4:
        return None, a, dict(audit, status="insufficient_mass_anchors")
    x = _mass_design(a.feat_mz.to_numpy(), a.feat_rt.to_numpy())
    y = a.ppm_error.to_numpy()
    groups = a["class"].astype(str).to_numpy()
    w = 1 / a.groupby("class")["class"].transform("size").to_numpy()
    w = w / w.mean()
    pred = np.full(len(a), np.nan)
    penalty = [1e-5, 0.2, 0.2, 0.5, 0.5]
    for group in sorted(set(groups)):
        train = groups != group
        test = ~train
        if train.sum() < 30:
            continue
        pred[test] = x[test] @ robust_fit(x[train], y[train], w[train], penalty, 0.25)
    valid = np.isfinite(pred)
    old = np.abs(y[valid] - baseline_center)
    new = np.abs(y[valid] - pred[valid])
    if valid.sum() < 40:
        return None, a, dict(audit, status="insufficient_mass_CV")
    old_med, new_med = float(np.median(old)), float(np.median(new))
    old90, new90 = np.quantile(old, 0.9), np.quantile(new, 0.9)
    audit.update(
        mass_CV_median_before=old_med,
        mass_CV_median_after=new_med,
        mass_CV_p90_before=float(old90),
        mass_CV_p90_after=float(new90),
    )
    pass_fraction = np.mean(
        [
            np.median(np.abs(y[groups == g] - pred[groups == g]))
            <= np.median(np.abs(y[groups == g] - baseline_center)) + 0.1
            for g in set(groups)
        ]
    )
    accepted = (
        new_med <= 0.95 * old_med
        and new90 <= old90 + 0.1
        and new90 <= 1.5
        and pass_fraction >= 0.6
    )
    if not accepted:
        return None, a, dict(audit, status="mass_CV_not_better")
    coeff = robust_fit(x, y, w, penalty, 0.25)
    return (
        coeff,
        a,
        dict(audit, status="mass_CV_pass", mass_uncertainty_ppm=float(new90)),
    )


def mass_predictions(refs, coeff, anchors, baseline_center):
    fallback = np.full(len(refs), baseline_center)
    supported = np.zeros(len(refs), bool)
    if coeff is None:
        return fallback, supported
    predictions = _mass_design(refs.ref_mz.to_numpy(), refs.ref_rt.to_numpy()) @ coeff
    for i, row in enumerate(refs.itertuples()):
        nearby = anchors[
            anchors.feat_mz.sub(row.ref_mz).abs().le(150)
            & anchors.feat_rt.sub(row.ref_rt).abs().le(5)
        ]
        within = (
            anchors.feat_mz.min() <= row.ref_mz <= anchors.feat_mz.max()
            and anchors.feat_rt.min() <= row.ref_rt <= anchors.feat_rt.max()
        )
        if (
            within
            and len(nearby) >= 10
            and nearby["class"].nunique() >= 3
            and np.isfinite(predictions[i])
            and abs(predictions[i]) <= 3
            and abs(predictions[i] - baseline_center) <= 1.5
        ):
            fallback[i] = predictions[i]
            supported[i] = True
    return fallback, supported


def _rt_prediction(anchors, carbon, db, kind="actual"):
    other = anchors[~(anchors.carbon.eq(carbon) & anchors.db.eq(db))].copy()
    other["distance"] = np.sqrt(
        ((other.carbon - carbon) / 2) ** 2 + (other.db - db) ** 2
    )
    other = (
        other[other.distance.le(6)]
        .sort_values(["distance", "name"], kind="stable")
        .head(16)
    )
    if len(other) < 10:
        return None, other
    coords = other[["carbon", "db"]].to_numpy(float)
    try:
        if Delaunay(coords).find_simplex([[carbon, db]])[0] < 0:
            return None, other
    except QhullError:
        return None, other
    c = (other.carbon.to_numpy() - carbon) / 2
    d = other.db.to_numpy() - db
    x = np.column_stack([np.ones(len(other)), c, d, c * c, d * d, c * d])
    response = (
        other.feat_rt.to_numpy() * 60
        if kind == "actual"
        else other.rt_error.to_numpy() * 60
    )
    origin = float(np.median(response))
    coef = robust_fit(
        x,
        response - origin,
        np.exp(-other.distance.to_numpy() / 3),
        [0.001, 0.05, 0.05, 0.2, 0.2, 0.2],
        0.5,
    )
    value = float(coef[0]) + origin
    if kind == "actual":
        valid = (
            coef[1] > 0 and coef[2] < 0 and response.min() <= value <= response.max()
        )
    else:
        valid = abs(value) <= 20
    return (value if valid else None), other


def fit_rt_calibration(pool: pd.DataFrame, annotations: pd.DataFrame):
    expected = np.log((pool.carbon + 5) * 0.0107 / 0.9893)
    isotope = pool.q_isotope_upper_correlation.ge(0.8) & (
        pool.q_isotope_upper_log_ratio - expected
    ).abs().le(0.7)
    counterpart = (pool.feat_rt - pool.pe_negative_rt).abs().mul(60).le(5)
    accepted_keys = set(
        _pe_keys(annotations[annotations["class"].eq("PE") & ~_protected(annotations)])
    )
    a = pool[
        _pe_keys(pool).isin(accepted_keys)
        & pool.pe_corrected_mass_error_ppm.abs().lt(1.5)
        & pool.pe_local_rt_error_seconds.abs().le(10)
        & pool.q_profile_samples.ge(4)
        & (isotope | counterpart)
    ].copy()
    a = a.drop_duplicates(["carbon", "db"], keep=False)
    audit = dict(RT_anchors=len(a))
    if len(a) < 14:
        return None, dict(audit, status="insufficient_RT_anchors")
    choices = []
    for kind in ("actual", "residual"):
        old = []
        new = []
        for row in a.itertuples():
            prediction, _ = _rt_prediction(a, row.carbon, row.db, kind)
            if prediction is None:
                continue
            observed = row.feat_rt * 60 if kind == "actual" else row.rt_error * 60
            old.append(abs(row.pe_local_rt_error_seconds))
            new.append(abs(observed - prediction))
        if len(new) < 10:
            continue
        om, nm = float(np.median(old)), float(np.median(new))
        o90, n90 = float(np.quantile(old, 0.9)), float(np.quantile(new, 0.9))
        record = dict(
            RT_model=kind,
            RT_CV_targets=len(new),
            RT_CV_median_before=om,
            RT_CV_median_after=nm,
            RT_CV_p90_before=o90,
            RT_CV_p90_after=n90,
        )
        record["passed"] = nm <= 0.9 * om and n90 <= o90 + 0.5 and n90 <= 8
        choices.append(record)
    if not choices:
        return None, dict(audit, status="insufficient_RT_CV")
    valid = [v for v in choices if v["passed"]]
    best = min(
        valid or choices, key=lambda v: (v["RT_CV_median_after"], v["RT_CV_p90_after"])
    )
    audit.update(best)
    if not best["passed"]:
        return None, dict(audit, status="RT_CV_not_better")
    return a, dict(
        audit, status="RT_CV_pass", RT_uncertainty_seconds=best["RT_CV_p90_after"]
    )


def prepare_local_pe_pool(
    annotations, reference, features, counterpart_annotations, counterpart_features
):
    """Frozen single pilot: RP positive only, unchanged composition limits and gates."""
    old, _, context = prepare_pe_rebuild_candidates(
        annotations, reference, features, counterpart_annotations, counterpart_features
    )
    if old.empty:
        return old, dict(status="no_PE_candidates")
    q = peak_evidence(features, old.feat_feature_id, polarity="Pos")
    old = old.merge(
        q, left_on="feat_feature_id", right_on="feature_id", validate="many_to_one"
    )
    center = context["pe_mass_center_ppm"]
    coeff, ma, mass_audit = fit_mass_calibration(annotations, center)
    from annotator.composition_ranges import DEFAULT_COMPOSITION_RANGE_SETTINGS

    refs = pe_search_reference(reference, DEFAULT_COMPOSITION_RANGE_SETTINGS)
    offsets, supported = mass_predictions(refs, coeff, ma, center)
    mapping = dict(zip(refs["name"], offsets))
    support_map = dict(zip(refs["name"], supported))
    if coeff is None:
        pool = old.copy()
    else:
        pool, _, _ = prepare_pe_rebuild_candidates(
            annotations,
            reference,
            features,
            counterpart_annotations,
            counterpart_features,
            mass_centers=mapping,
        )
        q = peak_evidence(features, pool.feat_feature_id, polarity="Pos")
        pool = pool.merge(
            q, left_on="feat_feature_id", right_on="feature_id", validate="many_to_one"
        )
    pool["pe_calibration_mass_supported"] = pool["name"].map(support_map).fillna(False)
    pool["pe_precalibration_mass_error_ppm"] = pool.ppm_error - center
    lookup = pd.Series(old.pe_local_rt_error_seconds.to_numpy(), index=_pe_keys(old))
    pool["pe_precalibration_rt_error_seconds"] = _pe_keys(pool).map(lookup)
    rt_anchors, rt_audit = fit_rt_calibration(old, annotations)
    pool["pe_calibration_RT_supported"] = False
    pool["pe_calibration_RT_uncertainty_seconds"] = np.nan
    for _name, group in pool.groupby("name", sort=True):
        if rt_anchors is None:
            break
        shift, support = _rt_prediction(
            rt_anchors, group.carbon.iloc[0], group.db.iloc[0], rt_audit["RT_model"]
        )
        if shift is None:
            continue
        observed = (
            group.feat_rt * 60
            if rt_audit["RT_model"] == "actual"
            else group.rt_error * 60
        )
        pool.loc[group.index, "pe_local_rt_error_seconds"] = observed - shift
        pool.loc[group.index, "pe_support_count"] = len(support)
        pool.loc[group.index, "pe_support_compositions"] = ";".join(support["name"])
        pool.loc[group.index, "pe_calibration_RT_supported"] = True
        pool.loc[group.index, "pe_calibration_RT_uncertainty_seconds"] = rt_audit[
            "RT_uncertainty_seconds"
        ]
    audit = {
        **context,
        **{f"mass_{k}": v for k, v in mass_audit.items()},
        **{f"RT_{k}": v for k, v in rt_audit.items()},
    }
    return pool, audit


def refine_local_pe(
    annotations,
    all_pairs,
    reference,
    features,
    counterpart_annotations,
    counterpart_features,
):
    """Test recalibrated coordinates with the unchanged earlier PE scorer/margin."""
    pool, audit = prepare_local_pe_pool(
        annotations, reference, features, counterpart_annotations, counterpart_features
    )
    if pool.empty:
        return annotations.copy(), all_pairs.copy(), pool, pd.DataFrame([audit])
    model = json.loads(
        (Path(REFERENCE_DATA_DIRECTORY) / "standard_pe_rebuild.json").read_text()
    )
    pool["pe_completion_score"] = score_pe_rebuild(pe_rebuild_evidence(pool), model)
    mass_changed = (
        pool.pe_mass_center_ppm.sub(audit["pe_mass_center_ppm"]).abs().ge(0.15)
        & pool.pe_calibration_mass_supported
    )
    rt_changed = (
        pool.pe_local_rt_error_seconds.sub(pool.pe_precalibration_rt_error_seconds)
        .abs()
        .ge(1)
        & pool.pe_calibration_RT_supported
    )
    reserved = set(annotations.feat_feature_id.astype(str)) | set(
        all_pairs.loc[_protected(all_pairs), "feat_feature_id"].astype(str)
    )
    locked = set(annotations["name"].astype(str)) | set(
        all_pairs.loc[_protected(all_pairs), "name"].astype(str)
    )
    protected_names = set(
        annotations.loc[_protected(annotations), "name"].astype(str)
    ) | set(all_pairs.loc[_protected(all_pairs), "name"].astype(str))
    locked |= {name.removesuffix(" (IS)") for name in protected_names}
    eligible = (
        (mass_changed | rt_changed)
        & pool.pe_corrected_mass_error_ppm.abs().lt(3)
        & pool.pe_local_rt_error_seconds.abs().le(10)
        & pool.pe_support_count.ge(5)
        & pool.q_profile_samples.ge(4)
        & ~pool.feat_feature_id.isin(reserved)
        & ~pool["name"].isin(locked)
    )
    mass_uncertainty = float(audit.get("mass_mass_uncertainty_ppm", 0.0))
    mass_budget = (
        ~pool.pe_calibration_mass_supported
        | pool.pe_corrected_mass_error_ppm.abs().add(mass_uncertainty).lt(3)
    )
    rt_budget = (
        ~pool.pe_calibration_RT_supported
        | pool.pe_local_rt_error_seconds.abs()
        .add(pool.pe_calibration_RT_uncertainty_seconds)
        .le(10)
    )
    eligible &= mass_budget & rt_budget
    pool["pe_calibration_mass_uncertainty_pass"] = mass_budget
    pool["pe_calibration_RT_uncertainty_pass"] = rt_budget
    pool["pe_calibration_eligible"] = eligible
    pool["pe_calibration_selected"] = False
    selected = []
    for _, group in pool[eligible].groupby("name", sort=True):
        g = group.sort_values(
            ["pe_completion_score", "feat_intensity", "feat_feature_id"],
            ascending=[False, False, True],
            kind="stable",
        )
        best = g.iloc[0]
        if best.pe_completion_score < 0.1:
            continue
        if (
            len(g) > 1
            and best.pe_completion_score - g.pe_completion_score.iloc[1] < 0.03
        ):
            continue
        selected.append(best.name)
    chosen = (
        pool.loc[selected]
        .sort_values("pe_completion_score", ascending=False)
        .drop_duplicates("feat_feature_id")
        .copy()
    )
    chosen["automatic_accepted"] = True
    chosen["automatic_candidate"] = True
    chosen["pe_calibration_evidence_pass"] = True
    chosen["pe_final_selected"] = True
    chosen["auto_annotation_status"] = "pe_local_calibration_supported"
    chosen["status"] = "pe_local_calibration_supported"
    chosen["rank_for_feature"] = 1
    chosen["standard_stream"] = "RP_Pos"
    chosen["chromatography"] = "RP"
    chosen["grid_coherence_exemption"] = "target_excluded_local_PE_calibration"
    after = (
        pd.concat([annotations, chosen], ignore_index=True, sort=False)
        if not chosen.empty
        else annotations.copy()
    )
    if after.feat_feature_id.astype(str).duplicated().any():
        raise ValueError("Local calibration collided on an occupied feature")
    pairs = (
        pd.concat(
            [all_pairs[~_pe_keys(all_pairs).isin(set(_pe_keys(chosen)))], chosen],
            ignore_index=True,
            sort=False,
        )
        if not chosen.empty
        else all_pairs.copy()
    )
    pool.loc[chosen.index, "pe_calibration_selected"] = True
    audit.update(
        status="completed",
        added_PE=len(chosen),
        baseline_annotations=len(annotations),
        final_annotations=len(after),
        manual_labels_used=False,
    )
    return after, pairs, pool, pd.DataFrame([audit])


# Frozen pilot candidate generator, isolated from runtime after rejection.
def _join_raw_pe(
    refs: pd.DataFrame,
    features: pd.DataFrame,
    negative_features: pd.DataFrame,
    negative_annotations: pd.DataFrame,
    mass_center: float,
    negative_center: float,
    settings: PERebuildSettings,
    mass_centers: dict[str, float] | None = None,
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
        offset = (
            mass_center
            if mass_centers is None
            else float(mass_centers.get(target["name"], mass_center))
        )
        if (
            not np.isfinite(offset)
            or abs(offset) > 3
            or abs(offset - mass_center) > 1.5
        ):
            offset = mass_center
        if not np.isfinite(rt) or not np.isfinite(mz) or mz <= 0:
            continue
        lo = np.searchsorted(
            masses, mz * (1 + (offset - settings.ppm_tolerance) * 1e-6)
        )
        hi = np.searchsorted(
            masses,
            mz * (1 + (offset + settings.ppm_tolerance) * 1e-6),
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
                or abs(ppm - offset) >= settings.ppm_tolerance
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
                pe_mass_center_ppm=offset,
                pe_corrected_mass_error_ppm=ppm - offset,
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
    mass_centers: dict[str, float] | None = None,
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
        mass_centers,
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
