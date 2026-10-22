"""Research-only peak choice AFTER an identity has already been accepted.

Separates 'is this lipid present?' from 'which original feature represents it?'.
Identity/adduct/mode sets cannot grow, so this step cannot create extra identities.
No curation is used at prediction time. Standards/manual rows stay untouched.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .standard_completion import _manual, _standards


def conditional_training_weights(pool, dataset, cohort):
    """Train ONLY resolved one-true-peak identities with alternative raw peaks.

    Each identity gives equal total weight to its true peak and ALL alternatives
    combined. Each project group then gets equal total weight. Constant class/
    composition occurrence cannot improve classification merely via prevalence.
    Zero weights mean this row is not conditional-peak training material.
    """
    if not pool.index.equals(dataset.index) or not pool.index.equals(cohort.index):
        raise ValueError("Conditional training metadata must match candidate indices")
    required = {
        "stream",
        "name",
        "adduct",
        "feat_feature_id",
        "training_positive",
        "training_label_known",
    }
    if not required.issubset(pool):
        raise ValueError("Conditional training needs resolved feature-level labels")
    if pool[["training_positive", "training_label_known"]].isna().any().any():
        raise ValueError("Training label flags must be explicit")
    keys = [
        dataset.astype(str),
        pool.stream.astype(str),
        pool.name.astype(str),
        pool.adduct.astype(str),
    ]
    known = pool.training_label_known.astype(bool)
    positive = pool.training_positive.astype(bool) & known
    npos = positive.groupby(keys).transform("sum")
    nnegative = (known & ~positive).groupby(keys).transform("sum")
    nfeatures = pool.feat_feature_id.astype(str).groupby(keys).transform("nunique")
    eligible = known & npos.eq(1) & nnegative.ge(1) & nfeatures.ge(2)
    weight = pd.Series(0.0, index=pool.index)
    weight.loc[eligible & positive] = 1.0
    weight.loc[eligible & ~positive] = 1.0 / nnegative[eligible & ~positive]
    groups = cohort.astype(str)
    totals = weight.groupby(groups).transform("sum")
    keep = weight.gt(0)
    if keep.any():
        weight.loc[keep] /= totals[keep]
        weight.loc[keep] *= int(keep.sum()) / groups[keep].nunique()
    return weight


def rerank_accepted_peaks(
    annotations,
    pool,
    scores,
    *,
    min_score=0.75,
    margin=0.15,
    allowed_classes=None,
    forbidden_features=None,
):
    """Conservatively replace an accepted identity's peak, NEVER its identity.

    New peak must beat ALL alternatives by the fixed margin and already satisfy
    the caller's physical candidate limits. Original occupied peaks cannot be
    borrowed; conflicting proposals are ALL withheld. No iterative cascade.
    """
    if not scores.index.equals(pool.index):
        raise ValueError("Conditional scores must match candidate rows")
    if not 0.5 <= min_score <= 1 or not 0 <= margin <= 1:
        raise ValueError("Invalid conditional peak score/margin")
    keys = ["stream", "name", "adduct", "is_is"]
    protected = _standards(annotations) | _manual(annotations)
    duplicate_identity = annotations.duplicated(keys, keep=False)
    if (duplicate_identity & ~protected).any():
        raise ValueError(
            "Unprotected accepted identities must be unique before peak reranking"
        )
    occupied_keys = ["stream", "feat_feature_id"]
    if (
        annotations.assign(feat_feature_id=annotations.feat_feature_id.astype(str))
        .duplicated(occupied_keys)
        .any()
    ):
        raise ValueError("Accepted original features must be unique within a stream")
    selected = annotations.copy()
    occupied = set(
        zip(selected.stream.astype(str), selected.feat_feature_id.astype(str))
    )
    forbidden = forbidden_features or {}
    ranked = pool.assign(_conditional_score=pd.to_numeric(scores, errors="coerce"))
    groups = {k: g for k, g in ranked.groupby(["stream", "name", "adduct"], sort=False)}
    proposals, audit = [], []
    for index, row in selected.iterrows():
        if protected.loc[index]:
            continue
        cls = str(row.get("class", ""))
        if (
            allowed_classes is not None
            and (str(row.stream), cls) not in allowed_classes
        ):
            continue
        part = groups.get((row.stream, row["name"], row.adduct))
        if part is None:
            continue
        part = part.sort_values(
            "_conditional_score", ascending=False, kind="stable"
        ).drop_duplicates("feat_feature_id")
        old_id = str(row.feat_feature_id)
        current = part[part.feat_feature_id.astype(str).eq(old_id)]
        if len(current) != 1 or len(part) < 2:
            continue
        best, runner = part.iloc[0], part.iloc[1]
        fid = str(best.feat_feature_id)
        score = float(best._conditional_score)
        old_score = float(current.iloc[0]._conditional_score)
        if (
            fid == old_id
            or not np.isfinite(
                [score, old_score, float(runner._conditional_score)]
            ).all()
        ):
            continue
        if (
            score < min_score
            or score - float(runner._conditional_score) < margin
            or score - old_score < margin
        ):
            continue
        if (str(row.stream), fid) in occupied or fid in forbidden.get(
            str(row.stream), set()
        ):
            continue
        proposals.append((index, row.stream, fid, old_id, best, old_score))
    claims = Counter((str(stream), fid) for _, stream, fid, _, _, _ in proposals)
    for index, stream, fid, old_id, best, old_score in proposals:
        if claims[(str(stream), fid)] > 1:
            audit.append(
                dict(
                    action="withhold_conflicting_peak_proposals",
                    stream=stream,
                    feature_id=fid,
                )
            )
            continue
        old_joint = selected.loc[index].get("class_selection_score", np.nan)
        for col, value in best.items():
            if col.startswith("training_") or col in {
                "_conditional_score",
                "calibration_identity_in_manual",
            }:
                continue
            if col not in selected:
                selected[col] = pd.Series(None, index=selected.index, dtype=object)
            elif selected[col].dtype != pool[col].dtype:
                # CSV round-trips infer all-empty text columns as float NaN.
                # Preserve values while allowing the original candidate type;
                # this has no influence on model scores or replacement rules.
                selected[col] = selected[col].astype(object)
            selected.loc[index, col] = value
        selected.loc[index, "previous_feature_id"] = old_id
        selected.loc[index, "previous_class_selection_score"] = old_joint
        selected.loc[index, "class_selection_score"] = np.nan
        for col in ("confidence", "confidence_score", "post_regression_confidence"):
            if col in selected:
                selected.loc[index, col] = np.nan
        selected.loc[index, "conditional_peak_score"] = float(best._conditional_score)
        selected.loc[index, "previous_conditional_peak_score"] = old_score
        selected.loc[index, "auto_annotation_status"] = (
            "conditional_original_peak_reranked"
        )
        selected.loc[index, "automatic_accepted"] = True
        audit.append(
            dict(
                action="replace_original_peak_only",
                stream=stream,
                name=selected.loc[index, "name"],
                previous_feature_id=old_id,
                feature_id=fid,
                score=float(best._conditional_score),
                previous_score=old_score,
            )
        )
    assert Counter(annotations[keys].itertuples(index=False, name=None)) == Counter(
        selected[keys].itertuples(index=False, name=None)
    )
    assert len(selected) == len(annotations)
    assert (
        not selected.assign(feat_feature_id=selected.feat_feature_id.astype(str))
        .duplicated(occupied_keys)
        .any()
    )
    return selected, pd.DataFrame(audit)
