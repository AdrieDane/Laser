"""Unused research: NP-wide context models, prospectively BMP-only additions.

No files or annotation defaults. Callers authorize TRAINING truth before
masking; final query scoring refuses supervision. All old annotations stay
outside these helpers and MUST be preserved unchanged by the caller.
"""

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .coherent_retention import _protected_mass
from .evidence_training import (
    fit_grouped_evidence_model,
    predict_grouped_evidence,
    validate_feature_scope,
)
from .mode_evidence_experts import _subset
from .operating_point import select_operating_candidates
from .partial_curation_training import mask_uncurated_opposite_np_bmp
from .standard_rt_library import ion_key

NP_MODES = ("NP_Pos", "NP_Neg")
FORMAT = "laser-NP-wide-context-BMP-completion-v1"


def mask_training_bmp(data, manual, *, training_groups, excluded_groups):
    """Caller checks authorization BEFORE opening manual; check it again here."""
    validate_feature_scope(
        data, training_groups, excluded_groups, query_is_training=True
    )
    masked, audit = mask_uncurated_opposite_np_bmp(
        data["pool"],
        manual,
        query_group=data["query_group"],
        permitted_groups=training_groups,
        excluded_groups=excluded_groups,
    )
    return dict(data, pool=masked, bmp_partial_supervision_audit=audit)


def fit_np_context(data_by_dataset, *, training_groups, excluded_groups, parameters):
    """Up to TWO fresh mode experts,with ALL NP classes,not a BMP-only fit.

    >=20knownBMPpositive pairs from>=3actual projects per mode;otherwise explicit
    UNKNOWN for completion in that mode. Positive support is not biological
    replication or identification confidence. Original cohort weights retained.
    """
    allowed = set(training_groups)
    if {d["query_group"] for d in data_by_dataset.values()} != allowed:
        raise ValueError("Every permitted training project must be represented")
    columns = None
    for data in data_by_dataset.values():
        validate_feature_scope(data, allowed, excluded_groups, query_is_training=True)
        if not {
            "class",
            "stream",
            "is_is",
            "training_positive",
            "training_label_known",
        }.issubset(data["pool"]):
            raise ValueError(
                "Separate physical classes/modes and explicit training flags required"
            )
        audit = data.get("bmp_partial_supervision_audit", {})
        if (
            audit.get("evaluation_labels_changed") is not False
            or audit.get("positive_rows_before") != audit.get("positive_rows_after")
            or audit.get("positive_rows_after")
            != int(data["pool"].training_positive.sum())
            or audit.get("known_rows_after")
            != int(data["pool"].training_label_known.sum())
        ):
            raise ValueError(
                "Explicit unchanged-positive partial-supervision audit required"
            )
        if (
            not data["pool"]
            .stream.astype(str)
            .equals(data["evidence"].stream.astype(str))
        ):
            raise ValueError("Physical and predictor acquisition routes disagree")
        if columns is None:
            columns = list(data["evidence"])
        elif set(columns) != set(data["evidence"]):
            raise ValueError("Unchanged predictor schema required")
    experts, support = {}, {}
    for mode in NP_MODES:
        records = {k: _subset(d, mode) for k, d in data_by_dataset.items()}
        positive_projects, positives, other_rows = set(), 0, 0
        for data in records.values():
            pool = data["pool"]
            bmp = pool["class"].map(canonical_lipid_class).eq(
                "BMP"
            ) & ~pool.is_is.astype(bool)
            known = pool.training_label_known.astype(bool)
            positive = bmp & known & pool.training_positive.astype(bool)
            positives += int(positive.sum())
            other_rows += int((~bmp & known).sum())
            if positive.any():
                positive_projects.add(data["query_group"])
        support[mode] = dict(
            known_BMP_positive_pairs=positives,
            BMP_positive_projects=sorted(positive_projects),
            other_class_known_context_rows=other_rows,
        )
        if other_rows == 0:
            raise ValueError(
                "ALL NP classes must supply context;not a BMP-only specialist"
            )
        if positives < 20 or len(positive_projects) < 3:
            experts[mode] = None
            continue
        experts[mode] = fit_grouped_evidence_model(
            records,
            training_groups=allowed,
            excluded_groups=excluded_groups,
            parameters=parameters,
        )
        assert experts[mode]["columns"] == columns
    return dict(
        format=FORMAT,
        experts=experts,
        columns=columns,
        training_cohorts=sorted(allowed),
        excluded_groups=sorted(excluded_groups),
        support=support,
        fresh_mode_classifier_fits=sum(m is not None for m in experts.values()),
        training_pairs=sum(
            m["training_pairs"] for m in experts.values() if m is not None
        ),
        positives=sum(m["positives"] for m in experts.values() if m is not None),
        note="NP-wide all-class TRAINING context,prospectively BMP-only completion. Research curator agreement,not analytical identification probability/FDR. RP unchanged upstream;old annotations must remain immutable.",
    )


def predict_np_context(data, model):
    if model.get("format") != FORMAT or set(model["experts"]) != set(NP_MODES):
        raise ValueError("Explicit complete NP context model required")
    if {
        "training_positive",
        "training_label_known",
        "manual_feature_id",
        "same_manual_peak",
        "calibration_identity_in_manual",
        "manual_confirmed",
        "manually_confirmed",
        "manual_annotation",
    } & set(data["pool"]):
        raise ValueError("Final query prediction refuses supervision")
    validate_feature_scope(
        data,
        model["training_cohorts"],
        model["excluded_groups"],
        query_is_training=False,
    )
    out = pd.Series(
        np.nan,
        index=data["pool"].index[data["pool"].stream.isin(NP_MODES)],
        dtype=float,
    )
    for mode in NP_MODES:
        expert = model["experts"][mode]
        if expert is None:
            continue
        if set(expert["training_cohorts"]) != set(model["training_cohorts"]) or set(
            expert["excluded_groups"]
        ) != set(model["excluded_groups"]):
            raise ValueError("Context expert scopes disagree")
        subset = _subset(data, mode)
        if len(subset["pool"]):
            out.loc[subset["pool"].index] = predict_grouped_evidence(subset, expert)
    return out


def select_context_bmp_additions(
    pool, scores, previous, *, stream, threshold, protected_features, declared_masses
):
    """GLOBAL all-label winners first,then BMP,then original-call occupancy.

    No weaker alternative promotion,no old call replacement,no mutation. Raw
    strict3ppm/10s and independently protected spike IDs/masses rechecked.
    Training flags may remain attached for calibration counts but are NOT read.
    """
    if (
        stream not in NP_MODES
        or not np.isfinite(threshold)
        or not 0.5 <= threshold <= 1
    ):
        raise ValueError("Explicit NP mode and declared operating grid required")
    required = {
        "stream",
        "name",
        "class",
        "adduct",
        "is_is",
        "feat_feature_id",
        "feat_intensity",
        "ref_mz",
        "feat_mz",
        "feat_rt",
    }
    if (
        not required.issubset(pool)
        or not pool.index.is_unique
        or not scores.index.equals(pool.index)
    ):
        raise ValueError("Aligned full physical all-class NP competition required")
    if not pool.stream.eq(stream).all() or (
        len(previous) and not previous.stream.eq(stream).all()
    ):
        raise ValueError("Competition and occupancy must use the same mode")
    if threshold == 1.0 or pool.empty:
        return pool.iloc[:0].copy()
    if not np.isfinite(scores.to_numpy(float)).all():
        raise ValueError("Unsupported context scores require disabled completion")
    winners = (
        pool.assign(_context_score=scores)
        .sort_values(
            ["_context_score", "feat_intensity"], ascending=False, kind="stable"
        )
        .drop_duplicates(["name", "adduct"])
    )
    winner_keys = set(
        winners[["name", "adduct", "feat_feature_id"]].itertuples(
            index=False, name=None
        )
    )
    selected = select_operating_candidates(
        pool, scores, threshold=threshold, ambiguity_margin=0.05
    )
    selected = selected.loc[
        selected["class"].map(canonical_lipid_class).eq("BMP")
        & ~selected.is_is.astype(bool)
    ].copy()
    selected = selected.loc[
        [
            k in winner_keys
            for k in selected[["name", "adduct", "feat_feature_id"]].itertuples(
                index=False, name=None
            )
        ]
    ].copy()
    if selected.empty:
        return selected
    previous_ids = {
        (str(n), ion_key(a))
        for n, a in previous[["name", "adduct"]].itertuples(index=False, name=None)
    }
    occupied = set(previous.feat_feature_id.astype(str)) | set(
        map(str, protected_features)
    )
    keep = ~selected.feat_feature_id.astype(str).isin(occupied)
    keep &= pd.Series(
        [
            (str(n), ion_key(a)) not in previous_ids
            for n, a in selected[["name", "adduct"]].itertuples(index=False, name=None)
        ],
        index=selected.index,
    )
    keep &= ~selected.name.astype(str).str.endswith(" (IS)") & ~selected.name.astype(
        str
    ).str.startswith("IS: ")
    masses = np.asarray(list(declared_masses), float)
    if not np.isfinite(masses).all() or (masses <= 0).any():
        raise ValueError("Finite positive declared spike masses required")
    keep &= ~_protected_mass(selected.feat_mz.to_numpy(float), masses)
    keep &= ~_protected_mass(selected.ref_mz.to_numpy(float), masses)
    keep &= np.isfinite(
        selected[["ref_mz", "feat_mz", "feat_rt", "feat_intensity"]].to_numpy(float)
    ).all(axis=1)
    keep &= (
        selected.ref_mz.gt(0)
        & selected.feat_mz.gt(0)
        & selected.feat_rt.gt(0)
        & selected.feat_intensity.gt(0)
    )
    keep &= ((selected.feat_mz / selected.ref_mz - 1) * 1e6).abs().lt(3)
    expected = []
    for c, scale in (("ref_rt", 60), ("rt_expected", 60), ("atlas_rt_seconds", 1)):
        if c in selected:
            expected.append(
                (
                    selected.feat_rt * 60
                    - pd.to_numeric(selected[c], errors="coerce") * scale
                ).abs()
            )
    if not expected:
        raise ValueError("Permitted fitted/atlas numerical RT reference required")
    keep &= pd.concat(expected, axis=1).min(axis=1).le(10 + 1e-9)
    result = selected.loc[keep].copy()
    result["auto_annotation_status"] = "global_context_BMP_completion"
    return result
