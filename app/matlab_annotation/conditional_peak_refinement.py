"""Unused research: propose peak moves, NEVER accept/drop a lipid identity.

No curation/files are read. Initial occupied features and standards are immutable.
All identities/adducts are preserved exactly. Numerical preferences are not an
analytical identity probability; this helper is not enabled in production.
"""

import numpy as np
import pandas as pd

from .conditional_peak_model import predict_peak_preferences
from .evidence_training import validate_feature_scope
from .mode_evidence_experts import MODES

KEYS = ["stream", "name", "adduct"]
MOVE_COLUMNS = [
    "annotation_index",
    "current_index",
    "candidate_index",
    *KEYS,
    "old_feature_id",
    "new_feature_id",
    "preference",
    "decision",
]


def peak_refinement_preferences(
    data,
    accepted,
    model,
    *,
    protected_features_by_stream,
    protected_masses_by_stream,
):
    """Score eligible free-feature challengers BEFORE calibration truth is attached.

    Same old strict3ppm/10s evidence only; exclude all declared spike masses
    within5.01ppm, independently measured spike features, and initially occupied
    features. Never reuse a vacated feature or infer specimen pairing.
    """
    validate_feature_scope(
        data,
        model["training_cohorts"],
        model["excluded_groups"],
        query_is_training=False,
    )
    if set(protected_features_by_stream) != set(MODES) or set(
        protected_masses_by_stream
    ) != set(MODES):
        raise ValueError("Explicit spike protections for ALL four modes required")
    required = {*KEYS, "feat_feature_id", "is_is"}
    if not required.issubset(accepted) or not accepted.index.is_unique:
        raise ValueError("Explicit unique accepted identity/standard rows required")
    if not accepted.is_is.isin([True, False]).all():
        raise ValueError("Explicit standard flags required")
    if (
        accepted.assign(_fid=accepted.feat_feature_id.astype(str))
        .duplicated(["stream", "_fid"])
        .any()
    ):
        raise ValueError("Base accepted original features must be unique per mode")
    pool, evidence = data["pool"], data["evidence"]
    if not pool.index.is_unique:
        raise ValueError("Candidate row indices must be unique")
    occupied = set(zip(accepted.stream, accepted.feat_feature_id.astype(str)))
    protected = {s: set(map(str, f)) for s, f in protected_features_by_stream.items()}
    masses = {
        s: np.asarray(list(v), float) for s, v in protected_masses_by_stream.items()
    }
    if any(not np.isfinite(a).all() or np.any(a <= 0) for a in masses.values()):
        raise ValueError("Declared spike masses must be finite and positive")
    groups = {key: g for key, g in pool.groupby(KEYS, sort=False)}
    rows = []
    standard_named = accepted.name.astype(str).str.startswith(
        "IS: "
    ) | accepted.name.astype(str).str.endswith(" (IS)")
    for ai, row in accepted[~(accepted.is_is.astype(bool) | standard_named)].iterrows():
        key = tuple(row[k] for k in KEYS)
        part = groups.get(key)
        if part is None:
            continue
        part = part.drop_duplicates("feat_feature_id", keep="first")
        current = part[part.feat_feature_id.astype(str).eq(str(row.feat_feature_id))]
        if current.empty:
            continue  # e.g. fixed027mass-only calls are not old-core candidates
        ci = current.index[0]
        ref = float(current.iloc[0].ref_mz)
        if not np.isfinite(ref) or ref <= 0:
            continue
        for ni, new in part.iterrows():
            fid = str(new.feat_feature_id)
            stream = new.stream
            if (stream, fid) in occupied or fid in protected[stream]:
                continue
            if not np.isclose(float(new.ref_mz), ref, rtol=0, atol=1e-8):
                continue
            mz = float(new.feat_mz)
            coordinates = np.asarray([mz, new.feat_rt, new.feat_intensity], float)
            if (
                not np.isfinite(coordinates).all()
                or mz <= 0
                or np.any(coordinates[1:] < 0)
            ):
                continue
            if abs((mz - ref) / ref * 1e6) >= 3:
                continue
            if (
                len(masses[stream])
                and np.min(np.abs(mz - masses[stream]) / masses[stream] * 1e6) < 5.01
            ):
                continue
            ev = evidence.loc[ni]
            spike = pd.to_numeric(
                pd.Series([ev.get("q_declared_standard_spike_feature", 0)]),
                errors="coerce",
            ).iloc[0]
            if np.isfinite(spike) and spike >= 0.5:
                continue
            rt = [
                ev.get(c, np.nan)
                for c in (
                    "absolute_RT_error_seconds",
                    "q_absolute_atlas_RT_error_seconds",
                    "q_absolute_fitted_RT_error_seconds",
                )
            ]
            rt = np.asarray(rt, float)
            if not (np.isfinite(rt) & (np.abs(rt) <= 10)).any():
                continue
            rows.append(
                dict(
                    annotation_index=ai,
                    current_index=ci,
                    candidate_index=ni,
                    **dict(zip(KEYS, key)),
                    old_feature_id=str(row.feat_feature_id),
                    new_feature_id=fid,
                    preference=np.nan,
                    decision="unscored",
                )
            )
    result = pd.DataFrame(rows, columns=MOVE_COLUMNS)
    if len(result):
        result["preference"] = predict_peak_preferences(
            data, model, result.candidate_index.tolist(), result.current_index.tolist()
        )
        result["decision"] = np.where(
            result.preference.notna(),
            "eligible",
            "unsupported_conditional_mode_or_class",
        )
    return result


def select_peak_refinements(preferences, thresholds, *, ambiguity_margin=0.05):
    """Fixed no-displacement competition; thresholds learned on INNER groups only."""
    if set(thresholds) != set(MODES) or any(
        not np.isfinite(t) or not 0.5 <= t <= 1 for t in thresholds.values()
    ):
        raise ValueError("Explicit [.5,1] move threshold for all four modes required")
    if not np.isfinite(ambiguity_margin) or not 0 <= ambiguity_margin <= 1:
        raise ValueError("Invalid conditional ambiguity margin")
    audit = preferences.copy()
    if not set(MOVE_COLUMNS).issubset(audit):
        raise ValueError("Complete blind physical preference table required")
    finite = audit.preference.dropna()
    if not np.isfinite(finite).all() or ((finite < 0) | (finite > 1)).any():
        raise ValueError("Invalid conditional preference values")
    choices = []
    for _, group in audit[audit.preference.notna()].groupby(
        "annotation_index", sort=False
    ):
        group = group.sort_values("preference", ascending=False, kind="stable")
        audit.loc[group.index, "decision"] = "lower_preference"
        best = group.iloc[0]
        idx = group.index[0]
        threshold = thresholds[best.stream]
        if threshold == 1:
            audit.loc[idx, "decision"] = "moves_disabled_by_inner_calibration"
        elif best.preference < threshold:
            audit.loc[idx, "decision"] = "below_move_threshold"
        elif (
            len(group) > 1
            and best.preference - group.iloc[1].preference < ambiguity_margin
        ):
            audit.loc[idx, "decision"] = "ambiguous_challenger_peaks"
        else:
            choices.append(idx)
    for _, group in audit.loc[choices].groupby(
        ["stream", "new_feature_id"], sort=False
    ):
        group = group.sort_values("preference", ascending=False, kind="stable")
        audit.loc[group.index, "decision"] = "other_identity_prefers_same_free_feature"
        if (
            len(group) > 1
            and group.iloc[0].preference - group.iloc[1].preference < ambiguity_margin
        ):
            audit.loc[group.index, "decision"] = "ambiguous_free_feature"
        else:
            audit.loc[group.index[0], "decision"] = "selected_peak_move"
    selected = audit[audit.decision.eq("selected_peak_move")].copy()
    assert not selected.annotation_index.duplicated().any()
    assert not selected.duplicated(["stream", "new_feature_id"]).any()
    return selected, audit


def apply_peak_refinements(accepted, pool, moves, *, features_by_stream):
    """Copy original candidate measurements, preserve ALL identity/standard keys.

    The original pointwise score must not masquerade as confidence in the new
    peak: it is retained separately, and the refined row's legacy score is NaN.
    No acceptance cleanup is rerun that could add/drop identities afterwards.
    """
    if moves.empty:
        return accepted.copy()
    if set(features_by_stream) != set(MODES):
        raise ValueError("Original raw feature tables for ALL four modes required")
    originals = {}
    for stream, features in features_by_stream.items():
        if not {"feature_id", "mz", "rt", "intensity"}.issubset(features):
            raise ValueError("Original raw feature coordinates required")
        if features.feature_id.astype(str).duplicated().any():
            raise ValueError("Original raw feature IDs must be unique per mode")
        originals[stream] = features.set_index(features.feature_id.astype(str))
    # Saved CSVs can represent an all-missing boolean column as float64.
    # Stage heterogeneous candidate scalars explicitly, then infer non-identity
    # columns and restore immutable columns with their EXACT original dtypes.
    result = accepted.astype(object).copy()
    added_columns = set(pool) | {
        "base_identity_acceptance_score",
        "base_identity_operating_threshold",
        "class_selection_score",
        "peak_refinement_score",
        "peak_refinement_old_feature_id",
        "auto_annotation_status",
        "automatic_accepted",
        "automatic_candidate",
    }
    added_columns.update(
        "feat_" + str(c) for f in features_by_stream.values() for c in f
    )
    for column in sorted(added_columns - set(result)):
        result[column] = pd.Series(np.nan, index=result.index, dtype=object)
    occupied = set(zip(accepted.stream, accepted.feat_feature_id.astype(str)))
    immutable = [
        c
        for c in (
            *KEYS,
            "class",
            "base",
            "carbon",
            "db",
            "ref_mz",
            "matlab_code",
            "matlab_compound",
            "isotopologue",
            "is_is",
            "display_name",
        )
        if c in accepted
    ]
    if (
        moves.annotation_index.duplicated().any()
        or moves.duplicated(["stream", "new_feature_id"]).any()
    ):
        raise ValueError("Peak moves must be one-to-one")
    for move in moves.itertuples():
        old = accepted.loc[move.annotation_index]
        new = pool.loc[move.candidate_index]
        named_standard = str(old["name"]).startswith("IS: ") or str(
            old["name"]
        ).endswith(" (IS)")
        if (
            bool(old.is_is)
            or named_standard
            or tuple(old[k] for k in KEYS) != tuple(new[k] for k in KEYS)
        ):
            raise ValueError("Cannot change standard or accepted lipid identity")
        if str(old.feat_feature_id) != str(move.old_feature_id) or str(
            new.feat_feature_id
        ) != str(move.new_feature_id):
            raise ValueError(
                "Move coordinates disagree with original candidate indices"
            )
        if (new.stream, str(new.feat_feature_id)) in occupied:
            raise ValueError("Cannot move onto an initially occupied original feature")
        raw = originals[new.stream].loc[str(new.feat_feature_id)]
        np.testing.assert_allclose(
            np.asarray([new.feat_mz, new.feat_rt, new.feat_intensity], float),
            np.asarray([raw.mz, raw.rt, raw.intensity], float),
            rtol=8 * np.finfo(float).eps,
            atol=0,
            equal_nan=True,
        )
        # Old peak-specific metadata must not silently describe the challenger.
        for column in result:
            if column not in immutable and column not in pool:
                result.loc[move.annotation_index, column] = np.nan
        for column, value in new.items():
            if column not in immutable:
                result.loc[move.annotation_index, column] = value
        for column in immutable:
            result.loc[move.annotation_index, column] = old[column]
        # Never retain quantification or peak bounds from the OLD feature.
        for column in result:
            if str(column).startswith("feat_"):
                result.loc[move.annotation_index, column] = np.nan
        for column, value in raw.items():
            result.loc[move.annotation_index, "feat_" + str(column)] = value
        result.loc[move.annotation_index, "feat_feature_id"] = str(raw.feature_id)
        result.loc[move.annotation_index, "base_identity_acceptance_score"] = old.get(
            "class_selection_score", np.nan
        )
        result.loc[move.annotation_index, "class_selection_score"] = np.nan
        result.loc[move.annotation_index, "base_identity_operating_threshold"] = (
            old.get("operating_threshold", np.nan)
        )
        if "operating_threshold" in result:
            result.loc[move.annotation_index, "operating_threshold"] = np.nan
        result.loc[move.annotation_index, "peak_refinement_score"] = move.preference
        result.loc[move.annotation_index, "peak_refinement_old_feature_id"] = str(
            old.feat_feature_id
        )
        result.loc[move.annotation_index, "auto_annotation_status"] = (
            "conditional_original_peak_refined"
        )
        result.loc[move.annotation_index, "automatic_accepted"] = True
        result.loc[move.annotation_index, "automatic_candidate"] = False
    result = result.infer_objects()
    for column in immutable:
        result[column] = accepted[column].copy()
    pd.testing.assert_frame_equal(
        accepted[immutable], result[immutable], check_exact=True
    )
    assert len(result) == len(accepted)
    assert (
        not result.assign(_fid=result.feat_feature_id.astype(str))
        .duplicated(["stream", "_fid"])
        .any()
    )
    return result
