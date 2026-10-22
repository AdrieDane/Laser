"""Research-only dual-RT candidate proposals; not enabled by the app.

Both RT sources propose ORIGINAL features inside 3 ppm / 10 seconds. A second
model can evaluate their evidence without discarding a valid existing proposal
merely because the atlas predicts another RT. No manual labels are read here.
"""

from __future__ import annotations

import pandas as pd

from annotator.composition_ranges import DEFAULT_COMPOSITION_RANGE_SETTINGS
from .automatic import matlab_annotation_adducts
from .class_candidates import _composition_allowed
from .retention_atlas import predict_aligned_retention_atlas

KEYS = ["name", "adduct", "feat_feature_id"]
REQUIRED = [
    "name",
    "class",
    "adduct",
    "carbon",
    "db",
    "ref_mz",
    "ref_rt",
    "feat_feature_id",
    "feat_mz",
    "feat_rt",
    "feat_intensity",
    "ppm_error",
    "rt_error",
]
DESCRIPTORS = [
    "post_regression_confidence",
    "grid_coherence_score",
    "grid_coherence_neighbors",
    "grid_coherence_pass",
    "positional_isomer_selected",
    "rp_isomer_evidence_score",
]


def combine_rt_sources(
    atlas_pool,
    fitted_pool,
    automatic,
    atlas,
    *,
    stream,
    ranges=DEFAULT_COMPOSITION_RANGE_SETTINGS,
):
    """A union, not unconditional acceptance; one row per feature/identity.

    Strict coordinates, non-isotope endogenous identities, existing composition
    limits and reserved standard features are enforced on the fitted proposals.
    Source-specific RT errors remain evidence even outside the other source's
    window. Neither a source indicator nor an existing acceptance is a label.
    """
    missing = set(REQUIRED) - set(fitted_pool)
    if missing:
        raise ValueError(f"Fitted candidates lack {sorted(missing)}")
    base = fitted_pool.copy()
    for frame in (base,):
        frame["feat_feature_id"] = frame.feat_feature_id.astype(str)
    standard = (
        base.get("is_is", pd.Series(False, index=base.index)).fillna(False).astype(bool)
    )
    isotope = (
        base.get("isotopologue", pd.Series("", index=base.index)).fillna("").astype(str)
    )
    declared = (
        automatic.get("is_is", pd.Series(False, index=automatic.index))
        .fillna(False)
        .astype(bool)
    )
    protected = set(automatic.loc[declared, "feat_feature_id"].astype(str))
    valid = (
        ~standard
        & isotope.eq("")
        & base.adduct.isin(matlab_annotation_adducts(stream[:2]))
        & pd.to_numeric(base.ppm_error, errors="coerce").abs().lt(3)
        & pd.to_numeric(base.rt_error, errors="coerce").abs().mul(60).le(10)
        & ~base.feat_feature_id.isin(protected)
    )
    base = base.loc[valid]
    allowed = [
        _composition_allowed(ranges, r["class"], r["carbon"], r["db"], r["name"])
        for r in base.to_dict("records")
    ]
    base = base.loc[allowed].copy()
    base = base.drop_duplicates(KEYS).reset_index(drop=True)
    fields = [*REQUIRED, *[c for c in DESCRIPTORS if c in base]]
    base = base[fields]
    base["baseline_ref_rt"] = base.ref_rt
    # The fitted engine can carry a positional correction in rt_expected.
    # Reconstruct its actual expected RT, not an uncorrected library coordinate.
    base["q_fitted_RT_error_seconds"] = (
        pd.to_numeric(base.rt_error, errors="coerce") * 60
    )
    base["q_fitted_source"] = 1.0
    for c in DESCRIPTORS:
        if c in base:
            base["q_fitted_" + c] = pd.to_numeric(base[c], errors="coerce")
    if len(base):
        predictions = predict_aligned_retention_atlas(
            base, automatic, atlas, stream=stream
        )
        for c in predictions:
            base[c] = predictions[c]
    else:
        base["atlas_rt_seconds"] = pd.Series(dtype=float)
    base["atlas_available"] = base.atlas_rt_seconds.notna()
    base["q_atlas_RT_error_seconds"] = base.feat_rt * 60 - base.atlas_rt_seconds
    base["q_atlas_source"] = 0.0
    # Keep atlas rows first to retain the candidate schema and their RT source.
    learned = atlas_pool.drop(
        columns=["training_positive", "training_label_known"], errors="ignore"
    ).copy()
    if len(learned):
        learned["feat_feature_id"] = learned.feat_feature_id.astype(str)
        learned = learned[~learned.feat_feature_id.isin(protected)].copy()
        learned["q_atlas_source"] = 1.0
        learned["q_atlas_RT_error_seconds"] = (
            learned.feat_rt * 60 - learned.atlas_rt_seconds
        )
    union = (
        pd.concat([learned, base], ignore_index=True)
        .drop_duplicates(KEYS)
        .reset_index(drop=True)
    )
    if union.empty:
        return union
    lookup = base.set_index(KEYS)
    indices = pd.MultiIndex.from_frame(union[KEYS])
    for c in [
        "q_fitted_source",
        "q_fitted_RT_error_seconds",
        *["q_fitted_" + s for s in DESCRIPTORS],
    ]:
        if c in lookup:
            union[c] = lookup[c].reindex(indices).to_numpy()
    union["q_fitted_source"] = union.get(
        "q_fitted_source", pd.Series(0.0, index=union.index)
    ).fillna(0.0)
    union["q_atlas_source"] = union.q_atlas_source.fillna(0.0)
    selected = (
        set(
            map(
                tuple,
                automatic.loc[~declared, KEYS]
                .assign(feat_feature_id=lambda x: x.feat_feature_id.astype(str))
                .to_numpy(),
            )
        )
        if set(KEYS).issubset(automatic)
        else set()
    )
    union["q_previously_selected"] = [
        float(k in selected) for k in union[KEYS].itertuples(index=False, name=None)
    ]
    union["q_absolute_fitted_RT_error_seconds"] = union.q_fitted_RT_error_seconds.abs()
    union["q_absolute_atlas_RT_error_seconds"] = union.q_atlas_RT_error_seconds.abs()
    union["stream"] = stream
    union["is_is"] = False
    union["isotopologue"] = ""
    union["automatic_accepted"] = False
    union["automatic_candidate"] = True
    union["auto_annotation_status"] = "dual_RT_source_candidate"
    assert not union.feat_feature_id.isin(protected).any()
    assert not union.duplicated(KEYS).any()
    return union
