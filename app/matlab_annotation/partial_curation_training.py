"""Training-only partial BMP supervision; never a runtime polarity rule.

Manual absence in the opposite NP polarity does not establish a negative when
the class was curated in just one NP polarity. Unknown is NOT positive. This
module is unused by production and cannot change evaluation/reference labels.
"""

from __future__ import annotations

from annotator.composition_ranges import canonical_lipid_class


def mask_uncurated_opposite_np_bmp(
    pool, manual, *, query_group, permitted_groups, excluded_groups
):
    """Return a copied supervised pool and audit, without adding predictors.

    Caller must check the same group authorization BEFORE reading the manual
    file. All manual BMP entries establish class presence, including unresolved
    features; standards do not. No policy is inferred for all-absent BMP, RP or
    other classes. Already unknown labels stay unknown; positives are immutable.
    """
    permitted, excluded = set(permitted_groups), set(excluded_groups)
    if (
        not permitted
        or not excluded
        or permitted & excluded
        or query_group not in permitted
        or query_group in excluded
    ):
        raise ValueError("Partial supervision requires an allowed training group")
    if not {"stream", "lipid_class", "internal_standard"}.issubset(manual):
        raise ValueError("Explicit manual class, mode and standard flags required")
    required = {"stream", "class", "is_is", "training_positive", "training_label_known"}
    if not required.issubset(pool):
        raise ValueError("Explicit separate supervised pool flags required")
    for frame, columns in (
        (manual, ["internal_standard"]),
        (pool, ["is_is", "training_positive", "training_label_known"]),
    ):
        if not all(frame[c].isin([True, False]).all() for c in columns):
            raise ValueError("Supervision and standard flags must be booleans")
    manual_bmp = manual.loc[
        ~manual.internal_standard.astype(bool)
        & manual.lipid_class.map(canonical_lipid_class).eq("BMP")
    ]
    present = set(manual_bmp.stream) & {"NP_Pos", "NP_Neg"}
    uncurated = ({"NP_Pos", "NP_Neg"} - present) if len(present) == 1 else set()
    affected = (
        pool.stream.isin(uncurated)
        & pool["class"].map(canonical_lipid_class).eq("BMP")
        & ~pool.is_is.astype(bool)
    )
    if (affected & pool.training_positive.astype(bool)).any():
        raise ValueError("A positive conflicts with claimed absent manual BMP mode")
    withheld = affected & pool.training_label_known.astype(bool)
    result = pool.copy()
    result.loc[withheld, "training_label_known"] = False
    assert result.training_positive.equals(pool.training_positive)
    assert not (result.training_label_known & ~pool.training_label_known).any()
    audit = dict(
        curated_np_modes=sorted(present),
        uncurated_opposite_np_modes=sorted(uncurated),
        withheld_known_negatives=int(withheld.sum()),
        positive_rows_before=int(pool.training_positive.sum()),
        positive_rows_after=int(result.training_positive.sum()),
        known_rows_before=int(pool.training_label_known.sum()),
        known_rows_after=int(result.training_label_known.sum()),
        evaluation_labels_changed=False,
    )
    return result, audit
