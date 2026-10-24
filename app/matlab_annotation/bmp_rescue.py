"""Research-only additive NP BMP specialist; never replace existing annotations."""

from annotator.composition_ranges import canonical_lipid_class
from .operating_point import select_operating_candidates

NP_MODES = ("NP_Pos", "NP_Neg")


def bmp_queries(data):
    """Keep native BMP classes in NP, retaining original row indices/provenance."""
    pool, evidence = data["pool"], data["evidence"]
    if not pool.index.is_unique or not pool.index.equals(evidence.index):
        raise ValueError("Aligned unique physical/predictor rows required")
    rows = (
        pool.stream.isin(NP_MODES)
        & pool["class"].map(canonical_lipid_class).eq("BMP")
        & ~pool.is_is.fillna(False).astype(bool)
    )
    return dict(data, pool=pool.loc[rows].copy(), evidence=evidence.loc[rows].copy())


def select_bmp_rescue(pool, scores, previous, *, stream, threshold):
    """Compete all strict BMP candidates before occupancy; no fallback promotion.

    Scores are learned exclusively on allowed training projects, never inferred
    from manual presence in a query. Caller supplies the unchanged strict
    physical candidate pool. Manual-absent calls remain extras in evaluation.
    """
    if stream not in NP_MODES:
        raise ValueError("BMP rescue applies only to NP measurement modes")
    if len(pool) and (
        not pool.stream.eq(stream).all()
        or not pool["class"].map(canonical_lipid_class).eq("BMP").all()
        or pool.is_is.fillna(False).astype(bool).any()
    ):
        raise ValueError("Only endogenous BMP candidates in the selected mode")
    if len(previous) and not previous.stream.eq(stream).all():
        raise ValueError("Occupancy must use the same measurement mode")
    selected = select_operating_candidates(
        pool, scores, threshold=threshold, ambiguity_margin=0.05
    )
    # An identity's strongest feature is fixed BEFORE ambiguity or occupancy.
    # Withholding that feature may not promote a weaker alternative.
    winners = (
        pool.assign(_score=scores)
        .sort_values(["_score", "feat_intensity"], ascending=False, kind="stable")
        .drop_duplicates(["name", "adduct"])
    )
    winner_keys = set(
        winners[["name", "adduct", "feat_feature_id"]].itertuples(
            index=False, name=None
        )
    )
    selected = selected.loc[
        [
            k in winner_keys
            for k in selected[["name", "adduct", "feat_feature_id"]].itertuples(
                index=False, name=None
            )
        ]
    ].copy()
    occupied = set(previous.feat_feature_id.astype(str))
    identities = set(previous[["name", "adduct"]].itertuples(index=False, name=None))
    free = ~selected.feat_feature_id.astype(str).isin(occupied) & ~selected[
        ["name", "adduct"]
    ].apply(tuple, axis=1).isin(identities)
    result = selected.loc[free].copy()
    result["auto_annotation_status"] = "independently_trained_bmp_rescue"
    return result
