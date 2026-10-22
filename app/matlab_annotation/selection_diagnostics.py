"""Read-only explanations of the EXISTING operating selector, not new choices.

All parameters must be the actual recorded settings. Never use diagnostic
curation flags as scoring features or silently change a cutoff during an audit.
This module is unused by production annotation entrypoints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .operating_point import select_operating_candidates


def audit_operating_candidates(pool, scores, *, threshold, ambiguity_margin=0.05):
    """Explain every candidate row and verify exact parity with the selector.

    Reasons are ordered by the existing selector's gates, not independent causal
    errors. A lower-ranked row can also fail the cutoff. Post-selector standard
    adjudication is outside this audit and must be distinguished by the caller.
    Call separately for each stream: feature IDs are not unique across modes.
    """
    if not pool.index.is_unique:
        raise ValueError("Unique query-row indices required for diagnostic alignment")
    if "stream" in pool and pool.stream.nunique() > 1:
        raise ValueError("Audit each measurement stream separately")
    expected = select_operating_candidates(
        pool, scores, threshold=threshold, ambiguity_margin=ambiguity_margin
    )
    ranked = pool.assign(_audit_score=scores).sort_values(
        ["_audit_score", "feat_intensity"], ascending=False, kind="stable"
    )
    pair_keys = ["feat_feature_id", "name", "adduct"]
    duplicate = ranked.duplicated(pair_keys)
    unique = ranked[~duplicate].copy()
    rank = unique.groupby("feat_feature_id").cumcount()
    next_score = unique.groupby("feat_feature_id")._audit_score.shift(-1)
    gap = unique._audit_score - next_score
    clear = next_score.isna() | gap.ge(ambiguity_margin)
    above = unique._audit_score.ge(threshold)
    passing = unique.loc[rank.eq(0) & clear & above]
    selected_indices = passing.drop_duplicates(["name", "adduct"]).index

    result = pd.DataFrame(index=pool.index)
    result["score"] = scores
    result["threshold"] = float(threshold)
    result["feature_rank"] = rank.reindex(pool.index)
    result["next_feature_score"] = next_score.reindex(pool.index)
    result["feature_score_gap"] = gap.reindex(pool.index)
    result["above_cutoff"] = scores.ge(threshold)
    result["clear_feature_margin"] = clear.reindex(pool.index, fill_value=False)
    result["selector_accepts"] = result.index.isin(selected_indices)
    result["primary_gate"] = "duplicate_candidate_pair"
    reasons = np.select(
        [rank.ne(0), ~clear, ~above, ~unique.index.isin(selected_indices)],
        [
            "another_label_ranks_above_on_feature",
            "ambiguous_feature_label",
            "below_acceptance_cutoff",
            "same_lipid_adduct_preferred_on_another_feature",
        ],
        default="selected",
    )
    result.loc[unique.index, "primary_gate"] = reasons
    if not result.index[result.selector_accepts].equals(
        pool.index[pool.index.isin(expected.index)]
    ):
        raise AssertionError(
            "Audit changed the existing selector's accepted query rows"
        )
    return result
