"""Shared exclusion semantics for annotation summaries and exports."""

from __future__ import annotations

import pandas as pd


# Internal annotation tables use the short names; stable CSV exports use the
# descriptive names.  Summary code must understand both so rejected matches
# cannot become species merely because a table crossed an export boundary.
EXCLUSION_FLAG_COLUMNS = (
    "rule_excluded",
    "msms_rule_excluded",
    "grid_excluded",
    "grid_coherence_excluded",
    "mzmine_shape_score_excluded",
    "boundary_apex_excluded",
)


def _boolean_series(values: pd.Series) -> pd.Series:
    """Interpret native and CSV-round-tripped boolean columns consistently."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalised = values.fillna("").astype(str).str.strip().str.lower()
    return normalised.isin({"1", "true", "yes", "y"})


def annotation_exclusion_mask(frame: pd.DataFrame) -> pd.Series:
    """True for every row explicitly rejected by an annotation criterion."""
    excluded = pd.Series(False, index=frame.index, dtype=bool)
    for column in EXCLUSION_FLAG_COLUMNS:
        if column in frame:
            excluded |= _boolean_series(frame[column])
    if "exclusion_reason" in frame:
        excluded |= (
            frame["exclusion_reason"].fillna("").astype(str).str.strip().ne("")
        )
    return excluded


def retained_annotations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only rows eligible to contribute to annotation totals."""
    return frame.loc[~annotation_exclusion_mask(frame)].copy()
