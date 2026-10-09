"""Shared primary-adduct selection for standard-lipidomics views."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


# This is also the order used for the initial adduct shown in the grid viewer.
# Include both formate spellings emitted by the conventional and MATLAB paths.
PRIMARY_STANDARD_ADDUCT_ORDER = (
    "[M+H]+",
    "[M-H]-",
    "[M+NH4]+",
    "[M+HCOO]-",
    "[M+CHO2]-",
    "[M+Cl]-",
    "[M-2H](2-)",
    "[M+Na-2H]-",
    "[M+H-H2O]+",
    "[M-H-H2O]-",
    "[M+Na]+",
    "[M+K]+",
)


def preferred_standard_adduct(values: Iterable[object]) -> str | None:
    """Return the primary available adduct using the standard viewer order."""
    available = {
        str(value).strip()
        for value in values
        if str(value).strip() and str(value).strip().casefold() != "nan"
    }
    if not available:
        return None
    return next(
        (adduct for adduct in PRIMARY_STANDARD_ADDUCT_ORDER if adduct in available),
        sorted(available)[0],
    )


def primary_adduct_rows(
    frame: pd.DataFrame,
    *,
    class_column: str = "lipid_class",
    adduct_column: str = "adduct",
    stream_column: str | None = "stream",
) -> pd.DataFrame:
    """Keep only the primary available adduct per stream and lipid class."""
    if (
        frame is None
        or frame.empty
        or class_column not in frame
        or adduct_column not in frame
    ):
        return frame.copy() if frame is not None else pd.DataFrame()
    keep = pd.Series(False, index=frame.index)
    classes = frame[class_column].fillna("(unknown)").astype(str)
    adducts = frame[adduct_column].fillna("").astype(str)
    groups = classes
    if stream_column is not None and stream_column in frame:
        groups = frame[stream_column].fillna("").astype(str) + "\0" + classes
    for group in groups.unique():
        class_mask = groups.eq(group)
        preferred = preferred_standard_adduct(adducts[class_mask])
        if preferred is not None:
            keep |= class_mask & adducts.eq(preferred)
    return frame.loc[keep].copy()


__all__ = [
    "PRIMARY_STANDARD_ADDUCT_ORDER",
    "preferred_standard_adduct",
    "primary_adduct_rows",
]
