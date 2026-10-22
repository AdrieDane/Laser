"""Research-only neutral-ion footprint ambiguity for NEW NP-positive calls.

Coeluting near-isobaric classes cannot be distinguished just by another adduct
or split XCMS feature. Withholding is not proof of a false chemical identity.
Existing annotations and same-class channels are not rewritten here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .standard_rt_library import ion_key
from .vendor.composition_initializer import _ADDUCTS
from .vendor.compound_model import PROTON_MASS

_ION = {
    ion_key(label): (float(addition), int(charge))
    for label, addition, charge in _ADDUCTS.values()
}


def neutral_ion_mass(observed_mz, adduct):
    """Invert the exact native catalogue convention; observed m/z is positive."""
    model = _ION.get(ion_key(adduct))
    try:
        observed = float(observed_mz)
    except (TypeError, ValueError):
        return np.nan
    if model is None or not np.isfinite(observed) or observed <= 0:
        return np.nan
    addition, charge = model
    return abs(charge) * (observed - np.sign(charge) * PROTON_MASS) - addition


def _coordinates(frame):
    neutral = np.asarray(
        [
            neutral_ion_mass(m, a)
            for m, a in frame[["feat_mz", "adduct"]].itertuples(index=False, name=None)
        ]
    )
    rt = pd.to_numeric(frame.feat_rt, errors="coerce").to_numpy(float) * 60
    cls = frame["class"].map(canonical_lipid_class).to_numpy(str)
    return neutral, rt, cls


def filter_new_np_positive_isobars(
    existing, additions, *, neutral_ppm=5.0, coelution_seconds=2.0
):
    """Keep old calls; withhold only NEW positive-NP cross-class conflicts.

    The5ppm/2s envelope is an EXCLUSION test for ambiguity, not a wider candidate
    admission window. Same canonical class is deliberately not filtered here;
    class/adduct channel selection is a separate question. New-vs-new conflicts
    are ALL withheld, never resolved greedily by input order.
    """
    if not 0 < neutral_ppm <= 10 or not 0 < coelution_seconds <= 5:
        raise ValueError("Invalid neutral-footprint exclusion window")
    required = {
        "stream",
        "name",
        "class",
        "adduct",
        "feat_feature_id",
        "feat_mz",
        "feat_rt",
    }
    if not required.issubset(existing) or not required.issubset(additions):
        raise ValueError("Ion footprints need original feature/identity coordinates")
    if not additions.index.is_unique:
        raise ValueError("New-call row indices must be unique")
    part = additions[additions.stream.eq("NP_Pos")]
    if part.empty:
        return additions.copy(), pd.DataFrame()
    old = existing[existing.stream.eq("NP_Pos")]
    mass, rt, cls = _coordinates(part)
    old_mass, old_rt, old_cls = _coordinates(old)
    finite = np.isfinite(mass) & (mass > 0) & np.isfinite(rt)
    old_valid = np.isfinite(old_mass) & (old_mass > 0) & np.isfinite(old_rt)
    with np.errstate(invalid="ignore", divide="ignore"):
        old_match = (
            (
                np.abs(mass[:, None] - old_mass[None, :]) / mass[:, None] * 1e6
                <= neutral_ppm
            )
            & (np.abs(rt[:, None] - old_rt[None, :]) <= coelution_seconds)
            & (cls[:, None] != old_cls[None, :])
            & old_valid[None, :]
        )
        new_match = (
            (
                np.abs(mass[:, None] - mass[None, :])
                / np.minimum(mass[:, None], mass[None, :])
                * 1e6
                <= neutral_ppm
            )
            & (np.abs(rt[:, None] - rt[None, :]) <= coelution_seconds)
            & (cls[:, None] != cls[None, :])
            & finite[:, None]
            & finite[None, :]
        )
    blocked = ~finite | old_match.any(axis=1) | new_match.any(axis=1)
    audit = []
    for position in np.flatnonzero(blocked):
        row = part.iloc[position]
        reason = (
            "withhold_unknown_neutral_ion_coordinates"
            if not finite[position]
            else "withhold_existing_cross_class_neutral_footprint"
            if old_match[position].any()
            else "withhold_competing_new_cross_class_footprints"
        )
        audit.append(
            dict(
                action=reason,
                stream=row.stream,
                name=row["name"],
                adduct=row.adduct,
                feature_id=str(row.feat_feature_id),
                neutral_mass=mass[position],
                existing_conflicts=int(old_match[position].sum()),
                new_conflicts=int(new_match[position].sum()),
            )
        )
    drop = part.index[blocked]
    result = additions.drop(index=drop).copy()
    assert set(result.index).issubset(additions.index)
    pd.testing.assert_frame_equal(
        result[~result.stream.eq("NP_Pos")], additions[~additions.stream.eq("NP_Pos")]
    )
    return result, pd.DataFrame(audit)
