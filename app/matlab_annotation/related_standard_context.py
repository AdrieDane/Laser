"""Research descriptors from related standards, never interchangeable identities.

An ether lipid can use a separately labelled diacyl-headgroup descriptor, and
positional lyso classes can use a separately labelled related lyso descriptor.
No RT order, shared RT, concentration equivalence or structural identification
is asserted. Mode-specific models must learn any useful offset on allowed
training projects. Candidate labels/formulas and the exact-class helper remain
unchanged. No file reads or annotation decisions occur here.
"""

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .declared_standard_context import (
    COLUMNS,
    _qualified_measurements,
    declared_standard_context_evidence,
)

PREFIX = "q_declared_standard_related_"
RELATIONS = {
    **{
        f"{root}[{link}]": (1, (root,))
        for root in ("PE", "PC")
        for link in ("O", "P", "O+P")
    },
    **{
        f"{position}-acyl {root}": (2, (root, f"{3 - position}-acyl {root}"))
        for root in ("LPC", "LPE")
        for position in (1, 2)
    },
    **{root: (2, (f"1-acyl {root}", f"2-acyl {root}")) for root in ("LPC", "LPE")},
}
NAMES = [
    "measured_spike_feature"
    if c == "spike_feature"
    else "related_compounds"
    if c == "same_class_compounds"
    else c
    for c in COLUMNS
]


def related_standard_context_evidence(pool, measurements):
    """Return 16 numeric context columns, without merging any candidate classes."""
    # Delegate the same query validation and missing-value convention.
    empty = declared_standard_context_evidence(pool, measurements.iloc[:0])
    output = empty.rename(columns=dict(zip(empty.columns, [PREFIX + c for c in NAMES])))
    output[PREFIX + "relation"] = 0.0
    if pool.empty or measurements.empty:
        return output
    qualified = _qualified_measurements(measurements)
    if qualified.empty:
        return output
    query_classes = pool["class"].map(canonical_lipid_class)
    standard_classes = qualified["class"].map(canonical_lipid_class)
    # Even a mislabelled query on ANY independently measured spike feature must
    # not use that spike or one of its companion ions as an endogenous anchor.
    spike_keys = set(zip(qualified.stream, qualified.feat_feature_id.astype(str)))
    for cls, (role, parents) in RELATIONS.items():
        rows = pool.loc[query_classes.eq(cls)]
        if rows.empty:
            continue
        anchors = qualified.loc[standard_classes.isin(parents)].copy()
        output.loc[rows.index, PREFIX + "relation"] = float(role)
        if anchors.empty:
            continue
        # A temporary matching key chooses the declared RELATED anchor set.
        # Never return this key or write it to physical candidates/annotations.
        context = declared_standard_context_evidence(
            rows.assign(**{"class": "_related_descriptor_only"}),
            anchors.assign(**{"class": "_related_descriptor_only"}),
        )
        context.columns = output.columns[:-1]
        blocked = pd.Series(
            [
                (s, str(fid)) in spike_keys
                for s, fid in rows[["stream", "feat_feature_id"]].itertuples(
                    index=False, name=None
                )
            ],
            index=rows.index,
        )
        context.loc[blocked, :] = np.nan
        for c in ("related_compounds", "same_adduct_compounds"):
            context.loc[blocked, PREFIX + c] = 0.0
        context.loc[blocked, PREFIX + "measured_spike_feature"] = 1.0
        output.loc[rows.index, context.columns] = context
    return output
