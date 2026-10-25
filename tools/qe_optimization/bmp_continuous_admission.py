"""046 admission-only ablation: hard native physics, continuous profile evidence.

No IO, labels, fitting, cut selection, app defaults or weaker fallback.
"""

import pandas as pd

from bmp_context_measurements import PREFIX


def select_additions(
    pool,
    scores,
    evidence,
    previous,
    *,
    stream,
    threshold,
    protected_features,
    declared_masses,
):
    from matlab_annotation.np_context_bmp_completion import select_context_bmp_additions

    if not pool.index.equals(evidence.index):
        raise ValueError("Aligned original pool/measurements required")
    if PREFIX + "source_physical" not in evidence:
        raise ValueError("Masked native physical eligibility required")
    selected = select_context_bmp_additions(
        pool,
        scores,
        previous,
        stream=stream,
        threshold=threshold,
        protected_features=protected_features,
        declared_masses=declared_masses,
    )
    # Strict native/global/identity-best decisions precede the independent
    # physical mask. No counterpart existence/correlation/metadata hard veto.
    return selected.loc[
        evidence[PREFIX + "source_physical"].eq(1).reindex(selected.index)
    ].copy()


def proposals(d, mask, raw, scores, previous, thresholds, supported):
    import run_qe_bmp_context_experiment as source

    pieces = []
    for s in source.NP_MODES:
        part = raw["pool"].loc[raw["pool"].stream.eq(s)]
        if not supported[s]:
            pieces.append(part.iloc[:0].copy())
            continue
        protection = source.original.inherited.common.physical_protection(
            d, mask, s, raw
        )
        pieces.append(
            select_additions(
                part,
                scores.loc[part.index],
                raw["evidence"].loc[part.index],
                previous.loc[previous.stream.eq(s)],
                stream=s,
                threshold=thresholds[s],
                **protection,
            )
        )
    return pd.concat(pieces, axis=0)
