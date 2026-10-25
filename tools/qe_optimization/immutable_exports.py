"""Pure frame assembly for research additions to an explicit retained checkpoint."""

import pandas as pd

from run_qe_np_context_bmp_experiment import accepted_keys


def assemble_immutable(previous, previous_candidates, added, *, method):
    from standard_lipidomics.validation import _boolean

    if added.empty:
        return previous.copy(deep=True), previous_candidates.copy(deep=True)
    oldkeys, newkeys = set(accepted_keys(previous)), set(accepted_keys(added))
    assert len(newkeys) == len(added) and not oldkeys & newkeys
    assert not {k[:4] for k in oldkeys} & {k[:4] for k in newkeys}
    assert not {(k[0], k[4]) for k in oldkeys} & {(k[0], k[4]) for k in newkeys}
    flags = (
        previous_candidates.automatic_accepted.map(_boolean).fillna(False).astype(bool)
    )
    assert set(accepted_keys(previous_candidates.loc[flags])) == oldkeys
    candidate = previous_candidates.copy(deep=True)
    prior_keys = accepted_keys(candidate)
    missing = newkeys - set(prior_keys)
    if missing:
        candidate = pd.concat(
            [candidate, added.loc[[k in missing for k in accepted_keys(added)]]],
            ignore_index=True,
        )
    candidate_keys = accepted_keys(candidate)
    mask = pd.Series([k in newkeys for k in candidate_keys], index=candidate.index)
    assert int(mask.sum()) == len(newkeys), (
        "New call requires one explicit candidate representative"
    )
    lookup = {k: r for k, (_, r) in zip(accepted_keys(added), added.iterrows())}
    if "previous_041_class_selection_score" in candidate:
        raise ValueError("Do not overwrite earlier factorization provenance")
    candidate["previous_041_class_selection_score"] = pd.Series(
        index=candidate.index, dtype=object
    )
    candidate.loc[mask, "previous_041_class_selection_score"] = candidate.loc[
        mask, "class_selection_score"
    ].to_numpy()
    for col in added:
        # Preserve old cell VALUES, including integer feature areas and literal
        # NA/empty metadata; no broad numeric/string coercion of old data.
        if col not in candidate:
            candidate[col] = pd.Series(index=candidate.index, dtype=object)
        else:
            candidate[col] = candidate[col].astype(object)
        candidate.loc[mask, col] = [
            lookup[k][col]
            for k in pd.Series(candidate_keys, index=candidate.index).loc[mask]
        ]
    candidate.loc[mask, "automatic_accepted"] = True
    candidate.loc[mask, "automatic_candidate"] = False
    candidate.loc[mask, "auto_annotation_status"] = method
    # All non-promoted existing candidate cells remain exact; only new columns
    # and the fully identified promoted row can change.
    untouched = [k not in newkeys for k in prior_keys]
    pd.testing.assert_frame_equal(
        candidate.iloc[: len(previous_candidates)]
        .loc[untouched, list(previous_candidates)]
        .reset_index(drop=True),
        previous_candidates.loc[untouched].reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    annotations = pd.concat([previous, added], ignore_index=True)
    pd.testing.assert_frame_equal(
        annotations.iloc[: len(previous)][list(previous)].reset_index(drop=True),
        previous.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    accepted = candidate.automatic_accepted.map(_boolean).fillna(False).astype(bool)
    assert set(accepted_keys(candidate.loc[accepted])) == oldkeys | newkeys
    assert int(accepted.sum()) == len(annotations)
    assert not annotations.duplicated(["stream", "feat_feature_id"]).any()
    return annotations, candidate
