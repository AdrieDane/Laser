"""All-cell/original-measurement checks shared by a prospective research audit."""

import numpy as np
import pandas as pd

from run_qe_np_context_bmp_experiment import accepted_keys


def check_export(path, previous, candidates, previous_candidates, baseline):
    from standard_lipidomics.validation import _boolean

    text = pd.read_csv(path, dtype=str, keep_default_na=False)
    actual = pd.read_csv(
        path,
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    pd.testing.assert_frame_equal(
        text.iloc[: len(previous)][list(previous)].reset_index(drop=True),
        previous.reset_index(drop=True),
        check_exact=True,
    )
    keys = set(accepted_keys(actual))
    oldkeys = set(accepted_keys(previous))
    assert oldkeys.issubset(keys) and len(keys) == len(actual)

    def flags(frame, column):
        return frame[column].map(_boolean).fillna(False).astype(bool)

    accepted = flags(candidates, "automatic_accepted")
    assert set(accepted_keys(candidates.loc[accepted])) == keys
    assert int(accepted.sum()) == len(actual)
    old_legacy = set(
        accepted_keys(
            previous_candidates.loc[
                flags(previous_candidates, "automatic_accepted")
                & flags(previous_candidates, "automatic_candidate")
            ]
        )
    )
    legacy = set(
        accepted_keys(
            candidates.loc[accepted & flags(candidates, "automatic_candidate")]
        )
    )
    assert legacy == old_legacy and not (keys - oldkeys) & legacy
    populated, inherited, roundoff = 0, 0, 0
    for stream, payload in baseline["streams"].items():
        rows = actual.loc[actual.stream.eq(stream)]
        assert not rows.feat_feature_id.astype(str).duplicated().any()
        original = payload["features"].copy()
        original.index = original.feature_id.astype(str)
        source = original.loc[rows.feat_feature_id.astype(str)]
        new = rows.index.to_numpy() >= len(previous)
        for destination, origin in (
            ("feat_mz", "mz"),
            ("feat_rt", "rt"),
            ("feat_intensity", "intensity"),
        ):
            a, b = rows[destination].to_numpy(float), source[origin].to_numpy(float)
            np.testing.assert_allclose(
                a, b, rtol=8 * np.finfo(float).eps, atol=0, equal_nan=True
            )
            roundoff += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
        for column in original:
            destination = "feat_" + column
            if destination not in text:
                assert not new.any(), (
                    stream,
                    destination,
                    "missing new original field",
                )
                continue
            values = text.loc[rows.index, destination]
            measured = source[column].reset_index(drop=True)
            present = values.ne("").to_numpy()
            if pd.api.types.is_numeric_dtype(original[column]):
                a = pd.to_numeric(values, errors="coerce").to_numpy(float)
                b = measured.to_numpy(float)
                np.testing.assert_allclose(
                    a[present | new],
                    b[present | new],
                    rtol=8 * np.finfo(float).eps,
                    atol=0,
                    equal_nan=True,
                    err_msg=destination,
                )
                inherited += int((~present & np.isfinite(b) & ~new).sum())
            else:
                a = values.astype(str).to_numpy()
                b = measured.fillna("").astype(str).to_numpy()
                assert (a[present | new] == b[present | new]).all(), destination
                inherited += int((~present & (b != "") & ~new).sum())
            populated += int(present.sum())
    return dict(
        legacy_flags=len(legacy),
        available_measurement_cells=populated,
        inherited_empty_old_measurement_cells=inherited,
        preexisting_roundoff_cells=roundoff,
        raw_stream_checks=len(baseline["streams"]),
    )
