import numpy as np
import pandas as pd
import pytest

from matlab_annotation.specimen_pairs import explicit_specimen_pairs


def fixture():
    metadata = dict(
        sample_id=np.array(["sample-A", "sample-B"]),
        group=np.array([1, 2]),
        group_label=np.array(["WT", "KO"]),
        filename=np.array(
            [
                ["negative-A.raw", "positive-A.mzML"],
                ["negative-B.raw", "positive-B.mzML"],
            ]
        ),
        column=np.array([["RP", "RP"], ["RP", "RP"]]),
        mode=np.array([[-1, 1], [-1, 1]]),
    )
    features = {}
    for stream, cols in [
        ("RP_Pos", ["positive-B", "positive-A"]),
        ("RP_Neg", ["negative-A", "negative-B"]),
    ]:
        f = pd.DataFrame([[1, 2]], columns=cols)
        f.attrs.update(
            sample_columns=cols,
            sample_group_mapping_status="validated",
            sample_group_map={
                c: "Group_02_KO" if c.endswith("B") else "Group_01_WT" for c in cols
            },
        )
        features[stream] = f
    return metadata, features


def test_explicit_pairing_is_independent_of_raw_column_and_measurement_order():
    m, f = fixture()
    result, audit = explicit_specimen_pairs(m, f)
    assert (
        result["RP_Pos"]["positive-A"] == result["RP_Neg"]["negative-A"] == "sample-A"
    )
    assert audit["RP_Pos"]["unique_specimens"] == 2
    for k in ("filename", "column", "mode"):
        m[k] = m[k][:, ::-1]
    assert explicit_specimen_pairs(m, f)[0] == result


def test_same_numbers_or_order_never_invent_missing_links():
    m, f = fixture()
    m["filename"] = np.full((2, 2), "other-file")
    # Different rows cannot reuse one filename for different sample IDs.
    with pytest.raises(ValueError, match="multiple specimen"):
        explicit_specimen_pairs(m, f)
    m, f = fixture()
    m["filename"] = np.array([["unknown-1", "unknown-2"], ["unknown-3", "unknown-4"]])
    result, audit = explicit_specimen_pairs(m, f)
    assert not result["RP_Pos"] and audit["RP_Pos"]["unmapped_columns"] == 2


def test_conflicting_group_or_ambiguous_file_links_fail_explicitly():
    m, f = fixture()
    f["RP_Pos"].attrs["sample_group_map"]["positive-A"] = "Group_02_KO"
    with pytest.raises(ValueError, match="group disagrees"):
        explicit_specimen_pairs(m, f)
    m, f = fixture()
    m["sample_id"] = np.array(["same", "same"])
    with pytest.raises(ValueError, match="conflicting biological groups"):
        explicit_specimen_pairs(m, f)


def test_only_explicit_phase_and_polarity_bind_streams_and_missing_schema_fails():
    m, f = fixture()
    m["mode"] = np.zeros((2, 2))
    with pytest.raises(ValueError, match="phase/mode"):
        explicit_specimen_pairs(m, f)
    m, f = fixture()
    del m["sample_id"]
    with pytest.raises(ValueError, match="fields are missing"):
        explicit_specimen_pairs(m, f)
