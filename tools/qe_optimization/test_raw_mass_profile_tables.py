"""Synthetic exact raw memoization/scoping checks, no annotation/model fits."""

import pandas as pd
import numpy as np
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources
from prepare_qe_raw_mass_profile_tables import masks_for, exact_lookup, physical_queries
from verify_qe_raw_mass_profile_tables import independent_numeric

configure_frozen_resources(REPO)


def fixture():
    return pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                **{"class": cls},
                adduct="[M+H]+",
                ref_mz=500.0,
                feat_feature_id=fid,
                feat_rt=5.0,
                feat_mz=500.0001,
            )
            for cls, fid in [("PC-O", "10"), ("PE", "11")]
        ],
        index=[8, 12],
    )


def test_all_nested_masks_include_own_fold_and_cover_exact363_tables():
    for own in range(5):
        masks = masks_for(own)
        assert len(masks) == len(set(masks)) == 11
        assert all(own in m for m in masks)
        assert [len(m) for m in masks].count(1) == 1
        assert [len(m) for m in masks].count(2) == 4
        assert [len(m) for m in masks].count(3) == 6
    assert 33 * len(masks_for(0)) == 363


def test_exact_lookup_preserves_original_order_duplicates_and_coordinates():
    pool = fixture()
    queries = physical_queries(pool).reset_index(drop=True)
    measured = pd.DataFrame(dict(value=[0.1, 0.9]), index=queries.index)
    wanted = pool.iloc[[1, 0, 1]].set_axis([33, 77, 88])
    result = exact_lookup(queries, measured, wanted)
    assert result.index.equals(wanted.index) and result.value.tolist() == [
        0.9,
        0.1,
        0.9,
    ]
    assert queries.loc[0, "class"] == "PC[O]"
    pd.testing.assert_frame_equal(pool, fixture())
    for bad in (
        pool.assign(feat_feature_id="missing"),
        pool.assign(feat_rt=5.1),
        pool.assign(ref_mz=501),
        pool.assign(feat_mz=501),
    ):
        with pytest.raises(ValueError, match="Missing raw"):
            exact_lookup(queries, measured, bad)
    with pytest.raises(ValueError, match="Unique aligned"):
        exact_lookup(pd.concat([queries, queries.iloc[[0]]]), measured, pool)


def test_legacy_one_bit_RT_versions_are_distinct_exact_keys_not_rounded_together():
    pool = fixture().iloc[[0]]
    alias = pool.assign(feat_rt=np.nextafter(5.0, 6.0)).set_axis([99])
    combined = pd.concat([pool, alias])
    queries = physical_queries(combined).reset_index(drop=True)
    assert not queries.duplicated().any()
    measured = pd.DataFrame(dict(value=[0.1, 0.2]))
    assert exact_lookup(queries, measured, combined).value.tolist() == [0.1, 0.2]


@pytest.mark.parametrize(
    "field",
    ["training_positive", "manual_feature_id", "same_manual_peak", "manual_confirmed"],
)
def test_memoization_refuses_all_curator_supervision(field):
    with pytest.raises(ValueError, match="curation"):
        physical_queries(fixture().assign(**{field: False}))


@pytest.mark.parametrize("paired", [True, False])
def test_independent_mass_profile_constructor_matches_all23_fields_and_charges(paired):
    from matlab_annotation.raw_counterpart_evidence import raw_counterpart_evidence

    features, maps, rows = {}, {}, []
    columns = [f"sample{i}" for i in range(8)]
    values = np.exp(np.array([1.0, 2.0, 1.5, 3.0, 2.5, 1.7, 3.4, 2.8]))
    for stream in ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"):
        positive = stream.endswith("Pos")
        masses = [500.0, 1258.880854] if positive else [543.990928, 619.416238]
        table = pd.DataFrame(
            dict(
                feature_id=["pc", "cl", "background1", "background2", "background3"],
                mz=[*masses, 2000.0, 2001.0, 2002.0],
                rt=[5.0, 8.0, 2.0, 3.0, 4.0],
                intensity=100.0,
                **{
                    c: [values[i], values[::-1][i], 100.0, 100.0, 100.0]
                    for i, c in enumerate(columns)
                },
            )
        )
        table.attrs = dict(
            sample_columns=list(columns),
            sample_group_mapping_status="validated",
            sample_group_map={c: "Group_1_Tissue" for c in columns},
        )
        features[stream] = table
        maps[stream] = {c: str(i) for i, c in enumerate(columns)}
        for cls, fid, mass, rt, ion in [
            ("PC", "pc", masses[0], 5.0, "[M+H]+" if positive else "[M+CHO2]-"),
            ("CL", "cl", masses[1], 8.0, "[M+NH4]+" if positive else "[M-2H](2-)"),
        ]:
            if cls == "CL" and stream.startswith("NP"):
                continue
            rows.append(
                dict(
                    stream=stream,
                    **{"class": cls},
                    adduct=ion,
                    ref_mz=mass,
                    feat_feature_id=fid,
                    feat_rt=rt,
                    feat_mz=mass,
                )
            )
    queries = pd.DataFrame(rows)
    mapping = maps if paired else {}
    independent = independent_numeric(queries, features, mapping)
    measured = (
        pd.concat(
            [
                raw_counterpart_evidence(
                    queries.loc[queries.stream.eq(s)],
                    features,
                    REPO / "reference_data/matlab_pipeline",
                    mapping,
                    stream=s,
                )
                for s in features
            ]
        )
        .reindex(queries.index)
        .loc[:, independent.columns]
    )
    np.testing.assert_allclose(
        measured.to_numpy(float),
        independent.to_numpy(float),
        rtol=2e-6,
        atol=2e-7,
        equal_nan=True,
    )
