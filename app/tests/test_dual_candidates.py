import pandas as pd

from matlab_annotation.dual_candidates import combine_rt_sources
from test_class_candidates import inputs


def fitted_rows():
    a, r, f, atlas = inputs()
    rows = pd.concat(
        [
            r.assign(
                feat_feature_id="0017",
                feat_mz=700.0001,
                feat_rt=8.01,
                feat_intensity=1e5,
                rt_error=0.01,
                ppm_error=0.14,
                is_is=False,
            ),
            r.assign(
                feat_feature_id="18",
                feat_mz=700.0001,
                feat_rt=8.5,
                feat_intensity=1e5,
                rt_error=0.5,
                ppm_error=0.14,
                is_is=False,
            ),
        ],
        ignore_index=True,
    )
    return a, r, f, atlas, rows


def test_fitted_proposal_outside_atlas_window_remains_evidence_not_acceptance():
    a, r, f, atlas, rows = fitted_rows()
    union = combine_rt_sources(pd.DataFrame(), rows, a, atlas, stream="RP_Pos")
    assert union.feat_feature_id.tolist() == ["0017"]
    assert union.q_fitted_source.tolist() == [1.0]
    assert union.q_atlas_source.tolist() == [0.0]
    assert union.q_absolute_atlas_RT_error_seconds.iloc[0] > 100
    assert not union.automatic_accepted.any()


def test_duplicate_sources_are_one_pair_and_standards_reserved():
    a, r, f, atlas, rows = fitted_rows()
    learned = rows.iloc[:1].assign(
        atlas_rt_seconds=480.0,
        atlas_available=True,
        atlas_training_cohorts=4,
        atlas_training_mad_seconds=1.0,
        atlas_local_support_count=0,
        atlas_local_shift_seconds=0.0,
    )
    union = combine_rt_sources(learned, rows, a, atlas, stream="RP_Pos")
    assert len(union) == 1
    assert union.q_atlas_source.iloc[0] == union.q_fitted_source.iloc[0] == 1
    standard = pd.DataFrame(
        {
            "name": ["X (IS)"],
            "adduct": ["[M+H]+"],
            "feat_feature_id": ["0017"],
            "is_is": [True],
        }
    )
    assert combine_rt_sources(
        pd.DataFrame(), rows, standard, atlas, stream="RP_Pos"
    ).empty
    assert combine_rt_sources(learned, rows, standard, atlas, stream="RP_Pos").empty


def test_no_manual_inputs_or_source_mutation_and_strict_mass_boundary():
    a, r, f, atlas, rows = fitted_rows()
    old = rows.copy(deep=True)
    bad = rows.assign(ppm_error=3.0)
    assert combine_rt_sources(pd.DataFrame(), bad, a, atlas, stream="RP_Pos").empty
    pd.testing.assert_frame_equal(rows, old)
