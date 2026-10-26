"""Pure completion graph/occupancy controls, no real example annotations run."""

import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
import matrix_pipeline_assembly as assembly


def row(name, fid, accepted):
    return dict(
        stream="RP_Pos",
        name=name,
        **{"class": "PC"},
        adduct="[M+H]+",
        feat_feature_id=fid,
        is_is=False,
        feat_mz=500.0,
        feat_rt=5.0,
        feat_intensity=100.0,
        automatic_accepted=accepted,
        automatic_candidate=not accepted,
    )


def test_completion_keeps_all_competitors_and_does_not_mutate_old_cells():
    previous = pd.DataFrame([row("PC(34:1)", "1", True)])
    graph = pd.DataFrame(
        [
            row("PC(34:1)", "1", True),
            row("PC(36:1)", "2", False),
            row("PC(36:1)", "3", False),
        ]
    )
    filled = pd.DataFrame([row("PC(36:1)", "2", True)])
    frozen = graph.copy(deep=True)
    annotations, candidate = assembly.append_completion(previous, graph, filled)
    assert list(annotations.feat_feature_id) == ["1", "2"]
    assert set(candidate.feat_feature_id) == {"1", "2", "3"}
    assert set(candidate.loc[candidate.automatic_accepted].feat_feature_id) == {
        "1",
        "2",
    }
    pd.testing.assert_frame_equal(graph, frozen, check_exact=True)


def test_empty_completion_is_exact_noop():
    previous = pd.DataFrame([row("PC(34:1)", "1", True)])
    annotations, candidates = assembly.append_completion(
        previous, previous, previous.iloc[:0]
    )
    pd.testing.assert_frame_equal(annotations, previous, check_exact=True)
    pd.testing.assert_frame_equal(candidates, previous, check_exact=True)


@pytest.mark.parametrize("name,fid", [("PC(34:1)", "2"), ("PC(36:1)", "1")])
def test_old_identity_or_original_feature_cannot_be_replaced(name, fid):
    previous = pd.DataFrame([row("PC(34:1)", "1", True)])
    with pytest.raises(AssertionError):
        assembly.append_completion(
            previous, previous, pd.DataFrame([row(name, fid, True)])
        )


def test_all_sample_cells_checked_on_same_original_feature():
    raw = pd.DataFrame(
        dict(feature_id=["01", "2"], SAMPLE=[100.0, 200.0], BLANK=[1.0, 2.0])
    )
    raw.attrs["sample_columns"] = ["SAMPLE", "BLANK"]
    base = dict(streams={"RP_Pos": dict(features=raw)})
    annotations = pd.DataFrame(
        [dict(row("PC(34:1)", "01", True), feat_SAMPLE=100.0, feat_BLANK=1.0)]
    )
    assert assembly.check_original_sample_cells(annotations, base) == 2
    altered = annotations.copy()
    altered.loc[0, "feat_BLANK"] = 2.0
    with pytest.raises(AssertionError):
        assembly.check_original_sample_cells(altered, base)


@pytest.mark.parametrize("raw_id,annotation_id", [("01", "01"), (1, "1")])
def test_missing_output_sample_values_restored_without_reselecting_or_rescaling(
    raw_id, annotation_id
):
    raw = pd.DataFrame(
        dict(
            feature_id=[raw_id],
            mz=[500.0],
            rt=[5.0],
            intensity=[100.0],
            SAMPLE=[1234.0],
        )
    )
    raw.attrs["sample_columns"] = ["SAMPLE"]
    base = dict(streams={"RP_Pos": dict(features=raw)})
    old = pd.DataFrame(
        [
            dict(
                row("PC(34:1)", annotation_id, True),
                feat_SAMPLE=float("nan"),
                class_selection_score=0.9,
            )
        ]
    )
    fixed = assembly.restore_original_cells(old, base)
    assert fixed.feat_SAMPLE.iloc[0] == 1234.0
    assert (
        fixed.name.iloc[0] == old.name.iloc[0]
        and fixed.feat_feature_id.iloc[0] == annotation_id
    )
    assert fixed.class_selection_score.iloc[0] == 0.9
    assert pd.isna(old.feat_SAMPLE.iloc[0])
    wrong_coordinate = old.copy()
    wrong_coordinate.loc[0, "feat_mz"] = 501.0
    with pytest.raises(AssertionError):
        assembly.restore_original_cells(wrong_coordinate, base)
