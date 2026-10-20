from dataclasses import replace
import json

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from adapters.standard_engines import build_automatic_anchor_settings
from matlab_annotation.pe_completion import (
    PECompletionSettings,
    complete_rp_positive_pe,
)


def fixture(count=7, odd=False):
    rows = []
    for i in range(count):
        carbon = 30 + 2 * i + int(odd)
        rows.append(
            dict(
                name=f"PE({carbon}:2)",
                **{"class": "PE"},
                adduct="[M+H]+",
                carbon=carbon,
                db=2,
                feat_feature_id=str(i + 1),
                feat_rt=10 + i * 0.25,
                rt_error=1 / 60,
                ppm_error=0.2,
                feat_intensity=1e7 / (i + 1),
                feat_mz=700 + i,
                ref_mz=700 + i,
                rt_expected=10 + i * 0.25 - 1 / 60,
                isotopologue="",
                is_is=False,
                grid_coherence_pass=True,
                grid_coherence_score=1.0,
                post_regression_confidence=0.01,
                automatic_accepted=False,
                automatic_candidate=False,
                auto_annotation_status="review_only_blue_candidate",
            )
        )
    pool = pd.DataFrame(rows)
    base = pool.iloc[[0]].copy().assign(feat_feature_id="old", name="PC(36:2)")
    base["class"] = "PC"
    base["automatic_accepted"] = True
    return base, pool


@pytest.fixture
def settings(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            dict(
                format="laser-pe-confidence-v1",
                intercept=5,
                numeric={},
                categorical={},
                composition_priors={},
            )
        )
    )
    return replace(PECompletionSettings(), model_path=model)


def test_adds_only_pe_without_global_count_cap_and_preserves_existing_rows(settings):
    before, pool = fixture()
    old_pool = pool.copy(deep=True)
    after, candidates, audit, anchors, details = complete_rp_positive_pe(
        before, pool, settings=settings
    )
    assert len(after) == 8 and audit.added_pe.iloc[0] == 7
    assert_frame_equal(after.iloc[:1][before.columns], before, check_dtype=False)
    assert_frame_equal(pool, old_pool)
    assert after.iloc[1:]["class"].eq("PE").all()
    assert after.iloc[1:].pe_completion_score.gt(0.9).all()
    assert after.iloc[1:].post_regression_confidence.eq(0.01).all()
    assert not after.feat_feature_id.duplicated().any()
    assert candidates.automatic_accepted.all()
    assert len(anchors) == 7
    for row in details.itertuples():
        assert row.pe_support_count >= 4
        assert (
            f"{int(row.carbon)}:{int(row.db)}"
            not in row.pe_support_compositions.split(";")
        )


def test_four_compositions_cannot_support_themselves(settings):
    before, pool = fixture(4)
    after, _, audit, _, _ = complete_rp_positive_pe(before, pool, settings=settings)
    assert_frame_equal(after, before)
    assert audit.added_pe.iloc[0] == 0


@pytest.mark.parametrize("stream", ["NP_Pos", "NP_Neg", "RP_Neg"])
def test_other_streams_are_exact_noop(stream, settings):
    before, pool = fixture()
    after, candidates, audit, _, _ = complete_rp_positive_pe(
        before, pool, stream=stream, settings=settings
    )
    assert_frame_equal(after, before)
    assert_frame_equal(candidates, pool)
    assert audit.status.iloc[0] == "not_applicable"


def test_disabled_is_exact_noop(settings):
    before, pool = fixture()
    after, candidates, audit, _, _ = complete_rp_positive_pe(
        before, pool, settings=replace(settings, enabled=False)
    )
    assert_frame_equal(after, before)
    assert_frame_equal(candidates, pool)
    assert audit.status.iloc[0] == "disabled"


def test_ambiguous_split_peak_is_not_selected_or_used_as_anchor(settings):
    before, pool = fixture(9)
    split = pool.iloc[[4]].assign(feat_feature_id="split")
    pool = pd.concat([pool, split], ignore_index=True)
    after, _, _, anchors, _ = complete_rp_positive_pe(before, pool, settings=settings)
    assert "split" not in set(after.feat_feature_id)
    assert "5" not in set(after.feat_feature_id)
    assert "5" not in set(anchors.feat_feature_id)


@pytest.mark.parametrize("defect", ["mass", "rt", "coherence", "isotope", "standard"])
def test_strict_safeguards_and_standards_cannot_be_bypassed(defect, settings):
    before, pool = fixture(10)
    if defect == "mass":
        pool.loc[5, "ppm_error"] = 3.0
    elif defect == "rt":
        pool.loc[5, "rt_error"] = 10.01 / 60
    elif defect == "coherence":
        pool.loc[5, "grid_coherence_pass"] = False
    elif defect == "isotope":
        pool.loc[5, "isotopologue"] = "13C"
    else:
        pool.loc[5, "internal_standard_name"] = "Declared spike"
    after, _, _, anchors, _ = complete_rp_positive_pe(before, pool, settings=settings)
    assert "6" not in set(after.feat_feature_id)
    assert "6" not in set(anchors.feat_feature_id)


def test_never_overwrites_occupied_feature_or_duplicates_existing_pe(settings):
    before, pool = fixture(10)
    before = pd.concat(
        [before, pool.iloc[[2]].assign(name="PS(40:0)"), pool.iloc[[4]]],
        ignore_index=True,
    )
    after, _, _, _, _ = complete_rp_positive_pe(before, pool, settings=settings)
    assert_frame_equal(
        after.iloc[: len(before)][before.columns], before, check_dtype=False
    )
    assert (after.feat_feature_id == "3").sum() == 1
    assert (after["name"] == pool["name"].iloc[4]).sum() == 1


def test_competing_valid_non_pe_identity_withholds_pe(settings):
    before, pool = fixture(10)
    other = pool.iloc[[5]].assign(name="PC(35:2)")
    other["class"] = "PC"
    pool = pd.concat([pool, other], ignore_index=True)
    after, _, _, _, _ = complete_rp_positive_pe(before, pool, settings=settings)
    assert "6" not in set(after.feat_feature_id)


def test_negative_mode_alignment_uses_other_compositions_and_is_optional(settings):
    before, pool = fixture()
    negative = pool.copy().assign(adduct="[M-H]-")
    negative["feat_rt"] -= 8 / 60
    after, _, _, _, details = complete_rp_positive_pe(
        before, pool, negative, settings=settings
    )
    assert len(after) == 8
    assert details.pe_cross_polarity_supported.all()
    assert details.pe_polarity_support_count.ge(4).all()
    assert details.pe_polarity_offset_seconds.round(8).eq(8).all()
    without, _, _, _, _ = complete_rp_positive_pe(before, pool, settings=settings)
    assert set(after.feat_feature_id) == set(without.feat_feature_id)


def test_no_hard_odd_carbon_whitelist_and_repeated_pass_is_idempotent(settings):
    before, pool = fixture(9, odd=True)
    after, candidates, _, _, _ = complete_rp_positive_pe(
        before, pool, settings=settings
    )
    assert len(after) == 10
    repeated, _, _, _, _ = complete_rp_positive_pe(after, candidates, settings=settings)
    assert_frame_equal(repeated, after)


def test_missing_model_fails_closed(settings):
    before, pool = fixture()
    after, candidates, audit, _, _ = complete_rp_positive_pe(
        before,
        pool,
        settings=replace(
            settings, model_path=settings.model_path.parent / "absent.json"
        ),
    )
    assert_frame_equal(after, before)
    assert_frame_equal(candidates, pool)
    assert audit.status.iloc[0] == "model_unavailable"


@pytest.mark.parametrize(
    "change",
    [
        dict(ppm_tolerance=4),
        dict(rt_tolerance_seconds=11),
        dict(minimum_support=3),
        dict(maximum_anchors=4),
        dict(local_ppm=-1),
        dict(feature_margin=-1),
    ],
)
def test_invalid_settings_are_rejected(change, settings):
    before, pool = fixture()
    with pytest.raises(ValueError):
        complete_rp_positive_pe(before, pool, settings=replace(settings, **change))


def test_public_settings_default_and_disable():
    assert build_automatic_anchor_settings({}).use_pe_completion
    assert not build_automatic_anchor_settings(
        {"use_pe_completion": False}
    ).use_pe_completion
    assert build_automatic_anchor_settings({}).pe_completion_confidence_threshold == 0.9


def test_pe_audits_and_evidence_survive_saved_project_roundtrip(tmp_path, settings):
    from saved_project import save_standard_project, load_project_table
    from completed_project import load_completed_project

    before, pool = fixture()
    after, candidates, audit, anchors, detail = complete_rp_positive_pe(
        before, pool, settings=settings
    )
    result = {
        "engine": "matlab",
        "automatic_annotations": after,
        "identified_lipids": after,
        "equivalent_lipids": pd.DataFrame(),
        "all_candidates": candidates,
        "matlab_pe_completion_audit": audit,
        "matlab_pe_anchors": anchors,
        "matlab_pe_candidate_decisions": detail,
        "streams": {
            "RP_Pos": {
                "annotations": after,
                "all_pairs": candidates,
                "pe_completion_audit": audit,
                "pe_anchors": anchors,
            }
        },
    }
    root = tmp_path / "saved"
    save_standard_project(root, result)
    restored = load_completed_project(root)
    assert len(load_project_table(restored, "automatic_annotations")) == len(after)
    assert len(load_project_table(restored, "matlab_pe_anchors")) == len(anchors)
    assert len(load_project_table(restored, "matlab_pe_candidate_decisions")) == len(
        detail
    )
    restored_calls = load_project_table(restored, "automatic_annotations")
    assert int(restored_calls.pe_completion_supported.fillna(False).sum()) == 7
