from dataclasses import replace
import json

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from annotator.composition_ranges import composition_range_settings_from_dict
from adapters.standard_engines import build_automatic_anchor_settings
from matlab_annotation.pe_rebuild import (
    PERebuildSettings,
    pe_search_reference,
    prepare_pe_rebuild_candidates,
    pe_rebuild_evidence,
    rebuild_rp_positive_pe,
    score_pe_rebuild,
    complete_pe_stream,
)


@pytest.fixture
def inputs(tmp_path):
    refs = pd.DataFrame(
        [
            dict(
                name=f"PE({30 + 2 * i}:2)",
                **{"class": "PE"},
                adduct="[M+H]+",
                carbon=30 + 2 * i,
                db=2,
                ref_mz=700 + i * 28.031300128,
                ref_rt=10 + i * 0.25,
                matlab_code=i + 1,
                matlab_compound=45,
            )
            for i in range(7)
        ]
    )
    features = pd.DataFrame(
        dict(
            feature_id=[str(i + 1) for i in range(7)],
            mz=refs.ref_mz + 0.0001,
            rt=refs.ref_rt + 1 / 60,
            intensity=np.linspace(1e7, 1e6, 7),
        )
    )
    old = pd.DataFrame(
        [
            dict(
                name="PC(34:1)",
                **{"class": "PC"},
                adduct="[M+H]+",
                feat_feature_id="old",
                feat_rt=5.0,
                ppm_error=0.1,
                rt_error=0.01,
                automatic_accepted=True,
            )
        ]
    )
    pool = pd.DataFrame(
        dict(
            name=refs.name,
            **{"class": refs["class"]},
            adduct=refs.adduct,
            feat_feature_id=features.feature_id,
            automatic_accepted=False,
            grid_coherence_pass=False,
        )
    )
    ranges = composition_range_settings_from_dict(
        dict(
            use_group_ranges=False,
            classes={"PE": dict(carbon_min=30, carbon_max=42, db_min=2, db_max=2)},
        )
    )
    model = tmp_path / "pe.json"
    model.write_text(
        json.dumps(
            dict(
                format="laser-pe-rebuild-v1",
                features=[],
                intercept=5.0,
                learning_rate=0.04,
                trees=[],
            )
        )
    )
    return (
        old,
        pool,
        refs,
        features,
        ranges,
        replace(PERebuildSettings(), model_path=model),
    )


def run(inputs, **overrides):
    old, pool, ref, features, ranges, settings = inputs
    args = dict(
        annotations=old,
        all_pairs=pool,
        reference=ref,
        features=features,
        negative_annotations=pd.DataFrame(),
        negative_features=pd.DataFrame(),
        ranges=ranges,
        settings=settings,
    )
    args.update(overrides)
    return rebuild_rp_positive_pe(**args)


def test_reconsiders_raw_pe_after_old_grid_rejected_it(inputs):
    old, pool, *_ = inputs
    before = pool.copy(deep=True)
    after, pairs, audit, anchors, detail = run(inputs)
    assert len(after) == 8 and int(audit.added_pe.iloc[0]) == 7
    assert_frame_equal(after.iloc[: len(old)][old.columns], old, check_dtype=False)
    assert_frame_equal(pool, before)
    assert pairs.automatic_accepted.all()
    assert after.iloc[1:].pe_final_selected.all()
    assert detail.pe_final_selected.all() and detail.pe_decision.eq("accepted").all()
    assert after.iloc[1:].rank_for_feature.eq(1).all()
    assert len(anchors) == 7
    for row in detail.itertuples():
        assert row.name not in row.pe_support_compositions.split(";")
        assert row.pe_support_count >= 5


def test_broader_discovery_is_not_a_mass_tolerance_bypass(inputs):
    old, pool, ref, features, ranges, settings = inputs
    features = features.copy()
    features.loc[3, "mz"] = float(ref.ref_mz.iloc[3]) * (1 + 3.1e-6)
    after, *_ = run(inputs, features=features)
    assert "4" not in set(after.feat_feature_id)


def test_mass_correction_uses_many_other_class_peaks_not_target(inputs):
    old, pool, ref, features, ranges, settings = inputs
    cal = pd.concat(
        [old.assign(feat_feature_id=f"cal{i}", ppm_error=2.0) for i in range(20)],
        ignore_index=True,
    )
    shifted = features.copy()
    shifted.mz = ref.ref_mz * (1 + 3e-6)
    shifted.loc[6, "mz"] = ref.ref_mz.iloc[6] * (1 + 4e-6)
    after, _, audit, _, detail = run(inputs, annotations=cal, features=shifted)
    assert len(after) == 27
    assert audit.pe_mass_center_ppm.iloc[0] == 2.0
    assert np.allclose(detail.pe_corrected_mass_error_ppm.iloc[:-1], 1.0, atol=1e-8)
    assert np.isclose(detail.pe_corrected_mass_error_ppm.iloc[-1], 2.0, atol=1e-8)
    no_cal, *_ = run(inputs, features=shifted)
    assert len(no_cal) == len(old)


def test_does_not_repurpose_features_standards_or_existing_pe(inputs):
    old, pool, ref, features, ranges, settings = inputs
    protected = pd.concat(
        [
            old,
            old.assign(feat_feature_id="3"),
            old.assign(name="PE(38:2)", **{"class": "PE"}, feat_feature_id="5"),
        ],
        ignore_index=True,
    )
    after, *_ = run(inputs, annotations=protected)
    assert_frame_equal(after.iloc[:3][protected.columns], protected, check_dtype=False)
    assert (after.feat_feature_id == "3").sum() == 1
    assert (after["name"] == "PE(38:2)").sum() == 1
    repeated = run(inputs, annotations=after)[0]
    assert_frame_equal(repeated, after)


def test_ambiguous_peaks_stay_unselected(inputs):
    old, pool, ref, features, *_ = inputs
    split = features.iloc[[3]].assign(feature_id="split")
    after, _, _, _, details = run(
        inputs, features=pd.concat([features, split], ignore_index=True)
    )
    assert not {"4", "split"} & set(after.feat_feature_id)
    assert (
        details.loc[details.feat_feature_id.isin(["4", "split"]), "pe_decision"]
        .eq("ambiguous_peak_choice")
        .all()
    )


@pytest.mark.parametrize("stream", ["NP_Pos", "NP_Neg", "RP_Neg"])
def test_other_streams_are_exact_noops(inputs, stream):
    old, pool, *_ = inputs
    after, pairs, *_ = run(inputs, stream=stream)
    assert_frame_equal(after, old)
    assert_frame_equal(pairs, pool)


def test_disabled_or_unavailable_model_is_exact_noop(inputs):
    old, pool, *_, settings = inputs
    for s in [
        replace(settings, enabled=False),
        replace(settings, model_path=settings.model_path.with_name("absent")),
    ]:
        after, pairs, *_ = run(inputs, settings=s)
        assert_frame_equal(after, old)
        assert_frame_equal(pairs, pool)


def test_raw_negative_evidence_does_not_require_a_negative_annotation(inputs):
    old, pool, ref, features, ranges, settings = inputs
    neg = features.copy()
    neg.mz = ref.ref_mz - 2.014552
    neg.rt = features.rt - 0.01
    _, _, _, _, detail = run(inputs, negative_features=neg)
    assert detail.pe_negative_raw_intensity_2s.gt(0).all()
    assert detail.pe_negative_rt.isna().all()


def test_independent_rt_rebuild_can_accept_a_peak_outside_old_ten_seconds(inputs):
    features = inputs[3].copy()
    features.rt += 14 / 60
    after, _, _, _, checks = run(inputs, features=features)
    assert len(after) == 8
    assert checks.rt_error.abs().mul(60).gt(10).all()
    assert checks.pe_support_count.ge(5).all()


def test_negative_evidence_can_choose_a_weaker_positive_peak(inputs):
    old, pool, ref, features, ranges, settings = inputs
    split = features.iloc[[3]].assign(
        feature_id="stronger", rt=features.rt.iloc[3] + 8 / 60
    )
    split.intensity *= 10
    features = pd.concat([features, split], ignore_index=True)
    negative = inputs[3].copy()
    negative.mz = ref.ref_mz - 2.014552
    settings.model_path.write_text(
        json.dumps(
            dict(
                format="laser-pe-rebuild-v1",
                features=["neg_present5"],
                intercept=0.0,
                learning_rate=1.0,
                trees=[
                    dict(
                        left=[1, -1, -1],
                        right=[2, -1, -1],
                        feature=[0, -2, -2],
                        threshold=[0.5, -2.0, -2.0],
                        value=[0.0, -5.0, 5.0],
                    )
                ],
            )
        )
    )
    after, _, _, _, _ = run(inputs, features=features, negative_features=negative)
    assert "4" in set(after.feat_feature_id)
    assert "stronger" not in set(after.feat_feature_id)


def test_user_composition_limits_apply_to_extrapolated_targets(inputs):
    _, _, ref, _, ranges, _ = inputs
    expanded = composition_range_settings_from_dict(
        dict(
            use_group_ranges=False,
            classes={"PE": dict(carbon_min=28, carbon_max=44, db_min=2, db_max=2)},
        )
    )
    targets = pe_search_reference(ref, expanded)
    assert targets.carbon.min() == 28 and targets.carbon.max() == 44
    assert targets.db.eq(2).all()
    assert targets.loc[targets.carbon.eq(28), "matlab_code"].isna().all()
    assert targets.loc[targets.carbon.eq(28), "name"].iloc[0] == "PE(28:2)"


def test_features_exclude_identifiers_and_curated_labels(inputs):
    old, _, ref, features, ranges, settings = inputs
    pool, _, _ = prepare_pe_rebuild_candidates(
        old,
        ref,
        features,
        pd.DataFrame(),
        pd.DataFrame(),
        ranges=ranges,
        settings=settings,
    )
    x = pe_rebuild_evidence(pool)
    changed = pool.assign(
        correct=True,
        dataset="test",
        feat_feature_id="another",
        name=pool["name"] + " renamed",
    )
    assert_frame_equal(x, pe_rebuild_evidence(changed))
    assert np.isfinite(x.to_numpy()).all()


def test_portable_tree_prediction_and_unknown_model_rejection():
    x = pd.DataFrame({"test": [-1.0, 1.0]})
    payload = dict(
        format="laser-pe-rebuild-v1",
        features=["test"],
        intercept=0.0,
        learning_rate=1.0,
        trees=[
            dict(
                left=[1, -1, -1],
                right=[2, -1, -1],
                feature=[0, -2, -2],
                threshold=[0.0, -2.0, -2.0],
                value=[0.0, -2.0, 2.0],
            )
        ],
    )
    np.testing.assert_allclose(
        score_pe_rebuild(x, payload), [1 / (1 + np.exp(2)), 1 / (1 + np.exp(-2))]
    )
    with pytest.raises(ValueError):
        score_pe_rebuild(x, {"format": "unknown"})
    payload["intercept"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        score_pe_rebuild(x, payload)


@pytest.mark.parametrize(
    "changes",
    [
        dict(score_threshold=-0.1),
        dict(feature_margin=float("nan")),
        dict(ppm_tolerance=4),
        dict(search_seconds=61),
        dict(search_seconds=float("nan")),
    ],
)
def test_invalid_settings_fail(inputs, changes):
    with pytest.raises(ValueError):
        run(inputs, settings=replace(inputs[-1], **changes))


def test_public_settings_can_restore_earlier_pe_method():
    assert build_automatic_anchor_settings({}).use_pe_rebuild
    s = build_automatic_anchor_settings(
        {
            "use_pe_rebuild": False,
            "pe_rebuild_score_threshold": 0.25,
            "pe_rebuild_feature_margin": 0.05,
            "pe_rebuild_search_seconds": 30,
        }
    )
    assert not s.use_pe_rebuild and s.pe_rebuild_score_threshold == 0.25
    assert s.pe_rebuild_feature_margin == 0.05 and s.pe_rebuild_search_seconds == 30


def test_saved_projects_and_grid_keep_rebuilt_pe_as_full_annotations(inputs, tmp_path):
    from completed_project import load_completed_project
    from saved_project import load_project_table, save_standard_project
    from standard_lipidomics.grid_browser import (
        _accepted_annotation_keys,
        annotation_identity,
    )

    after, pairs, audit, anchors, decisions = run(inputs)
    added = after.iloc[1:].copy()
    assert set(annotation_identity(added)) <= _accepted_annotation_keys(pairs)
    result = {
        "engine": "matlab",
        "automatic_annotations": after,
        "identified_lipids": after,
        "equivalent_lipids": pd.DataFrame(),
        "all_candidates": pairs,
        "matlab_pe_completion_audit": audit,
        "matlab_pe_anchors": anchors,
        "matlab_pe_candidate_decisions": decisions,
        "streams": {"RP_Pos": {"annotations": after, "all_pairs": pairs}},
    }
    save_standard_project(tmp_path / "project", result)
    saved = load_completed_project(tmp_path / "project")
    calls = load_project_table(saved, "automatic_annotations")
    checks = load_project_table(saved, "matlab_pe_candidate_decisions")
    assert calls.pe_final_selected.fillna(False).sum() == 7
    assert checks.pe_final_selected.sum() == 7


@pytest.mark.parametrize("stream", ["NP_Pos", "RP_Neg"])
def test_other_supported_streams_use_their_own_models_and_correct_adduct(
    inputs, stream
):
    old, pairs, refs, features, ranges, settings = inputs
    model = json.loads(settings.model_path.read_text())
    model["stream"] = stream
    settings.model_path.write_text(json.dumps(model))
    refs = refs.copy()
    if stream == "RP_Neg":
        refs.adduct = "[M-H]-"
    after, _, audit, _, _ = complete_pe_stream(
        old,
        pairs,
        refs,
        features,
        pd.DataFrame(),
        pd.DataFrame(),
        stream=stream,
        ranges=ranges,
        settings=settings,
    )
    assert len(after) == 8
    assert audit.stream.eq(stream).all()
    assert after.iloc[1:].chromatography.eq(stream.split("_")[0]).all()
    assert after.iloc[1:].adduct.eq("[M-H]-" if stream == "RP_Neg" else "[M+H]+").all()


def test_negative_stream_searches_positive_counterpart_at_higher_mass(inputs):
    old, pairs, refs, features, ranges, settings = inputs
    refs = refs.copy()
    refs.adduct = "[M-H]-"
    positive = features.copy()
    positive.mz = refs.ref_mz + 2.014552
    pool, _, _ = prepare_pe_rebuild_candidates(
        old,
        refs,
        features,
        pd.DataFrame(),
        positive,
        ranges=ranges,
        settings=settings,
        target_adduct="[M-H]-",
    )
    assert pool.pe_negative_raw_intensity_2s.gt(0).all()


def test_model_from_another_stream_fails_closed(inputs):
    old, pairs, refs, features, ranges, settings = inputs
    after, candidates, audit, _, _ = complete_pe_stream(
        old,
        pairs,
        refs,
        features,
        pd.DataFrame(),
        pd.DataFrame(),
        stream="NP_Pos",
        settings=settings,
        ranges=ranges,
    )
    assert_frame_equal(after, old)
    assert_frame_equal(candidates, pairs)
    assert audit.status.iloc[0] == "model_unavailable"


def test_review_grid_displays_the_accepted_pe_peak_and_its_actual_evidence(inputs):
    from standard_lipidomics.grid_browser import prepare_standard_grid_data, build_class_grid_state

    after, pairs, _, _, _ = run(inputs)
    refs = inputs[2]
    pairs = pairs.copy()
    index = pairs.index[pairs.feat_feature_id.eq("4")][0]
    pairs.loc[index, "ppm_error"] = 4.0
    pairs.loc[index, "pe_corrected_mass_error_ppm"] = 0.5
    pairs.loc[index, "pe_local_rt_error_seconds"] = 0.1
    after = after.copy()
    ai = after.index[after.feat_feature_id.eq("4")][0]
    for col in ["ppm_error", "pe_corrected_mass_error_ppm", "pe_local_rt_error_seconds"]:
        after.loc[ai, col] = pairs.loc[index, col]
    alternative = pairs.loc[[index]].copy()
    alternative["feat_feature_id"] = "closer_to_old_reference"
    alternative["feat_rt"] = alternative.ref_rt
    alternative["ppm_error"] = 0.01
    alternative["automatic_accepted"] = False
    alternative["automatic_candidate"] = False
    alternative["auto_annotation_status"] = "not_selected"
    pairs = pd.concat([pairs, alternative], ignore_index=True)
    reference, candidates = prepare_standard_grid_data(dict(
        reference=refs, all_pairs=pairs, annotations=after, rt_models=pd.DataFrame(),
    ))
    criteria = dict(ppm_tolerance=3.0, rt_tolerance=1/60, use_rt=True)
    state = build_class_grid_state(reference,candidates,lipid_class="PE",adduct="[M+H]+",
                                   confirmed=set(),run_criteria=criteria,review=criteria)
    point = state["plot_df"].loc[state["plot_df"]["name"].eq("PE(36:2)")].iloc[0]
    assert point.feat_feature_id == "4"
    assert point.automatic_candidate and point.run_candidate
    assert point.ppm_error == 4.0  # Raw display value is not falsified.
    assert np.isclose(point.grid_resid,0.1/60)
