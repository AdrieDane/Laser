from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from adapters.standard_engines import build_automatic_anchor_settings
from matlab_annotation.experimental_pe import (
    ExperimentalPESelectionSettings,
    MODEL_FILENAMES,
    apply_experimental_pe_selector,
    experimental_pe_evidence,
    pair_preference,
    rank_experimental_pe,
    refine_experimental_pe_stream,
    select_experimental_pe,
)


def model():
    # Prefer lower corrected mass error. The symmetric score cannot prefer
    # both sides of a duel or turn a constant model into false certainty.
    return dict(
        format="laser-pe-rebuild-v1",
        features=["mass_error"],
        intercept=0.0,
        learning_rate=1.0,
        trees=[
            dict(
                left=[1, -1, -1],
                right=[2, -1, -1],
                feature=[0, -2, -2],
                threshold=[0.0, -2.0, -2.0],
                value=[0.0, 6.0, -6.0],
            )
        ],
    )


def pool(stream="RP_Pos"):
    rows = []
    for fid, error, rt_error in [("old", 1.5, 4.0), ("new", 0.2, 1.0)]:
        row = dict(
            name="PE(36:2)",
            **{"class": "PE"},
            adduct="[M-H]-" if stream.endswith("Neg") else "[M+H]+",
            carbon=36,
            db=2,
            ref_mz=750.0,
            ref_rt=10.0,
            feat_feature_id=fid,
            feat_mz=750 * (1 + error * 1e-6),
            feat_rt=10 + rt_error / 60,
            feat_intensity=1e7,
            ppm_error=error,
            rt_error=rt_error / 60,
            pe_corrected_mass_error_ppm=error,
            pe_local_rt_error_seconds=rt_error,
            pe_support_count=8,
            pe_support_distance=1.0,
            pe_negative_rt=10.0,
            pe_negative_intensity=1e7,
            isotopologue="",
            is_is=False,
            automatic_accepted=False,
            automatic_candidate=False,
            grid_coherence_assessed=True,
            grid_coherence_score=0.1,
            grid_coherence_pass=False,
            q_profile_samples=4,
            q_detection_fraction=1.0,
            q_log_median=15.0,
            q_isotope_upper_correlation=0.95,
            q_isotope_upper_log_ratio=np.log(0.44),
            q_isotope_lower_correlation=np.nan,
            q_isotope_lower_log_ratio=np.nan,
            q_log_sample_blank_ratio=np.log(10),
        )
        for window in (2, 5, 10, 20):
            row[f"pe_negative_raw_intensity_{window}s"] = 1e7
            row[f"pe_negative_raw_rt_error_{window}s"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def baseline(p):
    old = p.iloc[[0]].copy()
    old["automatic_accepted"], old["automatic_candidate"] = True, True
    old["auto_annotation_status"] = "previous_automatic_method"
    return old


def run(p=None, ann=None, pairs=None, stream="RP_Pos", settings=None):
    p = pool(stream) if p is None else p
    ann = baseline(p) if ann is None else ann
    pairs = pd.concat([ann, p.iloc[1:]], ignore_index=True) if pairs is None else pairs
    return select_experimental_pe(
        ann,
        pairs,
        p,
        stream=stream,
        model=model(),
        settings=settings or ExperimentalPESelectionSettings(enabled=True),
    )


def test_pair_preference_is_antisymmetric_and_constant_ties_have_no_certainty():
    a, b = np.array([[0.1]]), np.array([[1.0]])
    assert pair_preference(a, b, model())[0] > 0.99
    assert np.isclose(
        pair_preference(a, b, model())[0] + pair_preference(b, a, model())[0], 1
    )
    assert pair_preference(a, a, model())[0] == 0.5


def test_batched_rank_scores_match_individual_duels():
    p = pool()
    second = p.copy()
    second["name"] = "PE(38:2)"
    second["feat_feature_id"] = ["other-old", "other-new"]
    p = pd.concat([p, second], ignore_index=True)
    ranked, x = rank_experimental_pe(p, model())
    for _, group in p.groupby("name"):
        a, b = group.index
        score = pair_preference(x.loc[[a]].to_numpy(), x.loc[[b]].to_numpy(), model())[
            0
        ]
        assert ranked.loc[a, "pe_experimental_rank_score"] == pytest.approx(score)
        assert ranked.loc[b, "pe_experimental_rank_score"] == pytest.approx(1 - score)


def test_experimental_feature_evidence_contains_no_manual_labels_or_identifiers():
    p = pool()
    p["correct"] = True
    p["dataset"] = "hidden_truth"
    p["was_selected"] = True
    e = experimental_pe_evidence(p)
    assert not {"correct", "dataset", "was_selected", "feat_feature_id", "name"} & set(
        e
    )
    assert all(np.isfinite(e.to_numpy().ravel()))


@pytest.mark.parametrize("stream", ["RP_Pos", "RP_Neg", "NP_Neg"])
def test_strong_supported_replacement_preserves_count_and_raw_evidence(stream):
    p = pool(stream)
    a = baseline(p)
    pairs = pd.concat([a, p.iloc[1:]], ignore_index=True)
    before_a, before_pairs = a.copy(deep=True), pairs.copy(deep=True)
    after, view, decisions, changes, audit = run(p, a, pairs, stream=stream)
    assert (
        after.feat_feature_id.tolist() == ["new"]
        and audit["replaced_PE"] == 1
        and audit["added_PE"] == 0
    )
    row = after.iloc[0]
    assert row.automatic_accepted and row.pe_experimental_evidence_pass
    assert (
        row.ppm_error == 0.2
        and not row.grid_coherence_pass
        and row.grid_coherence_score == 0.1
    )
    assert changes.old_feature_id.tolist() == [
        "old"
    ] and changes.new_feature_id.tolist() == ["new"]
    assert not view.loc[view.feat_feature_id.eq("old"), "automatic_accepted"].any()
    assert decisions.pe_experimental_final_selected.tolist() == [False, True]
    assert_frame_equal(a, before_a)
    assert_frame_equal(pairs, before_pairs)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("manual_confirmed", True),
        ("manual_override", True),
        ("annotation_source", "manual"),
        ("is_is", True),
        ("internal_standard_name", "PE spike"),
    ],
)
def test_manual_confirmations_and_standards_are_never_replaced(flag, value):
    p = pool()
    a = baseline(p)
    a[flag] = value
    after, _, _, changes, _ = run(p, ann=a)
    assert after.feat_feature_id.tolist() == ["old"] and changes.empty


def test_false_string_does_not_protect_all_csv_rows():
    p = pool()
    a = baseline(p)
    a["manual_confirmed"] = "False"
    assert run(p, ann=a)[0].feat_feature_id.tolist() == ["new"]


@pytest.mark.parametrize("other_class", ["PC", "PE"])
def test_other_accepted_features_cannot_be_consumed(other_class):
    p = pool()
    a = baseline(p)
    other = p.iloc[[1]].copy()
    other["name"] = "another_lipid"
    other["class"] = other_class
    a = pd.concat([a, other], ignore_index=True)
    after, _, _, changes, _ = run(p, ann=a)
    assert changes.empty and set(after.feat_feature_id) == {"old", "new"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("pe_corrected_mass_error_ppm", 3.0),
        ("pe_local_rt_error_seconds", 11.0),
        ("pe_support_count", 4),
        ("q_profile_samples", 3),
        ("adduct", "[M+Na]+"),
        ("isotopologue", "13C"),
        ("class", "PC"),
    ],
)
def test_candidate_gates_cannot_be_bypassed_by_high_pair_score(field, value):
    p = pool()
    p.loc[1, field] = value
    after, _, _, changes, _ = run(p)
    assert changes.empty and after.feat_feature_id.tolist() == ["old"]


def test_mass_RT_dominance_guard_preserves_incumbent_despite_model_preference():
    p = pool()
    p.loc[1, "pe_local_rt_error_seconds"] = 6.0
    after, _, detail, changes, _ = run(p)
    assert changes.empty and after.feat_feature_id.tolist() == ["old"]
    assert detail.pe_experimental_decision.eq("withheld_mass_RT_dominance_guard").all()


def test_RP_positive_needs_both_independent_fit_improvements():
    p = pool()
    p.loc[1, "pe_corrected_mass_error_ppm"] = 1.31  # Only 0.19 ppm better.
    after, _, _, changes, _ = run(p)
    assert changes.empty and after.feat_feature_id.tolist() == ["old"]
    p = pool("RP_Neg")
    p.loc[1, "pe_local_rt_error_seconds"] = (
        4.05  # Neutral RT, substantially better mass.
    )
    assert run(p, stream="RP_Neg")[0].feat_feature_id.tolist() == ["new"]


def test_supported_isotope_child_is_not_promoted():
    p = pool()
    p.loc[1, "q_isotope_lower_correlation"] = 0.99
    p.loc[1, "q_isotope_lower_log_ratio"] = -np.log(0.44)
    after, _, _, changes, _ = run(p)
    assert changes.empty and after.feat_feature_id.tolist() == ["old"]


def test_NP_negative_addition_needs_independent_admission_evidence():
    p = pool("NP_Neg").iloc[[1]].reset_index(drop=True)
    empty = baseline(p).iloc[:0]
    after, _, _, changes, audit = run(p, ann=empty, pairs=p, stream="NP_Neg")
    assert after.feat_feature_id.tolist() == ["new"] and audit["added_PE"] == 1
    assert changes.change.tolist() == ["addition"]
    assert changes.isotope_support.all() and changes.opposite_mode_support.all()
    p["q_isotope_upper_correlation"] = np.nan
    p["pe_negative_rt"] = np.nan
    assert run(p, ann=empty, pairs=p, stream="NP_Neg")[0].empty


@pytest.mark.parametrize(
    "field,value",
    [
        ("pe_corrected_mass_error_ppm", 2.5),
        ("pe_local_rt_error_seconds", 5.01),
        ("q_detection_fraction", 0.49),
        ("q_log_sample_blank_ratio", 0.0),
    ],
)
def test_new_NP_negative_calls_have_stricter_gates(field, value):
    p = pool("NP_Neg").iloc[[1]].reset_index(drop=True)
    p.loc[0, field] = value
    assert run(p, ann=baseline(p).iloc[:0], pairs=p, stream="NP_Neg")[0].empty


def test_no_unrequested_new_RP_identities():
    p = pool().iloc[[1]].reset_index(drop=True)
    assert run(p, ann=baseline(p).iloc[:0], pairs=p)[0].empty


def test_disabled_and_NP_positive_are_exact_no_ops():
    p = pool()
    a = baseline(p)
    result = run(p, ann=a, pairs=p, settings=ExperimentalPESelectionSettings())
    assert_frame_equal(result[0], a)
    assert_frame_equal(result[1], p)
    assert run(p, ann=a, pairs=p, stream="NP_Pos")[-1]["status"] == "not_applicable"


def test_wrong_stream_or_missing_models_fail_closed(tmp_path):
    p = pool()
    a = baseline(p)
    settings = ExperimentalPESelectionSettings(enabled=True, model_directory=tmp_path)
    for wrong in [
        None,
        dict(
            format="laser-experimental-pe-selector-v1",
            stream="RP_Neg",
            pair_model=model(),
        ),
    ]:
        if wrong:
            (tmp_path / MODEL_FILENAMES["RP_Pos"]).write_text(json.dumps(wrong))
        result = refine_experimental_pe_stream(
            a, p, p, p, p, p, stream="RP_Pos", settings=settings
        )
        assert_frame_equal(result[0], a)
        assert_frame_equal(result[1], p)
        assert result[-1].status.iloc[0] == "experiment_unavailable"


def test_invalid_inference_fails_closed_without_partial_changes(tmp_path, monkeypatch):
    import matlab_annotation.experimental_pe as module

    p = pool()
    a = baseline(p)
    bad = model()
    bad["intercept"] = float("nan")
    payload = dict(
        format="laser-experimental-pe-selector-v1", stream="RP_Pos", pair_model=bad
    )
    (tmp_path / MODEL_FILENAMES["RP_Pos"]).write_text(json.dumps(payload))
    monkeypatch.setattr(
        module,
        "prepare_experimental_pe_candidates",
        lambda *args, **kwargs: (p.copy(), pd.DataFrame(), {}),
    )
    result = refine_experimental_pe_stream(
        a,
        p,
        p,
        p,
        p,
        p,
        stream="RP_Pos",
        settings=ExperimentalPESelectionSettings(
            enabled=True, model_directory=tmp_path
        ),
    )
    assert_frame_equal(result[0], a)
    assert_frame_equal(result[1], p)
    assert result[-1].status.iloc[0] == "experiment_unavailable"


def test_counterpart_annotations_are_frozen_before_any_stream_changes(monkeypatch):
    import matlab_annotation.experimental_pe as module

    old_pos = baseline(pool())
    old_neg = baseline(pool("RP_Neg"))
    old_pos["feat_feature_id"] = "pos-before"
    old_neg["feat_feature_id"] = "neg-before"
    result = {
        "streams": {
            s: dict(
                annotations=a,
                all_pairs=a.copy(),
                reference=pd.DataFrame(),
                features=pd.DataFrame(),
            )
            for s, a in [("RP_Pos", old_pos), ("RP_Neg", old_neg)]
        }
    }
    observed = {}

    def refine(
        ann,
        pairs,
        reference,
        features,
        counterpart,
        counterpart_features,
        *,
        stream,
        **kwargs,
    ):
        observed[stream] = counterpart.feat_feature_id.tolist()
        ann = ann.copy()
        ann["feat_feature_id"] = "changed-" + stream
        return (
            ann,
            pairs,
            pd.DataFrame(),
            pd.DataFrame([dict(change="replacement")]),
            pd.DataFrame(),
        )

    monkeypatch.setattr(module, "refine_experimental_pe_stream", refine)
    apply_experimental_pe_selector(
        result, settings=ExperimentalPESelectionSettings(enabled=True)
    )
    assert observed == {"RP_Pos": ["neg-before"], "RP_Neg": ["pos-before"]}
    assert result["streams"]["RP_Pos"]["annotations"].feat_feature_id.tolist() == [
        "pos-before"
    ]


def test_default_public_setting_and_wrapper_preserve_current_results():
    assert not build_automatic_anchor_settings({}).use_experimental_pe_selector
    assert build_automatic_anchor_settings(
        {"use_experimental_pe_selector": True}
    ).use_experimental_pe_selector
    result = {"streams": {}, "marker": "preserved"}
    assert apply_experimental_pe_selector(result) == result
    result["experimental_pe_selector_applied"] = True
    assert (
        apply_experimental_pe_selector(
            result, settings=ExperimentalPESelectionSettings(enabled=True)
        )
        == result
    )


def test_text_false_never_accidentally_enables_experimental_selection():
    assert not build_automatic_anchor_settings(
        {"use_experimental_pe_selector": "false"}
    ).use_experimental_pe_selector
    assert build_automatic_anchor_settings(
        {"use_experimental_pe_selector": "true"}
    ).use_experimental_pe_selector
    with pytest.raises(ValueError):
        build_automatic_anchor_settings({"use_experimental_pe_selector": "maybe"})


def test_grid_displays_actual_experimental_choice_without_falsifying_old_coherence():
    from standard_lipidomics.grid_browser import (
        prepare_standard_grid_data,
        build_class_grid_state,
    )

    p = pool()
    after, pairs, _, _, _ = run(p)
    # Raw ppm >3 can be independently corrected below 3. Keep it visible.
    pairs.loc[pairs.feat_feature_id.eq("new"), "ppm_error"] = 4.0
    after.loc[:, "ppm_error"] = 4.0
    refs = p[
        ["name", "class", "adduct", "carbon", "db", "ref_mz", "ref_rt"]
    ].drop_duplicates("name")
    reference, candidates = prepare_standard_grid_data(
        dict(
            reference=refs, all_pairs=pairs, annotations=after, rt_models=pd.DataFrame()
        )
    )
    criteria = dict(
        ppm_tolerance=3.0,
        rt_tolerance=0.5,
        use_rt=True,
        use_grid_coherence=True,
        grid_coherence_threshold=0.45,
    )
    view = build_class_grid_state(
        reference,
        candidates,
        lipid_class="PE",
        adduct="[M+H]+",
        confirmed=set(),
        run_criteria=criteria,
        review=criteria,
    )
    assert view["plot_df"].feat_feature_id.tolist() == ["new"]
    assert view["plot_df"].ppm_error.iloc[0] == 4.0


def test_experimental_audits_and_final_flags_survive_saved_project(tmp_path):
    from completed_project import load_completed_project
    from saved_project import load_project_table, save_standard_project

    p = pool()
    after, pairs, decisions, changes, audit = run(p)
    result = dict(
        engine="matlab",
        automatic_annotations=after,
        identified_lipids=after,
        equivalent_lipids=pd.DataFrame(),
        all_candidates=pairs,
        matlab_experimental_pe_audit=pd.DataFrame([audit]),
        matlab_experimental_pe_decisions=decisions,
        matlab_experimental_pe_changes=changes,
        streams={"RP_Pos": dict(annotations=after, all_pairs=pairs)},
    )
    save_standard_project(tmp_path / "project", result)
    saved = load_completed_project(tmp_path / "project")
    assert load_project_table(
        saved, "automatic_annotations"
    ).pe_experimental_evidence_pass.all()
    assert load_project_table(
        saved, "matlab_experimental_pe_changes"
    ).new_feature_id.tolist() == ["new"]


@pytest.mark.parametrize(
    "settings",
    [
        ExperimentalPESelectionSettings(ppm_tolerance=4),
        ExperimentalPESelectionSettings(rt_seconds=11),
        ExperimentalPESelectionSettings(minimum_pair_score=0.4),
    ],
)
def test_experimental_settings_do_not_silently_relax_limits(settings):
    with pytest.raises(ValueError):
        apply_experimental_pe_selector({}, settings=settings)


def _calibration_tool():
    import importlib.util
    from pathlib import Path

    pytest.importorskip("sklearn")
    path = Path(__file__).resolve().parents[2] / "tools/calibrate_experimental_pe.py"
    spec = importlib.util.spec_from_file_location(
        "experimental_pe_calibration_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_excludes_unlisted_identities_and_internal_standard_labels():
    tool = _calibration_tool()
    known = pool()
    unknown = pool()
    standard = pool()
    unknown["name"] = "PE(40:2)"
    standard["name"] = "PE(28:0)"
    p = pd.concat([known, unknown, standard], ignore_index=True)
    cross = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                matlab_class="PE",
                matlab_species=n,
                xcms_feature_id="new",
                crosswalk_status="profile_match",
                matlab_is_internal_standard=standard_flag,
            )
            for n, standard_flag in [("PE(36:2)", "False"), ("PE(28:0)", "True")]
        ]
    )
    x, y, weights, columns, groups = tool.training_pairs(p, cross, "RP_Pos")
    assert len(x) == 2 and list(y) == [1, 0] and groups == 1
    assert not {"correct", "feature_id", "dataset", "name"} & set(columns)
    assert np.allclose(x[0], -x[1]) and list(weights) == [1, 1]


def test_fitted_pair_model_roundtrip_uses_portable_inference():
    tool = _calibration_tool()
    data = []
    truth = []
    for i in range(20):
        p = pool()
        p["name"] = f"PE({30 + i}:2)"
        p["carbon"] = 30 + i
        p["feat_feature_id"] = [f"old-{i}", f"new-{i}"]
        data.append(p)
        truth.append(
            dict(
                stream="RP_Pos",
                matlab_class="PE",
                matlab_species=f"PE({30 + i}:2)",
                xcms_feature_id=f"new-{i}",
                crosswalk_status="profile_match",
                matlab_is_internal_standard=False,
            )
        )
    p = pd.concat(data, ignore_index=True)
    fitted = tool.fit_pair_model([(p, pd.DataFrame(truth))], "RP_Pos")
    assert (
        fitted["training_identity_groups"] == 20 and fitted["training_pair_rows"] == 40
    )
    e = experimental_pe_evidence(p)[fitted["pair_model"]["features"]]
    score = pair_preference(
        e.iloc[[1]].to_numpy(), e.iloc[[0]].to_numpy(), fitted["pair_model"]
    )[0]
    assert score > 0.9
