from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tools.local_pe_calibration import (
    robust_fit,
    fit_mass_calibration,
    mass_predictions,
    _rt_prediction,
)


def mass_anchors():
    random = np.random.default_rng(9)
    mz = random.uniform(400, 1100, 200)
    rt = random.uniform(5, 20, 200)
    return pd.DataFrame(
        {
            "class": np.tile(["PC", "PS", "DG", "TG"], 50),
            "feat_feature_id": [f"f{i:03d}" for i in range(200)],
            "feat_mz": mz,
            "feat_rt": rt,
            "rt_error": 0.01,
            "ppm_error": 0.6
            + 0.35 * (mz - 800) / 200
            + 0.2 * (rt - 12) / 5
            + random.normal(0, 0.03, 200),
        }
    )


def test_mass_model_must_improve_class_held_out_predictions():
    a = mass_anchors()
    before = a.copy(deep=True)
    coefficients, anchors, audit = fit_mass_calibration(a, float(a.ppm_error.median()))
    assert coefficients is not None and audit["status"] == "mass_CV_pass"
    assert audit["mass_CV_median_after"] < 0.1 * audit["mass_CV_median_before"]
    assert len(anchors) == 200
    assert_frame_equal(a, before)


def test_target_PE_family_and_evaluation_labels_cannot_train_mass_model():
    a = mass_anchors()
    base = fit_mass_calibration(a, 0.6)[0]
    targets = a.iloc[:20].assign(**{"class": "PE", "ppm_error": 2.8})
    ethers = targets.assign(**{"class": "PE-O"})
    x = pd.concat([a, targets, ethers], ignore_index=True).assign(
        correct=True, dataset="truth"
    )
    coefficients, anchors, _ = fit_mass_calibration(x, 0.6)
    np.testing.assert_allclose(coefficients, base)
    assert not anchors["class"].str.startswith("PE").any()


def test_mass_sparse_or_missing_inputs_do_not_claim_a_calibration():
    for a in [
        pd.DataFrame(),
        mass_anchors().iloc[:20],
        mass_anchors().assign(**{"class": "PC"}),
    ]:
        assert fit_mass_calibration(a, 0.6)[0] is None


def test_local_mass_prediction_does_not_extrapolate_or_make_large_shifts():
    a = mass_anchors()
    coef, anchors, _ = fit_mass_calibration(a, 0.6)
    refs = pd.DataFrame(
        {"ref_mz": [750.0, 1800.0, 750.0], "ref_rt": [12.0, 12.0, 40.0]}
    )
    values, supported = mass_predictions(refs, coef, anchors, 0.6)
    assert supported.tolist() == [True, False, False]
    assert values[1] == values[2] == 0.6
    extreme = np.array([20.0, 0, 0, 0, 0])
    values, supported = mass_predictions(refs, extreme, anchors, 0.6)
    assert not supported.any() and np.all(values == 0.6)


def test_robust_fit_resists_a_single_extreme_anchor():
    x = np.ones((100, 1))
    y = np.full(100, 5.0)
    y[-1] = 9999.0
    value = robust_fit(x, y, np.ones(100), [0.0001], 0.25)[0]
    assert abs(value - 5) < 0.1


def test_RT_prediction_excludes_target_composition_and_requires_support():
    rows = []
    for carbon in range(30, 47, 2):
        for db in range(9):
            shift = (
                2
                + 0.3 * (carbon - 38)
                - 0.2 * (db - 4)
                + 0.02 * (carbon - 38) * (db - 4)
            )
            rows.append(
                dict(
                    name=f"PE({carbon}:{db})",
                    carbon=carbon,
                    db=db,
                    rt_error=shift / 60,
                    feat_rt=10 + shift / 60,
                )
            )
    anchors = pd.DataFrame(rows)
    anchors.loc[anchors.carbon.eq(38) & anchors.db.eq(4), "rt_error"] = 100.0
    anchors.loc[anchors.carbon.eq(38) & anchors.db.eq(4), "feat_rt"] = 1000.0
    predicted, support = _rt_prediction(anchors, 38, 4)
    assert abs(predicted - 602) < 0.1
    assert not (support.carbon.eq(38) & support.db.eq(4)).any()
    assert _rt_prediction(anchors, 52, 12)[0] is None
    assert _rt_prediction(anchors.iloc[:5], 38, 4)[0] is None


def selection_fixture(monkeypatch):
    import tools.local_pe_calibration as module

    pool = pd.DataFrame(
        [
            dict(
                name="PE(36:2)",
                **{"class": "PE"},
                adduct="[M+H]+",
                feat_feature_id="new",
                feat_intensity=1e8,
                pe_mass_center_ppm=1.4,
                pe_corrected_mass_error_ppm=0.5,
                pe_precalibration_rt_error_seconds=3.0,
                pe_local_rt_error_seconds=2.0,
                pe_calibration_mass_supported=True,
                pe_calibration_RT_supported=False,
                pe_calibration_RT_uncertainty_seconds=np.nan,
                pe_support_count=12,
                q_profile_samples=4,
            )
        ]
    )
    annotations = pd.DataFrame(
        [
            dict(
                name="PC(34:1)",
                **{"class": "PC"},
                adduct="[M+H]+",
                feat_feature_id="old",
            )
        ]
    )
    monkeypatch.setattr(
        module,
        "prepare_local_pe_pool",
        lambda *args: (
            pool.copy(),
            dict(pe_mass_center_ppm=1.0, mass_mass_uncertainty_ppm=0.8),
        ),
    )
    monkeypatch.setattr(module, "pe_rebuild_evidence", lambda x: x)
    monkeypatch.setattr(module, "score_pe_rebuild", lambda x, m: np.full(len(x), 0.9))

    def select(ann=None, pairs=None):
        return module.refine_local_pe(
            annotations if ann is None else ann,
            pool if pairs is None else pairs,
            None,
            None,
            None,
            None,
        )

    return pool, annotations, select


def test_only_supported_coordinate_change_can_add_a_peak(monkeypatch):
    p, a, select = selection_fixture(monkeypatch)
    after, pairs, decisions, audit = select()
    assert after.feat_feature_id.tolist() == ["old", "new"]
    assert pairs.loc[pairs.feat_feature_id.eq("new"), "automatic_accepted"].all()
    assert audit.added_PE.iloc[0] == 1 and decisions.pe_calibration_selected.all()
    p["pe_calibration_mass_supported"] = False
    assert_frame_equal(select()[0], a)


@pytest.mark.parametrize("kind", ["mass", "RT"])
def test_uncertainty_must_fit_inside_original_acceptance_window(monkeypatch, kind):
    p, a, select = selection_fixture(monkeypatch)
    if kind == "mass":
        p["pe_corrected_mass_error_ppm"] = 2.5
    else:
        p["pe_calibration_RT_supported"] = True
        p["pe_local_rt_error_seconds"] = 5.0
        p["pe_calibration_RT_uncertainty_seconds"] = 6.0
    assert_frame_equal(select()[0], a)


@pytest.mark.parametrize("protection", ["occupied", "manual", "standard"])
def test_calibration_cannot_replace_other_classes_or_protected_choices(
    monkeypatch, protection
):
    p, a, select = selection_fixture(monkeypatch)
    if protection == "occupied":
        a["feat_feature_id"] = "new"
    elif protection == "manual":
        p["manual_confirmed"] = True
    else:
        p["is_is"] = True
    assert_frame_equal(select()[0], a)


def test_ambiguous_new_peaks_do_not_fill_the_grid(monkeypatch):
    p, a, select = selection_fixture(monkeypatch)
    second = p.iloc[0].copy()
    second["feat_feature_id"] = "second"
    p.loc[1] = second
    assert_frame_equal(select()[0], a)


def test_declared_standard_is_not_reintroduced_as_an_ordinary_identity(monkeypatch):
    p, a, select = selection_fixture(monkeypatch)
    declared = p.assign(
        name="PE(36:2) (IS)", feat_feature_id="standard-feature", is_is=True
    )
    assert_frame_equal(select(pairs=pd.concat([p, declared], ignore_index=True))[0], a)


def test_research_candidate_generator_matches_runtime_without_local_offsets():
    from matlab_annotation.pe_rebuild import (
        _join_raw_pe as live_join,
        PERebuildSettings,
    )
    from tools.local_pe_calibration import _join_raw_pe

    refs = pd.DataFrame(
        [dict(name="PE(36:2)", ref_mz=700.0, ref_rt=10.0, adduct="[M+H]+")]
    )
    features = pd.DataFrame(
        [dict(feature_id="original", mz=700.0 * (1 + 4e-6), rt=10.01, intensity=1e6)]
    )
    args = (
        refs,
        features,
        pd.DataFrame(),
        pd.DataFrame(),
        2.0,
        0.0,
        PERebuildSettings(),
    )
    assert_frame_equal(_join_raw_pe(*args), live_join(*args))
    assert _join_raw_pe(*args, mass_centers={"PE(36:2)": 0.5}).empty
    assert_frame_equal(
        _join_raw_pe(*args, mass_centers={"PE(36:2)": 99.0}), live_join(*args)
    )


def test_evaluation_keys_use_real_stream_not_incomplete_standard_alias():
    from tools.evaluate_local_pe_calibration import selected_keys

    frame = pd.DataFrame(
        dict(
            stream=["RP_Pos", "NP_Pos"],
            standard_stream=[np.nan, "NP_Pos"],
            name=["PC(34:1)", "PC(34:1)"],
            adduct=["[M+H]+", "[M+H]+"],
            feat_feature_id=["123.0", "123.0"],
        )
    )
    assert selected_keys(frame) == {
        ("RP_Pos", "PC(34:1)", "[M+H]+", "123"),
        ("NP_Pos", "PC(34:1)", "[M+H]+", "123"),
    }
