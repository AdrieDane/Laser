import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from matlab_annotation.pe_standards import complete_pe_internal_standards


def fixture(stream="RP_Pos"):
    adduct = "[M-H]-" if stream.endswith("Neg") else "[M+H]+"
    base = pd.DataFrame(
        [
            dict(
                name="PC(34:1)",
                **{"class": "PC"},
                adduct=adduct,
                feat_feature_id="old",
                feat_rt=8.0,
                feat_mz=700.0,
                feat_intensity=1e8,
                isotopologue="",
                automatic_accepted=True,
                is_is=False,
                internal_standard_name="",
            )
        ]
    )
    pool = pd.DataFrame(
        [
            dict(
                name="PE(28:0) (IS)",
                **{"class": "PE"},
                adduct=adduct,
                carbon=28,
                db=0,
                feat_feature_id="standard",
                feat_rt=10.05,
                ref_rt=10.0,
                rt_expected=10.0,
                feat_mz=640.0,
                ref_mz=640.0,
                feat_intensity=1e7,
                ppm_error=0.2,
                rt_error=0.05,
                isotopologue="",
                is_is=True,
                internal_standard_name="PE(14:0)2",
                automatic_accepted=False,
                automatic_candidate=False,
                grid_coherence_assessed=True,
                grid_coherence_score=0.2,
                grid_coherence_outlier=True,
                grid_coherence_pass=False,
            )
        ]
    )
    return dict(
        streams={
            stream: dict(
                annotations=base,
                all_pairs=pool,
                reference=pool.rename(columns={"feat_feature_id": "unused"}),
                rt_models=pd.DataFrame(),
            )
        },
        matlab_pe_completion_audit=pd.DataFrame(),
        matlab_internal_standard_audit=pd.DataFrame(
            [
                dict(
                    phase=stream.split("_")[0],
                    internal_standard_name="PE(14:0)2",
                    accepted_matches=0,
                    status="review_candidate",
                )
            ]
        ),
    )


@pytest.mark.parametrize("stream", ["RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"])
def test_unique_declared_pe_standard_is_not_removed_by_endogenous_grid(stream):
    old = fixture(stream)
    result = complete_pe_internal_standards(old)
    after = result["streams"][stream]["annotations"]
    assert len(after) == 2
    assert len(old["streams"][stream]["annotations"]) == 1
    assert_frame_equal(
        after.iloc[:1][old["streams"][stream]["annotations"].columns],
        old["streams"][stream]["annotations"],
        check_dtype=False,
    )
    std = after.iloc[1]
    assert std.auto_annotation_status == "pe_declared_standard_supported"
    assert std.automatic_accepted and std.is_is
    assert not std.grid_coherence_pass  # Do not falsify the historical grid score.
    assert std.grid_coherence_exemption == "declared_PE_standard_unique_mass_RT_match"
    assert result["matlab_internal_standard_audit"].accepted_matches.iloc[0] == 1
    assert (
        result["matlab_pe_completion_audit"].restored_pe_internal_standards.iloc[0] == 1
    )
    assert len(result["automatic_annotations"]) == 2
    repeat = complete_pe_internal_standards(result)
    assert_frame_equal(repeat["automatic_annotations"], result["automatic_annotations"])


@pytest.mark.parametrize(
    "defect", ["mass", "rt", "undeclared", "not_standard", "isotope", "other_class"]
)
def test_declared_standard_check_does_not_relax_physical_or_identity_limits(defect):
    r = fixture()
    pool = r["streams"]["RP_Pos"]["all_pairs"]
    if defect == "mass":
        pool.loc[0, "ppm_error"] = 3.0
    elif defect == "rt":
        pool.loc[0, "rt_error"] = 10.01 / 60
    elif defect == "undeclared":
        pool.loc[0, "internal_standard_name"] = ""
    elif defect == "not_standard":
        pool.loc[0, "is_is"] = False
    elif defect == "isotope":
        pool.loc[0, "isotopologue"] = "13C"
    else:
        pool.loc[0, "class"] = "PE-P"
    result = complete_pe_internal_standards(r)
    assert len(result["streams"]["RP_Pos"]["annotations"]) == 1


def test_ambiguous_standard_is_not_made_unique_by_an_occupied_feature():
    r = fixture()
    p = r["streams"]["RP_Pos"]
    alt = p["all_pairs"].copy().assign(feat_feature_id="old")
    p["all_pairs"] = pd.concat([p["all_pairs"], alt], ignore_index=True)
    result = complete_pe_internal_standards(r)
    assert len(result["streams"]["RP_Pos"]["annotations"]) == 1


def test_existing_other_annotation_or_standard_is_never_replaced():
    for already_standard in [False, True]:
        r = fixture()
        p = r["streams"]["RP_Pos"]
        if already_standard:
            p["annotations"] = pd.concat(
                [p["annotations"], p["all_pairs"].assign(feat_feature_id="different")],
                ignore_index=True,
            )
        else:
            p["annotations"].loc[0, "feat_feature_id"] = "standard"
        before = p["annotations"].copy()
        after = complete_pe_internal_standards(r)["streams"]["RP_Pos"]["annotations"]
        assert_frame_equal(after, before)


def test_standard_remains_a_full_annotation_in_default_review_grid():
    from standard_lipidomics.grid_browser import (
        prepare_standard_grid_data,
        build_class_grid_state,
    )

    r = complete_pe_internal_standards(fixture())
    ref, candidates = prepare_standard_grid_data(r["streams"]["RP_Pos"])
    criteria = dict(
        ppm_tolerance=3.0,
        rt_tolerance=0.5,
        use_rt=True,
        use_grid_coherence=True,
        grid_coherence_threshold=0.45,
    )
    state = build_class_grid_state(
        ref,
        candidates,
        lipid_class="PE",
        adduct="[M+H]+",
        confirmed=set(),
        run_criteria=criteria,
        review=criteria,
    )
    assert len(state["plot_df"]) == 1
    assert state["plot_df"].automatic_candidate.iloc[0]


def test_disabled_is_exact_noop():
    r = fixture()
    after = complete_pe_internal_standards(r, enabled=False)
    assert_frame_equal(
        after["streams"]["RP_Pos"]["annotations"], r["streams"]["RP_Pos"]["annotations"]
    )
    assert "pe_standard_completion_applied" not in after


def test_public_engine_applies_standard_completion_after_phase_results(monkeypatch):
    import matlab_annotation.engine as engine

    r = fixture()
    monkeypatch.setattr(
        engine, "discover_standard_streams", lambda project: {"RP_Pos": "input.txt"}
    )
    monkeypatch.setattr(engine, "_run_matlab_annotation_phase", lambda *a, **k: r)
    result = engine.run_matlab_annotation("unused", phases=("RP",))
    assert len(result["streams"]["RP_Pos"]["annotations"]) == 2
