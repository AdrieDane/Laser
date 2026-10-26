"""RP gap prototype software checks; no QE annotation changes."""

import copy

import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
import paired_retention_gap as p


def training(family="LPC"):
    stream, ion, a, b, offset, _ = p.FAMILIES[family]
    rows, support = [], []
    for cohort in ("A", "B", "C"):
        for side, cls in enumerate((a, b)):
            rows.append(
                dict(
                    dataset=cohort,
                    cohort=cohort,
                    stream=stream,
                    lipid_class=cls,
                    carbon=18,
                    db=1 - (offset if side else 0),
                    adduct=ion,
                    feature_id=cohort + str(side),
                    rt_seconds=600 + (-6 if side else 0),
                    reference_mz=500.0,
                    observed_mz=500.0,
                )
            )
            support.append(
                dict(cohort=cohort, stream=stream, lipid_class=cls, manual_rows=40)
            )
    return pd.DataFrame(rows), pd.DataFrame(support)


def model(family="LPC"):
    rows, support = training(family)
    return p.fit_gaps(
        rows, support, training_groups={"A", "B", "C"}, excluded_groups={"Q"}
    )[0]


def query(family="LPC"):
    stream, ion, a, b, offset, _ = p.FAMILIES[family]
    targets = pd.DataFrame(
        [
            dict(
                stream=stream,
                name=f"{cls}(18:{1 - (offset if side else 0)})",
                **{"class": cls},
                adduct=ion,
                carbon=18,
                db=1 - (offset if side else 0),
                ref_mz=500.0,
            )
            for side, cls in enumerate((a, b))
        ]
    )
    anchors = (
        targets.iloc[[0]]
        .drop(columns="ref_mz")
        .assign(
            feat_feature_id="source",
            feat_mz=500.0,
            feat_rt=10.0,
            is_is=False,
            class_selection_score=0.99,
        )
    )
    return anchors, targets


def predict(anchors, targets, fitted=None):
    return p.predict_partners(
        anchors,
        targets,
        model() if fitted is None else fitted,
        query_group="Q",
        declared_masses={"RP_Pos": [], "RP_Neg": []},
    )


def test_true_partner_prediction_and_inputs_immutable():
    a, t = query()
    before_a, before_t = a.copy(deep=True), t.copy(deep=True)
    result = predict(a, t)
    assert len(result) == 1 and result.predicted_rt_seconds.iloc[0] == 594
    assert (
        result.signed_gap_seconds.iloc[0] == -6
        and result.training_projects.iloc[0] == 3
    )
    pd.testing.assert_frame_equal(a, before_a, check_exact=True)
    pd.testing.assert_frame_equal(t, before_t, check_exact=True)


def test_reverse_prediction_keeps_the_same_gap_orientation():
    _, targets = query()
    source = (
        targets.iloc[[1]]
        .drop(columns="ref_mz")
        .assign(
            feat_feature_id="two",
            feat_mz=500.0,
            feat_rt=9.9,
            is_is=False,
            class_selection_score=0.99,
        )
    )
    prediction = predict(source, targets)
    assert prediction.signed_gap_seconds.iloc[0] == 6
    assert prediction.predicted_rt_seconds.iloc[0] == 600


def test_one_vote_per_project_even_many_related_tissues():
    rows, support = training()
    extra = []
    for i in range(20):
        part = rows.loc[rows.cohort.eq("A")].copy()
        part["dataset"] = "related" + str(i)
        part.loc[part.lipid_class.eq("2-acyl LPC"), "rt_seconds"] = 592
        extra.append(part)
    fitted, pairs = p.fit_gaps(
        pd.concat([rows, *extra]),
        support,
        training_groups={"A", "B", "C"},
        excluded_groups={"Q"},
    )
    assert len(pairs) == 23
    item = fitted["models"]["LPC|18|1"]
    assert item["training_projects"] == 3 and item["gap_seconds"] == -6


def test_plasmalogen_db_offset_and_same_formula():
    fitted = model("PC_ether")
    assert "PC_ether|18|1" in fitted["models"]
    a, t = query("PC_ether")
    result = predict(a, t, fitted)
    assert result.db.iloc[0] == 0 and result["class"].iloc[0] == "PC[P]"
    assert result.predicted_rt_seconds.iloc[0] == 594
    t.loc[t["class"].eq("PC[P]"), "ref_mz"] += 1
    assert predict(a, t, fitted).empty


def test_query_scope_sparse_classes_mass_and_duplicate_pairs_withheld():
    rows, support = training()
    with pytest.raises(ValueError):
        p.fit_gaps(
            rows, support, training_groups={"A", "B"}, excluded_groups={"C", "Q"}
        )
    sparse = support.assign(manual_rows=1)
    assert not p.fit_gaps(
        rows, sparse, training_groups={"A", "B", "C"}, excluded_groups={"Q"}
    )[0]["models"]
    bad = rows.copy()
    bad.loc[bad.cohort.eq("A"), "observed_mz"] = 501
    assert not p.fit_gaps(
        bad, support, training_groups={"A", "B", "C"}, excluded_groups={"Q"}
    )[0]["models"]
    repeated = pd.concat([rows, rows.iloc[[0]]])
    assert not p.fit_gaps(
        repeated, support, training_groups={"A", "B", "C"}, excluded_groups={"Q"}
    )[0]["models"]
    fitted = model()
    a, t = query()
    with pytest.raises(ValueError):
        p.predict_partners(
            a, t, fitted, query_group="A", declared_masses={"RP_Pos": []}
        )


def test_no_query_gold_fitted_rt_weak_anchor_or_standard_self_anchor():
    a, t = query()
    with pytest.raises(ValueError):
        predict(a, t.assign(ref_rt=999))
    with pytest.raises(ValueError):
        predict(a.assign(same_manual_peak=True), t)
    assert predict(a.assign(class_selection_score=0.9), t).empty
    assert predict(a.assign(feat_mz=500.001), t).empty
    assert predict(a.assign(is_is=True), t).empty
    assert p.predict_partners(
        a, t, model(), query_group="Q", declared_masses={"RP_Pos": [500]}
    ).empty
    wrong = copy.deepcopy(model())
    wrong["models"]["LPC|18|1"]["contributing_groups"].append("Q")
    with pytest.raises(ValueError):
        predict(a, t, wrong)


def test_complete_raw_graph_retains_partner_and_other_competitors():
    a, t = query()
    predictions = predict(a, t)
    features = pd.DataFrame(
        dict(
            feature_id=["source", "true", "other", "far"],
            mz=[500.0, 500.0, 500.0, 501.0],
            rt=[10.0, 9.9, 9.85, 9.9],
            intensity=[1000.0, 200.0, 100.0, 200.0],
        )
    )
    result = p.raw_candidates(predictions, {"RP_Pos": features})
    assert set(result.feat_feature_id) == {"source", "true", "other"}
    assert not result.loc[
        result.feat_feature_id.eq("source"), "physical_partner_eligible"
    ].any()
    assert result.loc[
        result.feat_feature_id.eq("true"), "physical_partner_eligible"
    ].all()
    assert result.loc[
        result.feat_feature_id.eq("other"), "physical_partner_eligible"
    ].all()
    assert p.raw_candidates(predictions.iloc[:0], {}).empty
