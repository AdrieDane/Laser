"""Separate construction parity on synthetic RP partners, no query curation."""

import json
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import paired_retention_gap as producer
import paired_gap_independent as independent
from test_paired_retention_gap import training, query


def compare_models(a, b):
    assert set(a) == set(b)
    for k in a:
        if isinstance(a[k], dict):
            compare_models(a[k], b[k])
        elif isinstance(a[k], float):
            assert a[k] == pytest.approx(b[k], abs=1e-10, rel=0)
        else:
            assert a[k] == b[k]


@pytest.mark.parametrize("family", list(producer.FAMILIES))
def test_scalar_references_predictions_and_complete_graph(family):
    rows, support = training(family)
    rows = rows.astype({"rt_seconds": float})
    for cohort, shift in (("A", 0.1), ("B", -0.2), ("C", 0.3)):
        rows.loc[rows.cohort.eq(cohort), "rt_seconds"] += 20
        rows.loc[
            rows.cohort.eq(cohort) & rows.feature_id.str.endswith("1"), "rt_seconds"
        ] += shift
    a, pairs_a = producer.fit_gaps(
        rows, support, training_groups={"A", "B", "C"}, excluded_groups={"Q"}
    )
    b, pairs_b = independent.fit_reference(
        rows.sample(frac=1, random_state=1),
        support,
        training_groups={"A", "B", "C"},
        excluded_groups={"Q"},
    )
    compare_models(a, b)
    pd.testing.assert_frame_equal(pairs_a, pairs_b, check_exact=True)
    anchors, chemical = query(family)
    kwargs = dict(query_group="Q", declared_masses={"RP_Pos": [], "RP_Neg": []})
    x = producer.predict_partners(anchors, chemical, a, **kwargs)
    y = independent.predict_reference(anchors, chemical, b, **kwargs)
    pd.testing.assert_frame_equal(x, y, check_exact=False, rtol=0, atol=1e-10)
    features = pd.DataFrame(
        dict(
            feature_id=[1, 2, 3, 4],
            mz=[500.0, 500.0, 500.0, 501.0],
            rt=[10.0, 9.9, 9.85, 9.9],
            intensity=[1000.0, 200.0, 100.0, 10.0],
        )
    )
    first = producer.raw_candidates(x, {x.stream.iloc[0]: features})
    second = independent.raw_reference(y, {y.stream.iloc[0]: features})
    keys = ["prediction_index", "feat_feature_id"]
    pd.testing.assert_frame_equal(
        first.sort_values(keys).reset_index(drop=True),
        second.sort_values(keys).reset_index(drop=True),
        check_exact=False,
        rtol=0,
        atol=1e-10,
    )
    assert len(first) == 3
    json.dumps(a, allow_nan=False)


def test_normal_phase_cannot_receive_rp_partner_proposals():
    rows, support = training()
    model, _ = producer.fit_gaps(
        rows, support, training_groups={"A", "B", "C"}, excluded_groups={"Q"}
    )
    a, t = query()
    a["stream"], t["stream"] = "NP_Pos", "NP_Pos"
    kwargs = dict(query_group="Q", declared_masses={"NP_Pos": []})
    assert producer.predict_partners(a, t, model, **kwargs).empty
    assert independent.predict_reference(a, t, model, **kwargs).empty


def test_quality_and_one_project_vote_do_not_use_row_multiplicity():
    rows, support = training()
    extras = []
    for i in range(10):
        part = rows.loc[rows.cohort.eq("A")].assign(dataset="tissue" + str(i)).copy()
        part.loc[part.feature_id.str.endswith("1"), "rt_seconds"] -= 2
        extras.append(part)
    rows = pd.concat([rows, *extras], ignore_index=True)
    a, pa = producer.fit_gaps(
        rows, support, training_groups={"A", "B", "C"}, excluded_groups={"Q"}
    )
    b, pb = independent.fit_reference(
        rows, support, training_groups={"A", "B", "C"}, excluded_groups={"Q"}
    )
    compare_models(a, b)
    pd.testing.assert_frame_equal(pa, pb, check_exact=True)
    assert a["models"]["LPC|18|1"]["gap_seconds"] == -6
    assert np.isfinite(a["models"]["LPC|18|1"]["training_p90_seconds"])


def test_diagnostic_counts_correct_feature_not_just_identity():
    import diagnose_qe_paired_gap_pilot as diagnostic
    from standard_lipidomics.validation import _species

    rows, support = training()
    model, _ = producer.fit_gaps(
        rows, support, training_groups={"A", "B", "C"}, excluded_groups={"Q"}
    )
    a, t = query()
    predictions = producer.predict_partners(
        a, t, model, query_group="Q", declared_masses={"RP_Pos": []}
    )
    features = pd.DataFrame(
        dict(
            feature_id=["source", "true", "wrong"],
            mz=[500.0, 500.0, 500.0],
            rt=[10.0, 9.9, 9.85],
            intensity=[1000.0, 200.0, 100.0],
        )
    )
    graph = producer.raw_candidates(predictions, {"RP_Pos": features})
    graph["identity_is_free"] = True
    graph["feature_is_free"] = ~graph.feat_feature_id.eq("source")
    graph["novel_to_old029_core_pool"] = graph.feat_feature_id.eq("true")
    manual = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                species=_species(t.iloc[side]["name"]),
                lipid_class=t.iloc[side]["class"],
                adduct="[M+H]+",
                feature_id=fid,
                internal_standard=False,
                mapped=True,
                duplicate_manual_entry=False,
                same_manual_peak=side == 0,
            )
            for side, fid in enumerate(("source", "true"))
        ]
    )
    base = dict(streams={"RP_Pos": dict(features=features)})
    evaluated, _, _ = diagnostic.evaluate_query("Q", predictions, graph, manual, base)
    assert evaluated.partner_matches_manual.all()
    assert evaluated.target_is_missing_mapped.all()
    assert (
        evaluated.correct_target_in_graph.all() and evaluated.correct_target_free.all()
    )
    assert (
        evaluated.correct_target_new_to_core.all()
        and evaluated.true_gap_error_better.all()
    )
    changed = manual.copy()
    changed.loc[1, "feature_id"] = "wrong"
    evaluated, _, _ = diagnostic.evaluate_query("Q", predictions, graph, changed, base)
    assert evaluated.wrong_gap_error_better.all()
    assert not evaluated.correct_target_new_to_core.any()


def test_unresolved_manual_mapping_is_not_an_assumed_correct_peak():
    import diagnose_qe_paired_gap_pilot as diagnostic
    from standard_lipidomics.validation import _species

    a, t = query()
    rows, support = training()
    model, _ = producer.fit_gaps(
        rows, support, training_groups={"A", "B", "C"}, excluded_groups={"Q"}
    )
    pred = producer.predict_partners(
        a, t, model, query_group="Q", declared_masses={"RP_Pos": []}
    )
    features = pd.DataFrame(
        dict(
            feature_id=["source", "true"],
            mz=[500.0, 500.0],
            rt=[10.0, 9.9],
            intensity=[1000.0, 200.0],
        )
    )
    graph = producer.raw_candidates(pred, {"RP_Pos": features})
    for field in ("identity_is_free", "feature_is_free", "novel_to_old029_core_pool"):
        graph[field] = True
    manual = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                species=_species(t.iloc[1]["name"]),
                lipid_class=t.iloc[1]["class"],
                adduct="[M+H]+",
                feature_id="",
                internal_standard=False,
                mapped=False,
                duplicate_manual_entry=False,
                same_manual_peak=False,
            )
        ]
    )
    result, _, retained = diagnostic.evaluate_query(
        "Q", pred, graph, manual, dict(streams={"RP_Pos": dict(features=features)})
    )
    assert len(retained) == 1 and result.target_manual_rows.iloc[0] == 1
    assert (
        not result.target_is_unique_mapped.any()
        and not result.correct_target_in_graph.any()
    )


def test_workers_configure_final_snapshot_before_scientific_imports():
    import prepare_qe_paired_gap_pilot as prep
    import verify_qe_paired_gap_pilot as verifier

    for module, names in (
        (prep, ("train", "query")),
        (verifier, ("check_model", "check_query")),
    ):
        nodes = {
            n.name: n
            for n in ast.parse(Path(module.__file__).read_text()).body
            if isinstance(n, ast.FunctionDef)
        }
        for name in names:
            first = nodes[name].body[0]
            assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
            assert (
                isinstance(first.value.func, ast.Name)
                and first.value.func.id == "configure_frozen_resources"
            )
