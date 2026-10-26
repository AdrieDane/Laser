"""Synthetic tests; never reads query curation or creates annotations."""

import copy

import numpy as np
import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
import matrix_peer_composition as p
import matrix_peer_independent as oracle


def automatic(offset=0):
    return pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                **{"class": cls},
                carbon=base + 2 * i + offset,
                db=i % 3,
                is_is=False,
                ref_mz=500.0,
                feat_mz=500.0,
                feat_intensity=1000.0,
                post_regression_confidence=0.99,
            )
            for cls, base in (("PC", 30), ("PE", 30), ("TG", 50), ("SM-d", 32))
            for i in range(6)
        ]
    )


def example(n=6):
    signatures, rows = {}, []
    for k in range(n):
        cohort = f"24P{k:04d}"
        dataset = cohort + "_project"
        signatures[dataset] = p.signature(automatic(0 if k < 3 else 10))
        for _i in range(40):
            rows.append(
                dict(
                    dataset=dataset,
                    cohort=cohort,
                    stream="RP_Pos",
                    lipid_class="PC",
                    carbon=34 if k < 3 else 44,
                    db=1,
                )
            )
    training = {d[:7] for d in signatures}
    observations = pd.DataFrame(rows)
    model = p.fit_reference(
        observations, signatures, training_groups=training, excluded_groups={"QUERY"}
    )
    return observations, signatures, model


def targets():
    return pd.DataFrame(
        dict(
            stream=["RP_Pos"] * 3,
            **{"class": ["PC"] * 3},
            carbon=[34, 34, 44],
            db=[1, 1, 1],
            is_is=[False] * 3,
        ),
        index=[10, 22, 35],
    )


def test_signatures_scalar_parity_input_immutable_and_duplicate_adducts():
    a = automatic()
    before = a.copy(deep=True)
    sig = p.signature(a)
    assert sig == oracle.signature_scalar(a)
    assert p.signature(pd.concat([a, a])) == sig
    pd.testing.assert_frame_equal(a, before, check_exact=True)


def test_reference_prediction_scalar_parity_and_peak_invariance():
    rows, signatures, model = example()
    other = oracle.fit_scalar(
        rows,
        signatures,
        training_groups=model["training_groups"],
        excluded_groups={"QUERY"},
    )
    assert model == other
    q = p.signature(automatic())
    source, t = copy.deepcopy(model), targets()
    a = p.predict(t, q, model, query_group="QUERY")
    b = oracle.predict_scalar(t, q, other, query_group="QUERY")
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=0, atol=1e-12)
    pd.testing.assert_series_equal(a.iloc[0], a.iloc[1], check_names=False)
    assert (
        a.iloc[0][p.PREFIX + "peer_exact_frequency"]
        > a.iloc[2][p.PREFIX + "peer_exact_frequency"]
    )
    assert model == source and a.notna().all().all()


def test_target_family_is_withheld_in_every_stream():
    _, _, model = example()
    a = automatic()
    x = p.predict(targets(), p.signature(a), model, query_group="QUERY")
    a.loc[a["class"].eq("PC"), "carbon"] += 60
    a = pd.concat([a, a.loc[a["class"].eq("PC")].assign(stream="NP_Pos")])
    y = p.predict(targets(), p.signature(a), model, query_group="QUERY")
    pd.testing.assert_frame_equal(x, y, check_exact=True)
    assert p.root("2-acyl LPC") == p.root("PC-P") == p.root("PC")
    assert p.root("BMP") == p.root("PG")


def test_fifth_boundary_ties_cannot_select_projects_by_identifier():
    rows, signatures, model = example(7)
    for key in signatures:
        signatures[key] = p.signature(automatic())
    model = p.fit_reference(
        rows,
        signatures,
        training_groups=model["training_groups"],
        excluded_groups={"QUERY"},
    )
    out = p.predict(targets(), p.signature(automatic()), model, query_group="QUERY")
    assert out[p.PREFIX + "peer_projects"].eq(7).all()
    np.testing.assert_allclose(out[p.PREFIX + "peer_vs_global_logodds"], 0, atol=0.1)


def test_sparse_or_missing_context_unknown_and_standards_protected():
    _, _, model = example()
    sig = p.signature(automatic().loc[lambda t: t["class"].isin(["PC", "PE", "TG"])])
    assert p.predict(targets(), sig, model, query_group="QUERY").isna().all().all()
    query = p.signature(automatic())
    assert (
        p.predict(targets().assign(is_is=True), query, model, query_group="QUERY")
        .isna()
        .all()
        .all()
    )
    assert (
        p.predict(
            targets().assign(**{"class": "X4"}), query, model, query_group="QUERY"
        )
        .isna()
        .all()
        .all()
    )


def test_scope_and_supervision_refused_before_prediction():
    rows, signatures, model = example()
    with pytest.raises(ValueError):
        p.predict(
            targets(),
            p.signature(automatic()),
            model,
            query_group=model["training_groups"][0],
        )
    with pytest.raises(ValueError):
        p.predict(
            targets().assign(same_manual_peak=True),
            p.signature(automatic()),
            model,
            query_group="QUERY",
        )
    with pytest.raises(ValueError):
        p.fit_reference(
            rows, signatures, training_groups={"OTHER"}, excluded_groups={"QUERY"}
        )
    with pytest.raises(ValueError):
        p.signature(automatic().assign(training_positive=True))


def test_bad_mass_weak_and_standard_rows_do_not_supply_context():
    a = automatic()
    a.loc[a["class"].eq("SM-d"), "feat_mz"] = 501
    a.loc[a["class"].eq("TG"), "is_is"] = True
    a.loc[a["class"].eq("PE"), "post_regression_confidence"] = 0.8
    assert len(p.signature(a)["groups"]) == 1
    assert p.signature(a) == oracle.signature_scalar(a)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Cer(d34:1)", (34, 1)),
        ("SM(t42:2)", (42, 2)),
        ("PC(O+P-40:3)", (40, 3)),
        ("PE(P-38:4)", (38, 4)),
        ("TG(56:5)", (56, 5)),
        ("not-a-composition", None),
    ],
)
def test_native_composition_prefixes_are_not_silently_dropped(name, expected):
    from matrix_peer_pilot_data import parse_composition

    assert parse_composition(name) == expected
