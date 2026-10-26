from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
import competing_ion_family_measurements as m
from competing_ion_family_independent import independent_measure, independent_signature
from matlab_annotation.adduct_family_evidence import native_adduct_families
from matlab_annotation.decode import CodeDecoder
from matlab_annotation.vendor.compound_model import theoretical_mz
from matlab_annotation.xcms_project import load_bundled_compound_catalog

ROOT = REPO / "reference_data/matlab_pipeline"


def row(template, carbon, db, phase="RP", adduct="[M+H]+", fid="same"):
    cat = load_bundled_compound_catalog(ROOT)
    for idx, attr in enumerate(cat.composition_attr):
        if phase not in tuple(attr.get("column", ())):
            continue
        dec = CodeDecoder(SimpleNamespace(compounds=cat), idx + 1, phase)
        if dec.template != template or adduct not in dec.axis(0):
            continue
        code = dec.code_for(
            {
                0: dec.axis(0).index(adduct),
                3: dec.axis(3).index(str(carbon)),
                4: dec.axis(4).index(str(db)),
            }
        )
        return dict(
            stream=phase + ("_Pos" if "+" in adduct.split("]")[-1] else "_Neg"),
            name=dec.decode(code).species,
            **{"class": dec.lipid_class},
            adduct=adduct,
            ref_mz=abs(theoretical_mz(attr, phase, code)),
            feat_feature_id=fid,
        )
    raise ValueError(template)


def fixture():
    pool = pd.DataFrame([row("PE(x:x)", 36, 1), row("PC(x:x)", 33, 1)])
    evidence = pd.DataFrame(
        1.0,
        index=pool.index,
        columns=[
            m.RAW_PREFIX + f for f in (*m.MEASURES, "profile_measurement_available")
        ],
    )
    for f in m.CONTRASTS:
        evidence[m.RAW_PREFIX + f] = [0.9, 0.4]
    return (
        pool,
        evidence,
        native_adduct_families(str(ROOT)),
        pd.Series(True, index=pool.index),
    )


def compare(p, e, f, physical):
    output, links = m.measure(p, e, f, physical)
    checked, checked_links = independent_measure(
        p, e, f, physical, m.FIELDS, m.MEASURES, m.CONTRASTS, m.PREFIX, m.RAW_PREFIX
    )
    pd.testing.assert_frame_equal(
        output, checked, check_exact=False, atol=1e-12, rtol=1e-12
    )
    pd.testing.assert_frame_equal(links, checked_links)
    for _idx, r in p.iterrows():
        assert m.signature(r, f) == pytest.approx(independent_signature(r, f))
    return output, links


def test_isobaric_PE_PC_has_disjoint_negative_ion_evidence():
    p, e, f, physical = fixture()
    assert p.ref_mz.iloc[0] == pytest.approx(p.ref_mz.iloc[1], abs=1e-10)
    x, links = compare(p, e, f, physical)
    assert x[m.PREFIX + "alternative_families"].eq(1).all()
    assert x[
        m.PREFIX + "own_minus_alternative_best_raw_correlation"
    ].tolist() == pytest.approx([0.5, -0.5])
    assert len(links) == 2


@pytest.mark.parametrize(
    "left,right,phase,adduct",
    [
        (row("PC(O-x:x)", 36, 2), row("PC(P-x:x)", 36, 1), "RP", "[M+H]+"),
        (
            row("BMP(x:x)", 36, 1, "NP", "[M+NH4]+"),
            row("PG(x:x)", 36, 1, "NP", "[M+NH4]+"),
            "NP",
            "[M+NH4]+",
        ),
    ],
)
def test_identical_or_shared_ion_sets_cannot_discriminate_classes(
    left, right, phase, adduct
):
    _, e, f, _ = fixture()
    p = pd.DataFrame([left, right])
    x, links = compare(p, e, f, pd.Series(True, index=p.index))
    assert x[m.PREFIX + "alternative_families"].eq(0).all()
    assert x[m.PREFIX + "own_minus_alternative_best_raw_correlation"].isna().all()
    assert links.empty


def test_profile_unknown_protection_and_stream_namespaces():
    p, e, f, phy = fixture()
    e.loc[1, m.RAW_PREFIX + "profile_measurement_available"] = 0
    e.loc[1, m.RAW_PREFIX + "best_raw_correlation"] = np.nan
    x, _ = compare(p, e, f, phy)
    assert x.loc[0, m.PREFIX + "alternative_families"] == 1
    assert x.loc[0, m.PREFIX + "measured_alternative_families"] == 0
    assert np.isnan(x.loc[0, m.PREFIX + "own_minus_alternative_best_raw_correlation"])
    phy.loc[1] = False
    x, links = compare(p, e, f, phy)
    assert links.empty
    p.loc[1, "stream"] = "NP_Pos"
    phy.loc[1] = True
    x, links = compare(p, e, f, phy)
    assert links.empty


def test_duplicate_hypothesis_does_not_multiply_ion_families():
    p, e, f, phy = fixture()
    p = pd.concat([p, p.iloc[[1]]], ignore_index=True)
    e = pd.concat([e, e.iloc[[1]]], ignore_index=True)
    x, links = compare(p, e, f, pd.Series(True, index=p.index))
    assert x.loc[0, m.PREFIX + "alternative_families"] == 1
    assert len(links.loc[links.source_index.eq(0)]) == 1
