from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from matlab_annotation.raw_counterpart_rt import (
    independent_spike_pairs,
    fit_independent_rt_links,
    mapped_counterpart_rt,
    opposite,
)

ROOT = Path(__file__).resolve().parents[2] / "reference_data/matlab_pipeline"


def fixture(phase="RP"):
    source_mass = np.arange(500.0, 560.0, 10.0)
    source_seconds = np.array([180.0, 240.0, 300.0, 360.0, 480.0, 600.0])
    target_seconds = source_seconds * 1.01 + 2
    rows = []
    for i, (mass, rt) in enumerate(zip(source_mass, source_seconds)):
        rows.append(
            dict(
                stream=phase + "_Pos",
                name=f"declared{i} (IS)",
                **{"class": "PC"},
                adduct="[M+H]+",
                ref_mz=mass,
                feat_rt=rt / 60,
                feat_intensity=100.0,
                feat_feature_id=f"source{i}",
                standard_compound_id=f"spike{i}",
                is_is=True,
            )
        )
    # One independently detected shared spike seeds augmentation; the other
    # target-native ions are deliberately NOT annotated/detected beforehand.
    rows.append(
        dict(
            stream=phase + "_Neg",
            name="declared0 (IS)",
            **{"class": "PC"},
            adduct="[M+CHO2]-",
            ref_mz=source_mass[0] + 43.990928,
            feat_rt=target_seconds[0] / 60,
            feat_intensity=100.0,
            feat_feature_id="target0",
            standard_compound_id="spike0",
            is_is=True,
        )
    )
    features = pd.DataFrame(
        dict(
            feature_id=[f"target{i}" for i in range(6)],
            mz=source_mass + 43.990928,
            rt=target_seconds / 60,
            intensity=100.0,
        )
    )
    return pd.DataFrame(rows), features


def test_unique_review_only_native_ions_can_link_declared_spikes_without_annotations():
    measurements, features = fixture()
    old = measurements.copy(deep=True)
    pairs = independent_spike_pairs(
        measurements, features, ROOT, source_stream="RP_Pos"
    )
    assert (
        len(pairs) == 6
        and pairs.origin.eq("unique_native_alternate_ion_evidence_only").sum() == 5
    )
    assert not {"automatic_accepted", "name", "adduct"} & set(pairs)
    links = fit_independent_rt_links(pairs, source_stream="RP_Pos")
    assert links["global_map"] is not None
    result, mapping = mapped_counterpart_rt(links, 400, "PC")
    assert np.isclose(result, 406) and mapping["anchor_compounds"] == 6
    assert mapped_counterpart_rt(links, 1000, "PC") is None
    pd.testing.assert_frame_equal(measurements, old, check_exact=True)


def test_ambiguous_raw_peak_or_absent_seed_never_becomes_false_RT_confirmation():
    measurements, features = fixture()
    duplicated = features.iloc[[1]].assign(
        feature_id="rival", rt=features.rt.iloc[1] + 0.01
    )
    pairs = independent_spike_pairs(
        measurements,
        pd.concat([features, duplicated], ignore_index=True),
        ROOT,
        source_stream="RP_Pos",
    )
    assert len(pairs) == 5 and "spike1" not in set(pairs.standard_compound_id)
    assert fit_independent_rt_links(pairs, source_stream="RP_Pos")["global_map"] is None
    no_seed = measurements.loc[measurements.stream.eq("RP_Pos")]
    pairs = independent_spike_pairs(no_seed, features, ROOT, source_stream="RP_Pos")
    assert pairs.empty


def test_NP_maps_are_exact_class_only_never_RP_global_affine_assumptions():
    measurements, features = fixture("NP")
    pairs = independent_spike_pairs(
        measurements, features, ROOT, source_stream="NP_Pos"
    )
    links = fit_independent_rt_links(pairs, source_stream="NP_Pos")
    assert links["global_map"] is None and set(links["class_maps"]) == {"PC"}
    assert mapped_counterpart_rt(links, 400, "PC") is not None
    assert mapped_counterpart_rt(links, 400, "PE") is None
    assert mapped_counterpart_rt(links, 400, "PC-O") is None


def test_polarity_charge_and_curator_supervision_are_hard_guards():
    measurements, features = fixture()
    with pytest.raises(ValueError, match="polarity"):
        independent_spike_pairs(
            measurements.assign(adduct="[M+CHO2]-"),
            features,
            ROOT,
            source_stream="RP_Pos",
        )
    with pytest.raises(ValueError, match="supervision"):
        independent_spike_pairs(
            measurements.assign(training_positive=True),
            features,
            ROOT,
            source_stream="RP_Pos",
        )
    with pytest.raises(ValueError):
        opposite("RPtoNP")
    assert opposite("RP_Pos") == "RP_Neg" and opposite("NP_Neg") == "NP_Pos"


def test_bad_RP_affine_or_leave_one_out_disagreement_withholds_map():
    measurements, features = fixture()
    pairs = independent_spike_pairs(
        measurements, features, ROOT, source_stream="RP_Pos"
    )
    for changed in (
        pairs.assign(target_rt_seconds=pairs.target_rt_seconds * 1.5),
        pairs.assign(target_rt_seconds=pairs.target_rt_seconds + [0, 0, 0, 0, 0, 20]),
    ):
        assert (
            fit_independent_rt_links(changed, source_stream="RP_Pos")["global_map"]
            is None
        )


def test_one_physical_peak_cannot_masquerade_as_two_independent_spike_compounds():
    measurements, features = fixture()
    alias = measurements.iloc[[1]].assign(standard_compound_id="another_label")
    pairs = independent_spike_pairs(
        pd.concat([measurements, alias], ignore_index=True),
        features,
        ROOT,
        source_stream="RP_Pos",
    )
    assert len(pairs) == 5
    assert "spike1" not in set(
        pairs.standard_compound_id
    ) and "another_label" not in set(pairs.standard_compound_id)
    assert fit_independent_rt_links(pairs, source_stream="RP_Pos")["global_map"] is None
