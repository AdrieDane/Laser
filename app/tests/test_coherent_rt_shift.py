import numpy as np
import pandas as pd
import pytest

from matlab_annotation.coherent_rt_shift import FIELDS, estimate_coherent_rt_shift


def matches(offset=25.0, prefix="a"):
    return pd.DataFrame(
        [
            dict(
                feature_id=f"{prefix}{i}",
                composition_id=f"c{prefix}{i}",
                series_id="series",
                carbon=30 + 2 * (i % 4),
                db=1 + i // 4,
                mass_error_ppm=0.2,
                rt_residual_seconds=offset + [-0.2, 0.1, 0.2, 0][i % 4],
                intensity=1000.0 + 10 * i,
            )
            for i in range(16)
        ]
    )


def test_estimates_shared_shift_from_distinct_coherent_physical_features_without_mutation():
    data = matches()
    before = data.copy()
    result = estimate_coherent_rt_shift(data)
    assert result["supported"] and abs(result["shift_seconds"] - 25.0) <= 0.2
    assert (
        result["features"] >= 6
        and result["carbon_counts"] >= 3
        and result["doublebond_counts"] >= 2
    )
    pd.testing.assert_frame_equal(data, before, check_exact=True)


def test_duplicate_references_do_not_inflate_votes_and_identifiers_do_not_predict_offset():
    data = matches()
    original = estimate_coherent_rt_shift(data)
    copied = pd.concat([data] * 8, ignore_index=True)
    assert estimate_coherent_rt_shift(copied) == original
    renamed = data.assign(
        feature_id=[f"unrelated{i}" for i in range(len(data))],
        composition_id=[f"other{i}" for i in range(len(data))],
        series_id="different",
    )
    assert (
        estimate_coherent_rt_shift(renamed)["shift_seconds"]
        == original["shift_seconds"]
    )


def test_two_competing_coherent_modes_withhold_shift():
    data = pd.concat([matches(25.0, "a"), matches(-25.0, "b")], ignore_index=True)
    result = estimate_coherent_rt_shift(data)
    assert not result["supported"] and result["shift_seconds"] is None
    assert result["reason"] == "competing_RT_shift_modes"


def test_one_feature_many_labels_or_many_adducts_one_composition_cannot_invent_support():
    for data in (
        matches().assign(feature_id="one"),
        matches().assign(composition_id="one", carbon=30, db=1),
    ):
        assert not estimate_coherent_rt_shift(data)["supported"]


def test_noise_mass_failures_and_no_comparisons_do_not_force_a_prediction():
    empty = pd.DataFrame(columns=sorted(FIELDS))
    assert estimate_coherent_rt_shift(empty)["shift_seconds"] is None
    assert not estimate_coherent_rt_shift(matches().assign(mass_error_ppm=2.0))[
        "supported"
    ]
    assert not estimate_coherent_rt_shift(matches().assign(intensity=0.0))["supported"]
    data = matches()
    data["rt_residual_seconds"] = np.linspace(-80, 80, len(data))
    assert not estimate_coherent_rt_shift(data)["supported"]


def test_conflicting_physical_versions_and_supervision_columns_fail_closed():
    data = matches()
    conflict = pd.concat(
        [data, data.iloc[[0]].assign(rt_residual_seconds=10.0)], ignore_index=True
    )
    assert (
        estimate_coherent_rt_shift(conflict)["reason"]
        == "conflicting_duplicate_physical_hypotheses"
    )
    with pytest.raises(ValueError, match="label-free"):
        estimate_coherent_rt_shift(data.assign(training_positive=True))


def test_carbon_and_doublebond_span_and_neighbor_links_are_required():
    assert not estimate_coherent_rt_shift(matches().assign(db=0))["supported"]
    data = matches()
    data["carbon"] = 10 + 10 * np.arange(len(data))
    data["db"] = np.arange(len(data)) * 3
    assert not estimate_coherent_rt_shift(data)["supported"]


def test_adduct_aliases_cannot_invent_compositions_or_change_intensity_scale():
    data = matches()
    original = estimate_coherent_rt_shift(data)
    aliases = data.assign(composition_id="alias_" + data.composition_id)
    assert estimate_coherent_rt_shift(pd.concat([data, aliases])) == original
    aliases = matches().assign(carbon=30, db=1)
    assert not estimate_coherent_rt_shift(aliases)["supported"]


def test_equally_plausible_separated_peaks_do_not_get_selected_by_row_order():
    data = matches()
    data["rt_residual_seconds"] = 24.0
    other = data.assign(feature_id="other_" + data.feature_id, rt_residual_seconds=26.0)
    ambiguous = pd.concat([data, other], ignore_index=True)
    result = estimate_coherent_rt_shift(ambiguous)
    assert not result["supported"]
    assert estimate_coherent_rt_shift(ambiguous.iloc[::-1]) == result


def test_one_feature_inconsistent_intensity_is_withheld():
    data = matches()
    conflict = data.iloc[[0]].assign(composition_id="x", carbon=38, intensity=1e8)
    assert estimate_coherent_rt_shift(pd.concat([data, conflict]))["reason"] == (
        "inconsistent_physical_feature_intensity"
    )
