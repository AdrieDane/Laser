import numpy as np
import pandas as pd
import pytest

from matlab_annotation.run_context_evidence import run_context_evidence


def data():
    auto = pd.DataFrame(
        [
            dict(
                name=f"PC({c}:2)",
                **{"class": "PC"},
                carbon=c,
                db=2,
                feat_intensity=1e6,
                ppm_error=0.2,
                post_regression_confidence=0.99,
                is_is=False,
            )
            for c in [30, 32, 34, 36, 38, 40, 42]
        ]
    )
    targets = pd.DataFrame(
        {
            "class": ["PC-O", "PC"],
            "carbon": [36, 36],
            "db": [2, 2],
            "feat_intensity": [1e5, 1e5],
        }
    )
    return auto, targets


def test_context_uses_other_compositions_and_does_not_read_names_or_ids_as_features():
    auto, targets = data()
    expected = run_context_evidence(targets, auto[~auto.carbon.eq(36)])
    fake = auto[auto.carbon.eq(36)].assign(feat_intensity=1e12)
    actual = run_context_evidence(
        targets, pd.concat([auto, pd.concat([fake] * 20)], ignore_index=True)
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.q_context_carbon_median.iloc[1] == 36
    assert np.isnan(
        actual.q_context_carbon_median.iloc[0]
    )  # Distinct structural class.
    assert all(c.startswith("q_") for c in actual)


def test_only_declared_commensurate_standard_quantities_can_normalize_abundance():
    auto, targets = data()
    standards = pd.DataFrame(
        [
            dict(
                name=f"{cls}(14:0) (IS)",
                **{"class": cls},
                carbon=14,
                db=0,
                feat_intensity=1e6,
                ppm_error=0.1,
                is_is=True,
                internal_standard_name=cls,
                internal_standard_quantity=100,
                internal_standard_unit="nM",
            )
            for cls in ["PC", "PE", "PG"]
        ]
    )
    result = run_context_evidence(
        targets, pd.concat([auto, standards], ignore_index=True)
    )
    assert result.q_global_standard_calibrators.eq(3).all()
    assert result.q_class_standard_calibrators.eq(1).all()
    assert result.q_log_over_global_standard_response.iloc[0] == pytest.approx(
        np.log1p(1e5) - np.log(1e4)
    )
    bad = run_context_evidence(
        targets,
        pd.concat(
            [auto, standards.assign(internal_standard_unit="unknown")],
            ignore_index=True,
        ),
    )
    assert bad.q_log_over_global_standard_response.isna().all()


def test_sparse_or_weak_anchor_context_stays_unknown():
    auto, targets = data()
    output = run_context_evidence(targets, auto.assign(post_regression_confidence=0.8))
    assert output.q_other_composition_anchors.eq(0).all()
    assert output.q_context_carbon_median.isna().all()
