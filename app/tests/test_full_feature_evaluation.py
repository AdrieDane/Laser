"""Reference-pair reporting must use the declared cross-mode denominator."""
from pathlib import Path

import pandas as pd


def test_cross_mode_pair_metric_excludes_within_mode_pairs(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'tools'))
    from evaluate_full_feature_groups import reference_pair_metrics

    reference = pd.DataFrame([
        {'stream': 'RP_Pos', 'feature_id': 'h', 'mapped': True, 'name': 'PC synthetic',
         'adduct': '[M+H]+', 'manual_rt_seconds': 600},
        {'stream': 'RP_Pos', 'feature_id': 'na', 'mapped': True, 'name': 'PC synthetic',
         'adduct': '[M+Na]+', 'manual_rt_seconds': 600},
        {'stream': 'NP_Pos', 'feature_id': 'np', 'mapped': True, 'name': 'PC synthetic',
         'adduct': '[M+H]+', 'manual_rt_seconds': 120},
    ])
    predictions = reference[['stream', 'feature_id', 'adduct']].assign(compound_id='compound_0001')
    result = reference_pair_metrics(reference, predictions)
    assert result['reference_pair_agreement'] == {'n': 2, 'N': 2, 'percent': 100.0}
    assert result['compatible_mapped_reference_pair_recovery'] == {'n': 2, 'N': 2, 'percent': 100.0}
    assert result['discordant_reference_pairs'] == 0

    predictions.loc[predictions.feature_id.eq('np'), 'adduct'] = '[M+K]+'
    wrong = reference_pair_metrics(reference, predictions)
    assert wrong['reference_pair_agreement'] == {'n': 0, 'N': 2, 'percent': 0.0}
    assert wrong['discordant_reference_pairs'] == 2
