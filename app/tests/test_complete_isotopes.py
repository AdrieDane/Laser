"""Composition correctness and controlled measurement failures, not self-fit accuracy."""
from dataclasses import replace
import math

import pandas as pd
import pytest

from standard_lipidomics.isotope_patterns import (
    fine_pattern, ion_composition, theoretical_pattern, resolving_power,
)
from standard_lipidomics.isotope_quality import IsotopeIndex, score_envelope
from standard_lipidomics.untargeted import UntargetedComparisonCriteria

INSTRUMENT = {'model': 'Q Exactive Plus Orbitrap', 'resolution': 280000,
              'resolution_reference_mz': None}
FORMULA = 'C42H82NO8P'
CRITERIA = UntargetedComparisonCriteria('A', 'B', isotope_model='complete')


def fixture(formula=FORMULA, adduct='[M+H]+', *, missing=False, interference=False):
    pattern = theoretical_pattern(formula, adduct, INSTRUMENT)
    mono = pattern[0]
    rows = []
    for number, p in enumerate(pattern):
        ratio = p.probability/mono.probability
        if ratio < .001 or (missing and p.nominal > 0):
            continue
        rows.append({'feature_id': str(number), 'mz': p.mz, 'rt': 5., 'rtmin': 4.98, 'rtmax': 5.02,
                     **{f'b{i}': ratio*(i+1)*10000 for i in range(6)},
                     **{f'qc{i}': ratio*(i+1)*20000 for i in range(3)}, 'blank': 1e12})
    if interference:
        for r in rows[1:]:
            for i in range(6):
                r[f'b{i}'] *= 20
    frame = pd.DataFrame(rows)
    frame.attrs.update(sample_columns=[*[f'b{i}' for i in range(6)], *[f'qc{i}' for i in range(3)], 'blank'],
        sample_group_map={**{f'b{i}': 'A' if i < 3 else 'B' for i in range(6)},
                          **{f'qc{i}': 'QC_pool' for i in range(3)}, 'blank': 'Blank'},
        sample_type_map={**{f'b{i}': 'Sample' for i in range(6)},
                         **{f'qc{i}': 'QC' for i in range(3)}, 'blank': 'Blank'})
    return frame


def test_carbon_binomial_probabilities_and_electron_charge_independently():
    # Analytic binomial expectation, independent of the simulated envelope.
    import IsoSpecPy
    c = IsoSpecPy.Iso(formula='C1')
    p0, p1 = c.isotopeProbabilities[0]
    pattern = fine_pattern('C10', '[M+Na]+')
    first = next(p for p in pattern if p.nominal == 1)
    assert first.probability/pattern[0].probability == pytest.approx(10*p1/p0)
    assert pattern[0].mz == pytest.approx(142.9892207, abs=2e-5)
    negative = fine_pattern('C10H2', '[M-2H]2-')
    assert negative[0].mz == pytest.approx(60.0005485799, abs=2e-7)
    assert negative[1].mz-negative[0].mz == pytest.approx(1.0033548352/2)


@pytest.mark.parametrize('adduct,delta', [('[M+NH4]+', {'N': 1, 'H': 4}),
    ('[M+HCOO]-', {'C': 1, 'H': 1, 'O': 2}), ('[M+CH3COO]-', {'C': 2, 'H': 3, 'O': 2}),
    ('[M+Cl]-', {'Cl': 1}), ('[M+K]+', {'K': 1}), ('[M+H-H2O]+', {'H': -1, 'O': -1})])
def test_all_adduct_atoms_in_complete_composition(adduct, delta):
    from standard_lipidomics.isotope_patterns import parse_formula
    atoms, _ = ion_composition(FORMULA, adduct)
    neutral = parse_formula(FORMULA)
    for element, n in delta.items():
        assert atoms.get(element, 0) == neutral.get(element, 0)+n


def test_chloride_and_sulfur_are_not_carbon_only_patterns():
    cl = fine_pattern('C10H20O2', '[M+Cl]-')
    r2 = sum(p.probability for p in cl if p.nominal == 2)/cl[0].probability
    assert r2 > .3
    sulfur = fine_pattern('C48H93NO11S', '[M-H]-')
    assert sum(p.probability for p in sulfur if p.nominal == 2)/sulfur[0].probability > .2


def test_resolution_is_explicit_and_unknown_is_not_fine_structure():
    power, basis = resolving_power(INSTRUMENT, 800.)
    assert power == 140000 and 'manufacturer' in basis
    assert resolving_power({'model': 'unknown', 'resolution': 280000}, 800.)[0] is None
    coarse = theoretical_pattern(FORMULA, '[M+Cl]-', {})
    fine = theoretical_pattern(FORMULA, '[M+Cl]-', {**INSTRUMENT, 'resolution': 2000000})
    assert len(fine) > len(coarse)


@pytest.mark.parametrize('formula,adduct', [(FORMULA, '[M+H]+'), ('C39H76NO8P', '[M+HCOO]-'),
    ('C48H93NO11S', '[M-H]-'), ('C73H142O17P2', '[M-2H]2-'), (FORMULA, '[M+Cl]-')])
def test_controlled_envelopes_and_qc_separation(formula, adduct):
    frame = fixture(formula, adduct)
    index = IsotopeIndex('RP_Neg' if adduct.endswith('-') else 'RP_Pos', frame)
    score, peaks = score_envelope(index, '0', formula, adduct, CRITERIA, INSTRUMENT)
    assert score['status'] == 'compatible isotope evidence'
    assert score['score'] < 1e-7
    assert score['evaluable_components'] >= 2
    assert all(r['paired_samples'] == 6 and r['qc_pairs'] == 3 for r in peaks if r['selected'])
    assert 'blank' not in index.ratio_samples


def test_missing_isotopes_retain_hypothesis_and_mass_only_library_candidates():
    from standard_lipidomics.acquisition_method import default_acquisition_method
    from standard_lipidomics.complete_isotopes import candidate_hypotheses
    frame = fixture(missing=True)
    score, _ = score_envelope(IsotopeIndex('RP_Pos', frame), '0', FORMULA, '[M+H]+', CRITERIA, INSTRUMENT)
    assert score['status'].startswith('mass-only') and math.isfinite(score['score'])
    candidates = candidate_hypotheses({'mz': frame.mz.iloc[0]}, 'RP_Pos', CRITERIA, default_acquisition_method())
    assert (FORMULA, '[M+H]+') in {(f, a) for f, a, _ in candidates}


def test_interference_saturation_and_ratio_instability_flagged():
    f = fixture(interference=True)
    score, _ = score_envelope(IsotopeIndex('RP_Pos', f), '0', FORMULA, '[M+H]+', CRITERIA, INSTRUMENT)
    assert score['status'] == 'isotope ratio discordance' and 'interference' in score['quality_flags']
    f = fixture()
    f['saturated'] = f.feature_id.eq('0')
    score, _ = score_envelope(IsotopeIndex('RP_Pos', f), '0', FORMULA, '[M+H]+', CRITERIA, INSTRUMENT)
    assert score['evaluable_components'] == 0 and 'saturation' in score['quality_flags']
    assert score['saturation_status'] == 'supplied flag'


def test_isomer_candidates_not_silently_picked_by_row_order():
    f = fixture()
    duplicate = f.iloc[1].copy()
    duplicate.feature_id = 'isomer'
    g = pd.concat([f, duplicate.to_frame().T], ignore_index=True)
    g.attrs.update(f.attrs)
    score, peaks = score_envelope(IsotopeIndex('RP_Pos', g), '0', FORMULA, '[M+H]+', CRITERIA, INSTRUMENT)
    assert 'competing feature' in score['quality_flags']
    assert any(r['status'] == 'ambiguous' for r in peaks)


def test_natural_abundance_rejects_labelled_and_user_disabled_ions():
    for formula, adduct in [('C10H20D3', '[M+H]+'), ('[13C]10H20', '[M+H]+'),
                            (FORMULA, '[M+2H]2+'), (FORMULA, '[M+Na-2H]-')]:
        with pytest.raises(ValueError):
            fine_pattern(formula, adduct)
    with pytest.raises(ValueError):
        replace(CRITERIA, isotope_model='unknown')


def test_additive_save_reopen_preserves_all_original_values(tmp_path):
    from dataclasses import asdict
    from standard_lipidomics.acquisition_method import default_acquisition_method
    from standard_lipidomics.complete_isotopes import apply_complete_isotopes
    from standard_lipidomics.project_store import save_untargeted_workflow, load_untargeted_workflow
    features = fixture()
    method = default_acquisition_method()
    method['methods']['RP']['instrument'].update(INSTRUMENT)
    original = pd.DataFrame([{'stream': 'RP_Pos', 'feature_id': '0', 'mz': features.mz.iloc[0],
        'rt': 5., 'p_value': .001, 'raw_p_significant': True, 'significant': True,
        'compound_id': 'compound_0001', 'adduct': '[M+H]+', 'predicted_chemical_formula': 'old value'}])
    analysis = {'features': original, 'raw_p_features': original.copy(), 'significant_features': original.copy(),
                'compound_features': original.copy(), 'criteria': asdict(CRITERIA), 'acquisition_method': method}
    result = {'streams': {'RP_Pos': {'features': features}}, 'automatic_annotations': pd.DataFrame()}
    current = apply_complete_isotopes(result, analysis, replace(CRITERIA, isotope_model='current'))
    assert current is analysis
    complete = apply_complete_isotopes(result, analysis, CRITERIA)
    pd.testing.assert_frame_equal(complete['features'][original.columns], original)
    assert not complete['isotope_candidate_scores'].empty
    assert not complete['isotope_peak_evidence'].empty
    save_untargeted_workflow(tmp_path, result, complete)
    reopened, _ = load_untargeted_workflow(tmp_path)
    saved = reopened['untargeted_analysis']
    assert saved['criteria']['isotope_model'] == 'complete'
    assert saved['isotope_model_provenance'] == complete['isotope_model_provenance']
    assert len(saved['isotope_peak_evidence']) == len(complete['isotope_peak_evidence'])
    assert saved['features'].p_value.iloc[0] == .001


def test_complete_isotope_background_job_freezes_model_and_saves_evidence(tmp_path):
    from application.untargeted_jobs import start_untargeted_job
    from test_annotation_jobs import wait_job
    from standard_lipidomics.acquisition_method import default_acquisition_method
    from standard_lipidomics.project_store import load_untargeted_workflow
    method = default_acquisition_method()
    method['methods']['RP']['instrument'].update(INSTRUMENT)
    features = fixture()
    for i, multiplier in enumerate([1., 1.02, .98, 20., 20.2, 19.8]):
        features[f'b{i}'] = features.b0 * multiplier
    result = {'streams': {'RP_Pos': {'features': features}}, 'automatic_annotations': pd.DataFrame(),
              'acquisition_method': method}
    job = start_untargeted_job(tmp_path, CRITERIA, result=result)
    state = wait_job(job)
    assert state['status'] == 'complete', (job/'untargeted.log').read_text()
    loaded, _ = load_untargeted_workflow(tmp_path)
    analysis = loaded['untargeted_analysis']
    assert analysis['criteria']['isotope_model'] == 'complete'
    assert analysis['isotope_model_provenance']['engine_version'] == '2.6.1'
    assert not analysis['isotope_candidate_scores'].empty
