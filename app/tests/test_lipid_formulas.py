"""Candidate-space coverage and explicit scoring limits on synthetic ions."""
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from standard_lipidomics.acquisition_method import default_acquisition_method
from standard_lipidomics.isotope_patterns import fine_pattern
from standard_lipidomics.lipid_formulas import formula_candidates, neutral_valid, predict_formulas
from standard_lipidomics.untargeted import UntargetedComparisonCriteria
from test_complete_isotopes import fixture, INSTRUMENT

CRITERIA = UntargetedComparisonCriteria('A', 'B', isotope_model='complete', formula_model='lipid_aware')


@pytest.mark.parametrize('formula,adduct', [
    ('C42H82NO8P', '[M+H]+'), ('C42H84NO7P', '[M+H]+'),
    ('C42H82NO7P', '[M+H]+'), ('C42H82NO10P', '[M-H]-'),
    ('C81H150O17P2', '[M-2H]2-'), ('C120H236O6', '[M+NH4]+'),
    ('C27H44O2', '[M+H]+'),
])
def test_backbones_ethers_plasmalogens_oxidation_and_high_mass_without_library(monkeypatch, formula, adduct):
    monkeypatch.setattr('standard_lipidomics.complete_isotopes.formula_library',
                        lambda: (np.array([]), np.array([], str), 'synthetic-empty-library'))
    mode = 'RP_Neg' if adduct.endswith('-') else 'RP_Pos'
    mz = fine_pattern(formula, adduct)[0].mz
    candidates, audit = formula_candidates(mz, mode, CRITERIA, default_acquisition_method(), supplied_adduct=adduct)
    found = next(r for r in candidates if r['formula'] == formula)
    assert 'lipid backbone' in found['origins']
    assert 'public library' not in found['origins']
    assert audit['generated_candidates'] == len(candidates)
    assert abs(found['mass_error_ppm']) < .01


def test_de_novo_route_survives_absence_from_library_and_backbones(monkeypatch):
    monkeypatch.setattr('standard_lipidomics.complete_isotopes.formula_library',
                        lambda: (np.array([]), np.array([], str), 'empty'))
    formula, adduct = 'C25H42N3O4P', '[M+H]+'
    candidates, _ = formula_candidates(fine_pattern(formula, adduct)[0].mz, 'RP_Pos', CRITERIA,
                                       default_acquisition_method(), supplied_adduct=adduct)
    found = next(r for r in candidates if r['formula'] == formula)
    assert found['origins'] == 'de novo CHNOPS'


def test_valence_on_neutral_composition_handles_phosphorus_and_sulfur():
    assert neutral_valid({'C': 42, 'H': 82, 'N': 1, 'O': 8, 'P': 1})
    assert neutral_valid({'C': 48, 'H': 93, 'N': 1, 'O': 11, 'S': 1})
    assert not neutral_valid({'C': 42, 'H': 83, 'N': 1, 'O': 8, 'P': 1})
    assert not neutral_valid({'C': 2, 'H': 20, 'O': 1})
    assert not neutral_valid({'C': 42, 'H': 82, 'Cl': 1})


def test_explicit_budget_keeps_unscored_candidates_and_roundtrips(tmp_path):
    from standard_lipidomics.isotope_quality import IsotopeIndex
    from standard_lipidomics.complete_isotopes import apply_complete_isotopes
    from standard_lipidomics.project_store import save_untargeted_workflow, load_untargeted_workflow
    frame = fixture()
    method = default_acquisition_method()
    method['methods']['RP']['instrument'].update(INSTRUMENT)
    criteria = replace(CRITERIA, formula_scoring_limit=1)
    row = {'stream': 'RP_Pos', 'feature_id': '0', 'mz': frame.mz.iloc[0], 'rt': 5., 'p_value': .001,
           'compound_id': 'compound_0001', 'adduct': '[M+H]+', 'raw_p_significant': True, 'significant': True}
    scores, _, audit = predict_formulas(IsotopeIndex('RP_Pos', frame), row, 'RP_Pos', criteria, method)
    assert audit['generated_candidates'] == len(scores) > 1
    assert sum(s['isotope_scored'] for s in scores) == 1
    assert audit['scoring_budget_exhausted']
    original = pd.DataFrame([row])
    analysis = {'features': original, 'raw_p_features': original, 'significant_features': original,
                'compound_features': original, 'criteria': asdict(criteria), 'acquisition_method': method}
    result = {'streams': {'RP_Pos': {'features': frame}}, 'automatic_annotations': pd.DataFrame()}
    complete = apply_complete_isotopes(result, analysis, criteria)
    pd.testing.assert_frame_equal(complete['features'][original.columns], original)
    assert 'all retained' in complete['raw_p_features'].full_isotope_quality.iloc[0]
    save_untargeted_workflow(tmp_path, result, complete)
    loaded, _ = load_untargeted_workflow(tmp_path)
    saved = loaded['untargeted_analysis']
    assert saved['criteria']['formula_model'] == 'lipid_aware'
    assert saved['formula_model_provenance'] == complete['formula_model_provenance']
    assert saved['formula_search_audit'].generated_candidates.iloc[0] == len(scores)


def test_user_excluded_ions_and_criteria_dependency_remain_enforced():
    method = default_acquisition_method()
    for ion in ['[M+2H]2+', '[M+Na-2H]-']:
        candidates, _ = formula_candidates(700., 'RP_Neg' if ion.endswith('-') else 'RP_Pos', CRITERIA,
                                           method, supplied_adduct=ion)
        assert not candidates
    with pytest.raises(ValueError, match='require complete'):
        replace(CRITERIA, isotope_model='current')
    for limit in [0, 257, 1.5, True]:
        with pytest.raises(ValueError):
            replace(CRITERIA, formula_scoring_limit=limit)


def test_background_formula_model_saves_budget_audit(tmp_path):
    from application.untargeted_jobs import start_untargeted_job
    from test_annotation_jobs import wait_job
    from standard_lipidomics.project_store import load_untargeted_workflow
    frame=fixture()
    for i,m in enumerate([1.,1.02,.98,20.,20.2,19.8]):
        frame[f'b{i}']=frame.b0*m
    method=default_acquisition_method()
    method['methods']['RP']['instrument'].update(INSTRUMENT)
    job=start_untargeted_job(tmp_path,replace(CRITERIA,formula_scoring_limit=8),
        result={'streams':{'RP_Pos':{'features':frame}},'acquisition_method':method,'automatic_annotations':pd.DataFrame()})
    assert wait_job(job)['status']=='complete',(job/'untargeted.log').read_text()
    loaded,_=load_untargeted_workflow(tmp_path)
    analysis=loaded['untargeted_analysis']
    assert analysis['criteria']['formula_model']=='lipid_aware'
    assert analysis['criteria']['formula_scoring_limit']==8
    assert not analysis['formula_search_audit'].empty
    assert analysis['formula_model_provenance']['model']=='lipid_aware'
