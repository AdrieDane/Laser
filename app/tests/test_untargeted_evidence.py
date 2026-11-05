"""Independent mass examples and complete detached untargeted save/reopen flow."""
from pathlib import Path
import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from application.annotation_jobs import read_annotation_job
from application.untargeted_jobs import start_untargeted_job, load_untargeted_inputs
from presentation.standard_untargeted import _untargeted_sample_bar_figure
from standard_lipidomics.project_store import save_untargeted_workflow, load_untargeted_workflow, save_annotation_workflow
from standard_lipidomics.untargeted import (
    UntargetedComparisonCriteria, analyse_standard_untargeted, _elemental_formula_candidates,
)
from standard_lipidomics.untargeted_adducts import search_adduct_evidence
from standard_lipidomics.untargeted_series import attach_lipid_series
from test_annotation_jobs import wait_job
import test_standard_untargeted as fixtures

APP = Path(__file__).resolve().parents[1] / 'app.py'


def test_example_mass_links_all_modes_with_adduct_changes_and_phase_rt_rules():
    mz = 678.5068
    neutral = mz - 1.007276466621
    def source(ids, masses, rt):
        return pd.DataFrame({'feature_id': ids, 'mz': masses, 'rt': rt,
                              'already_annotated': [False] * len(ids)}).sort_values('mz')
    sources = {
        'RP_Pos': source(['hit', 'sodium', 'far'], [mz, neutral + 22.989220702, neutral + 38.963157907], [8., 8.01, 12.]),
        'NP_Pos': source(['ammonium'], [neutral + 18.033825553], [2.]),
        'RP_Neg': source(['formate'], [neutral + 44.998202851], [8.02]),
        'NP_Neg': source(['acetate'], [neutral + 59.013852915], [1.]),
    }
    hits = {'RP_Pos': pd.DataFrame({'feature_id': ['hit'], 'mz': [mz], 'rt': [8.],
        'raw_p_significant': [True], 'isotope_charge_hypothesis': [1]})}
    links = search_adduct_evidence(sources, hits, ppm=3., rt_tolerance=.1)
    proton = links[links.source_adduct.eq('[M+H]+')]
    assert {'sodium', 'ammonium', 'formate', 'acetate'} <= set(proton.target_feature_id)
    assert 'far' not in set(proton.target_feature_id)
    assert proton[proton.target_stream.str.startswith('NP')].rt_compared.eq(False).all()
    assert proton[proton.target_stream.str.startswith('RP')].rt_compared.eq(True).all()
    assert proton.mass_error_ppm.abs().max() < 1e-6
    assert set(proton.target_adduct) >= {'[M+NH4]+', '[M+HCOO]-', '[M+CH3COO]-'}


def test_doubly_charged_source_converts_to_singly_charged_negative_ion():
    source = pd.DataFrame({'feature_id': ['negative'], 'mz': [998.992723533379],
                           'rt': [5.], 'already_annotated': [True]})
    hits = {'RP_Pos': pd.DataFrame({'feature_id': ['hit'], 'mz': [501.007276466621],
        'rt': [5.], 'raw_p_significant': [True], 'isotope_charge_hypothesis': [2]})}
    links = search_adduct_evidence({'RP_Neg': source}, hits, ppm=3., rt_tolerance=.1)
    assert len(links) == 1
    assert links.iloc[0].source_adduct == '[M+2H]2+'
    assert links.iloc[0].target_adduct == '[M-H]-'
    assert links.iloc[0].neutral_mass == pytest.approx(1000.)
    assert bool(links.iloc[0].target_is_annotated)


def test_isotope_constrained_formula_includes_phospholipid_and_charge_conversion():
    # Independent C36H72NO8P neutral formula, close to the user's 678.5068 example.
    neutral = 36*12 + 72*1.00782503223 + 14.00307400443 + 8*15.99491461957 + 30.97376199842
    text, count = _elemental_formula_candidates(neutral+1.007276466621, 'pos', 36., 3., m_plus_2_percent=9.)
    assert 'C36H72N1O8P1' in text
    assert count > 1  # Accurate mass and isotope ratios need not give a unique formula.
    text2, _ = _elemental_formula_candidates((neutral+2*1.007276466621)/2, 'pos', 36., 3., charge=2)
    assert '[M+2H]2+ C36H72N1O8P1' in text2
    assert _elemental_formula_candidates(neutral, 'pos', np.nan, 3.) == ('', 0)


def test_raw_p_list_includes_hits_below_extra_fold_change_gate():
    result = fixtures.StandardUntargetedTests()._result()
    criteria = UntargetedComparisonCriteria('Group_01_control', 'Group_02_disease',
        p_value_threshold=.05, min_abs_log2_fold_change=20.)
    analysis = analyse_standard_untargeted(result, criteria)
    assert analysis['significant_features'].empty
    raw = analysis['raw_p_features'].set_index(['stream', 'feature_id'])
    assert ('RP_Pos', 'diff') in raw.index
    assert raw.loc[('RP_Pos', 'diff'), 'isotope_m_plus_1_feature_id'] == 'diff_m1'
    assert analysis['adduct_matches'].source_feature_id.eq('diff').any()
    assert not set(['ann1', 'ann2', 'ann3']) & set(raw.index.get_level_values('feature_id'))


def test_lipid_series_includes_annotated_ch2_and_c2h4_neighbors_and_rejects_far_rt():
    hits = {'RP_Pos': pd.DataFrame({'feature_id': ['hit'], 'mz': [678.5068], 'rt': [8.],
        'raw_p_significant': [True], 'isotope_charge_hypothesis': [1]})}
    source = pd.DataFrame({'feature_id': ['hit', 'pc1', 'pc2', 'far'],
        'mz': [678.5068, 692.522450064, 706.538100128, 720.553750192],
        'rt': [8., 8.1, 8.2, 12.], 'already_annotated': [False, True, True, False]})
    classes = pd.DataFrame({'feature_id': ['pc1', 'pc2'], 'lipid_class': ['PC', 'PC']})
    tables, links = attach_lipid_series(hits, {'RP_Pos': source}, {'RP_Pos': classes}, UntargetedComparisonCriteria('a', 'b'))
    assert set(links.target_feature_id) == {'pc1', 'pc2'}
    assert set(links.ch2_steps) == {1, 2}
    assert links.mass_error_da.abs().max() < 1e-9
    row = tables['RP_Pos'].iloc[0]
    assert row.lipid_series_size == 3
    assert row.lipid_series_class_context == 'PC'


def test_untargeted_bar_uses_original_sample_levels_and_metadata_groups():
    result = fixtures.StandardUntargetedTests()._result()
    figure = _untargeted_sample_bar_figure(result, pd.Series({'stream': 'RP_Pos', 'feature_id': 'diff'}))
    assert figure.layout.xaxis.showticklabels is False
    assert figure.layout.height == 280
    traces = {trace.name: trace for trace in figure.data}
    control = next(trace for name, trace in traces.items() if 'control' in name)
    np.testing.assert_allclose(control.y, [10., 10., 10.])
    assert set(control.customdata) == {'a1', 'a2', 'a3'}


def test_detached_untargeted_job_freezes_inputs_and_exports_all_evidence(tmp_path):
    result = fixtures.StandardUntargetedTests()._result()
    criteria = UntargetedComparisonCriteria('Group_01_control', 'Group_02_disease', p_value_threshold=.05)
    job = start_untargeted_job(tmp_path, criteria, result=result)
    result['streams']['RP_Pos']['features'].loc[:, 'mz'] = 100.
    state = wait_job(job)
    assert state['status'] == 'complete', (job/'untargeted.log').read_text()
    reopened, _ = load_untargeted_workflow(tmp_path)
    analysis = reopened['untargeted_analysis']
    assert analysis['evidence_version'] == 3
    assert not analysis['adduct_matches'].empty
    assert not analysis['raw_p_features'].empty
    assert analysis['features'].query("stream == 'RP_Pos' and feature_id == 'diff'").iloc[0].mz == 721.
    assert 'ann1' not in set(analysis['raw_p_features'].feature_id)
    assert (tmp_path/'Untargeted/tables/untargeted_adduct_matches.csv').is_file()
    assert (tmp_path/'Untargeted/tables/untargeted_homologous_matches.csv').is_file()
    assert read_annotation_job(job)['status'] == 'complete'


def test_duplicate_untargeted_start_and_completion_page(tmp_path):
    result = fixtures.StandardUntargetedTests()._result()
    criteria = UntargetedComparisonCriteria('Group_01_control', 'Group_02_disease')
    with (patch('application.annotation_jobs.subprocess.Popen') as process,
          patch('application.annotation_jobs.process_activity', return_value={'alive': True, 'cpu_seconds': 0.})):
        process.return_value.pid = os.getpid()
        first = start_untargeted_job(tmp_path, criteria, result=result)
        assert start_untargeted_job(tmp_path, criteria, result=result) == first
        assert process.call_count == 1
    (first/'status.json').write_text(json.dumps({'status': 'complete', 'message': 'Untargeted analysis complete'}))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.session_state['standard_pipeline_page'] = 'untargeted_job'
    app.session_state['standard_untargeted_job_path'] = str(first)
    app.run()
    assert not app.exception
    assert [s.value for s in app.success] == ['Untargeted analysis complete']
    assert not app.get('plotly_chart')


def test_completed_untargeted_viewer_never_recalculates(tmp_path):
    result = fixtures.StandardUntargetedTests()._result()
    analysis = analyse_standard_untargeted(result, UntargetedComparisonCriteria('Group_01_control', 'Group_02_disease'))
    save_untargeted_workflow(tmp_path, result, analysis)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.button(key='standard-module-open_untargeted').click().run()
    app.text_input(key='standard-open-project-folder').set_value(str(tmp_path))
    with patch('standard_lipidomics.untargeted.analyse_standard_untargeted', side_effect=AssertionError('Viewer must not run pipeline')):
        next(b for b in app.button if b.label == 'Open completed untargeted analysis').click().run()
    assert not app.exception
    assert app.get('plotly_chart')
    assert app.session_state['standard_untargeted_mode_payload'][1]['statistics']
    assert not any(b.label == 'Run untargeted analysis' for b in app.button)


def test_pipeline_launch_uses_completed_annotation_exclusions(tmp_path):
    from test_standard_project_store import _annotation_result
    saved = _annotation_result()
    save_annotation_workflow(tmp_path, saved)
    inputs, provenance = load_untargeted_inputs(tmp_path)
    assert 'LASER' in provenance
    pd.testing.assert_frame_equal(inputs['automatic_annotations'], saved['automatic_annotations'], check_dtype=False)


def test_launch_module_only_offers_pipeline_settings_and_run(tmp_path):
    from test_standard_project_store import _annotation_result
    save_annotation_workflow(tmp_path, {**_annotation_result(), **fixtures.StandardUntargetedTests()._result()})
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.button(key='standard-module-untargeted').click().run()
    app.text_input(key='standard-untargeted-input-project').set_value(str(tmp_path)).run()
    assert not app.exception
    assert any(button.label == 'Run untargeted analysis' for button in app.button)
    assert not app.get('plotly_chart')
    assert not any(header.value == 'Features with significant raw p-value' for header in app.subheader)


def test_empty_analysis_saves_and_reopens_without_hits(tmp_path):
    result = fixtures.StandardUntargetedTests()._result()
    criteria = UntargetedComparisonCriteria('Group_01_control', 'Group_02_disease', p_value_threshold=1e-30)
    analysis = analyse_standard_untargeted(result, criteria)
    assert analysis['raw_p_features'].empty
    save_untargeted_workflow(tmp_path, result, analysis)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.button(key='standard-module-open_untargeted').click().run()
    app.text_input(key='standard-open-project-folder').set_value(str(tmp_path))
    next(b for b in app.button if b.label == 'Open completed untargeted analysis').click().run()
    assert not app.exception
    assert any('No unannotated feature' in i.value for i in app.info)


@pytest.mark.parametrize('charge', [1,2])
def test_isotope_charge_uses_complete_envelope_and_rejects_impossible_ratio(charge):
    from standard_lipidomics.untargeted import _attach_isotope_evidence
    mz=(677.49955528+charge*1.007276466621)/charge
    ratio=.0107/.9893
    rows=[]
    for name,mass,intensity,rt in [('mono',mz,1e5,5.),('m1',mz+1.00335483507/charge,1e5*36*ratio,5.),
        ('m2',mz+2*1.00335483507/charge,1e5*36*35/2*ratio**2,5.),
        ('decoy_parent',mz,1e5,10.),('decoy',mz+1.00335483507/charge,1e6,10.)]:
        rows.append({'feature_id':name,'mz':mass,'rt':rt,**{s:intensity for s in ['a1','a2','b1','b2']}})
    features=pd.DataFrame(rows)
    features.attrs['sample_group_map']={'a1':'A','a2':'A','b1':'B','b2':'B'}
    hits=features[features.feature_id.isin(['mono','decoy_parent'])][['feature_id','mz','rt']].copy()
    hits['significant']=True
    hits['raw_p_significant']=True
    hits['mean_intensity_a']=1e5
    hits['mean_intensity_b']=1e5
    output=_attach_isotope_evidence({'streams':{'RP_Pos':{'features':features}}},{'RP_Pos':hits},UntargetedComparisonCriteria('A','B'))['RP_Pos'].set_index('feature_id')
    assert output.loc['mono','isotope_charge_hypothesis']==charge
    assert output.loc['mono','isotope_m_plus_1_feature_id']=='m1'
    assert output.loc['mono','estimated_carbon_count_from_m_plus_1']==pytest.approx(36.)
    assert output.loc['decoy_parent','isotope_m_plus_1_feature_id']==''


def test_same_mode_adducts_require_tighter_coelution_than_opposite_polarity():
    mz=678.5068
    sodium=mz-1.007276466621+22.989220702
    source=pd.DataFrame({'feature_id':['near','far'],'mz':[sodium,sodium], 'rt':[5.02,5.15], 'already_annotated':[False,False]})
    query=pd.DataFrame({'feature_id':['hit'],'mz':[mz],'rt':[5.], 'raw_p_significant':[True], 'isotope_charge_hypothesis':[1]})
    result=search_adduct_evidence({'RP_Pos':source},{'RP_Pos':query},ppm=3.,rt_tolerance=.2)
    assert set(result.query("source_adduct == '[M+H]+' and target_adduct == '[M+Na]+'").target_feature_id)=={'near'}
