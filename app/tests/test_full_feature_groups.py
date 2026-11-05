from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from standard_lipidomics.acquisition_method import default_acquisition_method
from standard_lipidomics.full_feature_groups import build_full_feature_groups, BoundedFamilies
from standard_lipidomics.full_feature_evidence import FeatureIndex, paired_columns, profile_metrics
from standard_lipidomics.untargeted import UntargetedComparisonCriteria, analyse_standard_untargeted
from standard_lipidomics.project_store import save_untargeted_workflow, load_untargeted_workflow


def fixture():
    neutral = 700.
    coordinates = {'RP_Pos': [('h', neutral+1.007276466621, 5.), ('na', neutral+22.989220702, 5.01),
        ('a_isomer', neutral+1.007276466621, 6.), ('iso', neutral+1.007276466621+1.00335483507, 5.005),
        ('homologue', neutral+1.007276466621+14.015650064, 5.1)],
        'RP_Neg': [('neg', neutral-1.007276466621, 5.02)],
        'NP_Pos': [('np', neutral+1.007276466621, 2.)],
        'NP_Neg': [('npneg', neutral+44.998202851, 2.01)]}
    values = np.array([1., 2., 4., 8., 20., 40., 80., 160.])
    streams = {}
    for mode, rows in coordinates.items():
        frame = pd.DataFrame(rows, columns=['feature_id', 'mz', 'rt'])
        samples = [mode+str(i) for i in range(8)]
        for i, sample in enumerate(samples):
            frame[sample] = values[i] * np.array([.4 if fid == 'iso' else 1. for fid in frame.feature_id])
        frame['qc'] = 1e12
        frame.attrs.update(sample_columns=samples+['qc'],
            sample_group_map={**{s:'A' if i < 4 else 'B' for i,s in enumerate(samples)}, 'qc':'QC_pool'},
            sample_type_map={**{s:'study' for s in samples}, 'qc':'QC'},
            sample_metadata_records={s:{'Sample ID':str(i)} for i,s in enumerate(samples)})
        streams[mode] = {'features':frame}
    return {'streams':streams,'automatic_annotations':pd.DataFrame({'stream':['RP_Pos'], 'feature_id':['na']})}


def criteria(mode='full_feature_soft'):
    return UntargetedComparisonCriteria('A', 'B', p_value_threshold=.05, ion_grouping_mode=mode)


def test_all_original_features_retained_with_correct_mode_links_and_isomer_separation():
    result = fixture()
    out = build_full_feature_groups(result, criteria())
    table = out['ion_feature_ledger'].set_index(['stream','feature_id'])
    assert len(table) == sum(len(x['features']) for x in result['streams'].values())
    family = table.loc[('RP_Pos','h'), 'compound_id']
    for key, ion in [(('RP_Pos','na'),'[M+Na]+'),(('RP_Neg','neg'),'[M-H]-'),
                     (('NP_Pos','np'),'[M+H]+'),(('NP_Neg','npneg'),'[M+HCOO]-')]:
        assert table.loc[key,'compound_id'] == family
        assert table.loc[key,'adduct'] == ion
    assert table.loc[('RP_Pos','a_isomer'),'compound_id'] != family
    assert table.loc[('RP_Pos','homologue'),'compound_id'] != family
    assert table.loc[('RP_Pos','iso'),'compound_id'] == family
    assert table.loc[('RP_Pos','iso'),'isotope_index'] == 1
    assert not out['ion_isotope_candidates'].envelope_verified.any()


def test_grouping_does_not_consume_p_values_significance_or_comparison_thresholds():
    result = fixture()
    before = build_full_feature_groups(result, criteria())
    changed = replace(criteria(), p_value_threshold=1e-8, min_abs_log2_fold_change=15., group_a='differentA', group_b='differentB')
    # Grouping uses all eligible biological specimens, not the selected contrast.
    after = build_full_feature_groups(result, changed)
    pd.testing.assert_frame_equal(before['ion_feature_ledger'], after['ion_feature_ledger'])
    for item in result['streams'].values():
        item['features'] = item['features'].iloc[:, ::-1]
    shuffled = build_full_feature_groups(result, criteria())
    pd.testing.assert_frame_equal(before['ion_feature_ledger'], shuffled['ion_feature_ledger'])


def test_profile_matching_excludes_qc_and_rejects_shared_group_effect():
    result = fixture()
    left = FeatureIndex('RP_Pos', result['streams']['RP_Pos']['features'])
    frame = result['streams']['NP_Pos']['features'].copy()
    sample = frame.attrs['sample_columns'][:-1]
    for s, value in zip(sample, [8.,4.,2.,1.,160.,80.,40.,20.]):
        frame[s] = value
    right = FeatureIndex('NP_Pos', frame)
    n, _, within, _ = profile_metrics(left, right, np.array([0]), np.array([0]), paired_columns(left,right))
    assert n[0] == 8
    assert within[0] < -.9


def test_bounded_union_does_not_follow_a_neutral_mass_chain():
    result = fixture()
    frame = result['streams']['RP_Pos']['features'].iloc[:3].copy()
    frame['feature_id'] = ['a','b','c']
    frame['rt'] = 5.
    index = FeatureIndex('RP_Pos',frame)
    groups = BoundedFamilies({'RP_Pos':index},criteria())
    assert groups.join(0,1,'[M+H]+','[M+Na]+',700.,700.002,1.) == 'selected'
    assert groups.join(1,2,'[M+Na]+','[M+K]+',700.002,700.004,1.) == 'neutral_mass_span'
    assert groups.root(0) != groups.root(2)


def test_peak_width_and_explicit_disabled_ions_constrain_full_search():
    result = fixture()
    frame = result['streams']['RP_Pos']['features']
    frame['rtmin'], frame['rtmax'] = frame.rt-.001, frame.rt+.001
    method = default_acquisition_method()
    method['methods']['RP']['adducts']['pos']['[M+Na]+'] = 'disabled'
    out = build_full_feature_groups(result,criteria(),method)
    edges = out['ion_relationships']
    assert not ((edges.source_stream.eq('RP_Pos') & edges.source_adduct.eq('[M+Na]+')) |
                (edges.target_stream.eq('RP_Pos') & edges.target_adduct.eq('[M+Na]+'))).any()
    same = edges[edges.source_stream.eq('RP_Pos') & edges.target_stream.eq('RP_Pos')]
    assert same.empty or same.rt_limit_seconds.le(.3+1e-8).all()


def test_no_significant_hits_still_builds_a_complete_reopenable_ledger(tmp_path):
    result = fixture()
    a = analyse_standard_untargeted(result, replace(criteria(),p_value_threshold=1e-15))
    assert a['raw_p_features'].empty
    assert len(a['ion_feature_ledger']) == 8
    assert a['compound_features'].empty
    # Accepted Na is excluded from the biological tests, retained for ion evidence.
    assert 'na' not in set(a['features'].feature_id)
    assert 'na' in set(a['ion_feature_ledger'].feature_id)
    save_untargeted_workflow(tmp_path,result,a)
    loaded,_ = load_untargeted_workflow(tmp_path)
    pd.testing.assert_frame_equal(a['ion_feature_ledger'],loaded['untargeted_analysis']['ion_feature_ledger'])
    assert loaded['untargeted_analysis']['criteria']['ion_grouping_mode'] == 'full_feature_soft'


def test_duplicate_original_feature_keys_are_never_silently_dropped():
    frame = pd.DataFrame({'feature_id':['same','same'],'mz':[500.,501.],'rt':[1.,1.]})
    with pytest.raises(ValueError,match='Duplicate'):
        FeatureIndex('RP_Pos',frame)


def test_soft_profiles_accept_moderate_within_group_evidence_without_hard_gate():
    x = np.tile(np.array([-3., -1., 1., 3.])/np.sqrt(5),2)
    orthogonal = np.tile([1., -1., -1., 1.],2)
    group = np.repeat([0., 8.],4)
    y = .1*x + np.sqrt(.99)*orthogonal
    frame = pd.DataFrame({'feature_id':['h','na'],'mz':[701.007276466621,722.989220702],'rt':[5.,5.]})
    samples = [f's{i}' for i in range(8)]
    for i,s in enumerate(samples):
        frame[s] = [np.exp(8+group[i]+x[i]),np.exp(8+group[i]+y[i])]
    frame.attrs.update(sample_columns=samples,sample_group_map={s:'A' if i<4 else 'B' for i,s in enumerate(samples)})
    result = {'streams':{'RP_Pos':{'features':frame}}}
    hard = build_full_feature_groups(result,criteria('full_feature_hard'))
    soft = build_full_feature_groups(result,criteria())
    assert hard['ion_feature_ledger'].compound_id.nunique() == 2
    assert soft['ion_feature_ledger'].compound_id.nunique() == 1
    cached = {**result,'_ion_candidate_cache':(hard['ion_relationships'],hard['ion_isotope_candidates']),
        '_ion_cache_provenance':{key:hard[key] for key in ('ion_group_source_signature','ion_group_settings')}}
    reused = build_full_feature_groups(cached,criteria())
    pd.testing.assert_frame_equal(soft['ion_feature_ledger'],reused['ion_feature_ledger'])


def test_full_feature_detached_worker_saves_all_ions_and_frozen_settings(tmp_path):
    from application.untargeted_jobs import start_untargeted_job
    from test_annotation_jobs import wait_job
    job = start_untargeted_job(tmp_path,criteria(),result=fixture())
    state = wait_job(job)
    assert state['status'] == 'complete', (job/'untargeted.log').read_text()
    result,_ = load_untargeted_workflow(tmp_path)
    analysis = result['untargeted_analysis']
    assert len(analysis['ion_feature_ledger']) == 8
    assert analysis['criteria']['ion_grouping_mode'] == 'full_feature_soft'
    assert analysis['ion_group_source_signature']
    assert not analysis['ion_relationships'].empty
    assert analysis['ion_feature_ledger'].feature_id.duplicated().sum() == 0
