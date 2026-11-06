"""Original-injection scan evidence, uncertainty and saved-project invariance."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from standard_lipidomics.trace_evidence import (
    trace_quality, shape_pair, candidate_pairs, collect_trace_evidence,
    annotate_trace_links, digest, freeze_sources,
)
from standard_lipidomics.untargeted import UntargetedComparisonCriteria
from standard_lipidomics.untargeted_compounds import assign_compound_ids, LABEL_COLUMNS
from test_untargeted_mode_view import example


def curve(center=.5, width=.04, scale=1e5):
    rt=np.arange(.2,.801,.01)
    y=scale*np.exp(-.5*((rt-center)/width)**2)
    quality,corrected=trace_quality(rt,y,np.ones(len(rt)),center-.15,center+.15)
    return {'scan':np.arange(len(rt)),'rt':rt,'intensity':y,'corrected':corrected,'quality':quality}


def test_shape_is_measured_on_actual_scans_and_is_scale_independent():
    a,b=curve(),curve(scale=31.)
    assert a['quality']['status']=='usable'
    measured=shape_pair(a,b)
    assert measured['status']=='measured'
    assert measured['shape_r']==pytest.approx(1.)
    assert measured['overlap']==pytest.approx(1.)


def test_shifted_peaks_are_not_lag_aligned_to_create_agreement():
    result=shape_pair(curve(.45,.025),curve(.55,.025))
    assert result['status']=='measured'
    assert result['shape_r'] < .5
    assert result['overlap'] < .25


@pytest.mark.parametrize('kind',['sparse','flat','multi_centroid','acquisition_gap','two_peaks'])
def test_unreliable_traces_cannot_supply_positive_evidence(kind):
    rt=np.arange(.2,.801,.01)
    y=1e5*np.exp(-.5*((rt-.5)/.04)**2)
    counts=np.ones(len(rt))
    if kind=='sparse':
        rt,y,counts=rt[::20],y[::20],counts[::20]
    if kind=='flat':
        y[:]=3.
    if kind=='multi_centroid':
        counts[np.argmax(y)]=2
    if kind=='acquisition_gap':
        rt[40:]+=.2
    if kind=='two_peaks':
        y=1e5*(np.exp(-.5*((rt-.4)/.02)**2)+np.exp(-.5*((rt-.6)/.02)**2))
    quality,_=trace_quality(rt,y,counts,.3,.7)
    assert quality['status']!='usable'


def test_no_common_apex_is_unavailable_not_interpolated():
    a,b=curve(),curve()
    b['scan']+=10000
    assert shape_pair(a,b)['status']=='not evaluable; no common scans'


def test_mode_identity_and_relationship_types_are_preserved():
    _,analysis=example()
    pairs=candidate_pairs(analysis)
    assert pairs.stream.str.startswith(('RP','NP')).all()
    assert not pairs.duplicated().any()
    linked=annotate_trace_links(analysis['adduct_matches'],pd.DataFrame())
    cross=linked[linked.source_stream.ne(linked.target_stream)]
    assert cross.trace_status.eq('different injections; no shape comparison').all()


def test_missing_trace_metadata_keeps_current_labels_and_statistics():
    result,analysis=example()
    baseline=assign_compound_ids(analysis,result=result)
    configured={**analysis,'criteria':{**analysis['criteria'],'trace_model':'peak_shape'}}
    enriched=collect_trace_evidence(result,configured,{'available':False,'reason':'no native RT mapping'})
    actual=assign_compound_ids(enriched,result=result)
    pd.testing.assert_frame_equal(baseline['compound_features'],actual['compound_features'])
    for name in ['features','raw_p_features','significant_features']:
        columns=[c for c in baseline[name] if c not in LABEL_COLUMNS]
        pd.testing.assert_frame_equal(baseline[name][columns],actual[name][columns])
    assert actual['trace_provenance']['status']=='unavailable'


def test_trace_criteria_defaults_and_untested_combinations():
    criteria=UntargetedComparisonCriteria('A','B')
    assert criteria.trace_model=='current'
    assert replace(criteria,trace_model='peak_shape').trace_model=='peak_shape'
    with pytest.raises(ValueError,match='requires current'):
        replace(criteria,trace_model='peak_shape',ion_grouping_mode='full_feature_soft')


def test_complete_two_injection_collection_and_source_drift(tmp_path,monkeypatch):
    raw=tmp_path/'Raw'
    raw.mkdir()
    (raw/'one.mzXML').write_text('synthetic source one')
    (raw/'two.mzXML').write_text('synthetic source two')
    table=tmp_path/'inputs.txt'
    table.write_text('original synthetic coordinates')
    bounds=tmp_path/'bounds.csv'
    pd.DataFrame([{'stream':'RP_Pos','feature_id':fid,'sample':sample,'target_mz':mz,
        'rt_start_corrected_min':.35,'rt_apex_corrected_min':.5,'rt_end_corrected_min':.65,'filled':False}
        for sample in ['one','two'] for fid,mz in [('a',500.),('b',522.)]]).to_csv(bounds,index=False)
    manifest={'schema_version':1,'peak_rt_basis':'corrected','streams':{'RP_Pos':{
        'boundaries':'bounds.csv','input_table':'inputs.txt','input_md5':digest(table,'md5'),
        'samples':[{'sample':n,'raw_path':'Raw/'+n+'.mzXML','raw_rt_min':[0.,1.],
                    'corrected_rt_min':[0.,1.]} for n in ['one','two']]}}}
    metadata=tmp_path/'manifest.json'
    metadata.write_text(json.dumps(manifest))
    context=freeze_sources(tmp_path,metadata)
    assert context['available']
    frame=pd.DataFrame({'feature_id':['a','b'],'mz':[500.,522.],'rt':[.5,.5],'one':[10.,20.],'two':[12.,24.]})
    frame.attrs.update(sample_columns=['one','two'],sample_group_map={'one':'A','two':'B'})
    result={'streams':{'RP_Pos':{'features':frame}}}
    links=pd.DataFrame([{'source_stream':'RP_Pos','source_feature_id':'a','target_stream':'RP_Pos','target_feature_id':'b'}])
    analysis={'adduct_matches':links}
    def scans(path):
        trace=curve()
        for time,z in zip(trace['rt'],trace['intensity']):
            yield time,np.array([500.,522.]),np.array([z,2*z])
    monkeypatch.setattr('standard_lipidomics.trace_evidence._iter_ms1_scans',scans)
    out=collect_trace_evidence(result,analysis,context)
    assert out['trace_provenance']['status']=='complete'
    assert out['trace_pair_evidence'].trace_status.tolist()==['supported']
    assert out['trace_pair_evidence'].usable_samples.tolist()==[2]
    pd.testing.assert_frame_equal(out['adduct_matches'],links)
    (raw/'one.mzXML').write_text('changed source')
    assert collect_trace_evidence(result,analysis,context)['trace_provenance']['status']=='unavailable'


def test_saved_trace_viewer_never_reextracts_raw_files(tmp_path,monkeypatch):
    from presentation.untargeted_mode import build_mode_payload
    from standard_lipidomics.project_store import save_untargeted_workflow,load_untargeted_workflow
    result,analysis=example()
    analysis['criteria']={**analysis['criteria'],'trace_model':'peak_shape'}
    analysis=collect_trace_evidence(result,analysis,{'available':False,'reason':'synthetic'})
    analysis['trace_pair_evidence']=pd.DataFrame([{'stream':'RP_Pos','source_feature_id':'a','target_feature_id':'na',
        'kind':'ion','trace_status':'supported','usable_samples':2,'median_shape_r':.932143757230147,'median_overlap':.85}])
    analysis=assign_compound_ids(analysis,result=result)
    payload=build_mode_payload(result,analysis)
    save_untargeted_workflow(tmp_path,result,analysis)
    loaded,_=load_untargeted_workflow(tmp_path)
    monkeypatch.setattr('standard_lipidomics.trace_evidence.collect_trace_evidence',lambda *a,**k:pytest.fail('Viewer must use saved evidence'))
    assert build_mode_payload(loaded,loaded['untargeted_analysis'])==payload
    pd.testing.assert_frame_equal(loaded['untargeted_analysis']['trace_pair_evidence'],analysis['trace_pair_evidence'],check_exact=True)
