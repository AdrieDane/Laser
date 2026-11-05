"""Shared compound/cluster labels and browser-local linked display behaviour."""
from pathlib import Path
import json
import shutil
import subprocess

import pandas as pd
import pytest

from presentation.untargeted_mode import build_mode_payload
from standard_lipidomics.untargeted_compounds import assign_compound_ids
from standard_lipidomics.project_store import save_untargeted_workflow, load_untargeted_workflow

ROOT = Path(__file__).resolve().parents[1]


def example():
    proton = 1.007276466621
    mass = 678.5068
    neutral = mass-proton
    ch2 = 14.015650064
    coordinates = {
        'RP_Pos': [('a',mass,10.5),('na',neutral+22.989220702,10.5),('c',mass+ch2,10.55),
                   ('cs',mass+2*ch2,10.6),('u',mass+5*ch2,10.65),('bg',1100.,13.)],
        'RP_Neg': [('neg',neutral-proton,10.5),('cl',neutral+34.969401262,10.5),
                   ('cn',neutral+ch2-proton,10.55),('csneg',neutral+2*ch2-proton,10.6)],
        'NP_Pos': [('np',mass,5.3),('npc',mass+ch2,5.4)],
        'NP_Neg': [('npneg',neutral-proton,5.3)],
    }
    streams, rows = {}, []
    hit_keys = {('RP_Pos','a'),('RP_Pos','cs'),('RP_Pos','u')}
    for stream, entries in coordinates.items():
        features=pd.DataFrame(entries,columns=['feature_id','mz','rt'])
        features['control']=10.
        features['case']=100.
        features.attrs.update(sample_columns=['control','case'],sample_group_map={'control':'Control','case':'Case'})
        streams[stream]={'features':features}
        for feature,mz,rt in entries:
            hit=(stream,feature) in hit_keys
            rows.append({'stream':stream,'feature_id':feature,'mz':mz,'rt':rt,'p_value':.0001 if hit else .4,
                         'log2_fold_change_b_vs_a':2.,'raw_p_significant':hit,'significant':hit,'isotope_charge_hypothesis':1})
    tested=pd.DataFrame(rows)
    raw=tested[tested.raw_p_significant].copy()
    links=[]
    for source_stream,source_id,target_stream,target_id,source_adduct,target_adduct in [
        ('RP_Pos','a','RP_Pos','na','[M+H]+','[M+Na]+'),
        ('RP_Pos','a','RP_Neg','neg','[M+H]+','[M-H]-'),
        ('RP_Pos','a','RP_Neg','cl','[M+H]+','[M+Cl]-'),
        ('RP_Pos','a','NP_Pos','np','[M+H]+','[M+H]+'),
        ('RP_Pos','a','NP_Neg','npneg','[M+H]+','[M-H]-'),
        ('RP_Pos','cs','RP_Neg','csneg','[M+H]+','[M-H]-')]:
        source=tested[tested.stream.eq(source_stream)&tested.feature_id.eq(source_id)].iloc[0]
        target=tested[tested.stream.eq(target_stream)&tested.feature_id.eq(target_id)].iloc[0]
        links.append({'source_stream':source_stream,'source_feature_id':source_id,'source_mz':source.mz,'source_rt':source.rt,
                      'source_adduct':source_adduct,'neutral_mass':source.mz-proton,'target_stream':target_stream,
                      'target_feature_id':target_id,'target_mz':target.mz,'target_rt':target.rt,'target_adduct':target_adduct,
                      'mass_error_ppm':0.,'rt_delta_seconds':(target.rt-source.rt)*60,
                      'rt_compared':source_stream[:2]==target_stream[:2],'target_is_annotated':False})
    homologous=pd.DataFrame([{'stream':'RP_Pos','source_feature_id':'a','target_feature_id':target,'ch2_steps':step,
                             'mass_error_ppm':0.,'rt_delta_seconds':step*3.,'target_lipid_classes':''}
                            for target,step in [('c',1),('cs',2)]])
    homologous=pd.concat([homologous,pd.DataFrame([{'stream':'RP_Pos','source_feature_id':'c','target_feature_id':'u',
        'ch2_steps':4,'mass_error_ppm':0.,'rt_delta_seconds':6.,'target_lipid_classes':''}])],ignore_index=True)
    analysis={'features':tested,'raw_p_features':raw,'significant_features':raw.copy(),'adduct_matches':pd.DataFrame(links),
              'homologous_matches':homologous,'criteria':{'group_a':'Control','group_b':'Case','cross_mode_ppm_tolerance':3.,
              'opposite_polarity_rt_tolerance_min':.1},'by_stream':{s:f.copy() for s,f in tested.groupby('stream')}}
    return {'streams':streams,'automatic_annotations':pd.DataFrame()},analysis


def test_compound_ids_link_adducts_and_modes_but_keep_cluster_compounds_distinct():
    _,analysis=example()
    output=assign_compound_ids(analysis)
    rows=output['compound_features'].set_index(['stream','feature_id'])
    compound=rows.loc[('RP_Pos','a'),'compound_id']
    assert compound.startswith('compound_')
    for key in [('RP_Pos','na'),('RP_Neg','neg'),('RP_Neg','cl'),('NP_Pos','np'),('NP_Neg','npneg')]:
        assert rows.loc[key,'compound_id']==compound
    assert rows.loc[('RP_Pos','cs'),'compound_id']!=compound
    cluster=rows.loc[('RP_Pos','a'),'cluster_id']
    assert cluster.startswith('cluster_')
    assert rows.loc[('RP_Pos','cs'),'cluster_id']==cluster
    assert rows.loc[('NP_Pos','np'),'cluster_id']==cluster
    assert ('RP_Pos','c') in set(zip(output['cluster_members'].stream,output['cluster_members'].feature_id))
    assert rows.loc[('RP_Pos','u'),'cluster_id']==cluster
    assert rows.loc[('RP_Neg','cl'),'adduct']=='[M+Cl]-'


def test_grouping_does_not_merge_same_mass_rt_isomers_in_other_chromatography():
    _,analysis=example()
    duplicate=analysis['adduct_matches'].query("target_feature_id == 'np'").iloc[0].copy()
    duplicate['target_feature_id']='np_isomer'
    duplicate['target_rt']=9.
    duplicate['mass_error_ppm']=.5
    analysis['adduct_matches']=pd.concat([analysis['adduct_matches'],duplicate.to_frame().T],ignore_index=True)
    output=assign_compound_ids(analysis)
    assert 'np_isomer' not in set(output['compound_features'].feature_id)


def test_labels_are_deterministic_and_saved_with_the_project(tmp_path):
    result,analysis=example()
    first=assign_compound_ids(analysis)
    shuffled={**analysis,'adduct_matches':analysis['adduct_matches'].sample(frac=1,random_state=4),
              'raw_p_features':analysis['raw_p_features'].sample(frac=1,random_state=7)}
    second=assign_compound_ids(shuffled)
    columns=['stream','feature_id','compound_id','cluster_id','adduct']
    pd.testing.assert_frame_equal(first['compound_features'][columns].sort_values(columns).reset_index(drop=True),
                                  second['compound_features'][columns].sort_values(columns).reset_index(drop=True))
    save_untargeted_workflow(tmp_path,result,first)
    opened,_=load_untargeted_workflow(tmp_path)
    assert not opened['untargeted_analysis']['compound_features'].empty
    assert not opened['untargeted_analysis']['cluster_members'].empty
    assert opened['untargeted_analysis']['raw_p_features'].compound_id.notna().all()


def test_browser_model_colours_modes_background_and_sample_levels(tmp_path):
    node=shutil.which('node')
    bundled=Path('/Applications/ChatGPT.app/Contents/Resources/cua_node/bin/node')
    node=node or (str(bundled) if bundled.is_file() else None)
    if not node:
        pytest.skip('Node runtime required for browser-model checks')
    result,analysis=example()
    result['automatic_annotations']=pd.DataFrame([{'stream':'RP_Pos','feature_id':'c','name':'PC(36:2)','adduct':'[M+H]+'}])
    payload=build_mode_payload(result,analysis)
    (tmp_path/'payload.json').write_text(json.dumps(payload,allow_nan=False))
    source=ROOT/'presentation/untargeted_mode'
    (tmp_path/'viewer.mjs').write_text((source/'model.js').read_text()+'\n'+(source/'viewer.js').read_text())
    script=r'''
import {readFileSync} from 'node:fs';
import {ModeViewerModel,featureKey,modeFigure,sampleFigure,sampleFigures,linkedModeZoom,axisResetZoom,axisFromClasses} from './viewer.mjs';
const data=JSON.parse(readFileSync('./payload.json'));
const model=new ModeViewerModel(data), key=(s,id)=>featureKey(s,id), selected=key('RP_Pos','a'), context=model.context(selected);
const check=(condition,message)=>{if(!condition) throw Error(message);};
check(model.style(selected,context)==='focus','selected green');
check(model.style(key('RP_Neg','neg'),context)==='focus','negative counterpart red');
check(model.style(key('NP_Pos','np'),context)==='focus','NP counterpart red');
for(const point of [key('RP_Pos','na'),key('RP_Neg','cl')]) check(model.style(point,context)==='adduct','additional adduct yellow');
for(const point of [key('RP_Pos','c'),key('RP_Neg','cn'),key('NP_Pos','npc')]) check(model.style(point,context)==='cluster','non-significant cluster and counterparts purple');
check(model.style(key('RP_Pos','cs'),context)==='significantCluster','significant cluster orange with purple border');
check(model.style(key('RP_Pos','u'),context)==='significantUp','unrelated hit keeps significance colour');
const state={model,selected,view:'RP POS and NEG',grey:false,canvas:'#fff',ink:'#111',grid:'#ddd'};
const initial=modeFigure({...state,selected:null});
check(initial.layout.dragmode==='pan','click and drag pans by default');
check(modeFigure({...state,dragMode:'zoom'}).layout.dragmode==='zoom','zoom toolbar choice retained');
check(initial.data.filter(t=>!t.name.startsWith('Significantly')).every(t=>t.x.length===0),'initial plot contains only significant points');
let figure=modeFigure(state);
check(figure.data.every(t=>t.marker.symbol==='circle'&&t.marker.size===(t.name==='Selected / possible counterpart'?12:8)),'circles equal except larger selection');
const annotated=figure.data.flatMap(t=>t.customdata).find(row=>row[0]===key('RP_Pos','c'));
check(annotated[2]==='c'&&annotated[4].startsWith('compound_')&&annotated[7].startsWith('cluster_'),'feature, compound and cluster IDs in hover');
check(annotated[8]==='Lipid: PC(36:2) [M+H]+','accepted lipid ID in hover');
check(figure.data.find(t=>t.name==='Other adduct candidate').marker.color==='#f2c230','adducts remain yellow');
check(figure.data.find(t=>t.name==='Selected / possible counterpart').marker.color==='#00e51b','selection bright green');
const clusterMarker=figure.data.find(t=>t.name==='Significant cluster member').marker;
check(clusterMarker.color==='#f28c28'&&clusterMarker.line.color==='#8b5cf6'&&clusterMarker.line.width===2,'significant cluster orange with purple border');
model.rows.get(key('RP_Pos','u')).log2_fold_change_b_vs_a=-2;
check(model.style(key('RP_Pos','u'),context)==='significantDown','downregulated significant blue');
model.rows.get(key('RP_Pos','u')).log2_fold_change_b_vs_a=2;
check(figure.data.find(t=>t.name==='Not significant').x.length===0,'grey off by default');
check(figure.data.flatMap(t=>t.customdata).some(row=>row[0]===key('RP_Pos','c')),'related non-hit visible with grey off');
check(!figure.data.flatMap(t=>t.customdata).some(row=>row[1].startsWith('NP')),'RP view isolated');
state.grey=true;figure=modeFigure(state);
check(figure.data.find(t=>t.name==='Not significant').x.length>0,'grey toggle on');
state.view='All 4 modes';figure=modeFigure(state);
check(figure.layout.yaxis4,'all four modes have separate panels');
check(figure.data.some(t=>t.yaxis==='y4'&&t.customdata.some(row=>row[1]==='NP_Neg')),'NP negative is present');
const sample=sampleFigure(state);
check(sample.layout.height===280,'same bar height');
check(sample.data.flatMap(t=>t.y).includes(100),'original sample values');
const bars=sampleFigures(state);
check(bars.length===4,'automatic sample plot per detected mode');
check(new Set(bars.map(item=>item.mode)).size===4,'one plot per mode');
check(bars.every(item=>item.figure.layout.height===280),'all sample plots same height');
check(bars.every(item=>item.figure.layout.title.text.includes(item.mode.replace('_',' '))),'mode in each title');
const zoom=linkedModeZoom({'xaxis.range[0]':9,'xaxis.range[1]':11,'yaxis.range[0]':600,'yaxis.range[1]':800},figure.layout);
check(JSON.stringify(zoom['xaxis2.range'])==='[9,11]','RT linked within RP');
check(JSON.stringify(zoom['yaxis2.range'])==='[-800,-600]','mirrored RP mass zoom');
check(!Object.keys(zoom).some(key=>/axis[34]/.test(key)),'RP zoom leaves NP independent');
check(axisFromClasses(['text','x2tick'])==='xaxis2','tick double click identifies axis');
check(axisFromClasses(['nsdrag','x3y3'])==='yaxis3','drag band identifies mass axis');
const reset=axisResetZoom('yaxis4',figure.layout);
check(reset.phase==='NP'&&reset.dimension==='mz','axis reset limited to dimension and chromatography');
check(JSON.stringify(reset.range)===JSON.stringify(figure.layout.meta.pairs[1].defaultMZ),'reset to original mass extent');
const reverse=linkedModeZoom({'yaxis4.range':[-900,-500]},figure.layout);
check(JSON.stringify(reverse['yaxis3.range'])==='[500,900]','NP negative zoom mirrors into NP positive');
check(!Object.keys(reverse).some(key=>/^.[a-z]*axis2?\./.test(key)),'NP zoom leaves RP independent');
// Stale saved evidence cannot highlight remote RT isomers as adducts or cluster members.
const rp=model.streams.get('RP_Pos');
const naIndex=rp.ids.indexOf('na'),cIndex=rp.ids.indexOf('c');
rp.rt[naIndex]=11.5;rp.rt[cIndex]=9.;
model.contextCache.clear();
const strict=model.context(selected);
check(!strict.adducts.has(key('RP_Pos','na')),'distant same-mode adduct hidden');
check(!strict.cluster.has(key('RP_Pos','c')),'distant cluster neighbour hidden');
console.log('browser model checks passed');
'''
    completed=subprocess.run([node,'--input-type=module'],input=script,cwd=tmp_path,capture_output=True,text=True,check=False)
    assert completed.returncode==0,completed.stderr


def test_cross_mode_links_cannot_merge_distant_adducts_within_one_mode():
    _, analysis = example()
    # A source in the other chromatography sees two equal-neutral-mass ions,
    # but the delayed ion cannot be an adduct of the main coeluting family.
    row = analysis['adduct_matches'].query("target_feature_id == 'np'").iloc[0].copy()
    row['target_feature_id'] = 'np_delayed'
    row['target_adduct'] = '[M+Na]+'
    row['target_mz'] = float(row.target_mz) - 1.007276466621 + 22.989220702
    row['target_rt'] = 5.38
    analysis['adduct_matches'] = pd.concat([analysis['adduct_matches'], row.to_frame().T], ignore_index=True)
    output = assign_compound_ids(analysis)
    assert 'np_delayed' not in set(output['compound_features'].feature_id)
