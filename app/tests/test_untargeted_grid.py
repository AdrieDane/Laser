"""Carbon/double-bond grids need occupied, RT-consistent connections."""
import pandas as pd
from standard_lipidomics.untargeted_grid import LipidGridIndex, CH2, H2, search_lipid_grids
from standard_lipidomics.untargeted import UntargetedComparisonCriteria
from standard_lipidomics.project_store import save_untargeted_workflow, load_untargeted_workflow


def grid(phase='RP'):
    c_rt,h_rt=(.16,.30) if phase=='RP' else (.001,.002)
    rows=[{'feature_id':f'{c},{h}','mz':900.74283+c*CH2+h*H2,'rt':14+c*c_rt+h*h_rt,'already_annotated':False}
          for c in range(-3,4) for h in range(-2,3)]
    return pd.DataFrame(rows).sort_values('mz').reset_index(drop=True)


def test_double_bond_and_mixed_steps_extend_beyond_the_first_ring():
    rows=LipidGridIndex(grid()).family('0,0')
    ids={row['target_feature_id'] for row in rows}
    assert {'0,-1','0,1','2,-1','3,2','-3,-2'} <= ids
    assert len(ids)==34
    assert all(row['grid_size']==35 for row in rows)


def test_empty_carbon_column_and_distant_rt_isomer_do_not_bridge():
    frame=grid()
    frame=frame[~frame.feature_id.str.startswith('-1,')].copy()
    isomers=grid();isomers['feature_id']='isomer_'+isomers.feature_id;isomers['rt']-=1.5
    frame=pd.concat([frame,isomers],ignore_index=True).sort_values('mz')
    ids={row['target_feature_id'] for row in LipidGridIndex(frame).family('0,0')}
    assert ids
    assert not any(fid.startswith(('isomer_','-2,','-3,')) for fid in ids)


def test_a_single_ch2_series_does_not_invent_a_two_axis_grid():
    frame=grid();frame=frame[frame.feature_id.str.endswith(',0')]
    assert LipidGridIndex(frame).family('0,0')==[]


def test_flat_np_grids_are_supported_and_saved(tmp_path):
    source=grid('NP')
    hits=pd.DataFrame([{'feature_id':'0,0','mz':900.74283,'rt':14.,'raw_p_significant':True,'isotope_charge_hypothesis':1}])
    links=search_lipid_grids({'NP_Pos':hits},{'NP_Pos':source},UntargetedComparisonCriteria('A','B'))
    assert len(links)==34
    result={'streams':{'NP_Pos':{'features':source}},'automatic_annotations':pd.DataFrame()}
    analysis={'features':hits.assign(stream='NP_Pos'),'significant_features':pd.DataFrame(),'raw_p_features':hits.assign(stream='NP_Pos'),
        'lipid_grid_matches':links,'lipid_grid_version':1,'criteria':{'group_a':'A','group_b':'B'}}
    save_untargeted_workflow(tmp_path,result,analysis)
    saved,_=load_untargeted_workflow(tmp_path)
    assert saved['untargeted_analysis']['lipid_grid_version']==1
    assert len(saved['untargeted_analysis']['lipid_grid_matches'])==34
