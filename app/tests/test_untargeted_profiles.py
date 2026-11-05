import numpy as np
import pandas as pd

from standard_lipidomics.untargeted_profiles import link_profile_evidence
from standard_lipidomics.untargeted_formula_labels import add_formula_labels
from standard_lipidomics.untargeted_compounds import assign_compound_ids


def test_profiles_pair_specimens_and_reject_shared_group_effect_without_qc():
    streams = {}
    values = [1., 2., 4., 8., 100., 200., 400., 800.]
    for mode in ['RP_Pos', 'NP_Pos']:
        samples = [mode+str(i) for i in range(8)]
        frame = pd.DataFrame({'feature_id': ['source', 'good', 'confound'], 'mz': [500., 500., 500.], 'rt': [1., 2., 3.]})
        for i, sample in enumerate(samples):
            frame[sample] = [values[i], values[i]*2, values[3-i] if i < 4 else values[11-i]]
        frame['qc'] = [1e9, 1., 1e9]
        frame.attrs.update(sample_columns=samples+['qc'],
            sample_group_map={**{s:'A' if i < 4 else 'B' for i,s in enumerate(samples)}, 'qc':'QC_pool'},
            sample_metadata_records={s:{'Sample ID':str(i)} for i,s in enumerate(samples)})
        streams[mode] = {'features':frame}
    links = pd.DataFrame([{'source_stream':'RP_Pos','source_feature_id':'source',
                           'target_stream':'NP_Pos','target_feature_id':target} for target in ['good', 'confound']])
    out = link_profile_evidence({'streams':streams}, {'criteria':{'group_a':'A','group_b':'B'}}, links)
    assert list(out.paired_sample_count) == [8, 8]
    assert list(out.profile_status) == ['supported', 'discordant']
    assert out.iloc[0].within_group_correlation > .99
    assert out.iloc[1].within_group_correlation < -.9
    assert out.pairing_method.eq('Excel metadata: Sample ID').all()
    # Column order never substitutes for specimen metadata.
    streams['NP_Pos']['features'] = streams['NP_Pos']['features'].iloc[:, ::-1]
    shuffled = link_profile_evidence({'streams':streams}, {'criteria':{'group_a':'A','group_b':'B'}}, links)
    np.testing.assert_allclose(out.within_group_correlation, shuffled.within_group_correlation)


def test_formula_labels_follow_assigned_adduct_and_do_not_invent_missing_predictions():
    compounds = pd.DataFrame([
        {'stream':'RP_Pos','feature_id':'a','compound_id':'compound_1','adduct':'[M+H]+'},
        {'stream':'NP_Neg','feature_id':'b','compound_id':'compound_1','adduct':'[M+HCOO]-'},
        {'stream':'NP_Pos','feature_id':'c','compound_id':'compound_2','adduct':''},
    ])
    raw = pd.DataFrame([{'stream':'RP_Pos','feature_id':'a','chemical_composition_hypothesis':
                        'candidate formulas: [M+Na]+ C10H20O2 (+1.0 ppm; DBE 1); [M+H]+ C12H24O2 (-0.2 ppm; DBE 1)'}])
    output = add_formula_labels(compounds, raw)
    assert list(output.predicted_chemical_formula) == ['C12H24O2', 'C12H24O2', '']
    assert output.iloc[1].formula_status == 'tentative linked-compound candidate'
    conflict = pd.concat([raw, pd.DataFrame([{'stream':'NP_Neg','feature_id':'b','chemical_composition_hypothesis':
                                            '[M+HCOO]- C11H22O2 (+0.1 ppm; DBE 1)'}])], ignore_index=True)
    assert add_formula_labels(compounds, conflict).iloc[0].formula_status == 'conflicting compound candidates'


def test_water_loss_without_same_mode_parent_remains_unresolved():
    raw = pd.DataFrame([{'stream':'RP_Pos','feature_id':'a','mz':500.,'rt':1.,'p_value':.0001,'log2_fold_change_b_vs_a':2.}])
    links = pd.DataFrame([{'source_stream':'RP_Pos','source_feature_id':'a','source_mz':500.,'source_rt':1.,
                          'source_adduct':'[M+H-H2O]+','neutral_mass':517.003288,
                          'target_stream':'NP_Pos','target_feature_id':'b','target_mz':500.,'target_rt':3.,
                          'target_adduct':'[M+H-H2O]+','mass_error_ppm':0.,'target_is_annotated':False}])
    result = assign_compound_ids({'raw_p_features':raw,'features':raw,'adduct_matches':links})
    assert len(result['compound_features']) == 1
    assert result['compound_features'].iloc[0].adduct == ''
    assert result['compound_features'].iloc[0].association_status == 'adduct unresolved'
